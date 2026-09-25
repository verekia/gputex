// bc7 encoder — f16 variant (requires the shader-f16 feature). The host
// selects this module when the device reports shader-f16, falling back to
// bc7.wgsl (the same algorithm in f32) otherwise.
//
// TWO BC7 MODES, chosen per block from the covariance and then encoded ONCE
// by the same per-pixel code:
//   • mode 6: one RGBA line, 16 levels, 7-bit + p-bit endpoints — the best
//     single-subset mode when the block's colours lie near one line.
//   • mode 4: "rotation": one channel (ch) becomes its own scalar plane with
//     6-bit endpoints at its exact extremes, the other three share a 5-bit
//     line; a 2-bit and a 3-bit index set, the 3-bit set going to whichever
//     plane carries more variance (idxMode). Decorrelated channels — normal
//     maps, channel-packed atlases, noisy chroma in photos, independent
//     alpha — are exactly where any single 4-D line fails.
//   Measured on the /eval corpus vs the mode-6-only encoder (hardware
//   decode, RGBA PSNR): rock colour +1.1..+2.2 dB, normal maps +2.4..+4.5,
//   packed atlases +0.1..+4.4, proc/cutout/alpha test images +2.6..+7.8;
//   gray maps unchanged (own tail below), wood colour ±0.03. MSE −33%
//   geomean over all 41 images. An exhaustive oracle (every mode-6/4/5
//   candidate encoded, exact error, best kept) sits only 0.1–0.9 dB higher.
//
// MODE DECISION (per block, no pixel pass): score = variance a candidate
// explains net of its quantisation, in the ×256 covariance units below:
//   mode 6:  λ·(1 − 1/225)                     (λ = principal eigenvalue)
//   mode 4:  (48/49)·max(λ3, σ²ch) + (8/9)·min(λ3, σ²ch) − N4
// with σ²ch the scalar channel's variance, λ3 the principal eigenvalue of
// the other three channels, N4 a flat charge for mode 4's coarser endpoint
// codes (tuned: 0.1 over-picks mode 4 on smooth wood, 0.3 under-picks it on
// normal maps). λ3 for all four candidate channels comes from ONE
// closed-form power step: with Ma = λa, M·a_m = λa − a_c·col_c for the axis
// restricted to the plane (a_m), and |M·a_m| / |a_m| bounds λ3 from below;
// the plane's largest variance is a second lower bound that takes over for
// the channel dominating the axis (a_m ≈ 0 there). Exact per-plane power
// iterations measured within 0.05 dB of this at +60% GPU. (A per-block
// per-candidate noise model from the actual 5/6-bit rounding errors of the
// bbox corners was within ±0.05 dB of the flat N4 — not worth its ALU.)
//
// PERFORMANCE DESIGN — both modes run the SAME instructions wherever
// possible, because content that benefits from mode 4 mixes it with mode 6
// in nearly every SIMD group (5–10% scattered coverage poisons most warps;
// the previous gated/opt-in mode 4 paid 1.4–1.5× for exactly that):
//   • Covariance over d = (px − pixel0)·16, stored back into the pixel
//     array (block-relative, so f16 sums don't cancel), 10 symmetric
//     products per pixel instead of 16.
//   • Power iteration on C/trace seeded with the largest-variance column
//     (a free step) and ONE step pair; λ = |M²a|^½ comes out of the pair's
//     normalisation. (The old bbox seed needed 8 iterations; a single step
//     loses 0.1–0.2 dB.) One further masked step refines the fit axis: mode
//     4 from its closed-form plane vector, mode 6 on its own axis.
//   • ONE extents pass along the fit axis keeps every pixel's projection t;
//     the indices come straight from those (decoded endpoints' positions
//     along the axis set the level scale — the off-axis part of endpoint
//     quantisation enters only at second order; exact projection onto the
//     quantised line measured ≤0.05 dB better). The same pass computes the
//     mode-4 scalar-plane indices (zero weights for mode 6).
//   • One quantiser for both endpoint formats (7-bit + p / 5-bit).
//   • The anchor rule is applied BEFORE indexing: pixel 0 is the d-space
//     origin, so its index is ⌊offset⌋ and a flip just mirrors the map — no
//     post-hoc index inversion in either packing path.
//   • Indices accumulate as float nibble fields (≤ 2^24, exact) for both
//     modes; mode 4 squeezes its nibbles into 2/3-bit fields at pack time.
//   Net: GPU time ×0.98 geomean vs the mode-6-only kernel over the corpus
//   (1K colour/normal textures ≤ +3%, 2K/4K ≈ par, low-mode-4 content
//   faster). All in f16 ([0,1] domain) except the float index fields.
//
//   • GRAY + opaque blocks (every texel R == G == B, A == 1) are a 1-D
//     problem and take their own tail in the integer domain (no power
//     iteration, no extents pass, no covariance — −34% GPU on roughness/
//     AO/displacement content vs the generic path):
//       – span ≤ 15: LOSSLESS. With integer endpoints ≤ 15 apart, mode 6's
//         rounded palette covers every integer between them and
//         round(15·(v − e0)/d) selects it (exhaustively verified, either
//         tie rounding). Even endpoints step outward while the span stays
//         ≤ 15 so both p-bits are 1 and alpha decodes to exactly 255 —
//         free for RGB (alpha = 254 + p is otherwise wrong on up to half
//         the texels, which capped RGBA PSNR at ~51 dB on smooth maps).
//       – span > 15: closed-form scalar LSQ refit off BC5-style moments
//         (ΣL, ΣL², Σv, ΣL·v of the seed levels), with all four p-bit
//         combinations priced INCLUDING the alpha term, accept-if-better.
//
// HISTORY: a MODE 1 (2-subset) candidate bought ~+1.3 dB on multi-modal
// content for up to ~3× the pass (2026-07, dropped). Mode 4 then shipped as
// an opt-in behind a pipeline constant with a variance-ratio gate and
// separate per-mode passes (+2..3 dB where used, 1.4–1.5× GPU from warp
// divergence); it is replaced by this always-on design. Mode 5 was priced
// by the oracle: +0.0–0.3 dB on real textures — not worth a third packing.
// The CPU reference decoder handles modes 1, 4 and 6 (bc7_ref.ts).
//
// MODE 6 BIT LAYOUT (LSB-first): see bc7.wgsl. Summary:
//   w0: mode(7 bits, 0x40) R0 R1 G0 G1[3:0]
//   w1: G1[6:4] B0 B1 A0 A1 P0
//   w2: P1, pixel0 index (3 bits), pixels 1..7 (4 bits each)
//   w3: pixels 8..15 (4 bits each)
// MODE 4 BIT LAYOUT (LSB-first): mode 0b00001, rotation @5 (channel swapped
// with alpha), idxMode @7 (0 = colour → 2-bit set, scalar → 3-bit set),
// colour endpoints 6×5 bits @8, alpha endpoints 2×6 @38, 31-bit 2-bit index
// field @50 (pixel 0 anchored to 1 bit), 47-bit 3-bit index field @81
// (pixel 0 anchored to 2 bits). Validated bit-exact against hardware
// bc7-rgba-unorm sampling; decode reference in bc7_ref.ts.
enable f16;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, y0: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
alias h = f16;
alias h4 = vec4<f16>;

// Mode-4 endpoint-precision charge (×256 covariance units; see header).
const N4: f16 = 0.15;

// Nibble-slot compaction for the mode-4 index fields: both index streams are
// accumulated 4 bits per pixel like mode 6's, then squeezed here.
// compact2: eight 2-bit values in nibbles → 16 bits; compact3: eight 3-bit
// values in nibbles → 24 bits.
fn compact2(x: u32) -> u32 {
  var y = (x | (x >> 2u)) & 0x0F0F0F0Fu;
  y = (y | (y >> 4u)) & 0x00FF00FFu;
  return (y | (y >> 8u)) & 0x0000FFFFu;
}
fn compact3(x: u32) -> u32 {
  var y = (x & 0x07070707u) | ((x >> 1u) & 0x38383838u);
  y = (y & 0x003F003Fu) | ((y >> 2u) & 0x0FC00FC0u);
  return (y & 0x00000FFFu) | ((y >> 4u) & 0x00FFF000u);
}

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid_raw: vec3<u32>) {
  // Row-band encodes dispatch a slice of the block grid starting at row y0.
  let gid = vec3<u32>(gid_raw.x, gid_raw.y + params.y0, gid_raw.z);
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);
  let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  // Load pass: texels, bbox and the gray test (edge blocks of odd-sized
  // images replicate the last row/column). The covariance is accumulated
  // only on the colour path below (gray blocks never use it).
  var pix: array<h4, 16>;
  var lo = h4(1.0);
  var hi = h4(0.0);
  var gd = h(0.0);
  let xs = min(vec4<i32>(base.x) + vec4<i32>(0, 1, 2, 3), vec4<i32>(mx.x));
  let ys = min(vec4<i32>(base.y) + vec4<i32>(0, 1, 2, 3), vec4<i32>(mx.y));
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let px = h4(textureLoad(src_tex, vec2<i32>(xs[i & 3u], ys[i >> 2u]), 0));
    pix[i] = px; lo = min(lo, px); hi = max(hi, px);
    gd = max(gd, max(abs(px.x - px.y), abs(px.x - px.z)));
  }

  if (lo.w == h(1.0) && gd == h(0.0)) {
    // GRAY + opaque: always mode 6, fully specialised tail (see header) —
    // gray textures are warp-uniform, and routing them through the shared
    // colour path measured +28% on displacement content. Integer domain: f16 holds k/255 to ±0.06 levels, so the
    // rounding recovers the exact 8-bit value. 8-bit endpoints E = 2q + p;
    // RGB share q, alpha is 254 + p.
    var gv: array<f32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) { gv[k] = floor(f32(pix[k].x) * 255.0 + 0.5); }
    let vmin = floor(f32(lo.x) * 255.0 + 0.5);
    let vmax = floor(f32(hi.x) * 255.0 + 0.5);
    var e0 = vmin;
    var e1 = vmax;
    if (vmax - vmin <= 15.0) {
      // LOSSLESS: with integer endpoints ≤ 15 apart, the rounded palette
      // covers every integer in [e0, e1] and round(15·(v − e0)/d) picks it
      // (exhaustively verified, either tie rounding). Odd endpoints keep
      // alpha exactly 255 (p = 1), so even ones step outward while the
      // span stays ≤ 15 — free for RGB.
      if (fract(e0 * 0.5) == 0.0 && e0 > 0.0 && e1 - e0 < 15.0) { e0 = e0 - 1.0; }
      if (fract(e1 * 0.5) == 0.0 && e1 < 255.0 && e1 - e0 < 15.0) { e1 = e1 + 1.0; }
    } else {
      // Closed-form scalar LSQ refit (BC5-style moments): seed levels
      // L = round(15·(v − vmin)/d) against the exact extremes, then the
      // endpoint pair minimising Σ(v − (1−t)e0 − t·e1)² (t = L/15), priced
      // for all four p-bit combinations INCLUDING the alpha channel
      // (254 + p vs 255) and accepted only if it beats the seed.
      let k1 = 15.0 / (vmax - vmin);
      let k0 = 0.5 - vmin * k1;
      var sL = 0.0;
      var sLL = 0.0;
      var sv = 0.0;
      var sLv = 0.0;
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let v = gv[k];
        let L = floor(v * k1 + k0);
        sL = sL + L;
        sLL = sLL + L * L;
        sv = sv + v;
        sLv = sLv + L * v;
      }
      let C = sLL * (1.0 / 225.0);
      let B = sL * (1.0 / 15.0) - C;
      let A = 16.0 - sL * (2.0 / 15.0) + C;
      let Y = sLv * (1.0 / 15.0);
      let X = sv - Y;
      let det = A * C - B * B;
      if (det > 1e-3) {
        let s0 = clamp((C * X - B * Y) / det, 0.0, 255.0);
        let s1 = clamp((A * Y - B * X) / det, 0.0, 255.0);
        // price(e0, e1, p0, p1) − Σv²·3, RGB ×3 plus alpha.
        let ps0 = vmin - 2.0 * floor(vmin * 0.5);
        let ps1 = vmax - 2.0 * floor(vmax * 0.5);
        var best = 3.0 * (A * vmin * vmin + 2.0 * B * vmin * vmax + C * vmax * vmax - 2.0 * (X * vmin + Y * vmax))
          + A * (1.0 - ps0) + 2.0 * B * (1.0 - ps0) * (1.0 - ps1) + C * (1.0 - ps1);
        for (var pc: u32 = 0u; pc < 4u; pc = pc + 1u) {
          let p0 = f32(pc & 1u);
          let p1 = f32(pc >> 1u);
          let c0 = 2.0 * clamp(floor((s0 - p0) * 0.5 + 0.5), 0.0, 127.0) + p0;
          let c1 = 2.0 * clamp(floor((s1 - p1) * 0.5 + 0.5), 0.0, 127.0) + p1;
          let pr = 3.0 * (A * c0 * c0 + 2.0 * B * c0 * c1 + C * c1 * c1 - 2.0 * (X * c0 + Y * c1))
            + A * (1.0 - p0) + 2.0 * B * (1.0 - p0) * (1.0 - p1) + C * (1.0 - p1);
          if (pr < best) {
            best = pr;
            e0 = c0;
            e1 = c1;
          }
        }
      }
    }
    // Index pass against the final endpoints: levels accumulate as float
    // nibble fields (≤ 2^24, exact), 6 + 6 + 4 pixels per accumulator.
    var ilo = 0u;
    var ihi = 0u;
    if (e1 != e0) {
      let k1 = 15.0 / (e1 - e0);
      let k0 = 0.5 - e0 * k1;
      var fa = 0.0;
      var fb = 0.0;
      var fc = 0.0;
      var w = 1.0;
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let sg = clamp(floor(gv[k] * k1 + k0), 0.0, 15.0);
        if (k < 6u) { fa = fa + sg * w; } else if (k < 12u) { fb = fb + sg * w; } else { fc = fc + sg * w; }
        w = select(w * 16.0, 1.0, k == 5u || k == 11u);
      }
      let ua = u32(fa);
      let ub = u32(fb);
      let uc = u32(fc);
      ilo = ua | (ub << 24u);
      ihi = (ub >> 8u) | (uc << 16u);
    }
    var u0 = u32(e0);
    var u1 = u32(e1);
    if ((ilo & 0x8u) != 0u) {
      let t = u0; u0 = u1; u1 = t;
      ilo = ~ilo; ihi = ~ihi;
    }
    let q0 = u0 >> 1u;
    let q1 = u1 >> 1u;
    let og = bi * 4u;
    dst[og] = 0x40u | (q0 << 7u) | (q1 << 14u) | (q0 << 21u) | (q1 << 28u);
    dst[og + 1u] = (q1 >> 4u) | (q0 << 3u) | (q1 << 10u) | (127u << 17u) | (127u << 24u) | ((u0 & 1u) << 31u);
    dst[og + 2u] = (u1 & 1u) | ((ilo & 0x7u) << 1u) | (ilo & 0xFFFFFFF0u);
    dst[og + 3u] = ihi;
    return;
  }

  // Covariance (10 symmetric products) over d = (px − p0)·16, stored back
  // into pix: every later pass works block-relative. The ×16 pre-scale
  // lifts shallow blocks (span ~1/255 → d² ≈ 4e-3) off the f16 subnormal
  // floor while full-range sums stay ≤ 4096. C = Σddᵀ − (Σd)(Σd)ᵀ/16 is the
  // ×256-scaled covariance.
  let p0v = pix[0];
  let p016 = p0v * h(16.0);
  pix[0] = h4(0.0);
  var sd = h4(0.0);
  var cx = h4(0.0);
  var cy = vec3<f16>(0.0);
  var cz = vec2<f16>(0.0);
  var cw = h(0.0);
  for (var i: u32 = 1u; i < 16u; i = i + 1u) {
    let d = pix[i] * h(16.0) - p016;
    pix[i] = d;
    sd = sd + d;
    cx = cx + d.x * d;
    cy = cy + d.y * d.yzw;
    cz = cz + d.z * d.zw;
    cw = cw + d.w * d.w;
  }
  let md = sd * h(1.0 / 16.0);
  // Mean correction via sd4 = Σd/4: every product ≤ 4096 (Σd·Σdᵀ could
  // reach 65536 and overflow f16).
  let sd4 = sd * h(0.25);
  cx = cx - sd4.x * sd4;
  cy = cy - sd4.y * sd4.yzw;
  cz = cz - sd4.z * sd4.zw;
  cw = cw - sd4.w * sd4.w;
  let diag = h4(cx.x, cy.x, cz.x, cw);
  let trace = diag.x + diag.y + diag.z + diag.w;

  // Mode decision + fit axis (see header). Flat blocks (trace ≈ 0) keep
  // axis 0: every index 0, both endpoints at the mean.
  var use4 = false;
  var idx1 = false;
  var ch = 0u;
  var cmask = h4(1.0);
  var axisF = h4(0.0);
  if (trace > h(1e-3)) {
    // M = C/trace: eigenvalues ≤ 1, the dominant one ≥ ¼.
    let s = h(1.0) / trace;
    let m0 = cx * s;
    let m1 = h4(cx.y, cy) * s;
    let m2 = h4(cx.z, cy.y, cz) * s;
    let m3 = h4(cx.w, cy.z, cz.y, cw) * s;
    var axis = m0;
    var dm = diag.x;
    if (diag.y > dm) { axis = m1; dm = diag.y; }
    if (diag.z > dm) { axis = m2; dm = diag.z; }
    if (diag.w > dm) { axis = m3; }
    axis = axis * inverseSqrt(dot(axis, axis));
    axis = h4(dot(m0, axis), dot(m1, axis), dot(m2, axis), dot(m3, axis));
    axis = h4(dot(m0, axis), dot(m1, axis), dot(m2, axis), dot(m3, axis));
    let a4 = max(dot(axis, axis), h(6.2e-5));
    axis = axis * inverseSqrt(a4);
    let lamn = sqrt(sqrt(a4));
    let lam = lamn * trace;

    // λ3 for all four scalar-channel candidates at once (lane c).
    let a2 = axis * axis;
    var nn = h4(0.0);
    { let wr = lamn * axis.x - axis * m0; nn = nn + wr * wr * h4(0.0, 1.0, 1.0, 1.0); }
    { let wr = lamn * axis.y - axis * m1; nn = nn + wr * wr * h4(1.0, 0.0, 1.0, 1.0); }
    { let wr = lamn * axis.z - axis * m2; nn = nn + wr * wr * h4(1.0, 1.0, 0.0, 1.0); }
    { let wr = lamn * axis.w - axis * m3; nn = nn + wr * wr * h4(1.0, 1.0, 1.0, 0.0); }
    var l3v = sqrt(nn / max(h4(1.0) - a2, h4(1e-3))) * trace;
    // Bounds: ≥ the plane's largest variance, ≤ the plane's trace.
    let d1 = max(max(diag.x, diag.y), max(diag.z, diag.w));
    let oh1 = diag == h4(d1);
    let dr = select(diag, h4(-1.0), oh1);
    let d2 = max(max(dr.x, dr.y), max(dr.z, dr.w));
    l3v = clamp(l3v, select(h4(d1), h4(d2), oh1), h4(trace) - diag);

    let sb = max(l3v, diag) * h(48.0 / 49.0) + min(l3v, diag) * h(8.0 / 9.0);
    let smax = max(max(sb.x, sb.y), max(sb.z, sb.w));
    use4 = smax - N4 > lam * h(224.0 / 225.0);
    if (sb.y == smax) { ch = 1u; }
    if (sb.z == smax) { ch = 2u; }
    if (sb.w == smax) { ch = 3u; }
    let ohc = select(h4(0.0), h4(1.0), vec4<u32>(ch) == vec4<u32>(0u, 1u, 2u, 3u));
    idx1 = dot(l3v - diag, ohc) > h(0.0);

    // Fit axis, branch-free for both modes: mode 4 starts from its plane's
    // closed-form vector (plus a whisker of the bbox diagonal for the
    // dominant-channel case where that vector vanishes), mode 6 from its
    // own axis; one masked power step each.
    let col = select(select(select(m0, m1, ch == 1u), m2, ch == 2u), m3, ch == 3u);
    cmask = select(h4(1.0), h4(1.0) - ohc, use4);
    var v = select(axis, (lamn * axis - dot(axis, ohc) * col) * cmask + (hi - lo) * cmask * h(1e-3), use4);
    v = v * inverseSqrt(max(dot(v, v), h(6.2e-5)));
    v = h4(dot(m0, v), dot(m1, v), dot(m2, v), dot(m3, v)) * cmask;
    let vv = dot(v, v);
    axisF = select(h4(0.0), v * inverseSqrt(max(vv, h(6.2e-5))), vv > h(1e-6));
  }

  // Mode-4 scalar plane: 6-bit codes at the channel's exact extremes, index
  // map v = ⌊d·ks + os⌋ in d space. chs = 0 leaves ks = 0 for mode 6.
  let chs = h4(1.0) - cmask;
  let Ls = select(h(7.0), h(3.0), idx1);
  var A0 = u32(floor(dot(lo, chs) * h(63.0) + h(0.5)));
  var A1 = u32(floor(dot(hi, chs) * h(63.0) + h(0.5)));
  var ks = h4(0.0);
  var os = h(0.0);
  {
    let d0a = h(f32((A0 << 2u) | (A0 >> 4u))) * h(1.0 / 255.0);
    let d1a = h(f32((A1 << 2u) | (A1 >> 4u))) * h(1.0 / 255.0);
    let aspan = d1a - d0a;
    if (aspan > h(0.001)) {
      let sca = Ls / aspan;
      ks = chs * (sca * h(1.0 / 16.0));
      os = (dot(p0v, chs) - d0a) * sca + h(0.5);
    }
    // Anchor rule up front: pixel 0 (the d-space origin) indexes at ⌊os⌋;
    // if its MSB would be set, swap the endpoints and mirror the map.
    if (floor(os) >= (Ls + h(1.0)) * h(0.5)) {
      let t = A0; A0 = A1; A1 = t;
      ks = -ks;
      os = Ls + h(1.0) - os;
    }
  }

  // ONE extents pass along the fit axis: projections kept for the colour
  // indices; the scalar-plane indices ride along as float nibble fields
  // (pixels 0–5, 6–11, 12–15; ≤ 2^24, exact).
  var tv: array<h, 16>;
  var t_min = h(64.0);
  var t_max = h(-64.0);
  var ga = 0.0;
  var gb = 0.0;
  var gc = 0.0;
  var w3 = 1.0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let t = dot(pix[k], axisF);
    tv[k] = t;
    t_min = min(t_min, t);
    t_max = max(t_max, t);
    var v = f32(clamp(floor(dot(pix[k], ks) + os), h(0.0), Ls));
    // (an exact tie at the mirrored anchor rounds down — equal error)
    if (k == 0u) { v = min(v, f32(floor(Ls * h(0.5)))); }
    if (k < 6u) { ga = ga + v * w3; } else if (k < 12u) { gb = gb + v * w3; } else { gc = gc + v * w3; }
    w3 = select(w3 * 16.0, 1.0, k == 5u || k == 11u);
  }
  let tm = dot(md, axisF);
  let base01 = p0v + md * h(1.0 / 16.0);
  let seed_lo = clamp(base01 + (t_min - tm) * h(1.0 / 16.0) * axisF, h4(0.0), h4(1.0));
  let seed_hi = clamp(base01 + (t_max - tm) * h(1.0 / 16.0) * axisF, h4(0.0), h4(1.0));

  // Endpoint codes, one quantiser for both modes: mode 6 = 7-bit + p-bit
  // (chosen per endpoint by quantisation error, alpha included), mode 4
  // colour = 5-bit.
  let sc = select(h(127.5), h(31.0), use4);
  let cmax = select(h(127.0), h(31.0), use4);
  let y0 = seed_lo * sc;
  let y1 = seed_hi * sc;
  let r0 = min(floor(y0 + h(0.5)), h4(cmax));
  let r1 = min(floor(y1 + h(0.5)), h4(cmax));
  let f0 = min(floor(y0), h4(cmax));
  let f1 = min(floor(y1), h4(cmax));
  let e0r = r0 - y0;
  let e0f = f0 + h(0.5) - y0;
  let e1r = r1 - y1;
  let e1f = f1 + h(0.5) - y1;
  let pp0 = !use4 && dot(e0f, e0f) < dot(e0r, e0r);
  let pp1 = !use4 && dot(e1f, e1f) < dot(e1r, e1r);
  let g0 = select(r0, f0, pp0);
  let g1 = select(r1, f1, pp1);
  var q0c = vec4<u32>(g0);
  var q1c = vec4<u32>(g1);
  var P0 = u32(pp0);
  var P1 = u32(pp1);
  // Decoded: mode 6 (2q + p)/255, mode 4 (q << 3 | q >> 2)/255 = (8q + ⌊q/4⌋)/255.
  let d0 = select(g0 * h(2.0) + h(f32(P0)), g0 * h(8.0) + floor(g0 * h(0.25)), use4) * h(1.0 / 255.0);
  let d1 = select(g1 * h(2.0) + h(f32(P1)), g1 * h(8.0) + floor(g1 * h(0.25)), use4) * h(1.0 / 255.0);
  let Lc = select(h(15.0), select(h(3.0), h(7.0), idx1), use4);

  // Index map from the stored projections: the decoded endpoints'
  // positions along the axis (tau) set the level scale.
  let tau0 = dot((d0 - p0v) * h(16.0), axisF);
  let tau1 = dot((d1 - p0v) * h(16.0), axisF);
  let span = tau1 - tau0;
  var kc = h(0.0);
  var oc = h(0.0);
  if (abs(span) > h(1.0 / 64.0)) {
    kc = Lc / span;
    oc = h(0.5) - tau0 * kc;
  }
  // Anchor rule up front (pixel 0 projects to t = 0, index ⌊oc⌋).
  if (min(floor(oc), Lc) >= (Lc + h(1.0)) * h(0.5)) {
    let tq = q0c; q0c = q1c; q1c = tq;
    let tp = P0; P0 = P1; P1 = tp;
    oc = h(0.5) + tau1 * kc;
    kc = -kc;
  }
  var fa = 0.0;
  var fb = 0.0;
  var fc = 0.0;
  var w = 1.0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    var sg = f32(clamp(floor(tv[k] * kc + oc), h(0.0), Lc));
    if (k == 0u) { sg = min(sg, f32(floor(Lc * h(0.5)))); }
    if (k < 6u) { fa = fa + sg * w; } else if (k < 12u) { fb = fb + sg * w; } else { fc = fc + sg * w; }
    w = select(w * 16.0, 1.0, k == 5u || k == 11u);
  }
  let ub = u32(fb);
  let ilo = u32(fa) | (ub << 24u);
  let ihi = (ub >> 8u) | (u32(fc) << 16u);

  let o = bi * 4u;
  if (!use4) {
    dst[o] = 0x40u | (q0c.x << 7u) | (q1c.x << 14u) | (q0c.y << 21u) | (q1c.y << 28u);
    dst[o + 1u] = (q1c.y >> 4u) | (q0c.z << 3u) | (q1c.z << 10u) | (q0c.w << 17u) | (q1c.w << 24u) | (P0 << 31u);
    dst[o + 2u] = P1 | ((ilo & 0x7u) << 1u) | (ilo & 0xFFFFFFF0u);
    dst[o + 3u] = ihi;
  } else {
    let vb = u32(gb);
    let slo = u32(ga) | (vb << 24u);
    let shi = (vb >> 8u) | (u32(gc) << 16u);
    // 2-bit field ← colour (idxMode 0) or scalar (1); 3-bit field ← the other.
    let c2 = compact2(select(ilo, slo, idx1)) | (compact2(select(ihi, shi, idx1)) << 16u);
    let iA = compact3(select(slo, ilo, idx1));
    let iB = compact3(select(shi, ihi, idx1));
    // Rotated-space RGB: position ch carries the original alpha.
    let R0 = select(q0c.x, q0c.w, ch == 0u);
    let G0 = select(q0c.y, q0c.w, ch == 1u);
    let B0 = select(q0c.z, q0c.w, ch == 2u);
    let R1 = select(q1c.x, q1c.w, ch == 0u);
    let G1 = select(q1c.y, q1c.w, ch == 1u);
    let B1 = select(q1c.z, q1c.w, ch == 2u);
    let rot = (ch + 1u) & 3u;
    // Index fields drop the anchors' MSBs: 31 bits (2-bit set) and 47 bits
    // (3-bit set).
    let field2 = (c2 & 1u) | ((c2 >> 2u) << 1u);
    let f_lo = (iA & 3u) | ((iA >> 3u) << 2u) | (iB << 23u);
    let f_hi = iB >> 9u;
    dst[o] = 0x10u | (rot << 5u) | (u32(idx1) << 7u) | (R0 << 8u) | (R1 << 13u) | (G0 << 18u) | (G1 << 23u) | (B0 << 28u);
    dst[o + 1u] = (B0 >> 4u) | (B1 << 1u) | (A0 << 6u) | (A1 << 12u) | ((field2 & 0x3FFFu) << 18u);
    dst[o + 2u] = (field2 >> 14u) | (f_lo << 17u);
    dst[o + 3u] = (f_lo >> 15u) | (f_hi << 17u);
  }
}
