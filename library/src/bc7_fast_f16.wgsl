// bc7 "fast" encoder — f16 variant (requires the shader-f16 feature).
// Same algorithm family as the f32 fast path in bc7.wgsl (principal-axis
// seed at the exact projection extents → quantise → one projection-based
// index-assignment pass), tuned for throughput:
//
//   • All projection math in f16 ([0,1] domain). ~2× ALU throughput on
//     f16-capable GPUs. The projection direction is pre-scaled by 32:
//     a shallow block (endpoints ~1/255 apart) has dd = dot(dir,dir) ≈ 1.5e-5,
//     where 15/dd ≈ 10⁶ overflows f16 (max 65504) to +inf and the products
//     inside the projection dot are subnormal — the indices turn to garbage
//     (visible as banding on smooth gradients). Scaling dir by 32 multiplies
//     the dots by 32 and dd by 1024; s = dot·(32·L/dd₃₂) is the same
//     quantity with every intermediate in f16's normal range (worst case
//     inv = 480/0.0157 ≈ 3.0e4 < 65504).
//   • TWO MODES (mode 4 OPT-IN via the enable_mode4 override constant,
//     default off and dead-coded at pipeline creation — see the constant's
//     comment for the measured cost/benefit), decided per block BEFORE
//     encoding — never encoded both:
//     mode 6 (single RGBA line, 4-bit indices) by default, mode 4 (rotation:
//     one channel split into its own scalar plane with 3-bit indices, the
//     remaining three on a 2-bit line) when the principal axis leaves a
//     large share of the block's variance unexplained — decorrelated data
//     (normal maps, channel-packed atlases) where any single 4-D line fails.
//     The decision reads the covariance already in registers (λ = axisᵀCa,
//     residual = trace − λ) and costs no extra pass. An encode-both-and-
//     compare trial was priced at ~2× on exactly this content (see the mode
//     1 postmortem below) — deciding first keeps it at ~1.2×.
//   • The two modes SHARE the per-pixel passes (axis matvecs, projection
//     extents, the index/weight pass runs once with per-thread level count,
//     index width and packing split) so warps holding a mix of mode-4 and
//     mode-6 blocks do not execute two disjoint kernels back to back — a
//     first cut with separate per-mode passes measured 1.77× on normal maps
//     from exactly that divergence; the only mode-4-extra 16-pixel work is
//     the cheap scalar-plane pass.
//   • GRAY + opaque blocks (every texel R == G == B, A == 1) have their
//     principal axis analytically: (1,1,1,0)/√3, with projection extents at
//     the luma min/max. They skip the power iteration AND the extents pass
//     (−34% GPU on roughness/AO/displacement content) and always take
//     mode 6 — a gray single line fits gray data exactly.
//   • NO least-squares refit, unlike the BC1/BC5/ASTC fast paths: with the
//     seed already on the principal axis at the exact projection extents,
//     mode 6's fine 16-level palette leaves the refit ≤0.05 dB on the colour
//     card, ≤0.15 dB on the normal card and +0.03 dB on the channel-packed
//     packed-materials atlas — not worth its two extra 16-pixel passes. The
//     coarse 4-level formats DO need it (dropping it there costs 0.5–1.3 dB).
//   • A MODE 1 (2-subset) candidate was built and evaluated (2026-07): it
//     buys ~+1.3 dB on multi-modal content but its candidate evaluation
//     costs up to ~3× the mode-6 pass on exactly that content — dropped in
//     favour of the decided (not compared) mode 4 above, which covers the
//     decorrelated-channel share of that content at a fraction of the cost.
//     The CPU reference decoder keeps mode 1 support (bc7_ref.ts).
//   • Indices are packed into two u32 words ON THE FLY during the
//     projection pass — no array<u32,16> private array. The BC7 anchor
//     reflection is then just a bitwise NOT of the packed words.
//   • The 128-bit block is assembled with straight-line constant shifts
//     instead of a generic write_bits() helper (whose dynamic word indexing
//     defeats register promotion of the output array).
//
// The host selects this module only when the device reports shader-f16,
// falling back to bc7.wgsl otherwise.
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
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
alias h = f16;
alias h4 = vec4<f16>;

// Mode-4 gate: encode mode 4 when the principal axis leaves more than
// MODE4_THETA of the (×256-scaled) total variance unexplained and the block
// isn't near-flat. Tuned against per-content mode histograms and PSNR.
const MODE4_THETA: f16 = 0.2;
const MODE4_FLOOR: f16 = 1.0;
const MODE4_CONC: f16 = 0.5;

// OPT-IN adaptive mode 4, folded at pipeline creation (WebGPU override
// constant; default OFF dead-codes the whole path — measured at exact par
// with the mode-6-only kernel). Rationale: the quality is real (+2.5–2.9 dB
// on normal maps, +1.9–2.4 on channel-packed atlases) but any warp holding
// one mode-4 block executes both modes' passes, and content that benefits
// runs 1.4–1.5×; a θ sweep showed quality and warp-poisoning scale together
// (no per-block middle ground without subgroup ballots). So the trade is
// the CALLER's: BC7Encoder({ adaptiveMode4: true }).
override enable_mode4: bool = false;

// Quantise an ideal endpoint (h4 in [0,1]) to 7-bit + p-bit, choosing the
// p-bit with the lower quantisation error. `eight` is the decoded value the
// hardware will interpolate with, back in [0,1].
struct Ep { seven: vec4<u32>, eight: h4, p: u32 };
fn pick_ep(ideal01: h4) -> Ep {
  let ideal = ideal01 * h(255.0);
  let q0 = clamp(floor(ideal * h(0.5) + h(0.5)), h4(0.0), h4(127.0));        // p=0
  let e0 = q0 * h(2.0);
  let q1 = clamp(floor((ideal - h(1.0)) * h(0.5) + h(0.5)), h4(0.0), h4(127.0)); // p=1
  let e1 = q1 * h(2.0) + h(1.0);
  let d0 = e0 - ideal; let d1 = e1 - ideal;
  if (dot(d1, d1) < dot(d0, d0)) { return Ep(vec4<u32>(q1), e1 * h(1.0 / 255.0), 1u); }
  return Ep(vec4<u32>(q0), e0 * h(1.0 / 255.0), 0u);
}

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);
  let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  // Load pass, with the covariance moments FUSED in (no separate 16-pixel
  // pass): d = (px − pixel0)·16, relative to the block's first pixel so the
  // accumulators scale with the block's span — raw Σv·vᵀ moments would
  // cancel catastrophically in f16 — and pre-scaled ×16 so shallow blocks
  // (span ~1/255 → d² ≈ 1e-3) clear the subnormal floor while full-range
  // sums stay ≤4096. C = Σddᵀ − (Σd)(Σd)ᵀ/16 is the ×256-scaled covariance.
  var pix: array<h4, 16>;
  var lo = h4(1.0);
  var hi = h4(0.0);
  var gd = h(0.0);
  var p0v = h4(0.0);
  var sd = h4(0.0);
  var c0v = h4(0.0);
  var c1v = h4(0.0);
  var c2v = h4(0.0);
  var c3v = h4(0.0);
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0), mx);
    let px = h4(textureLoad(src_tex, p, 0));
    pix[i] = px; lo = min(lo, px); hi = max(hi, px);
    gd = max(gd, max(abs(px.x - px.y), abs(px.x - px.z)));
    if (i == 0u) { p0v = px; }
    let d = (px - p0v) * h(16.0);
    sd = sd + d;
    c0v = c0v + d.x * d;
    c1v = c1v + d.y * d;
    c2v = c2v + d.z * d;
    c3v = c3v + d.w * d;
  }
  let mean = p0v + sd * h(1.0 / 256.0);
  // Mean-correction via sd4·sd4ᵀ with sd4 = Σd/4: (Σd)(Σd)ᵀ/16 with every
  // product ≤4096 (a direct Σd·Σdᵀ could hit 65536 and overflow f16).
  let sd4 = sd * h(0.25);
  c0v = c0v - sd4.x * sd4;
  c1v = c1v - sd4.y * sd4;
  c2v = c2v - sd4.z * sd4;
  c3v = c3v - sd4.w * sd4;

  // Seed endpoints + per-block mode decision (see header).
  var seed_lo = lo;
  var seed_hi = hi;
  var use4 = false;
  var cmask = h4(1.0);
  var ch = 0u;
  if (lo.w == h(1.0) && gd == h(0.0)) {
    // GRAY + opaque: analytic axis (1,1,1,0)/√3, extents at luma min/max,
    // always mode 6 — and a fully specialised tail: gray textures are
    // warp-uniform, and routing them through the parametric shared loop
    // below (runtime index width/split) measured +28% on displacement
    // content purely from the lost constant-shift codegen.
    var ep0g = pick_ep(h4(lo.x, lo.x, lo.x, h(1.0)));
    var ep1g = pick_ep(h4(hi.x, hi.x, hi.x, h(1.0)));
    var ilo = 0u;
    var ihi = 0u;
    let dirg = (ep1g.eight - ep0g.eight) * h(32.0);
    let ddg = dot(dirg, dirg);
    if (ddg >= h(0.008)) {
      let invg = h(480.0) / ddg;
      for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let sg = clamp(floor(dot(pix[k] - ep0g.eight, dirg) * invg + h(0.5)), h(0.0), h(15.0));
        ilo = ilo | (u32(sg) << (k * 4u));
      }
      for (var k: u32 = 8u; k < 16u; k = k + 1u) {
        let sg = clamp(floor(dot(pix[k] - ep0g.eight, dirg) * invg + h(0.5)), h(0.0), h(15.0));
        ihi = ihi | (u32(sg) << ((k - 8u) * 4u));
      }
    }
    if ((ilo & 0x8u) != 0u) {
      let t = ep0g; ep0g = ep1g; ep1g = t;
      ilo = ~ilo; ihi = ~ihi;
    }
    let e0g = ep0g.seven;
    let e1g = ep1g.seven;
    let og = bi * 4u;
    dst[og] = 0x40u | (e0g.x << 7u) | (e1g.x << 14u) | (e0g.y << 21u) | (e1g.y << 28u);
    dst[og + 1u] = (e1g.y >> 4u) | (e0g.z << 3u) | (e1g.z << 10u) | (e0g.w << 17u) | (e1g.w << 24u) | (ep0g.p << 31u);
    dst[og + 2u] = ep1g.p | ((ilo & 0x7u) << 1u) | (ilo & 0xFFFFFFF0u);
    dst[og + 3u] = ihi;
    return;
  }
  {
    var axis = hi - lo;
    var axis_ok = true;
    // 8 iterations: 4 was under-converged on noisy 4-D blocks (heavily
    // downscaled photographic/channel-packed content) — going to 8 measured
    // +0.75 dB on the normal card, +0.12 colour, +0.08 packed-materials, and
    // matches the f32 fallback's iteration count. Four extra 4-dot matvecs
    // per block are noise next to the index pass.
    for (var it: u32 = 0u; it < 8u; it = it + 1u) {
      let nv = h4(dot(c0v, axis), dot(c1v, axis), dot(c2v, axis), dot(c3v, axis));
      let m = max(max(abs(nv.x), abs(nv.y)), max(abs(nv.z), abs(nv.w)));
      if (m < h(1e-4)) { axis_ok = false; break; }
      axis = nv / m;
    }
    if (axis_ok) {
      axis = axis / length(axis);
      var axisF = axis;

      // Mode decision from the covariance already in registers: λ is the
      // variance the mode-6 line explains, trace − λ what it cannot.
      let Ca = h4(dot(c0v, axis), dot(c1v, axis), dot(c2v, axis), dot(c3v, axis));
      let lam = dot(Ca, axis);
      let diag = h4(c0v.x, c1v.y, c2v.z, c3v.w);
      let trace = diag.x + diag.y + diag.z + diag.w;
      let resid = trace - lam;
      let rc = diag - lam * axis * axis;
      var rbest = rc.x;
      if (rc.y > rbest) { ch = 1u; rbest = rc.y; }
      if (rc.z > rbest) { ch = 2u; rbest = rc.z; }
      if (rc.w > rbest) { ch = 3u; rbest = rc.w; }
      use4 = enable_mode4 && resid > MODE4_THETA * trace && trace > MODE4_FLOOR && rbest > MODE4_CONC * resid;
      if (use4) {
        // The colour plane is the remaining three channels, handled as
        // masked 4-vectors so every vec4 pass below applies unchanged.
        // Branchless mask build — a dynamic component store spills the
        // vector to scratch on some compilers.
        cmask = h4(1.0) - h4(h(f32(u32(ch == 0u))), h(f32(u32(ch == 1u))), h(f32(u32(ch == 2u))), h(f32(u32(ch == 3u))));
        var a3 = (hi - lo) * cmask;
        var ok3 = true;
        for (var it: u32 = 0u; it < 2u; it = it + 1u) {
          let nv = h4(dot(c0v, a3), dot(c1v, a3), dot(c2v, a3), dot(c3v, a3)) * cmask;
          let m = max(max(abs(nv.x), abs(nv.y)), max(abs(nv.z), abs(nv.w)));
          if (m < h(1e-4)) { ok3 = false; break; }
          a3 = nv / m;
        }
        if (ok3) {
          axisF = a3 / length(a3);
        } else {
          use4 = false;
          cmask = h4(1.0);
        }
      }

      // Exact projection extents along the fit axis — ONE shared pass for
      // both modes (for mode 4 axisF[ch] = 0, so the scalar plane is
      // invisible to it). (A Rayleigh-quotient span estimate was tried in
      // place of this pass — it saves 16 dots but costs 0.1–0.8 dB and
      // 4–10× on the worst-easy-block gate.)
      var t_min = h(4.0);
      var t_max = h(-4.0);
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let t = dot(pix[k] - mean, axisF);
        t_min = min(t_min, t);
        t_max = max(t_max, t);
      }
      seed_lo = clamp(mean + t_min * axisF, h4(0.0), h4(1.0));
      seed_hi = clamp(mean + t_max * axisF, h4(0.0), h4(1.0));
    }
  }

  // Endpoints, per mode. d0/d1 are the DECODED values the weight pass
  // projects against.
  var ep0: Ep;
  var ep1: Ep;
  var q0c = vec4<u32>(0u);
  var q1c = vec4<u32>(0u);
  var A0 = 0u;
  var A1 = 0u;
  var iA = 0u;
  var iB = 0u;
  var d0: h4;
  var d1: h4;
  var d0a = h(0.0);
  var sca = h(0.0);
  let chs = h4(1.0) - cmask;
  ep0 = pick_ep(seed_lo);
  ep1 = pick_ep(seed_hi);
  d0 = ep0.eight;
  d1 = ep1.eight;
  if (use4) {
    // Scalar plane (3-bit index set): 6-bit endpoints at the channel's
    // exact extremes. Its projection is FUSED into the shared weight pass
    // below — a separate 16-pixel pass here measured +46% on normal maps
    // (mixed warps paid it wholesale); fused, the marginal cost is one dot
    // per pixel under a warp-uniform predicate.
    let a0q = u32(floor(dot(lo, chs) * h(63.0) + h(0.5)));
    let a1q = u32(floor(dot(hi, chs) * h(63.0) + h(0.5)));
    A0 = a0q;
    A1 = a1q;
    let d0av = h(f32((a0q << 2u) | (a0q >> 4u))) * h(1.0 / 255.0);
    let d1av = h(f32((a1q << 2u) | (a1q >> 4u))) * h(1.0 / 255.0);
    let aspan = d1av - d0av;
    if (aspan > h(0.001)) {
      d0a = d0av;
      sca = h(7.0) / aspan;
    }
    // Colour plane: 5-bit endpoints from the masked extents seed.
    q0c = vec4<u32>(clamp(floor(seed_lo * h(31.0) + h(0.5)), h4(0.0), h4(31.0)));
    q1c = vec4<u32>(clamp(floor(seed_hi * h(31.0) + h(0.5)), h4(0.0), h4(31.0)));
    d0 = h4(vec4<f32>((q0c << vec4<u32>(3u)) | (q0c >> vec4<u32>(2u)))) * h(1.0 / 255.0) * cmask;
    d1 = h4(vec4<f32>((q1c << vec4<u32>(3u)) | (q1c >> vec4<u32>(2u)))) * h(1.0 / 255.0) * cmask;
  }

  // Index/weight pass: per-mode SPECIALISED loops (constant level counts
  // and shifts, so each unrolls cleanly — a single parametric loop with
  // runtime width/split measured +22% on pure mode-6 photo content).
  // Mixed warps execute both loops; the mode-4 one carries the fused
  // scalar-plane projection. For mode 4 pix[ch]·dir[ch] = 0, so the
  // scalar plane never perturbs the colour projection.
  var a_lo = 0u;
  var a_hi = 0u;
  // Same ×32 pre-scale as the extents math; distinct quantised endpoints
  // are ≥1/255 apart (dd₃₂ ≥ 0.0157), so the flat-block threshold only
  // catches truly identical ones.
  let dir = (d1 - d0) * h(32.0);
  let dd = dot(dir, dir);
  let live = dd >= h(0.008);
  if (use4) {
    if (live) {
      let inv = h(96.0) / dd;
      for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let s = clamp(floor(dot(pix[k] - d0, dir) * inv + h(0.5)), h(0.0), h(3.0));
        let sv = clamp(floor((dot(pix[k], chs) - d0a) * sca + h(0.5)), h(0.0), h(7.0));
        a_lo = a_lo | (u32(s) << (k * 2u));
        iA = iA | (u32(sv) << (k * 3u));
      }
      for (var k: u32 = 8u; k < 16u; k = k + 1u) {
        let s = clamp(floor(dot(pix[k] - d0, dir) * inv + h(0.5)), h(0.0), h(3.0));
        let sv = clamp(floor((dot(pix[k], chs) - d0a) * sca + h(0.5)), h(0.0), h(7.0));
        a_lo = a_lo | (u32(s) << (k * 2u));
        iB = iB | (u32(sv) << ((k - 8u) * 3u));
      }
    }
  } else if (live) {
    let inv = h(480.0) / dd;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      let s = clamp(floor(dot(pix[k] - d0, dir) * inv + h(0.5)), h(0.0), h(15.0));
      a_lo = a_lo | (u32(s) << (k * 4u));
    }
    for (var k: u32 = 8u; k < 16u; k = k + 1u) {
      let s = clamp(floor(dot(pix[k] - d0, dir) * inv + h(0.5)), h(0.0), h(15.0));
      a_hi = a_hi | (u32(s) << ((k - 8u) * 4u));
    }
  }

  // Anchors + packing. Mode 6 packs unconditionally (one-sided branches
  // compile better than two-sided divergence); mode-4 threads overwrite.
  let o = bi * 4u;
  {
    var ilo = a_lo;
    var ihi = a_hi;
    if ((ilo & 0x8u) != 0u) {
      let t = ep0; ep0 = ep1; ep1 = t;
      ilo = ~ilo; ihi = ~ihi;
    }
    let e0 = ep0.seven;
    let e1 = ep1.seven;
    dst[o] = 0x40u | (e0.x << 7u) | (e1.x << 14u) | (e0.y << 21u) | (e1.y << 28u);
    dst[o + 1u] = (e1.y >> 4u) | (e0.z << 3u) | (e1.z << 10u) | (e0.w << 17u) | (e1.w << 24u) | (ep0.p << 31u);
    dst[o + 2u] = ep1.p | ((ilo & 0x7u) << 1u) | (ilo & 0xFFFFFFF0u);
    dst[o + 3u] = ihi;
  }
  if (use4) {
    // 3-bit anchor: pixel 0's MSB must be 0; reflect = bitwise NOT.
    if ((iA & 4u) != 0u) {
      let tA = A0;
      A0 = A1;
      A1 = tA;
      iA = ~iA & 0xFFFFFFu;
      iB = ~iB & 0xFFFFFFu;
    }
    var c2 = a_lo;
    // 2-bit anchor: pixel 0's MSB must be 0.
    if ((c2 & 2u) != 0u) {
      let tq = q0c;
      q0c = q1c;
      q1c = tq;
      c2 = ~c2;
    }
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
    dst[o] = 0x10u | (rot << 5u) | (R0 << 8u) | (R1 << 13u) | (G0 << 18u) | (G1 << 23u) | (B0 << 28u);
    dst[o + 1u] = (B0 >> 4u) | (B1 << 1u) | (A0 << 6u) | (A1 << 12u) | ((field2 & 0x3FFFu) << 18u);
    dst[o + 2u] = (field2 >> 14u) | (f_lo << 17u);
    dst[o + 3u] = (f_lo >> 15u) | (f_hi << 17u);
  }
}
