// astc4x4 encoder — f16 variant (requires the shader-f16 feature). Same
// algorithm as the f32 fallback in astc4x4.wgsl, with the projection / fit
// math in f16:
//
//   • THREE block classes, picked per block from the loaded pixels (the
//     ASTC bit budget trades endpoint bits against weight bits, so a block
//     only pays for the channels it uses):
//       gray + opaque → CEM 0  (luminance), 5-bit weights, mode 0x253
//       opaque        → CEM 8  (RGB), two budgets:
//                         span > 12  → QUANT_192 endpoints + 4-bit
//                                      weights (QUANT_16), mode 0x242
//                         span ≤ 12  → QUANT_256 endpoints + 3-bit
//                                      weights (QUANT_8), mode 0x053
//       translucent   → CEM 12 (RGBA), 2-bit weights, mode 0x042
//     "gray" = every texel R == G == B exactly (f16 equality is exact for
//     8-bit sources), "opaque" = every texel A == 1. Every class fills the
//     128-bit block; the old opaque layout (8-bit endpoints + 3-bit weights
//     for every block) left 15 bits unused.
//   • OPAQUE BUDGETS. 16 weight levels need QUANT_192 endpoints (46 bits)
//     to fit — trit-ISE coded, 1/4 of the 8-bit values fall on excluded
//     slots and round to a neighbour. On wide blocks the finer weights win
//     by far (+0.6..1.2 dB on photographic colour); when the block's colour
//     span is small the endpoint error dominates instead (the weight error
//     scales with the span, the endpoint error doesn't), and exact 8-bit
//     endpoints with 8 levels win (the 12-level threshold was swept:
//     8/12/16/24 on the /eval corpus). Both budgets share ONE weight loop
//     (weights accumulate as 4-bit nibbles; the 3-bit stream is compacted
//     afterwards) — two loops measured ~8% slower on mixed warps.
//   • The opaque path runs in 3-channel math (alpha is constant): the dead
//     4th lane in the covariance, power iteration, extents and weight
//     loops cost ~10% of the kernel. Opaque endpoints are the PCA extents
//     (8 power iterations; 6 measured −0.1 dB on normal maps), bbox-clamped
//     — no LSQ refit: with 8–16 weight levels a converged axis carries the
//     quality.
//   • QUANT_16 weight levels are not uniform (0 4 8 12 17 21 25 29 35 …);
//     the weights are the uniformly rounded projection anyway — the exact
//     nearest-level mapping measured +0.03..0.06 dB for +15% GPU.
//   • The GRAY path is scalar — no covariance, no power iteration — in
//     [0,255]-integer f16 math (exact endpoints, and 64/span ≤ 64 never
//     overflows f16, unlike a [0,1]-domain 1/dd).
//   • TRANSLUCENT blocks: PCA seed (4 power iterations) → fused projection
//     + LSQ refit (the 4-level grid is coarse enough to need it, and the fit
//     pulls the line through the dominant cluster of multi-cluster blocks)
//     → the fit-pass weights are shipped without a reprojection. A
//     3-bit-weight + QUANT_192 translucent budget measured +1.5 dB on
//     translucent content but +10..20% GPU there — not taken.
//   • f16 range: projection directions are pre-scaled by 32 — a shallow
//     block (endpoints ~1/255 apart) has dd ≈ 1.5e-5, where L/dd overflows
//     f16 (max 65504) and the projection dots go subnormal, turning weights
//     and the refit to garbage (banding on smooth gradients). Scaled, every
//     intermediate stays in f16's normal range (worst case, 16 levels:
//     inv = 480/0.0157 ≈ 3.1e4).
//   • Endpoint ordering (the blue-contraction rule: sum(e0.rgb) must not
//     exceed sum(e1.rgb), compared on the UNQUANTISED values) is applied
//     BEFORE the weight pass, so no weight-reflection pass is needed.
//   • Weight streams are accumulated LSB-first into u32 words and placed
//     into the block's reversed-bit-order field with reverseBits() —
//     stream bit q lives at block bit 127 − q.
//
// BLOCK LAYOUTS + ISE: see astc4x4_ref.ts (single partition, CEM 0/8/12,
// trit-ISE QUANT_192 endpoints, plain-bit weights, block modes
// 0x253/0x242/0x053/0x042).
//
// The host selects this module only when the device reports shader-f16,
// falling back to astc4x4.wgsl otherwise.
enable f16;
alias h = f16;
alias h3 = vec3<f16>;
alias h4 = vec4<f16>;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, y0: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// Float -> u32 for exact integers in [0, 2^23): x + 2^23 holds x in its
// mantissa, so bitcast(x + MAGIC) ^ MAGIC_BITS == x (or masked, for small
// fields). WGSL's u32(float) is a SATURATING conversion (compares + selects
// around the convert): the per-texel weight conversions alone cost ~1.5% GPU.
const MAGIC = 8388608.0;
const MAGIC_BITS = 0x4B000000u;
fn hbits(x: f16, mask: u32) -> u32 {
  return bitcast<u32>(f32(x) + MAGIC) & mask;
}
fn fu(x: f32) -> u32 {
  return bitcast<u32>(x + MAGIC) ^ MAGIC_BITS;
}
fn fu3(x: vec3<f32>) -> vec3<u32> {
  return bitcast<vec3<u32>>(x + MAGIC) ^ vec3<u32>(MAGIC_BITS);
}
fn fu4(x: vec4<f32>) -> vec4<u32> {
  return bitcast<vec4<u32>>(x + MAGIC) ^ vec4<u32>(MAGIC_BITS);
}

// Fused projection + least-squares refit against the 4-level QUANT_4
// palette (translucent blocks). Returns the refit endpoints and the
// projection's packed weight stream.
struct Fit { e0: h4, e1: h4, valid: bool, wstream: u32 };
fn proj_fit(pix: ptr<function, array<h4, 16>>, e0: h4, e1: h4) -> Fit {
  var out: Fit;
  out.valid = false;
  out.wstream = 0u;
  // Spans below ~0.7 of an 8-bit step (dd₃₂ < 0.008, possible only for
  // non-8-bit sources) are treated as flat.
  let dir = (e1 - e0) * h(32.0);
  let dd = dot(dir, dir);
  if (dd < h(0.008)) { return out; }
  let inv = h(96.0) / dd; // 32·3/dd₃₂ ≡ 3/dd
  var sAA = h(0.0); var sBB = h(0.0); var sAB = h(0.0);
  var sAV = h4(0.0); var sBV = h4(0.0);
  var s_min = h(3.0); var s_max = h(0.0);
  // Value sums accumulate v − e0 (the basis is affine, a + b = 1, so the fit
  // commutes with the shift): accumulators scale with the block span, keeping
  // f16 rounding a fraction of the span instead of ±1 level at high absolute
  // values.
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let vr = (*pix)[k] - e0;
    let s = clamp(floor(dot(vr, dir) * inv + h(0.5)), h(0.0), h(3.0));
    out.wstream = out.wstream | (hbits(s, 3u) << (2u * k));
    s_min = min(s_min, s); s_max = max(s_max, s);
    let b = s * h(1.0 / 3.0);
    let a = h(1.0) - b;
    sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b;
    sAV = sAV + a * vr; sBV = sBV + b * vr;
  }
  // Rank-1 guard: if every pixel projects to ONE level the system is
  // singular — det/numerators are pure f16 rounding noise. With ≥2 distinct
  // levels det = Σ_i<j (b_j − b_i)² ≥ 15·(1/3)² ≈ 1.67 — 0.5 separates
  // cleanly.
  if (s_min == s_max) { return out; }
  let det = sAA * sBB - sAB * sAB;
  if (abs(det) < h(0.5)) { return out; }
  out.e0 = clamp(e0 + (sBB * sAV - sAB * sBV) / det, h4(0.0), h4(1.0));
  out.e1 = clamp(e0 + (sAA * sBV - sAB * sAV) / det, h4(0.0), h4(1.0));
  out.valid = true;
  return out;
}

// ISE trit-block encoder: 5 trits → the 8-bit T field (the inverse of the
// spec's trit-block decode; all 243 tuples round-trip — see the ref tests).
fn trit_enc(t0: u32, t1: u32, t2: u32, t3: u32, t4: u32) -> u32 {
  let c = select(select((t2 << 4u) | (t1 << 2u) | t0, (t1 << 4u) | (t0 << 2u) | 3u, t2 == 2u), 12u | t0, t2 == 2u && t1 == 2u);
  return select(select((t4 << 7u) | (t3 << 5u) | c, (t3 << 7u) | 96u | c, t4 == 2u), ((c >> 2u) << 5u) | 28u | (c & 3u), t3 == 2u && t4 == 2u);
}

// Nearest QUANT_192 endpoint (one trit + 6 bits) to x ∈ [0,255]. The 192
// unquantised levels are every value ≤ 127 that is not ≡ 3 (mod 4), plus
// their mirror images 255 − u above: lower-half value u = 4q + t (t < 3)
// is ISE value trit t, bits q << 1; the upper half sets bit 0 (the spec's
// A-mask inversion). A value that falls on an excluded slot moves to the
// nearer representable neighbour (a fixed direction instead costs up to
// 0.4 dB on smooth content). Returns (ISE value = trit·64 + bits, level).
fn q192(x: f32) -> vec2<u32> {
  let v = fu(clamp(floor(x + 0.5), 0.0, 255.0));
  let up = v > 127u;
  var u = select(v, 255u - v, up);
  if ((u & 3u) == 3u) {
    let xu = select(x, 255.0 - x, up);
    u = select(u - 1u, u + 1u, xu > f32(u) && u < 127u);
  }
  return vec2<u32>(((u & 3u) << 6u) | ((u >> 2u) << 1u) | u32(up), select(u, 255u - u, up));
}

fn q8(e: h4) -> vec4<u32> {
  return fu4(vec4<f32>(clamp(floor(e * h(255.0) + h(0.5)), h4(0.0), h4(255.0))));
}

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid_raw: vec3<u32>) {
  // Row-band encodes dispatch a slice of the block grid starting at row y0.
  let gid = vec3<u32>(gid_raw.x, gid_raw.y + params.y0, gid_raw.z);
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);
  let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  // Load pass. gd tracks the largest chroma deviation — 0 iff the block is
  // exactly grayscale (equal 8-bit channels convert to identical f16s).
  var pix: array<h4, 16>;
  var lo = h4(1.0);
  var hi = h4(0.0);
  var mean = h4(0.0);
  var gd = h(0.0);
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0), mx);
    let px = h4(textureLoad(src_tex, p, 0));
    pix[i] = px; lo = min(lo, px); hi = max(hi, px);
    mean = mean + px;
    gd = max(gd, max(abs(px.x - px.y), abs(px.x - px.z)));
  }
  mean = mean * h(1.0 / 16.0);
  let opaque = lo.w == h(1.0);

  var w0: u32; var w1: u32; var w2: u32; var w3: u32;

  if (opaque && gd == h(0.0)) {
    // ---------------- Luminance path: CEM 0, 5-bit weights ----------------
    // Scalar [0,255]-integer domain. Endpoints at the exact extremes (the
    // 8-bit values round-trip f16 exactly); 32 palette levels make an LSQ
    // refit unnecessary.
    let L0 = hbits(floor(lo.x * h(255.0) + h(0.5)), 255u);
    let L1 = hbits(floor(hi.x * h(255.0) + h(0.5)), 255u);
    var s0 = 0u; var s1 = 0u; var s2 = 0u;
    if (L1 > L0) {
      let l0f = h(f32(L0));
      let sc = h(64.0) / h(f32(L1 - L0)); // span ≥ 1 → sc ≤ 64, no overflow
      // Exact nearest entry of the QUANT_32 grid: unq = 2w for w ≤ 15,
      // 2w + 2 for w ≥ 16 (the grid has a 4-wide gap at the middle, so
      // uniform rounding is wrong there). Evaluate the best candidate of
      // each half and keep the closer.
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let v = floor(pix[k].x * h(255.0) + h(0.5)) - l0f; // exact integer
        let u = clamp(v * sc, h(0.0), h(64.0));
        let wlo = clamp(floor(u * h(0.5) + h(0.5)), h(0.0), h(15.0));
        let whi = clamp(floor((u - h(2.0)) * h(0.5) + h(0.5)), h(16.0), h(31.0));
        let pick = abs(u - wlo * h(2.0)) <= abs(u - (whi * h(2.0) + h(2.0)));
        let w = hbits(select(whi, wlo, pick), 31u);
        // Stream bit q = 5k + j; straddles handled with constant shifts.
        let off = 5u * k;
        if (off < 28u) { s0 = s0 | (w << off); }
        else if (off == 30u) { s0 = s0 | (w << 30u); s1 = s1 | (w >> 2u); }
        else if (off < 60u) { s1 = s1 | (w << (off - 32u)); }
        else if (off == 60u) { s1 = s1 | (w << 28u); s2 = s2 | (w >> 4u); }
        else { s2 = s2 | (w << (off - 64u)); }
      }
    }
    // Mode 0x253, partitions−1 = 0, CEM 0, L0 @17, L1 @25 (top bit spills
    // into w1 bit 0); stream word q∈[0,31] → block bits 127…96 via
    // reverseBits, q∈[32,63] → 95…64, q∈[64,79] → 63…48.
    w0 = 0x253u | (L0 << 17u) | (L1 << 25u);
    w1 = (L1 >> 7u) | reverseBits(s2);
    w2 = reverseBits(s1);
    w3 = reverseBits(s0);
  } else if (opaque) {
    // ------------- Opaque colour: CEM 8, two bit budgets (header) -------------
    // 3-channel math throughout (alpha is constant 1 here, so a 4-lane
    // pass would carry a dead lane through the covariance, iteration,
    // extents and weight loops).
    //
    // Seed endpoints from the block's principal colour axis (covariance
    // power-iteration, seeded with the bbox diagonal). The bbox diagonal is
    // sign-blind: on anti-correlated channels (normal maps, hue edges) it
    // points across the data instead of along it. Deviations are
    // pre-scaled ×16 so covariance entries for shallow blocks stay in f16's
    // normal range (span ~1/255 → d² ≈ 1e-3) while full-range sums stay
    // ≤4096; the iteration renormalises by the max component (a plain
    // length() of the matvec output could overflow f16), so only the
    // direction survives.
    let lo3 = lo.xyz;
    let hi3 = hi.xyz;
    let mean3 = mean.xyz;
    var c0v = h3(0.0);
    var c1v = h3(0.0);
    var c2v = h3(0.0);
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let d = (pix[k].xyz - mean3) * h(16.0);
      c0v = c0v + d.x * d;
      c1v = c1v + d.y * d;
      c2v = c2v + d.z * d;
    }
    var seed_lo = lo3;
    var seed_hi = hi3;
    var axis = hi3 - lo3;
    var axis_ok = true;
    // 8 iterations: the axis IS the endpoint quality here (no refit).
    for (var it: u32 = 0u; it < 8u; it = it + 1u) {
      let nv = h3(dot(c0v, axis), dot(c1v, axis), dot(c2v, axis));
      let m = max(max(abs(nv.x), abs(nv.y)), abs(nv.z));
      if (m < h(1e-4)) { axis_ok = false; break; }
      axis = nv / m;
    }
    if (axis_ok) {
      axis = axis / length(axis);
      var t_min = h(4.0);
      var t_max = h(-4.0);
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let t = dot(pix[k].xyz - mean3, axis);
        t_min = min(t_min, t);
        t_max = max(t_max, t);
      }
      seed_lo = mean3 + t_min * axis;
      seed_hi = mean3 + t_max * axis;
    }
    // Endpoints: the PCA extents, bbox-clamped (on multi-cluster blocks the
    // axis extents overshoot the data per channel and would decode to
    // colours that exist nowhere in the block).
    let x0 = vec3<f32>(clamp(seed_lo, lo3, hi3)) * 255.0;
    let x1 = vec3<f32>(clamp(seed_hi, lo3, hi3)) * 255.0;
    // Bit budget: span ≤ 12 → exact QUANT_256 endpoints + 8 weight levels
    // (mode 0x053), wider → QUANT_192 endpoints + 16 levels (mode 0x242).
    let span3 = hi3 - lo3;
    let small = max(max(span3.x, span3.y), span3.z) <= h(12.0 / 255.0);
    var r0: vec2<u32>; var g0: vec2<u32>; var b0: vec2<u32>;
    var r1: vec2<u32>; var g1: vec2<u32>; var b1: vec2<u32>;
    if (small) {
      let e0 = fu3(clamp(floor(x0 + 0.5), vec3<f32>(0.0), vec3<f32>(255.0)));
      let e1 = fu3(clamp(floor(x1 + 0.5), vec3<f32>(0.0), vec3<f32>(255.0)));
      r0 = vec2<u32>(e0.x); g0 = vec2<u32>(e0.y); b0 = vec2<u32>(e0.z);
      r1 = vec2<u32>(e1.x); g1 = vec2<u32>(e1.y); b1 = vec2<u32>(e1.z);
    } else {
      r0 = q192(x0.x); g0 = q192(x0.y); b0 = q192(x0.z);
      r1 = q192(x1.x); g1 = q192(x1.y); b1 = q192(x1.z);
    }
    // Blue-contraction ordering on the UNQUANTISED levels, before the
    // weight pass so the weights come out oriented (no reflection).
    if (r0.y + g0.y + b0.y > r1.y + g1.y + b1.y) {
      let tr = r0; r0 = r1; r1 = tr;
      let tg = g0; g0 = g1; g1 = tg;
      let tb = b0; b0 = b1; b1 = tb;
    }
    let d0 = h3(vec3<f32>(vec3<u32>(r0.y, g0.y, b0.y))) * h(1.0 / 255.0);
    let d1 = h3(vec3<f32>(vec3<u32>(r1.y, g1.y, b1.y))) * h(1.0 / 255.0);
    // Weight pass — ONE loop for both budgets (mixed warps would otherwise
    // run two): weights land as 4-bit nibbles, levels = 16 or 8. ×32
    // pre-scale as in the other formats (distinct levels are ≥1/255 apart,
    // so dd₃₂ ≥ 0.0157 and the flat threshold only catches identical ones).
    let lmax = select(h(15.0), h(7.0), small);
    let dir = (d1 - d0) * h(32.0);
    let dd = dot(dir, dir);
    var s0 = 0u; var s1 = 0u;
    if (dd >= h(0.008)) {
      let inv = lmax * h(32.0) / dd;
      for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let w = hbits(clamp(floor(dot(pix[k].xyz - d0, dir) * inv + h(0.5)), h(0.0), lmax), 15u);
        s0 = s0 | (w << (4u * k));
      }
      for (var k: u32 = 8u; k < 16u; k = k + 1u) {
        let w = hbits(clamp(floor(dot(pix[k].xyz - d0, dir) * inv + h(0.5)), h(0.0), lmax), 15u);
        s1 = s1 | (w << (4u * (k - 8u)));
      }
    }
    if (small) {
      // Mode 0x053: endpoints as plain bytes from bit 17; the 3-bit weight
      // stream (48 bits) is the nibbles compacted: 8 nibbles → 24 bits.
      var c0 = (s0 & 0x07070707u) | ((s0 & 0x70707070u) >> 1u);
      c0 = (c0 & 0x003F003Fu) | ((c0 & 0x3F003F00u) >> 2u);
      c0 = (c0 & 0x00000FFFu) | ((c0 & 0x0FFF0000u) >> 4u);
      var c1 = (s1 & 0x07070707u) | ((s1 & 0x70707070u) >> 1u);
      c1 = (c1 & 0x003F003Fu) | ((c1 & 0x3F003F00u) >> 2u);
      c1 = (c1 & 0x00000FFFu) | ((c1 & 0x0FFF0000u) >> 4u);
      w0 = 0x053u | (8u << 13u) | (r0.x << 17u) | (r1.x << 25u);
      w1 = (r1.x >> 7u) | (g0.x << 1u) | (g1.x << 9u) | (b0.x << 17u) | (b1.x << 25u);
      w2 = (b1.x >> 7u) | reverseBits(c1 >> 8u);
      w3 = reverseBits(c0 | (c1 << 24u));
    } else {
      // Mode 0x242, CEM 8 @13. Endpoint ISE (v0..v5 = R0 R1 G0 G1 B0 B1,
      // 46 bits from bit 17): group 1 = v0..v4 with trit field T, group 2 =
      // v5 with its lone trit as 2 bits (T of (t5,0,0,0,0) is t5 itself).
      // Weight stream bit q = 4k + j lives at block bit 127 − q.
      let tg = trit_enc(r0.x >> 6u, r1.x >> 6u, g0.x >> 6u, g1.x >> 6u, b0.x >> 6u);
      w0 = 0x242u | (8u << 13u) | ((r0.x & 63u) << 17u) | ((tg & 3u) << 23u) | ((r1.x & 63u) << 25u) | (((tg >> 2u) & 1u) << 31u);
      w1 = ((tg >> 3u) & 1u) | ((g0.x & 63u) << 1u) | (((tg >> 4u) & 1u) << 7u) | ((g1.x & 63u) << 8u)
        | (((tg >> 5u) & 3u) << 14u) | ((b0.x & 63u) << 16u) | ((tg >> 7u) << 22u) | ((b1.x & 63u) << 23u)
        | ((b1.x >> 6u) << 29u);
      w2 = reverseBits(s1);
      w3 = reverseBits(s0);
    }
  } else {
    // ------ Translucent: CEM 12, 2-bit weights, PCA seed + LSQ refit ------
    // Seed from the principal RGBA axis (covariance power iteration, bbox
    // diagonal start — sign-blind on anti-correlated channels, hence the
    // iteration). Deviations pre-scaled ×16 (f16 normal range for shallow
    // blocks, full-range sums ≤4096); renormalised by the max component.
    var c0v = h4(0.0);
    var c1v = h4(0.0);
    var c2v = h4(0.0);
    var c3v = h4(0.0);
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let d = (pix[k] - mean) * h(16.0);
      c0v = c0v + d.x * d;
      c1v = c1v + d.y * d;
      c2v = c2v + d.z * d;
      c3v = c3v + d.w * d;
    }
    var seed_lo = lo;
    var seed_hi = hi;
    var axis = hi - lo;
    var axis_ok = true;
    // 4 iterations: the refit absorbs residual axis error (4 more measured
    // 0.000 dB on the alpha card).
    for (var it: u32 = 0u; it < 4u; it = it + 1u) {
      let nv = h4(dot(c0v, axis), dot(c1v, axis), dot(c2v, axis), dot(c3v, axis));
      let m = max(max(abs(nv.x), abs(nv.y)), max(abs(nv.z), abs(nv.w)));
      if (m < h(1e-4)) { axis_ok = false; break; }
      axis = nv / m;
    }
    if (axis_ok) {
      axis = axis / length(axis);
      var t_min = h(4.0);
      var t_max = h(-4.0);
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let t = dot(pix[k] - mean, axis);
        t_min = min(t_min, t);
        t_max = max(t_max, t);
      }
      seed_lo = clamp(mean + t_min * axis, h4(0.0), h4(1.0));
      seed_hi = clamp(mean + t_max * axis, h4(0.0), h4(1.0));
    }
    // Refit result clamped to the block bbox: on multi-cluster blocks the
    // unconstrained solve extrapolates outside the block's colours and the
    // per-channel [0,1] clamp would bend the hue.
    var e0 = lo;
    var e1 = hi;
    var fitStream = 0u;
    var haveFitWeights = false;
    let r = proj_fit(&pix, seed_lo, seed_hi);
    if (r.valid) {
      e0 = clamp(r.e0, lo, hi);
      e1 = clamp(r.e1, lo, hi);
      fitStream = r.wstream;
      haveFitWeights = true;
    }
    var E0 = q8(e0);
    var E1 = q8(e1);
    var swapped = false;
    if (E0.x + E0.y + E0.z > E1.x + E1.y + E1.z) {
      let t = E0; E0 = E1; E1 = t;
      swapped = true;
    }
    // Valid fits ship the FIT-PASS weights instead of reprojecting (a whole
    // 16-pixel pass for −0.09 dB on the alpha card); the swap is a full
    // reflection w → 3 − w, i.e. bitwise NOT of the stream. Degenerate fits
    // reproject against the bbox endpoints.
    var s0 = 0u;
    if (haveFitWeights) {
      s0 = select(fitStream, ~fitStream, swapped);
    } else {
      let d0 = h4(vec4<f32>(E0)) * h(1.0 / 255.0);
      let d1 = h4(vec4<f32>(E1)) * h(1.0 / 255.0);
      let dir = (d1 - d0) * h(32.0);
      let dd = dot(dir, dir);
      if (dd >= h(0.008)) {
        let inv = h(96.0) / dd;
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
          let w = hbits(clamp(floor(dot(pix[k] - d0, dir) * inv + h(0.5)), h(0.0), h(3.0)), 3u);
          s0 = s0 | (w << (2u * k));
        }
      }
    }
    // Mode 0x042, CEM 12 @13, endpoints R0 R1 G0 G1 B0 B1 A0 A1 from 17.
    w0 = 0x042u | (12u << 13u) | (E0.x << 17u) | (E1.x << 25u);
    w1 = (E1.x >> 7u) | (E0.y << 1u) | (E1.y << 9u) | (E0.z << 17u) | (E1.z << 25u);
    w2 = (E1.z >> 7u) | (E0.w << 1u) | (E1.w << 9u);
    w3 = reverseBits(s0);
  }

  let o = bi * 4u;
  dst[o] = w0; dst[o + 1u] = w1; dst[o + 2u] = w2; dst[o + 3u] = w3;
}
