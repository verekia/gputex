// ASTC 4×4 LDR compute shader encoder.
//
// One invocation per 4×4 block. Emits 16 bytes = 4 u32s into the storage
// buffer at `dst[block_index * 4 .. + 3]`. This is the f32 fallback;
// astc4x4_fast_f16.wgsl is the same algorithm (and documents the design
// measurements) and is preferred when the device reports shader-f16.
//
// ALGORITHM: per-block class selection, then a single-line fit:
//   gray + opaque → CEM 0  (luminance), 5-bit weights, block mode 0x253 —
//                   scalar path: exact min/max endpoints, 32-level weight
//                   assignment, no covariance / iteration / refit needed
//   opaque        → CEM 8  (RGB), 3-channel PCA extents, two bit budgets:
//                   span > 12 → QUANT_192 endpoints (trit ISE) + 4-bit
//                   weights, mode 0x242; span ≤ 12 → 8-bit endpoints +
//                   3-bit weights, mode 0x053
//   translucent   → CEM 12 (RGBA), 2-bit weights, block mode 0x042 — PCA
//                   seed → fused projection + least-squares refit, the fit
//                   pass's weights shipped
// The palette entries of a single-line fit are colinear, so the nearest is
// the rounded projection — no per-entry search. The endpoint ordering rule
// is applied before the weight pass, so no reflection is needed.
//
// RESTRICTED SUBSET + BLOCK LAYOUT: see astc4x4_ref.ts (single partition,
// no dual-plane, CEM 0/8/12, plain-bit weights; block mode derivations,
// the QUANT_192 trit ISE and the weight-stream bit order are documented
// there).
//
// WEIGHT PLACEMENT: stream bit q (bit j of weight k, q = nBits·k + j) lives
// at block bit 127 − q, so a stream word assembled LSB-first maps onto a
// block word with a single reverseBits().
//
// ENDPOINT ORDERING: CEM 8/12 decoders branch into blue contraction when
// sum(e0.rgb) > sum(e1.rgb) (unquantised values); the encoder swaps
// endpoints up front (weights are assigned after the swap, so no
// reflection pass). CEM 0 has no rule (L0 ≤ L1 by construction).

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
  y0:       u32, // first block row of this dispatch (row-band encodes)
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn to8(v: vec4<f32>) -> vec4<i32> {
  return vec4<i32>(clamp(floor(v * 255.0 + 0.5), vec4<f32>(0.0), vec4<f32>(255.0)));
}

// One pass over the block: project every pixel onto the e0→e1 line
// (lmax + 1 colinear levels, so the nearest entry is the rounded
// projection) and accumulate the least-squares normal-equation sums;
// solve for the refit endpoints. Weights are not produced here — the
// caller reprojects against the quantised refit endpoints anyway.
struct Fit { e0: vec4<i32>, e1: vec4<i32>, valid: bool, wstream: u32 };
fn proj_fit(pixels: ptr<function, array<vec4<i32>, 16>>, e0: vec4<i32>, e1: vec4<i32>) -> Fit {
  var out: Fit;
  out.valid = false;
  out.wstream = 0u;
  let dir = vec4<f32>(e1 - e0);
  let dd = dot(dir, dir);
  if (dd == 0.0) { return out; }
  let e0f = vec4<f32>(e0);
  let inv = 3.0 / dd;
  var sAA: f32 = 0.0; var sBB: f32 = 0.0; var sAB: f32 = 0.0;
  var sAV: vec4<f32> = vec4<f32>(0.0); var sBV: vec4<f32> = vec4<f32>(0.0);
  var s_min = 3.0; var s_max = 0.0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let v = vec4<f32>((*pixels)[k]);
    let s = clamp(floor(dot(v - e0f, dir) * inv + 0.5), 0.0, 3.0);
    out.wstream = out.wstream | (u32(s) << (2u * k));
    s_min = min(s_min, s); s_max = max(s_max, s);
    let b = s * (1.0 / 3.0); let a = 1.0 - b;
    sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b;
    sAV = sAV + a * v; sBV = sBV + b * v;
  }
  // Rank-1 guard: if every pixel projects to ONE level the system is
  // singular — det and the numerators are pure float rounding noise and the
  // solve returns garbage endpoints. With ≥2 levels det ≥ 15·(1/3)² ≈ 1.67.
  if (s_min == s_max) { return out; }
  let det = sAA * sBB - sAB * sAB;
  if (abs(det) < 1e-3) { return out; }
  out.e0 = vec4<i32>(clamp(round((sBB * sAV - sAB * sBV) / det), vec4<f32>(0.0), vec4<f32>(255.0)));
  out.e1 = vec4<i32>(clamp(round((sAA * sBV - sAB * sAV) / det), vec4<f32>(0.0), vec4<f32>(255.0)));
  out.valid = true;
  return out;
}

// Principal colour axis via covariance power-iteration (RGBA, 8-bit integer
// pixel domain), seeded with the bbox diagonal. Returns a unit axis, or
// vec4(0) for a degenerate (constant) block. Used to seed the LSQ fit — the
// bbox diagonal is sign-blind and points across anti-correlated data (normal
// maps, hue edges) instead of along it.
fn principal_axis4(
  pixels: ptr<function, array<vec4<i32>, 16>>,
  mean: vec4<f32>,
  seed: vec4<f32>,
  iters: u32,
) -> vec4<f32> {
  var c0v = vec4<f32>(0.0);
  var c1v = vec4<f32>(0.0);
  var c2v = vec4<f32>(0.0);
  var c3v = vec4<f32>(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let d = vec4<f32>((*pixels)[k]) - mean;
    c0v = c0v + d.x * d;
    c1v = c1v + d.y * d;
    c2v = c2v + d.z * d;
    c3v = c3v + d.w * d;
  }
  var v = seed;
  var len = length(v);
  if (len < 1e-9) { return vec4<f32>(0.0); }
  v = v / len;
  for (var iter: u32 = 0u; iter < iters; iter = iter + 1u) {
    let nv = vec4<f32>(dot(c0v, v), dot(c1v, v), dot(c2v, v), dot(c3v, v));
    len = length(nv);
    if (len < 1e-12) { return vec4<f32>(0.0); }
    v = nv / len;
  }
  return v;
}

// Principal RGB axis for opaque blocks (alpha is constant there, so the
// 4th covariance lane is dead weight). Same iteration as principal_axis4.
fn principal_axis3(
  pixels: ptr<function, array<vec4<i32>, 16>>,
  mean: vec3<f32>,
  seed: vec3<f32>,
) -> vec3<f32> {
  var c0v = vec3<f32>(0.0);
  var c1v = vec3<f32>(0.0);
  var c2v = vec3<f32>(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let d = vec3<f32>((*pixels)[k].xyz) - mean;
    c0v = c0v + d.x * d;
    c1v = c1v + d.y * d;
    c2v = c2v + d.z * d;
  }
  var v = seed;
  var len = length(v);
  if (len < 1e-9) { return vec3<f32>(0.0); }
  v = v / len;
  for (var iter: u32 = 0u; iter < 8u; iter = iter + 1u) {
    let nv = vec3<f32>(dot(c0v, v), dot(c1v, v), dot(c2v, v));
    len = length(nv);
    if (len < 1e-12) { return vec3<f32>(0.0); }
    v = nv / len;
  }
  return v;
}

// ISE trit-block encoder: 5 trits → the 8-bit T field (the inverse of the
// spec's trit-block decode; see astc4x4_ref.ts).
fn trit_enc(t0: u32, t1: u32, t2: u32, t3: u32, t4: u32) -> u32 {
  let c = select(select((t2 << 4u) | (t1 << 2u) | t0, (t1 << 4u) | (t0 << 2u) | 3u, t2 == 2u), 12u | t0, t2 == 2u && t1 == 2u);
  return select(select((t4 << 7u) | (t3 << 5u) | c, (t3 << 7u) | 96u | c, t4 == 2u), ((c >> 2u) << 5u) | 28u | (c & 3u), t3 == 2u && t4 == 2u);
}

// Nearest QUANT_192 endpoint to x ∈ [0,255]; returns (ISE value =
// trit·64 + bits, unquantised level). See astc4x4_fast_f16.wgsl.
fn q192(x: f32) -> vec2<u32> {
  let v = u32(clamp(floor(x + 0.5), 0.0, 255.0));
  let up = v > 127u;
  var u = select(v, 255u - v, up);
  if ((u & 3u) == 3u) {
    let xu = select(x, 255.0 - x, up);
    u = select(u - 1u, u + 1u, xu > f32(u) && u < 127u);
  }
  return vec2<u32>(((u & 3u) << 6u) | ((u >> 2u) << 1u) | u32(up), select(u, 255u - u, up));
}

// ------------------------------- Entry ---------------------------------- //

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid_raw: vec3<u32>) {
  // Row-band encodes dispatch a slice of the block grid starting at row y0.
  let gid = vec3<u32>(gid_raw.x, gid_raw.y + params.y0, gid_raw.z);
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) {
    return;
  }

  let bx = gid.x;
  let by = gid.y;
  let block_index = by * params.blocks_x + bx;

  let base   = vec2<i32>(i32(bx) * 4, i32(by) * 4);
  let max_xy = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  var pixels: array<vec4<i32>, 16>;
  var lo = vec4<i32>(255);
  var hi = vec4<i32>(0);
  var isum = vec4<i32>(0);
  var gd = 0; // max |R−G|, |R−B| over the block; 0 ⇔ exactly grayscale
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0, 0), max_xy);
    let px = to8(textureLoad(src_tex, p, 0));
    pixels[i] = px;
    lo = min(lo, px);
    hi = max(hi, px);
    isum = isum + px;
    gd = max(gd, max(abs(px.x - px.y), abs(px.x - px.z)));
  }
  let opaque = lo.w == 255;

  var w0: u32; var w1: u32; var w2: u32; var w3: u32;

  if (opaque && gd == 0) {
    // ---------------- Luminance path: CEM 0, 5-bit weights ----------------
    // Endpoints at the exact extremes; 32 palette levels make an LSQ refit
    // unnecessary.
    let L0 = u32(lo.x);
    let L1 = u32(hi.x);
    var s0 = 0u; var s1 = 0u; var s2 = 0u;
    if (L1 > L0) {
      let sc = 64.0 / f32(hi.x - lo.x);
      // Exact nearest entry of the QUANT_32 grid: unq = 2w for w ≤ 15,
      // 2w + 2 for w ≥ 16 (4-wide gap at the middle, so uniform rounding
      // is wrong there). Best candidate of each half, keep the closer.
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let u = clamp(f32(pixels[k].x - lo.x) * sc, 0.0, 64.0);
        let wlo = clamp(floor(u * 0.5 + 0.5), 0.0, 15.0);
        let whi = clamp(floor((u - 2.0) * 0.5 + 0.5), 16.0, 31.0);
        let pick = abs(u - wlo * 2.0) <= abs(u - (whi * 2.0 + 2.0));
        let w = u32(select(whi, wlo, pick));
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
    // into w1 bit 0); stream words map onto block words via reverseBits.
    w0 = 0x253u | (L0 << 17u) | (L1 << 25u);
    w1 = (L1 >> 7u) | reverseBits(s2);
    w2 = reverseBits(s1);
    w3 = reverseBits(s0);
  } else if (opaque) {
    // ------------- Opaque colour: CEM 8, two bit budgets -------------------
    // span > 12 → QUANT_192 endpoints + 4-bit weights (mode 0x242), else
    // exact 8-bit endpoints + 3-bit weights (mode 0x053); endpoints are the
    // bbox-clamped PCA extents. See astc4x4_fast_f16.wgsl for the design
    // and measurements.
    let mean3 = vec3<f32>(isum.xyz) * (1.0 / 16.0);
    let lo3 = vec3<f32>(lo.xyz);
    let hi3 = vec3<f32>(hi.xyz);
    var x0 = lo3;
    var x1 = hi3;
    let axis = principal_axis3(&pixels, mean3, hi3 - lo3);
    if (dot(axis, axis) > 0.0) {
      var t_min: f32 = 1e30;
      var t_max: f32 = -1e30;
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let t = dot(vec3<f32>(pixels[k].xyz) - mean3, axis);
        t_min = min(t_min, t);
        t_max = max(t_max, t);
      }
      x0 = clamp(mean3 + t_min * axis, lo3, hi3);
      x1 = clamp(mean3 + t_max * axis, lo3, hi3);
    }
    let span3 = hi.xyz - lo.xyz;
    let small = max(max(span3.x, span3.y), span3.z) <= 12;
    var r0: vec2<u32>; var g0: vec2<u32>; var b0: vec2<u32>;
    var r1: vec2<u32>; var g1: vec2<u32>; var b1: vec2<u32>;
    if (small) {
      let q0 = vec3<u32>(clamp(floor(x0 + 0.5), vec3<f32>(0.0), vec3<f32>(255.0)));
      let q1 = vec3<u32>(clamp(floor(x1 + 0.5), vec3<f32>(0.0), vec3<f32>(255.0)));
      r0 = vec2<u32>(q0.x); g0 = vec2<u32>(q0.y); b0 = vec2<u32>(q0.z);
      r1 = vec2<u32>(q1.x); g1 = vec2<u32>(q1.y); b1 = vec2<u32>(q1.z);
    } else {
      r0 = q192(x0.x); g0 = q192(x0.y); b0 = q192(x0.z);
      r1 = q192(x1.x); g1 = q192(x1.y); b1 = q192(x1.z);
    }
    // Blue-contraction ordering on the unquantised levels.
    if (r0.y + g0.y + b0.y > r1.y + g1.y + b1.y) {
      let tr = r0; r0 = r1; r1 = tr;
      let tg = g0; g0 = g1; g1 = tg;
      let tb = b0; b0 = b1; b1 = tb;
    }
    let d0 = vec3<f32>(vec3<u32>(r0.y, g0.y, b0.y));
    let d1 = vec3<f32>(vec3<u32>(r1.y, g1.y, b1.y));
    // One weight loop for both budgets; weights as 4-bit nibbles.
    let lmax = select(15.0, 7.0, small);
    let dir = d1 - d0;
    let dd = dot(dir, dir);
    var s0 = 0u; var s1 = 0u;
    if (dd > 0.0) {
      let inv = lmax / dd;
      for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let w = u32(clamp(floor(dot(vec3<f32>(pixels[k].xyz) - d0, dir) * inv + 0.5), 0.0, lmax));
        s0 = s0 | (w << (4u * k));
      }
      for (var k: u32 = 8u; k < 16u; k = k + 1u) {
        let w = u32(clamp(floor(dot(vec3<f32>(pixels[k].xyz) - d0, dir) * inv + 0.5), 0.0, lmax));
        s1 = s1 | (w << (4u * (k - 8u)));
      }
    }
    if (small) {
      // Mode 0x053: plain 8-bit endpoints; the 3-bit weight stream is the
      // nibbles compacted (8 nibbles → 24 bits).
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
      // Mode 0x242: trit-ISE QUANT_192 endpoints (v0..v5 = R0 R1 G0 G1 B0
      // B1, 46 bits from bit 17: group 1 = v0..v4 with trit field T, group
      // 2 = v5 with its lone trit as 2 bits), 4-bit weights.
      let tg = trit_enc(r0.x >> 6u, r1.x >> 6u, g0.x >> 6u, g1.x >> 6u, b0.x >> 6u);
      w0 = 0x242u | (8u << 13u) | ((r0.x & 63u) << 17u) | ((tg & 3u) << 23u) | ((r1.x & 63u) << 25u) | (((tg >> 2u) & 1u) << 31u);
      w1 = ((tg >> 3u) & 1u) | ((g0.x & 63u) << 1u) | (((tg >> 4u) & 1u) << 7u) | ((g1.x & 63u) << 8u)
        | (((tg >> 5u) & 3u) << 14u) | ((b0.x & 63u) << 16u) | ((tg >> 7u) << 22u) | ((b1.x & 63u) << 23u)
        | ((b1.x >> 6u) << 29u);
      w2 = reverseBits(s1);
      w3 = reverseBits(s0);
    }
  } else {
    // ------------- Translucent: CEM 12, 2-bit weights, PCA seed + refit ----
    let mean = vec4<f32>(isum) * (1.0 / 16.0);

    // Fused LSQ fit seeded from the block's principal RGBA axis (4 power
    // iterations — the refit absorbs residual axis error) at the exact
    // projection extents. The refit is clamped to the block bbox: on
    // multi-cluster blocks the unconstrained solve extrapolates outside the
    // block's colours and the per-channel clamp would bend the hue.
    var seed0 = lo;
    var seed1 = hi;
    let axis = principal_axis4(&pixels, mean, vec4<f32>(hi - lo), 4u);
    if (dot(axis, axis) > 0.0) {
      var t_min: f32 = 1e30;
      var t_max: f32 = -1e30;
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let t = dot(vec4<f32>(pixels[k]) - mean, axis);
        t_min = min(t_min, t);
        t_max = max(t_max, t);
      }
      seed0 = vec4<i32>(clamp(round(mean + t_min * axis), vec4<f32>(0.0), vec4<f32>(255.0)));
      seed1 = vec4<i32>(clamp(round(mean + t_max * axis), vec4<f32>(0.0), vec4<f32>(255.0)));
    }
    var e0 = lo;
    var e1 = hi;
    var fitStream = 0u;
    var haveFitWeights = false;
    let r = proj_fit(&pixels, seed0, seed1);
    if (r.valid) {
      e0 = clamp(r.e0, lo, hi);
      e1 = clamp(r.e1, lo, hi);
      fitStream = r.wstream;
      haveFitWeights = true;
    }
    var swapped = false;
    if (e0.x + e0.y + e0.z > e1.x + e1.y + e1.z) {
      let tmp = e0; e0 = e1; e1 = tmp;
      swapped = true;
    }
    let E0 = vec4<u32>(e0);
    let E1 = vec4<u32>(e1);
    // Valid fits ship the fit-pass weights; the blue-contraction swap is a
    // full reflection w → 3−w = bitwise NOT of the packed stream.
    var s0 = 0u;
    if (haveFitWeights) {
      s0 = select(fitStream, ~fitStream, swapped);
    } else {
      let dir = vec4<f32>(e1 - e0);
      let dd = dot(dir, dir);
      let e0f = vec4<f32>(e0);
      if (dd > 0.0) {
        let inv = 3.0 / dd;
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
          let w = u32(clamp(floor(dot(vec4<f32>(pixels[k]) - e0f, dir) * inv + 0.5), 0.0, 3.0));
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

  let out = block_index * 4u;
  dst[out + 0u] = w0;
  dst[out + 1u] = w1;
  dst[out + 2u] = w2;
  dst[out + 3u] = w3;
}
