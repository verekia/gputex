// ASTC 4×4 LDR compute shader encoder.
//
// One invocation per 4×4 block. Emits 16 bytes = 4 u32s into the storage
// buffer at `dst[block_index * 4 .. + 3]`. This is the f32 fallback;
// astc4x4_fast_f16.wgsl is the same algorithm and is preferred when the
// device reports shader-f16.
//
// ALGORITHM: principal-axis seed (covariance power-iteration; bbox on
// degenerate blocks) at the exact projection extents → one fused pass that
// projects each pixel onto the endpoint line (the 4 palette entries are
// colinear, so the nearest is the rounded projection — no per-entry search)
// while accumulating the least-squares refit sums, then a reprojection
// against the quantised refit endpoints with the weights packed on the fly.
// The endpoint ordering rule is applied before the weight pass, so no
// reflection is needed.
//
// RESTRICTED SUBSET: single partition, no dual-plane, CEM 12 (LDR RGBA direct),
// 4×4 weight grid with 2-bit weights (QUANT_4), 8-bit endpoints (QUANT_256).
//
// BLOCK LAYOUT (128 bits, LSB-first)
//   bits [10:0]   block mode = 0x042
//   bits [12:11]  partition count − 1 = 0
//   bits [16:13]  CEM = 12
//   bits [80:17]  endpoints: R0 R1 G0 G1 B0 B1 A0 A1 (8-bit each)
//   bits [127:96] 16 × 2-bit weights; weight k: bit(127−2k)=lsb, bit(126−2k)=msb
//
// ENDPOINT ORDERING: if sum(e0.rgb) > sum(e1.rgb) swap endpoints and reflect
// indices (w' = 3 − w) to keep the decoder out of blue contraction.

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn to8(v: vec4<f32>) -> vec4<i32> {
  return vec4<i32>(clamp(floor(v * 255.0 + 0.5), vec4<f32>(0.0), vec4<f32>(255.0)));
}

// One pass over the block: project every pixel onto the e0→e1 line (4 levels,
// QUANT_4 ≈ thirds) and accumulate the least-squares normal-equation sums;
// solve for the refit endpoints. Weights are not produced here — the caller
// reprojects against the quantised refit endpoints anyway.
struct Fit { e0: vec4<i32>, e1: vec4<i32>, valid: bool };
fn proj_fit(pixels: ptr<function, array<vec4<i32>, 16>>, e0: vec4<i32>, e1: vec4<i32>) -> Fit {
  var out: Fit;
  out.valid = false;
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
  for (var iter: u32 = 0u; iter < 8u; iter = iter + 1u) {
    let nv = vec4<f32>(dot(c0v, v), dot(c1v, v), dot(c2v, v), dot(c3v, v));
    len = length(nv);
    if (len < 1e-12) { return vec4<f32>(0.0); }
    v = nv / len;
  }
  return v;
}

// ------------------------------- Entry ---------------------------------- //

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
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
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0, 0), max_xy);
    let px = to8(textureLoad(src_tex, p, 0));
    pixels[i] = px;
    lo = min(lo, px);
    hi = max(hi, px);
    isum = isum + px;
  }
  let mean = vec4<f32>(isum) * (1.0 / 16.0);

  // Fused LSQ fit seeded from the block's principal colour axis at the exact
  // projection extents, quantised refit endpoints, ordering applied BEFORE
  // the weight pass so no reflection is needed.
  // The refit is clamped to the block bbox: on multi-cluster blocks the
  // unconstrained solve extrapolates far outside the block's colours and the
  // per-channel [0,255] clamp then bends the hue — fringe pixels decode to
  // colours that exist nowhere in the block. Constraining to the bbox also
  // measures better in plain SSE (+1.8 dB on the colour test card).
  var seed0 = lo;
  var seed1 = hi;
  let axis = principal_axis4(&pixels, mean, vec4<f32>(hi - lo));
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
  let r = proj_fit(&pixels, seed0, seed1);
  var e0 = lo;
  var e1 = hi;
  if (r.valid) { e0 = clamp(r.e0, lo, hi); e1 = clamp(r.e1, lo, hi); }
  if (e0.x + e0.y + e0.z > e1.x + e1.y + e1.z) {
    let tmp = e0; e0 = e1; e1 = tmp;
  }

  // Weight pass, packing on the fly (weight k's lsb at bit 31−2k, msb at
  // bit 30−2k).
  var w3: u32 = 0u;
  let dir = vec4<f32>(e1 - e0);
  let dd = dot(dir, dir);
  if (dd > 0.0) {
    let e0f = vec4<f32>(e0);
    let inv = 3.0 / dd;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let s = u32(clamp(floor(dot(vec4<f32>(pixels[k]) - e0f, dir) * inv + 0.5), 0.0, 3.0));
      w3 = w3 | ((s & 1u) << (31u - 2u * k)) | (((s >> 1u) & 1u) << (30u - 2u * k));
    }
  }

  // Straight-line packing: block mode 0x042 @0, partitions−1=0 @11, CEM 12
  // @13, endpoints R0 R1 G0 G1 B0 B1 A0 A1 (8 bits each) from bit 17,
  // weights in the last word.
  let E0 = vec4<u32>(e0);
  let E1 = vec4<u32>(e1);
  let w0 = 0x042u | (12u << 13u) | (E0.x << 17u) | (E1.x << 25u);
  let w1 = (E1.x >> 7u) | (E0.y << 1u) | (E1.y << 9u) | (E0.z << 17u) | (E1.z << 25u);
  let w2 = (E1.z >> 7u) | (E0.w << 1u) | (E1.w << 9u);

  let out = block_index * 4u;
  dst[out + 0u] = w0;
  dst[out + 1u] = w1;
  dst[out + 2u] = w2;
  dst[out + 3u] = w3;
}
