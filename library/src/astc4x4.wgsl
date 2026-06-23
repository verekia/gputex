// ASTC 4×4 LDR compute shader encoder.
//
// One invocation per 4×4 block. Emits 16 bytes = 4 u32s into the storage
// buffer at `dst[block_index * 4 .. + 3]`.
//
// QUALITY LEVELS (pipeline-overridable constant `QUALITY_HIGH`)
//   fast (0, default): O(N) bounding-box seed → endpoints fitted by a single
//     least-squares pass whose sums are accumulated during a projection-based
//     weight assignment (the 4 palette entries are colinear, so the nearest is
//     found by projecting onto the endpoint line — no per-entry search).
//     Profiled ~4× faster than `high` for ~0.36 dB PSNR.
//   high (1): O(N²) farthest-pair seed, full 4-entry nearest search, one LSQ
//     refit — byte-for-byte identical to astc4x4_ref.ts.
// The fast branch is selected at pipeline-compile time; the driver eliminates
// the unused (high) code.
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

// 0 = fast (default), 1 = exhaustive/high-quality. Set via pipeline constants.
override QUALITY_HIGH: u32 = 0u;

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn weight_unq(i: u32) -> i32 {
  switch i {
    case 0u: { return  0; }
    case 1u: { return 21; }
    case 2u: { return 43; }
    default: { return 64; }  // case 3u
  }
}

fn interp4(e0: vec4<i32>, e1: vec4<i32>, w: i32) -> vec4<i32> {
  return ((64 - w) * e0 + w * e1 + vec4<i32>(32)) >> vec4<u32>(6u);
}

fn to8(v: vec4<f32>) -> vec4<i32> {
  return vec4<i32>(clamp(floor(v * 255.0 + 0.5), vec4<f32>(0.0), vec4<f32>(255.0)));
}

fn dist2(a: vec4<i32>, b: vec4<i32>) -> i32 {
  let d = a - b;
  let e = d * d;
  return e.x + e.y + e.z + e.w;
}

// ============================ FAST PATH ================================ //

// Projection weight assignment over 4 levels (QUANT_4 ≈ thirds), with the LSQ
// normal-equation sums accumulated in the same pass for a fused refit.
struct Fit { e0: vec4<i32>, e1: vec4<i32>, valid: bool };
fn proj_assign(
  pixels: ptr<function, array<vec4<i32>, 16>>,
  e0: vec4<i32>, e1: vec4<i32>,
  out_idx: ptr<function, array<u32, 16>>,
  fit: bool,
) -> Fit {
  var out: Fit;
  let dir = e1 - e0;
  let dd = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + dir.w * dir.w;
  if (dd == 0) {
    for (var k: u32 = 0u; k < 16u; k = k + 1u) { (*out_idx)[k] = 0u; }
    out.valid = false;
    return out;
  }
  let inv = 3.0 / f32(dd);
  var sAA: f32 = 0.0; var sBB: f32 = 0.0; var sAB: f32 = 0.0;
  var sAV: vec4<f32> = vec4<f32>(0.0); var sBV: vec4<f32> = vec4<f32>(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let q = (*pixels)[k] - e0;
    let s = clamp(floor(f32(q.x * dir.x + q.y * dir.y + q.z * dir.z + q.w * dir.w) * inv + 0.5), 0.0, 3.0);
    (*out_idx)[k] = u32(s);
    if (fit) {
      let v = vec4<f32>((*pixels)[k]);
      let b = s / 3.0; let a = 1.0 - b;
      sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b; sAV = sAV + a * v; sBV = sBV + b * v;
    }
  }
  if (!fit) { out.valid = false; return out; }
  let det = sAA * sBB - sAB * sAB;
  if (abs(det) < 1e-9) { out.valid = false; return out; }
  out.e0 = vec4<i32>(clamp(round((sBB * sAV - sAB * sBV) / det), vec4<f32>(0.0), vec4<f32>(255.0)));
  out.e1 = vec4<i32>(clamp(round((sAA * sBV - sAB * sAV) / det), vec4<f32>(0.0), vec4<f32>(255.0)));
  out.valid = true;
  return out;
}

// ============================ HIGH PATH ================================ //

struct Pair { a: vec4<i32>, b: vec4<i32> };
fn farthest_pair(pixels: ptr<function, array<vec4<i32>, 16>>) -> Pair {
  var best_d: i32 = 0;
  var pa = (*pixels)[0];
  var pb = (*pixels)[1];
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let xi = (*pixels)[i];
    for (var j: u32 = i + 1u; j < 16u; j = j + 1u) {
      let d = dist2(xi, (*pixels)[j]);
      if (d > best_d) { best_d = d; pa = xi; pb = (*pixels)[j]; }
    }
  }
  return Pair(pa, pb);
}

fn build_palette(e0: vec4<i32>, e1: vec4<i32>, pal: ptr<function, array<vec4<i32>, 4>>) {
  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    (*pal)[i] = interp4(e0, e1, weight_unq(i));
  }
}

fn assign_all(
  pixels:  ptr<function, array<vec4<i32>, 16>>,
  pal:     ptr<function, array<vec4<i32>, 4>>,
  out_idx: ptr<function, array<u32, 16>>,
) -> i32 {
  var err: i32 = 0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let px = (*pixels)[k];
    var best_i: u32 = 0u;
    var best_d: i32 = 2147483647;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      let d = dist2(px, (*pal)[i]);
      if (d < best_d) { best_d = d; best_i = i; }
    }
    (*out_idx)[k] = best_i;
    err = err + best_d;
  }
  return err;
}

struct RefitResult { e0: vec4<i32>, e1: vec4<i32>, valid: bool };
fn refit_endpoints(
  pixels:  ptr<function, array<vec4<i32>, 16>>,
  indices: ptr<function, array<u32, 16>>,
) -> RefitResult {
  var sAA: f32 = 0.0;
  var sBB: f32 = 0.0;
  var sAB: f32 = 0.0;
  var sAV: vec4<f32> = vec4<f32>(0.0);
  var sBV: vec4<f32> = vec4<f32>(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let unq = weight_unq((*indices)[k]);
    let a = f32(64 - unq) / 64.0;
    let b = f32(unq)      / 64.0;
    let v = vec4<f32>((*pixels)[k]);
    sAA = sAA + a * a;
    sBB = sBB + b * b;
    sAB = sAB + a * b;
    sAV = sAV + a * v;
    sBV = sBV + b * v;
  }
  let det = sAA * sBB - sAB * sAB;
  var out: RefitResult;
  if (abs(det) < 1e-9) {
    out.valid = false;
    return out;
  }
  let e0f = (sBB * sAV - sAB * sBV) / det;
  let e1f = (sAA * sBV - sAB * sAV) / det;
  out.e0 = vec4<i32>(clamp(round(e0f), vec4<f32>(0.0), vec4<f32>(255.0)));
  out.e1 = vec4<i32>(clamp(round(e1f), vec4<f32>(0.0), vec4<f32>(255.0)));
  out.valid = true;
  return out;
}

// -------------------------- Bit-packing helper -------------------------- //

fn write_bits(block: ptr<function, array<u32, 4>>, pos: u32, n_bits: u32, value: u32) {
  let v = value & ((1u << n_bits) - 1u);
  let word_lo = pos / 32u;
  let bit_lo = pos % 32u;
  let bits_in_lo = min(n_bits, 32u - bit_lo);
  let mask_lo = ((1u << bits_in_lo) - 1u) << bit_lo;
  (*block)[word_lo] = ((*block)[word_lo] & ~mask_lo) | ((v << bit_lo) & mask_lo);
  if (bits_in_lo < n_bits) {
    let bits_in_hi = n_bits - bits_in_lo;
    let mask_hi = (1u << bits_in_hi) - 1u;
    let val_hi = v >> bits_in_lo;
    (*block)[word_lo + 1u] = ((*block)[word_lo + 1u] & ~mask_hi) | (val_hi & mask_hi);
  }
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
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0, 0), max_xy);
    let px = to8(textureLoad(src_tex, p, 0));
    pixels[i] = px;
    lo = min(lo, px);
    hi = max(hi, px);
  }

  var e0: vec4<i32>;
  var e1: vec4<i32>;
  var indices: array<u32, 16>;

  if (QUALITY_HIGH != 0u) {
    let fp = farthest_pair(&pixels);
    e0 = fp.a;
    e1 = fp.b;
    var pal: array<vec4<i32>, 4>;
    build_palette(e0, e1, &pal);
    var err = assign_all(&pixels, &pal, &indices);
    let refit = refit_endpoints(&pixels, &indices);
    if (refit.valid) {
      build_palette(refit.e0, refit.e1, &pal);
      var idx2: array<u32, 16>;
      let err2 = assign_all(&pixels, &pal, &idx2);
      if (err2 < err) {
        e0 = refit.e0;
        e1 = refit.e1;
        indices = idx2;
        err = err2;
      }
    }
  } else {
    e0 = lo;
    e1 = hi;
    let r = proj_assign(&pixels, e0, e1, &indices, true);
    if (r.valid) {
      e0 = r.e0;
      e1 = r.e1;
      proj_assign(&pixels, e0, e1, &indices, false);
    }
  }

  // Endpoint ordering so the decoder doesn't apply blue contraction.
  if (e0.x + e0.y + e0.z > e1.x + e1.y + e1.z) {
    let tmp = e0; e0 = e1; e1 = tmp;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      indices[k] = 3u - indices[k];
    }
  }

  var block: array<u32, 4>;
  block[0] = 0u; block[1] = 0u; block[2] = 0u; block[3] = 0u;
  write_bits(&block, 0u,  11u, 0x042u);
  write_bits(&block, 11u, 2u,  0u);
  write_bits(&block, 13u, 4u,  12u);
  write_bits(&block, 17u + 0u * 8u, 8u, u32(e0.x));
  write_bits(&block, 17u + 1u * 8u, 8u, u32(e1.x));
  write_bits(&block, 17u + 2u * 8u, 8u, u32(e0.y));
  write_bits(&block, 17u + 3u * 8u, 8u, u32(e1.y));
  write_bits(&block, 17u + 4u * 8u, 8u, u32(e0.z));
  write_bits(&block, 17u + 5u * 8u, 8u, u32(e1.z));
  write_bits(&block, 17u + 6u * 8u, 8u, u32(e0.w));
  write_bits(&block, 17u + 7u * 8u, 8u, u32(e1.w));
  var w3: u32 = 0u;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let w = indices[k] & 0x3u;
    w3 = w3 | ((w & 1u) << (31u - 2u * k)) | (((w >> 1u) & 1u) << (30u - 2u * k));
  }
  block[3] = w3;

  let out = block_index * 4u;
  dst[out + 0u] = block[0];
  dst[out + 1u] = block[1];
  dst[out + 2u] = block[2];
  dst[out + 3u] = block[3];
}
