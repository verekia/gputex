// BC7 (BPTC) mode 6 compute shader encoder.
//
// One invocation per 4×4 block. Emits 16 bytes = 4 u32s into the storage
// buffer at `dst[block_index * 4 .. + 3]`.
//
// QUALITY LEVELS (pipeline-overridable constant `QUALITY_HIGH`)
//   fast (0, default): O(N) bounding-box seed → endpoints fitted by a single
//     least-squares pass whose normal-equation sums are accumulated *during* a
//     projection-based index assignment. The 16 palette entries are colinear
//     (pal[i] = lerp(e0,e1,w[i])), so the nearest index is found by projecting
//     each pixel onto the endpoint line — O(1) per pixel, no palette build and
//     no 16-entry search. Profiled ~20× faster than `high` for ~0.4 dB PSNR.
//   high (1): farthest-pair seed, exhaustive p-bit search over all four
//     (p0,p1) ∈ {0,1}² combos, full 16-entry nearest search, one LSQ refit —
//     byte-for-byte identical to bc7_ref.ts.
//
// Both paths run in the i32 domain. The fast path's branch is selected at
// pipeline-compile time, so the driver eliminates the unused (high) code.
//
// MODE 6 LAYOUT (LSB-first, bit 0 = byte 0's bit 0)
//   bits 0..6    mode field      (0b0000001 — only bit 6 is 1)
//   bits 7..13   R0 (7-bit)   bits 14..20 R1   bits 21..27 G0   bits 28..34 G1
//   bits 35..41  B0   bits 42..48 B1   bits 49..55 A0   bits 56..62 A1
//   bit  63      P0   bit 64 P1
//   bits 65..67  pixel 0 index (3 bits; anchor, MSB implicit 0)
//   bits 68..71  pixel 1 index (4 bits) ... bits 124..127 pixel 15 index
//
// Effective 8-bit endpoint channel = (7_bit_value << 1) | p_bit.
// Palette[i] = ((64 − W4[i]) × e0_8 + W4[i] × e1_8 + 32) >> 6, integer.

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

// Mode 6 interpolation weights (× 1/64), fixed by the spec (`W4` in bc7_ref.ts).
fn w4(i: u32) -> u32 {
  switch i {
    case 0u:  { return  0u; }
    case 1u:  { return  4u; }
    case 2u:  { return  9u; }
    case 3u:  { return 13u; }
    case 4u:  { return 17u; }
    case 5u:  { return 21u; }
    case 6u:  { return 26u; }
    case 7u:  { return 30u; }
    case 8u:  { return 34u; }
    case 9u:  { return 38u; }
    case 10u: { return 43u; }
    case 11u: { return 47u; }
    case 12u: { return 51u; }
    case 13u: { return 55u; }
    case 14u: { return 60u; }
    default:  { return 64u; }   // case 15u
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

// Quantize an 8-bit ideal endpoint to (7-bit value, reconstructed 8-bit) under
// a fixed p-bit, all four channels at once. q7 = round((ideal8 − p)/2); used by
// both paths.
struct QuantPair { seven: vec4<i32>, eight: vec4<i32> };
fn quantize_endpoint(ideal8: vec4<i32>, p: u32) -> QuantPair {
  let q = vec4<i32>(clamp(
    floor((vec4<f32>(ideal8) - f32(p)) / 2.0 + 0.5),
    vec4<f32>(0.0), vec4<f32>(127.0),
  ));
  let eff = (q << vec4<u32>(1u)) | vec4<i32>(i32(p));
  return QuantPair(q, eff);
}

// ============================ FAST PATH ================================ //

// Endpoint with its chosen p-bit, picked by minimum quantisation error.
struct Ep { seven: vec4<i32>, eight: vec4<i32>, p: u32 };
fn pick_ep(ideal: vec4<i32>) -> Ep {
  let a = quantize_endpoint(ideal, 0u);
  let b = quantize_endpoint(ideal, 1u);
  if (dist2(b.eight, ideal) < dist2(a.eight, ideal)) { return Ep(b.seven, b.eight, 1u); }
  return Ep(a.seven, a.eight, 0u);
}

// Projection index assignment. The palette is colinear, so the nearest entry is
// found by projecting onto the endpoint line — O(1) per pixel. When `fit`, the
// LSQ normal-equation sums are accumulated in the same pass for a fused refit
// (uniform weight i/15 — within a fraction of a code of the exact w4 table).
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
  let inv = 15.0 / f32(dd);
  var sAA: f32 = 0.0; var sBB: f32 = 0.0; var sAB: f32 = 0.0;
  var sAV: vec4<f32> = vec4<f32>(0.0); var sBV: vec4<f32> = vec4<f32>(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let q = (*pixels)[k] - e0;
    let s = clamp(floor(f32(q.x * dir.x + q.y * dir.y + q.z * dir.z + q.w * dir.w) * inv + 0.5), 0.0, 15.0);
    (*out_idx)[k] = u32(s);
    if (fit) {
      let v = vec4<f32>((*pixels)[k]);
      let b = s / 15.0; let a = 1.0 - b;
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

fn build_palette_6(e0: vec4<i32>, e1: vec4<i32>, pal: ptr<function, array<vec4<i32>, 16>>) {
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    (*pal)[i] = interp4(e0, e1, i32(w4(i)));
  }
}

fn assign_all(
  pixels:  ptr<function, array<vec4<i32>, 16>>,
  pal:     ptr<function, array<vec4<i32>, 16>>,
  out_idx: ptr<function, array<u32, 16>>,
) -> i32 {
  var err: i32 = 0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let px = (*pixels)[k];
    var best_i: u32 = 0u;
    var best_d: i32 = 2147483647;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
      let d = dist2(px, (*pal)[i]);
      if (d < best_d) { best_d = d; best_i = i; }
    }
    (*out_idx)[k] = best_i;
    err = err + best_d;
  }
  return err;
}

struct BestMode6 {
  e0_7: vec4<i32>, e1_7: vec4<i32>,
  p0: u32,         p1: u32,
  indices: array<u32, 16>,
  err: i32,
};

// Exhaustive p-bit search (high path); commits to `*best` only on improvement.
fn try_pbit_combos(
  pixels: ptr<function, array<vec4<i32>, 16>>,
  ideal0: vec4<i32>,
  ideal1: vec4<i32>,
  best:   ptr<function, BestMode6>,
) {
  var local_best = (*best).err;
  var pal: array<vec4<i32>, 16>;
  var tmp: array<u32, 16>;
  for (var p0: u32 = 0u; p0 < 2u; p0 = p0 + 1u) {
    let q0 = quantize_endpoint(ideal0, p0);
    for (var p1: u32 = 0u; p1 < 2u; p1 = p1 + 1u) {
      let q1 = quantize_endpoint(ideal1, p1);
      build_palette_6(q0.eight, q1.eight, &pal);
      let err = assign_all(pixels, &pal, &tmp);
      if (err < local_best) {
        local_best = err;
        (*best).e0_7 = q0.seven;
        (*best).e1_7 = q1.seven;
        (*best).p0 = p0;
        (*best).p1 = p1;
        (*best).indices = tmp;
        (*best).err = err;
      }
    }
  }
}

// Exact-weight LSQ refit (high path); matches bc7_ref.ts `refitEndpointsMode6`.
struct RefitResult { e0: vec4<i32>, e1: vec4<i32>, valid: bool };
fn refit_endpoints(
  pixels: ptr<function, array<vec4<i32>, 16>>,
  indices: ptr<function, array<u32, 16>>,
) -> RefitResult {
  var sAA: f32 = 0.0;
  var sBB: f32 = 0.0;
  var sAB: f32 = 0.0;
  var sAV: vec4<f32> = vec4<f32>(0.0);
  var sBV: vec4<f32> = vec4<f32>(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let i = (*indices)[k];
    let a = f32(64u - w4(i)) / 64.0;
    let b = f32(w4(i)) / 64.0;
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

// ------------------------------- Entry --------------------------------- //

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

  // Load 16 RGBA pixels (8-bit integer domain) and the per-channel bbox.
  var pixels: array<vec4<i32>, 16>;
  var lo = vec4<i32>(255);
  var hi = vec4<i32>(0);
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let lx = i32(i & 3u);
    let ly = i32(i >> 2u);
    let p  = clamp(base + vec2<i32>(lx, ly), vec2<i32>(0, 0), max_xy);
    let px = to8(textureLoad(src_tex, p, 0));
    pixels[i] = px;
    lo = min(lo, px);
    hi = max(hi, px);
  }

  var e0_7: vec4<i32>;
  var e1_7: vec4<i32>;
  var p0: u32;
  var p1: u32;
  var indices: array<u32, 16>;

  if (QUALITY_HIGH != 0u) {
    let fp = farthest_pair(&pixels);
    var best: BestMode6;
    best.err = 2147483647;
    try_pbit_combos(&pixels, fp.a, fp.b, &best);
    let refit = refit_endpoints(&pixels, &best.indices);
    if (refit.valid) {
      try_pbit_combos(&pixels, refit.e0, refit.e1, &best);
    }
    e0_7 = best.e0_7; e1_7 = best.e1_7; p0 = best.p0; p1 = best.p1; indices = best.indices;
  } else {
    var ep0 = pick_ep(lo);
    var ep1 = pick_ep(hi);
    let r = proj_assign(&pixels, ep0.eight, ep1.eight, &indices, true);
    if (r.valid) {
      ep0 = pick_ep(r.e0);
      ep1 = pick_ep(r.e1);
      proj_assign(&pixels, ep0.eight, ep1.eight, &indices, false);
    }
    e0_7 = ep0.seven; e1_7 = ep1.seven; p0 = ep0.p; p1 = ep1.p;
  }

  // Anchor rule — pixel 0's index MSB must be 0. If not, swap endpoints and
  // reflect every index (new_i = 15 − old_i); decoded image is unchanged.
  if ((indices[0] & 0x8u) != 0u) {
    let t7 = e0_7; e0_7 = e1_7; e1_7 = t7;
    let tp = p0;   p0   = p1;   p1   = tp;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      indices[k] = 15u - indices[k];
    }
  }

  // Pack into 128 bits = 4 u32s.
  var block: array<u32, 4>;
  block[0] = 0u; block[1] = 0u; block[2] = 0u; block[3] = 0u;
  var pos: u32 = 0u;
  write_bits(&block, pos, 7u, 0x40u); pos = pos + 7u;
  write_bits(&block, pos, 7u, u32(e0_7.x)); pos = pos + 7u;
  write_bits(&block, pos, 7u, u32(e1_7.x)); pos = pos + 7u;
  write_bits(&block, pos, 7u, u32(e0_7.y)); pos = pos + 7u;
  write_bits(&block, pos, 7u, u32(e1_7.y)); pos = pos + 7u;
  write_bits(&block, pos, 7u, u32(e0_7.z)); pos = pos + 7u;
  write_bits(&block, pos, 7u, u32(e1_7.z)); pos = pos + 7u;
  write_bits(&block, pos, 7u, u32(e0_7.w)); pos = pos + 7u;
  write_bits(&block, pos, 7u, u32(e1_7.w)); pos = pos + 7u;
  write_bits(&block, pos, 1u, p0); pos = pos + 1u;
  write_bits(&block, pos, 1u, p1); pos = pos + 1u;
  write_bits(&block, pos, 3u, indices[0] & 0x7u); pos = pos + 3u;
  for (var k: u32 = 1u; k < 16u; k = k + 1u) {
    write_bits(&block, pos, 4u, indices[k] & 0xFu);
    pos = pos + 4u;
  }

  let out = block_index * 4u;
  dst[out + 0u] = block[0];
  dst[out + 1u] = block[1];
  dst[out + 2u] = block[2];
  dst[out + 3u] = block[3];
}
