// ASTC 4×4 LDR compute shader encoder.
//
// One invocation per 4×4 block. Emits 16 bytes = 4 u32s into the storage
// buffer at `dst[block_index * 4 .. + 3]`.
//
// Numerically identical to `astc4x4_ref.ts` (same farthest-pair seed, same
// single LSQ refit accepted only on strict error decrease, same endpoint
// ordering rule). The rewrite only reschedules the arithmetic:
//   • Pixels and the 4-entry palette live in the i32 domain, so the nearest-
//     entry search is pure integer math with no per-comparison conversions.
//   • Endpoint interpolation runs once per 4-channel vec4 instead of four
//     scalar calls.
//   • The farthest-pair seed returns endpoint *values*, keeping `pixels[]`
//     indexed only by loop counters (register-resident).
//   • The 16 two-bit weights pack with a single shift-OR into the top word
//     instead of 32 individual masked read-modify-write `write_bits` calls.
//
// RESTRICTED SUBSET (both CPU ref and this shader):
//   • Single partition, no dual-plane
//   • CEM 12 (LDR RGBA, direct)
//   • Weight grid 4×4 (no upsampling), 2-bit weights (QUANT_4)
//   • 8-bit endpoints (QUANT_256 — bit-replication is a no-op)
//
// BLOCK LAYOUT (128 bits, LSB-first)
//   bits [10:0]   block mode = 0x042
//   bits [12:11]  partition count − 1 = 0
//   bits [16:13]  CEM = 12
//   bits [80:17]  endpoints: R0 R1 G0 G1 B0 B1 A0 A1 (8-bit each)
//   bits [95:81]  unused (zero padding)
//   bits [127:96] 16 × 2-bit weights; for weight k ∈ [0,15]:
//                   block_bit(127 − 2k) = weight_k[0]   (LSB)
//                   block_bit(126 − 2k) = weight_k[1]   (MSB)
//
// ENDPOINT ORDERING: after fitting, if sum(e0.rgb) > sum(e1.rgb) we swap
// endpoints and reflect indices (w' = 3 − w). This keeps the decoder out
// of the blue-contraction branch (see CPU ref file header for the full
// decoder behaviour).

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// QUANT_4 weight unquantisation: q ∈ [0,3] → unq ∈ [0, 21, 43, 64].
fn weight_unq(i: u32) -> i32 {
  switch i {
    case 0u: { return  0; }
    case 1u: { return 21; }
    case 2u: { return 43; }
    default: { return 64; }  // case 3u
  }
}

// Hardware-exact integer interpolation, all four channels at once. Operands
// are in [0,255]; the i32 arithmetic shift matches the unsigned formula.
fn interp4(e0: vec4<i32>, e1: vec4<i32>, w: i32) -> vec4<i32> {
  return ((64 - w) * e0 + w * e1 + vec4<i32>(32)) >> vec4<u32>(6u);
}

// Normalised [0, 1] → clamped 8-bit, all four channels at once. Same
// rounding rule (floor(v + 0.5)) as the CPU reference's Math.round.
fn to8(v: vec4<f32>) -> vec4<i32> {
  return vec4<i32>(clamp(floor(v * 255.0 + 0.5), vec4<f32>(0.0), vec4<f32>(255.0)));
}

// 4-channel L2 distance squared in the i32 domain (≤ 4 · 255² = 260 100).
fn dist2(a: vec4<i32>, b: vec4<i32>) -> i32 {
  let d = a - b;
  let e = d * d;
  return e.x + e.y + e.z + e.w;
}

// -------------------------- Farthest-pair seed -------------------------- //

struct Pair { a: vec4<i32>, b: vec4<i32> };

// O(N²) = 120 comparisons returning the two most distant pixel *values*.
// Same rationale as BC7's `farthest_pair`: bbox corners aren't safe initial
// endpoints when channels vary in different directions along the data line.
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

// -------------------------- Palette + assignment ------------------------ //

// Build the 4-entry RGBA palette (i32 domain) from 8-bit endpoints. Uses the
// same integer interpolation as decode, so assignments round-trip exactly.
fn build_palette(e0: vec4<i32>, e1: vec4<i32>, pal: ptr<function, array<vec4<i32>, 4>>) {
  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    (*pal)[i] = interp4(e0, e1, weight_unq(i));
  }
}

// Assign all 16 texels to their nearest palette entry (full 4-way L2),
// writing indices into `out_idx` and returning the total squared error.
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

// ---------------------- Least-squares endpoint refit -------------------- //

// Given current indices, solve the per-channel 2×2 normal equations for
// (e0, e1). See the CPU reference's `refitEndpoints` for the derivation.
// `valid = false` signals a degenerate system (all texels on one palette
// entry) and the caller keeps the farthest-pair seed.
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

// Write `n_bits` LSBs of `value` at bit position `pos` of a 128-bit field
// represented as `array<u32, 4>`. Handles word-boundary straddles.
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

  // 1. Load 16 RGBA texels in 8-bit integer domain.
  var pixels: array<vec4<i32>, 16>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let lx = i32(i & 3u);
    let ly = i32(i >> 2u);
    // Clamp to edge for non-multiple-of-4 input sizes.
    let p  = clamp(base + vec2<i32>(lx, ly), vec2<i32>(0, 0), max_xy);
    pixels[i] = to8(textureLoad(src_tex, p, 0));
  }

  // 2. Farthest-pair seed → initial endpoints (values, not indices).
  let fp = farthest_pair(&pixels);
  var e0 = fp.a;
  var e1 = fp.b;

  // 3. Initial assignment against the seed endpoints.
  var pal: array<vec4<i32>, 4>;
  build_palette(e0, e1, &pal);
  var indices: array<u32, 16>;
  var err = assign_all(&pixels, &pal, &indices);

  // 4. One LSQ refit pass. Accept only if the squared error strictly
  //    decreases — matches the CPU reference. The palette buffer is reused;
  //    on rejection the current indices/endpoints are left untouched.
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

  // 5. Endpoint ordering so the decoder doesn't apply blue contraction.
  //    Strict '>' avoids a gratuitous swap on ties.
  let s0 = e0.x + e0.y + e0.z;
  let s1 = e1.x + e1.y + e1.z;
  if (s0 > s1) {
    let tmp = e0; e0 = e1; e1 = tmp;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      // w' = 3 − w reflects the palette; decoded colour unchanged.
      indices[k] = 3u - indices[k];
    }
  }

  // 6. Pack 128 bits.
  var block: array<u32, 4>;
  block[0] = 0u; block[1] = 0u; block[2] = 0u; block[3] = 0u;

  // Config header.
  write_bits(&block, 0u,  11u, 0x042u);   // block mode: 4×4 grid, QUANT_4 weights
  write_bits(&block, 11u, 2u,  0u);       // partition count − 1
  write_bits(&block, 13u, 4u,  12u);      // CEM 12: LDR RGBA direct

  // Endpoints in the CEM 12 value order: R0 R1 G0 G1 B0 B1 A0 A1.
  write_bits(&block, 17u + 0u * 8u, 8u, u32(e0.x));
  write_bits(&block, 17u + 1u * 8u, 8u, u32(e1.x));
  write_bits(&block, 17u + 2u * 8u, 8u, u32(e0.y));
  write_bits(&block, 17u + 3u * 8u, 8u, u32(e1.y));
  write_bits(&block, 17u + 4u * 8u, 8u, u32(e0.z));
  write_bits(&block, 17u + 5u * 8u, 8u, u32(e1.z));
  write_bits(&block, 17u + 6u * 8u, 8u, u32(e0.w));
  write_bits(&block, 17u + 7u * 8u, 8u, u32(e1.w));

  // Weights occupy exactly the top 32-bit word (bits [127:96]). For weight k,
  // block_bit(127 − 2k) is its LSB and block_bit(126 − 2k) its MSB; relative
  // to word 3 (bit 96) those land at local bits (31 − 2k) and (30 − 2k). One
  // shift-OR per weight reproduces the two per-weight 1-bit writes exactly.
  var w3: u32 = 0u;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let w = indices[k] & 0x3u;
    w3 = w3 | ((w & 1u) << (31u - 2u * k)) | (((w >> 1u) & 1u) << (30u - 2u * k));
  }
  block[3] = w3;

  // 7. Store as 4 u32s.
  let out = block_index * 4u;
  dst[out + 0u] = block[0];
  dst[out + 1u] = block[1];
  dst[out + 2u] = block[2];
  dst[out + 3u] = block[3];
}
