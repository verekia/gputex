// BC5 (RGTC2) compute shader encoder.
//
// Each invocation encodes one 4x4 pixel block into a 16-byte BC5 block
// written as 4 x u32 into the destination storage buffer.
//
// BC5 = two BC4 blocks concatenated:
//   block bytes  0..7  : BC4 of R channel (normal.x for tangent-space normals)
//   block bytes  8..15 : BC4 of G channel (normal.y)
//
// Each BC4 half-block (8 bytes):
//   byte 0     : red0 (8-bit endpoint)
//   byte 1     : red1 (8-bit endpoint)
//   bytes 2..7 : 16 × 3-bit indices, LSB-first, pixel 0 at bit 0
//
// We always produce the 6-interpolation mode (red0 > red1). See
// `bc4_ref.js` for the reasoning and the CPU reference this shader is
// ported from — the algorithm and edge cases mirror it line-for-line.
//
// Pipeline per channel:
//   1. Load 16 single-channel values, find min/max → initial endpoints.
//   2. Quantize to 8-bit. Nudge apart if equal (forces 6-interp mode).
//   3. Assign each texel an index: fast projects onto the endpoint line
//      (O(1) per texel); high does the full 8-entry L2 nearest search.
//   4. (high only) One-pass least-squares refinement: solve the 2×2 normal
//      equations for the (r0, r1) that minimizes Σ(palette[i_k] − v_k)².
//      Accept only if quantized endpoints still satisfy r0 > r1 AND total
//      squared error decreased.
//   5. Pack 2 endpoint bytes + 48 bits of indices into the 8-byte block.
//
// In the high path the candidate endpoints/indices/error are tracked in place
// — the refit overwrites them only when accepted — so no 16-entry index array
// is ever copied across a function return.
//
// QUALITY LEVELS (pipeline-overridable constant `QUALITY_HIGH`)
//   fast (0, default): bbox endpoints, then an O(1) projection assignment. The
//     8 palette entries are uniformly spaced between the endpoints, so each
//     texel's nearest entry is found by projecting it onto the endpoint line
//     and rounding to one of 8 levels — no palette build, no 8-entry search.
//     This is error-identical to the full L2 search (only exact-midpoint ties,
//     which carry equal error, may pick the other of two equidistant entries).
//     The LSQ refit — the bulk of the kernel, worth ~0.36 dB — is skipped.
//   high (1): full 8-entry L2 search + the refit, byte-for-byte identical to
//     bc4_ref/bc5_ref.

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

// 6-interpolation-mode palette weights. palette[j] = W0_6[j]*r0 + W1_6[j]*r1.
// Expressed as a switch so we don't rely on module-scope const arrays.
fn w0_6(j: u32) -> f32 {
  switch j {
    case 0u:  { return 1.0; }
    case 1u:  { return 0.0; }
    case 2u:  { return 6.0 / 7.0; }
    case 3u:  { return 5.0 / 7.0; }
    case 4u:  { return 4.0 / 7.0; }
    case 5u:  { return 3.0 / 7.0; }
    case 6u:  { return 2.0 / 7.0; }
    default:  { return 1.0 / 7.0; }   // case 7u
  }
}

fn w1_6(j: u32) -> f32 {
  switch j {
    case 0u:  { return 0.0; }
    case 1u:  { return 1.0; }
    case 2u:  { return 1.0 / 7.0; }
    case 3u:  { return 2.0 / 7.0; }
    case 4u:  { return 3.0 / 7.0; }
    case 5u:  { return 4.0 / 7.0; }
    case 6u:  { return 5.0 / 7.0; }
    default:  { return 6.0 / 7.0; }   // case 7u
  }
}

fn quantize8(v: f32) -> u32 {
  // Round-to-nearest, clamp to [0, 255]. floor(x + 0.5) is the same
  // rounding rule the CPU reference uses.
  return u32(clamp(floor(v * 255.0 + 0.5), 0.0, 255.0));
}

// Build the 8-entry palette for endpoints (r0f, r1f) in normalised space.
fn build_pal(r0f: f32, r1f: f32, pal: ptr<function, array<f32, 8>>) {
  for (var j: u32 = 0u; j < 8u; j = j + 1u) {
    (*pal)[j] = w0_6(j) * r0f + w1_6(j) * r1f;
  }
}

// Assign each of the 16 values its nearest palette entry (full 8-entry L2),
// writing indices into `out_idx` and returning the total squared error.
fn assign_all(
  values:  ptr<function, array<f32, 16>>,
  pal:     ptr<function, array<f32, 8>>,
  out_idx: ptr<function, array<u32, 16>>,
) -> f32 {
  var err: f32 = 0.0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let v = (*values)[k];
    var best_j: u32 = 0u;
    var best_d: f32 = 1e20;
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
      let d  = (*pal)[j] - v;
      let d2 = d * d;
      if (d2 < best_d) {
        best_d = d2;
        best_j = j;
      }
    }
    (*out_idx)[k] = best_j;
    err = err + best_d;
  }
  return err;
}

// Map a projection level (0 = nearest r1/min, 7 = nearest r0/max) to the BC4
// palette index. BC4's index order is non-monotonic: index 0 = r0, index 1 =
// r1, indices 2..7 descend from just below r0 down to just above r1. So the six
// interior levels map to 8 - level, with the two endpoints as special cases.
fn level_to_index(level: u32) -> u32 {
  switch level {
    case 0u:  { return 1u; }
    case 7u:  { return 0u; }
    default:  { return 8u - level; }   // levels 1..6 → indices 7..2
  }
}

// Projection index assignment (fast path). The 8 6-interp palette entries are
// uniformly spaced scalar values between r1 (min) and r0 (max), so the nearest
// entry equals projecting v onto [r1, r0] and rounding to one of 8 levels —
// O(1) per texel, no palette build and no 8-entry search. r0f > r1f is
// guaranteed by the endpoint nudge, so the span is ≥ 1/255 and the reciprocal
// is finite.
fn proj_assign(
  values:  ptr<function, array<f32, 16>>,
  r0f: f32, r1f: f32,
  out_idx: ptr<function, array<u32, 16>>,
) {
  let inv = 7.0 / (r0f - r1f);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let level = clamp(floor(((*values)[k] - r1f) * inv + 0.5), 0.0, 7.0);
    (*out_idx)[k] = level_to_index(u32(level));
  }
}

// Encode 16 single-channel values into an 8-byte BC4 block, packed as
// two little-endian u32s (u32[0] = bytes 0..3, u32[1] = bytes 4..7).
fn encode_bc4(values: ptr<function, array<f32, 16>>) -> vec2<u32> {
  // ---------------- 1. Initial endpoints: bbox of input ----------------
  var vmin: f32 = 1.0;
  var vmax: f32 = 0.0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    vmin = min(vmin, (*values)[k]);
    vmax = max(vmax, (*values)[k]);
  }
  var r0: u32 = quantize8(vmax);
  var r1: u32 = quantize8(vmin);
  // Force 6-interp mode: red0 > red1 strictly.
  if (r0 == r1) {
    if (r1 > 0u) { r1 = r1 - 1u; }
    else         { r0 = r0 + 1u; }
  }

  // ---------------- 2. Assign indices ---------------------------------
  // The fast and high branches resolve at pipeline-compile time, so the driver
  // keeps only one. Fast uses the O(1) projection; high builds the palette and
  // runs the full 8-entry L2 search (whose error feeds the refit below).
  var indices: array<u32, 16>;
  if (QUALITY_HIGH != 0u) {
    var pal: array<f32, 8>;
    build_pal(f32(r0) / 255.0, f32(r1) / 255.0, &pal);
    var err = assign_all(values, &pal, &indices);

    // -------------- 3. Refinement: least-squares on (r0, r1) ----------
    // Normal equations for palette[j] = a_j * r0 + b_j * r1:
    //   [ΣAA  ΣAB] [r0]   [ΣAV]
    //   [ΣAB  ΣBB] [r1] = [ΣBV]
    var sAA: f32 = 0.0; var sBB: f32 = 0.0; var sAB: f32 = 0.0;
    var sAV: f32 = 0.0; var sBV: f32 = 0.0;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let a = w0_6(indices[k]);
      let b = w1_6(indices[k]);
      let v = (*values)[k];
      sAA = sAA + a * a;
      sBB = sBB + b * b;
      sAB = sAB + a * b;
      sAV = sAV + a * v;
      sBV = sBV + b * v;
    }
    let det = sAA * sBB - sAB * sAB;
    // Degenerate system → skip refinement.
    if (abs(det) > 1e-9) {
      let new_r0 = clamp((sBB * sAV - sAB * sBV) / det, 0.0, 1.0);
      let new_r1 = clamp((sAA * sBV - sAB * sAV) / det, 0.0, 1.0);
      let qR0 = quantize8(new_r0);
      let qR1 = quantize8(new_r1);
      // Only accept refinements that stay in 6-interp mode. A refinement
      // that flips or equalizes the endpoints would change decode mode.
      if (qR0 > qR1) {
        build_pal(f32(qR0) / 255.0, f32(qR1) / 255.0, &pal);
        var idx2: array<u32, 16>;
        let err2 = assign_all(values, &pal, &idx2);
        if (err2 < err) {
          r0 = qR0;
          r1 = qR1;
          indices = idx2;
          err = err2;
        }
      }
    }
  } else {
    proj_assign(values, f32(r0) / 255.0, f32(r1) / 255.0, &indices);
  }

  // ---------------- 4. Pack 48-bit index field + 2 endpoint bytes -----
  // The 48-bit index field spans block bytes 2..7. Split into idx_lo
  // (low 32 bits of the field) and idx_hi (high 16 bits). An index at
  // bit position 3k straddles the 32-bit boundary iff 3k < 32 < 3k+3
  // (only k = 10, 11 straddle: bits 30..32 and 33..35; actually k=10
  // is bits 30..32, k=11 is 33..35 — so k=10 straddles). We handle
  // straddles by writing to both halves.
  var idx_lo: u32 = 0u;
  var idx_hi: u32 = 0u;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let bit = 3u * k;
    let v = indices[k] & 7u;
    if (bit + 3u <= 32u) {
      idx_lo = idx_lo | (v << bit);
    } else if (bit >= 32u) {
      idx_hi = idx_hi | (v << (bit - 32u));
    } else {
      // Straddle: low part into idx_lo's top, high part into idx_hi's bottom.
      idx_lo = idx_lo | (v << bit);
      idx_hi = idx_hi | (v >> (32u - bit));
    }
  }

  // Final u32s, both little-endian:
  //   u32[0] bytes = red0, red1, idx_lo[7:0], idx_lo[15:8]
  //   u32[1] bytes = idx_lo[23:16], idx_lo[31:24], idx_hi[7:0], idx_hi[15:8]
  let out_lo = r0 | (r1 << 8u) | ((idx_lo & 0xFFFFu) << 16u);
  let out_hi = (idx_lo >> 16u) | (idx_hi << 16u);

  return vec2<u32>(out_lo, out_hi);
}

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

  // Load 4×4 RG values, splitting into per-channel arrays so each can
  // be handed to encode_bc4 independently.
  var r_values: array<f32, 16>;
  var g_values: array<f32, 16>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let lx = i32(i & 3u);
    let ly = i32(i >> 2u);
    // Clamp to edge for non-multiple-of-4 input sizes.
    let p  = clamp(base + vec2<i32>(lx, ly), vec2<i32>(0, 0), max_xy);
    let c  = textureLoad(src_tex, p, 0);
    r_values[i] = c.r;
    g_values[i] = c.g;
  }

  let r_block = encode_bc4(&r_values);
  let g_block = encode_bc4(&g_values);

  // BC5 block = R half (bytes 0..7) || G half (bytes 8..15) = 4 u32s.
  let out = block_index * 4u;
  dst[out + 0u] = r_block.x;
  dst[out + 1u] = r_block.y;
  dst[out + 2u] = g_block.x;
  dst[out + 3u] = g_block.y;
}
