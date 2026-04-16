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
//   3. Build palette, assign each texel its nearest entry (full L2).
//   4. One-pass least-squares refinement: solve the 2×2 normal equations
//      for the (r0, r1) that minimizes Σ(palette[i_k] − v_k)². Accept
//      only if quantized endpoints still satisfy r0 > r1 AND total
//      squared error decreased.
//   5. Pack 2 endpoint bytes + 48 bits of indices into the 8-byte block.

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

// Nearest-palette-index search over the 8-entry palette, returning the
// index and squared error. `palette` is stored in function memory so we
// pass by pointer.
fn nearest_index(v: f32, palette: ptr<function, array<f32, 8>>) -> vec2<f32> {
  // x = best index (encoded as f32), y = best squared error.
  var best_j: u32 = 0u;
  var best_d: f32 = 1e20;
  for (var j: u32 = 0u; j < 8u; j = j + 1u) {
    let d  = (*palette)[j] - v;
    let d2 = d * d;
    if (d2 < best_d) {
      best_d = d2;
      best_j = j;
    }
  }
  return vec2<f32>(f32(best_j), best_d);
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

  // ---------------- 2. Initial palette + indices + error --------------
  var palette: array<f32, 8>;
  let r0f = f32(r0) / 255.0;
  let r1f = f32(r1) / 255.0;
  for (var j: u32 = 0u; j < 8u; j = j + 1u) {
    palette[j] = w0_6(j) * r0f + w1_6(j) * r1f;
  }
  var indices: array<u32, 16>;
  var err: f32 = 0.0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let sel = nearest_index((*values)[k], &palette);
    indices[k] = u32(sel.x);
    err = err + sel.y;
  }

  var best_r0 = r0;
  var best_r1 = r1;
  var best_indices = indices;
  var best_err = err;

  // ---------------- 3. Refinement: least-squares on (r0, r1) ----------
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
      var pal2: array<f32, 8>;
      let r0f2 = f32(qR0) / 255.0;
      let r1f2 = f32(qR1) / 255.0;
      for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        pal2[j] = w0_6(j) * r0f2 + w1_6(j) * r1f2;
      }
      var idx2: array<u32, 16>;
      var err2: f32 = 0.0;
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let sel = nearest_index((*values)[k], &pal2);
        idx2[k] = u32(sel.x);
        err2 = err2 + sel.y;
      }
      if (err2 < best_err) {
        best_r0 = qR0;
        best_r1 = qR1;
        best_indices = idx2;
        best_err = err2;
      }
    }
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
    let v = best_indices[k] & 7u;
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
  let out_lo = best_r0 | (best_r1 << 8u) | ((idx_lo & 0xFFFFu) << 16u);
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
