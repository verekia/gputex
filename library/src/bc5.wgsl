// BC5 (RGTC2) compute shader encoder.
//
// Each invocation encodes one 4x4 pixel block into a 16-byte BC5 block
// written as 4 x u32 into the destination storage buffer. This is the f32
// fallback; bc5_fast_f16.wgsl is the same algorithm and is preferred when
// the device reports shader-f16.
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
// We always produce the 6-interpolation mode (red0 > red1). See `bc4_ref.ts`
// for the reference this encoder is validated against.
//
// Pipeline per channel: bbox endpoints + O(1) projection assignment per
// texel. The 8-entry palette in 6-interp mode is COLINEAR and EVENLY spaced
// from r0 to r1 (levels 0..7 in palette order 0,2,3,4,5,6,7,1), so the
// nearest entry is the rounded projection onto the r0→r1 axis — no 8-entry
// search, and the 3-bit indices are packed on the fly. The LSQ refit sums
// are accumulated in the same fused pass; the requantised refit is accepted
// only if it lowers the block error (worth ~1.3 dB on the normal-map card).

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn quantize8(v: f32) -> u32 {
  // Round-to-nearest, clamp to [0, 255]. floor(x + 0.5) is the same
  // rounding rule the CPU reference uses.
  return u32(clamp(floor(v * 255.0 + 0.5), 0.0, 255.0));
}

// Encode 16 single-channel values into an 8-byte BC4 block, packed as
// two little-endian u32s (u32[0] = bytes 0..3, u32[1] = bytes 4..7).
// vmin/vmax are the channel's min/max, computed in the caller's load loop —
// fusing that scan there saves a 16-value pass per channel.
fn encode_bc4(values: ptr<function, array<f32, 16>>, vmin: f32, vmax: f32) -> vec2<u32> {
  var r0: u32 = quantize8(vmax);
  var r1: u32 = quantize8(vmin);
  // Force 6-interp mode: red0 > red1 strictly.
  if (r0 == r1) {
    if (r1 > 0u) { r1 = r1 - 1u; }
    else         { r0 = r0 + 1u; }
  }

  // ONE fused pass: level = round(7·(v − r0)/(r1 − r0)) projection
  // assignment (the 8-entry 6-interp palette is colinear and evenly
  // spaced, so the rounded projection IS the nearest-entry search), the
  // seed solution's packed indices and squared error, and the least-
  // squares normal-equation sums. The refit endpoints are re-quantised,
  // reprojected, and accepted only if the block error decreases.
  // Level → BC4 index LUT (0,2,3,4,5,6,7,1) packed as 3-bit entries in
  // 0x3F58D0. Pixel k's 3 bits start at bit 3k+16 of the (w0,w1) pair
  // (bytes 0..1 are the endpoints); k = 5 straddles the word boundary.
  let r0f = f32(r0) / 255.0;
  let dir = f32(r1) / 255.0 - r0f;
  let scale = 7.0 / dir;
  var w0 = r0 | (r1 << 8u);
  var w1 = 0u;
  var sAA = 0.0; var sBB = 0.0; var sAB = 0.0;
  var sAV = 0.0; var sBV = 0.0;
  var s_min = 7.0; var s_max = 0.0;
  var seed_err = 0.0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let vr = (*values)[k] - r0f;
    let L = clamp(floor(vr * scale + 0.5), 0.0, 7.0);
    s_min = min(s_min, L); s_max = max(s_max, L);
    let b = L * (1.0 / 7.0); let a = 1.0 - b;
    sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b;
    sAV = sAV + a * vr; sBV = sBV + b * vr;
    let e = vr - b * dir;
    seed_err = seed_err + e * e;
    let idx = (0x3F58D0u >> (u32(L) * 3u)) & 7u;
    let bit = 3u * k + 16u;
    if (bit <= 29u) {
      w0 = w0 | (idx << bit);
    } else if (bit >= 32u) {
      w1 = w1 | (idx << (bit - 32u));
    } else {
      w0 = w0 | (idx << bit);
      w1 = w1 | (idx >> (32u - bit));
    }
  }

  // Rank-1 guard: with every pixel on ONE level the system is singular
  // (det is float rounding noise); with ≥2 distinct levels
  // det = Σ_i<j (b_j − b_i)² ≥ 15/49 ≈ 0.306.
  if (s_min < s_max) {
    let det = sAA * sBB - sAB * sAB;
    if (abs(det) > 1e-3) {
      // Clamp to [0,1], NOT the block's value range: for a scalar channel,
      // endpoints beyond the data range are often genuinely optimal (they
      // centre the palette levels on the data) and there is no colour axis
      // to bend — the bbox clamp the colour formats need costs ~0.3 dB
      // here. The accept-if-better guard still protects against a refit
      // that loses after quantisation.
      let e0 = clamp(r0f + (sBB * sAV - sAB * sBV) / det, 0.0, 1.0);
      let e1 = clamp(r0f + (sAA * sBV - sAB * sAV) / det, 0.0, 1.0);
      let n0 = quantize8(e0);
      let n1 = quantize8(e1);
      // Keep 6-interp mode (r0 > r1 strictly); skip the no-op refit.
      if (n0 > n1 && !(n0 == r0 && n1 == r1)) {
        let n0f = f32(n0) / 255.0;
        let ndir = f32(n1) / 255.0 - n0f;
        let nscale = 7.0 / ndir;
        var nw0 = n0 | (n1 << 8u);
        var nw1 = 0u;
        var refit_err = 0.0;
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
          let vr = (*values)[k] - n0f;
          let L = clamp(floor(vr * nscale + 0.5), 0.0, 7.0);
          let e = vr - L * (1.0 / 7.0) * ndir;
          refit_err = refit_err + e * e;
          let idx = (0x3F58D0u >> (u32(L) * 3u)) & 7u;
          let bit = 3u * k + 16u;
          if (bit <= 29u) {
            nw0 = nw0 | (idx << bit);
          } else if (bit >= 32u) {
            nw1 = nw1 | (idx << (bit - 32u));
          } else {
            nw0 = nw0 | (idx << bit);
            nw1 = nw1 | (idx >> (32u - bit));
          }
        }
        if (refit_err < seed_err) {
          return vec2<u32>(nw0, nw1);
        }
      }
    }
  }
  return vec2<u32>(w0, w1);
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
  // be handed to encode_bc4 independently; the per-channel min/max scan is
  // fused into the same loop.
  var r_values: array<f32, 16>;
  var g_values: array<f32, 16>;
  var r_min: f32 = 1.0; var r_max: f32 = 0.0;
  var g_min: f32 = 1.0; var g_max: f32 = 0.0;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let lx = i32(i & 3u);
    let ly = i32(i >> 2u);
    // Clamp to edge for non-multiple-of-4 input sizes.
    let p  = clamp(base + vec2<i32>(lx, ly), vec2<i32>(0, 0), max_xy);
    let c  = textureLoad(src_tex, p, 0);
    r_values[i] = c.r;
    g_values[i] = c.g;
    r_min = min(r_min, c.r); r_max = max(r_max, c.r);
    g_min = min(g_min, c.g); g_max = max(g_max, c.g);
  }

  let r_block = encode_bc4(&r_values, r_min, r_max);
  let g_block = encode_bc4(&g_values, g_min, g_max);

  // BC5 block = R half (bytes 0..7) || G half (bytes 8..15) = 4 u32s.
  let out = block_index * 4u;
  dst[out + 0u] = r_block.x;
  dst[out + 1u] = r_block.y;
  dst[out + 2u] = g_block.x;
  dst[out + 3u] = g_block.y;
}
