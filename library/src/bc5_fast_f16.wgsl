// bc5 "fast" encoder — f16 variant (requires the shader-f16 feature).
// Two BC4 halves (R and G) — same output family as bc5.wgsl's fast branch,
// tuned for throughput:
//
//   • The 8-entry palette in 6-interpolation mode is COLINEAR and EVENLY
//     spaced from r0 to r1 (levels 0..7 in palette order 0,2,3,4,5,6,7,1),
//     so the nearest entry is the rounded projection of v onto the r0→r1
//     axis — O(1) per pixel instead of an 8-entry distance search.
//   • Math runs in the exact-integer [0,255] f16 domain: endpoints and pixel
//     values are whole numbers ≤ 255 (exact in f16), so the only rounding is
//     the single 1/(r1−r0) division.
//   • ONE fused pass per channel: projection assignment + the least-squares
//     refit sums + the seed solution's packed indices and squared error. The
//     refit endpoints are re-quantised, reprojected, and accepted only if
//     the block error decreases (same accept-if-better family as the BC1
//     fast path) — worth ~1.3 dB on the normal-map card.
//   • 3-bit indices are packed into the 48-bit field on the fly — no
//     array<u32,16> private array and no separate packing loop.
//
// f16 range notes: value sums accumulate v − r0 (the affine-basis shift trick
// from the BC7/ASTC fast paths) scaled by 1/16, and error residuals are
// scaled by 1/16 before squaring — worst-case magnitudes stay ≲4k, well
// inside f16's 65504 max, with rounding a small fraction of a level. The
// accept-if-better guard makes any residual f16 noise fail-safe (worst case:
// the refit is rejected and the seed solution ships).
//
// Level → BC4 index (0→r0 ... 7→r1): 0,2,3,4,5,6,7,1 — packed 3-bit LUT
// 0x3F58D0 = sum(idx[L] << 3L).
//
// The host selects this module only when the device reports shader-f16,
// falling back to bc5.wgsl otherwise. "high" never uses this.
enable f16;
alias h = f16;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// Project the 16 values onto the r0→r1 axis, packing the 3-bit indices on the
// fly (pixel k's index starts at bit 3k of the 48-bit field, i.e. bit 3k+16
// of the (w0,w1) pair; k = 5 straddles the word boundary) and accumulating
// the squared error (residuals scaled by 1/16 before squaring). Returns the
// packed words with the endpoint bytes already in place.
struct Proj { w0: u32, w1: u32, err: h };
fn project_pack(values: ptr<function, array<h, 16>>, r0: u32, r1: u32) -> Proj {
  let r0f = h(f32(r0));
  let dir = h(f32(r1)) - r0f;
  let scale = h(7.0) / dir;
  var out: Proj;
  out.w0 = r0 | (r1 << 8u);
  out.w1 = 0u;
  out.err = h(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let vr = (*values)[k] - r0f;
    let L = clamp(floor(vr * scale + h(0.5)), h(0.0), h(7.0));
    let e = (vr - L * h(1.0 / 7.0) * dir) * h(1.0 / 16.0);
    out.err = out.err + e * e;
    let idx = (0x3F58D0u >> (u32(L) * 3u)) & 7u;
    let bit = 3u * k + 16u;
    if (bit <= 29u) {
      out.w0 = out.w0 | (idx << bit);
    } else if (bit >= 32u) {
      out.w1 = out.w1 | (idx << (bit - 32u));
    } else {
      out.w0 = out.w0 | (idx << bit);
      out.w1 = out.w1 | (idx >> (32u - bit));
    }
  }
  return out;
}

// Encode one channel (16 values in exact-integer [0,255] f16) to a BC4 half.
// vmin/vmax are the channel's min/max, computed in the caller's load loop —
// fusing that scan there saves a 16-value pass per channel.
fn encode_bc4(values: ptr<function, array<h, 16>>, vmin: h, vmax: h) -> vec2<u32> {
  var r0 = u32(vmax); // values are exact integers — no rounding needed
  var r1 = u32(vmin);
  if (r0 == r1) {
    // Flat block: nudge to keep 6-interp mode (r0 > r1 strictly).
    if (r1 > 0u) { r1 = r1 - 1u; } else { r0 = r0 + 1u; }
  }

  // Fused seed pass: projection assignment (level = round(7·(v − r0)/
  // (r1 − r0)), clamped — |v − r0| ≤ r0 − r1 for in-block values) + packed
  // indices + seed error + the LSQ normal-equation sums. Value sums
  // accumulate (v − r0)/16: the shift keeps the accumulators proportional to
  // the block's span, the exact power-of-two scale keeps products ≤ 4080.
  let r0f = h(f32(r0));
  let dir = h(f32(r1)) - r0f;
  let scale = h(7.0) / dir;
  var seed: Proj;
  seed.w0 = r0 | (r1 << 8u);
  seed.w1 = 0u;
  seed.err = h(0.0);
  var sAA = h(0.0); var sBB = h(0.0); var sAB = h(0.0);
  var sAV = h(0.0); var sBV = h(0.0);
  var s_min = h(7.0); var s_max = h(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let vr = (*values)[k] - r0f;
    let L = clamp(floor(vr * scale + h(0.5)), h(0.0), h(7.0));
    s_min = min(s_min, L); s_max = max(s_max, L);
    let b = L * h(1.0 / 7.0); let a = h(1.0) - b;
    let vr16 = vr * h(1.0 / 16.0);
    sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b;
    sAV = sAV + a * vr16; sBV = sBV + b * vr16;
    let e = vr16 - b * dir * h(1.0 / 16.0);
    seed.err = seed.err + e * e;
    let idx = (0x3F58D0u >> (u32(L) * 3u)) & 7u;
    let bit = 3u * k + 16u;
    if (bit <= 29u) {
      seed.w0 = seed.w0 | (idx << bit);
    } else if (bit >= 32u) {
      seed.w1 = seed.w1 | (idx << (bit - 32u));
    } else {
      // k = 5 straddles the word boundary (bits 31..33).
      seed.w0 = seed.w0 | (idx << bit);
      seed.w1 = seed.w1 | (idx >> (32u - bit));
    }
  }

  // LSQ refit, accepted only if the requantised endpoints lower the block
  // error. Rank-1 guard: with every pixel on ONE level the system is
  // singular and det is pure f16 rounding noise (≲0.03); with ≥2 distinct
  // levels det = Σ_i<j (b_j − b_i)² ≥ 15/49 ≈ 0.306 — 0.1 separates cleanly.
  if (s_min < s_max) {
    let det = sAA * sBB - sAB * sAB;
    if (abs(det) > h(0.1)) {
      // ×16 undoes the accumulator scale; clamp to the block's value range
      // (a strict-SSE win vs clamping to [0,255], same as the other formats).
      let e0 = clamp(r0f + (sBB * sAV - sAB * sBV) * h(16.0) / det, vmin, vmax);
      let e1 = clamp(r0f + (sAA * sBV - sAB * sAV) * h(16.0) / det, vmin, vmax);
      let n0 = u32(floor(e0 + h(0.5)));
      let n1 = u32(floor(e1 + h(0.5)));
      // Keep 6-interp mode (r0 > r1 strictly); skip the no-op refit.
      if (n0 > n1 && !(n0 == r0 && n1 == r1)) {
        let refit = project_pack(values, n0, n1);
        if (refit.err < seed.err) {
          return vec2<u32>(refit.w0, refit.w1);
        }
      }
    }
  }
  return vec2<u32>(seed.w0, seed.w1);
}

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);
  let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
  var rv: array<h, 16>;
  var gv: array<h, 16>;
  var rmin = h(255.0); var rmax = h(0.0);
  var gmin = h(255.0); var gmax = h(0.0);
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0), mx);
    let c = textureLoad(src_tex, p, 0);
    let r = h(c.r * 255.0);
    let g = h(c.g * 255.0);
    rv[i] = r; rmin = min(rmin, r); rmax = max(rmax, r);
    gv[i] = g; gmin = min(gmin, g); gmax = max(gmax, g);
  }
  let rb = encode_bc4(&rv, rmin, rmax);
  let gb = encode_bc4(&gv, gmin, gmax);
  let o = bi * 4u;
  dst[o] = rb.x; dst[o + 1u] = rb.y; dst[o + 2u] = gb.x; dst[o + 3u] = gb.y;
}
