// bc5 "fast" encoder — f16 variant (requires the shader-f16 feature).
// Two BC4 halves (R and G) — same output family as bc5.wgsl, tuned for
// throughput:
//
//   • The 8-entry palette in 6-interpolation mode is COLINEAR and EVENLY
//     spaced from r0 to r1 (levels 0..7 in palette order 0,2,3,4,5,6,7,1),
//     so the nearest entry is the rounded projection of v onto the r0→r1
//     axis — O(1) per pixel instead of an 8-entry distance search.
//   • Math runs in the exact-integer [0,255] f16 domain: endpoints and pixel
//     values are whole numbers ≤ 255 (exact in f16), so the only rounding is
//     the single 1/(r1−r0) division.
//   • BOTH channels ride the same fused pass as vec2<f16> lanes — one loop
//     computes projections, packed indices and refit sums for R and G at
//     once instead of two scalar encode_bc4 calls.
//   • The LSQ refit is accepted or rejected CLOSED-FORM, with no trial
//     projection pass. The normal-equation sums are accumulated in RESIDUAL
//     coordinates (r = v − seed prediction, all small numbers, so the f16
//     sums don't cancel): the refit solve is e = seed + M⁻¹(sAR,sBR), and
//     the error of the re-quantised refit endpoints ON THE SEED'S INDICES is
//       E(δ) = sErr − 2(δ0·sAR + δ1·sBR) + δ0²sAA + 2δ0δ1·sAB + δ1²sBB
//     with δ = quantised endpoint − seed endpoint. E < sErr accepts; the one
//     reprojection pass that follows can only improve on E (it re-optimises
//     the indices for the new endpoints), so the accept decision is safe.
//     A rejected refit costs no per-pixel work at all (the old scheme paid a
//     full trial pass either way — this restructure is −14% GPU wall on
//     4096², measured interleaved, at equal PSNR).
//   • 3-bit indices are packed BRANCH-FREE: pixels 0..7 accumulate into a
//     24-bit word, pixels 8..15 into another, recombined with constant
//     shifts into the 48-bit field (w0 gets field bits 0..15 above the two
//     endpoint bytes, w1 gets field bits 16..47) — no per-pixel straddle
//     branches.
//
// f16 range notes: residual sums accumulate r/16 (|r| ≤ half a level ≈ 18),
// and endpoint deltas enter E() as δ/16, so every accumulator and product
// stays ≲4k — well inside f16's 65504 max — while the quantities being
// compared (block errors) are small numbers with plenty of mantissa.
//
// Level → BC4 index (0→r0 ... 7→r1): 0,2,3,4,5,6,7,1 — packed 3-bit LUT
// 0x3F58D0 = sum(idx[L] << 3L).
//
// The host selects this module only when the device reports shader-f16,
// falling back to bc5.wgsl otherwise.
enable f16;
alias h = f16;
alias h2 = vec2<f16>;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const IDX_LUT: u32 = 0x3F58D0u;

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);
  let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  // Load 4×4 R/G pairs (x = R, y = G throughout), min/max fused in.
  var v: array<h2, 16>;
  var vmin = h2(255.0);
  var vmax = h2(0.0);
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0), mx);
    let c = textureLoad(src_tex, p, 0);
    let val = h2(h(c.r * 255.0), h(c.g * 255.0));
    v[i] = val; vmin = min(vmin, val); vmax = max(vmax, val);
  }

  // Seed endpoints at the exact per-channel extremes (values are exact
  // integers — no rounding needed). Flat blocks get nudged apart to keep
  // the 6-interp mode (r0 > r1 strictly).
  var r0 = vec2<u32>(vmax);
  var r1 = vec2<u32>(vmin);
  if (r0.x == r1.x) { if (r1.x > 0u) { r1.x = r1.x - 1u; } else { r0.x = r0.x + 1u; } }
  if (r0.y == r1.y) { if (r1.y > 0u) { r1.y = r1.y - 1u; } else { r0.y = r0.y + 1u; } }

  let r0f = h2(vec2<f32>(r0));
  let r1f = h2(vec2<f32>(r1));
  let dir = r1f - r0f;
  let scale = h2(7.0) / dir;

  // ONE fused pass, both channels: level = round(7·(v − r0)/(r1 − r0))
  // projection, branch-free packed indices, and the least-squares sums in
  // residual coordinates. iA holds pixels 0..7 (3 bits each), iB pixels
  // 8..15.
  var sAA = h2(0.0); var sBB = h2(0.0); var sAB = h2(0.0);
  var sAR = h2(0.0); var sBR = h2(0.0); var sErr = h2(0.0);
  var lmin = h2(7.0); var lmax = h2(0.0);
  var iAx = 0u; var iBx = 0u; var iAy = 0u; var iBy = 0u;
  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let vr = v[k] - r0f;
    let L = clamp(floor(vr * scale + h2(0.5)), h2(0.0), h2(7.0));
    lmin = min(lmin, L); lmax = max(lmax, L);
    let b = L * h2(1.0 / 7.0);
    let a = h2(1.0) - b;
    let r16 = (vr - b * dir) * h2(1.0 / 16.0);
    sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b;
    sAR = sAR + a * r16; sBR = sBR + b * r16; sErr = sErr + r16 * r16;
    iAx = iAx | (((IDX_LUT >> (u32(L.x) * 3u)) & 7u) << (k * 3u));
    iAy = iAy | (((IDX_LUT >> (u32(L.y) * 3u)) & 7u) << (k * 3u));
  }
  for (var k: u32 = 8u; k < 16u; k = k + 1u) {
    let vr = v[k] - r0f;
    let L = clamp(floor(vr * scale + h2(0.5)), h2(0.0), h2(7.0));
    lmin = min(lmin, L); lmax = max(lmax, L);
    let b = L * h2(1.0 / 7.0);
    let a = h2(1.0) - b;
    let r16 = (vr - b * dir) * h2(1.0 / 16.0);
    sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b;
    sAR = sAR + a * r16; sBR = sBR + b * r16; sErr = sErr + r16 * r16;
    iBx = iBx | (((IDX_LUT >> (u32(L.x) * 3u)) & 7u) << ((k - 8u) * 3u));
    iBy = iBy | (((IDX_LUT >> (u32(L.y) * 3u)) & 7u) << ((k - 8u) * 3u));
  }

  // Closed-form LSQ refit per channel. Rank-1 guard: with every pixel on ONE
  // level the system is singular and det is pure f16 rounding noise (≲0.03);
  // with ≥2 distinct levels det = Σ_i<j (b_j − b_i)² ≥ 15/49 ≈ 0.306 — 0.1
  // separates cleanly. Endpoints clamp to [0,255], NOT the block's value
  // range: for a scalar channel, endpoints beyond the data range are often
  // genuinely optimal and there is no colour axis to bend (the bbox clamp
  // the colour formats need costs ~0.3 dB here).
  let det = sAA * sBB - sAB * sAB;
  let d0 = (sBB * sAR - sAB * sBR) * h2(16.0);
  let d1 = (sAA * sBR - sAB * sAR) * h2(16.0);
  var n0 = r0; var n1 = r1;
  var accept = false;
  for (var c: u32 = 0u; c < 2u; c = c + 1u) {
    if (lmin[c] < lmax[c] && abs(det[c]) > h(0.1)) {
      let e0 = clamp(r0f[c] + d0[c] / det[c], h(0.0), h(255.0));
      let e1 = clamp(r1f[c] + d1[c] / det[c], h(0.0), h(255.0));
      let q0 = u32(floor(e0 + h(0.5)));
      let q1 = u32(floor(e1 + h(0.5)));
      // Keep 6-interp mode (q0 > q1 strictly); skip the no-op refit.
      if (q0 > q1 && !(q0 == r0[c] && q1 == r1[c])) {
        let dd0 = (h(f32(q0)) - r0f[c]) * h(1.0 / 16.0);
        let dd1 = (h(f32(q1)) - r1f[c]) * h(1.0 / 16.0);
        let eNew = sErr[c] - h(2.0) * (dd0 * sAR[c] + dd1 * sBR[c])
          + dd0 * dd0 * sAA[c] + h(2.0) * dd0 * dd1 * sAB[c] + dd1 * dd1 * sBB[c];
        if (eNew < sErr[c]) {
          n0[c] = q0; n1[c] = q1;
          accept = true;
        }
      }
    }
  }

  // One reprojection pass against the final endpoints when either channel's
  // refit was accepted; the unchanged channel just reproduces its seed
  // output. Re-optimising the indices only lowers the error below the E()
  // that justified the accept.
  if (accept) {
    let f0 = h2(vec2<f32>(n0));
    let fscale = h2(7.0) / (h2(vec2<f32>(n1)) - f0);
    iAx = 0u; iBx = 0u; iAy = 0u; iBy = 0u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      let L = clamp(floor((v[k] - f0) * fscale + h2(0.5)), h2(0.0), h2(7.0));
      iAx = iAx | (((IDX_LUT >> (u32(L.x) * 3u)) & 7u) << (k * 3u));
      iAy = iAy | (((IDX_LUT >> (u32(L.y) * 3u)) & 7u) << (k * 3u));
    }
    for (var k: u32 = 8u; k < 16u; k = k + 1u) {
      let L = clamp(floor((v[k] - f0) * fscale + h2(0.5)), h2(0.0), h2(7.0));
      iBx = iBx | (((IDX_LUT >> (u32(L.x) * 3u)) & 7u) << ((k - 8u) * 3u));
      iBy = iBy | (((IDX_LUT >> (u32(L.y) * 3u)) & 7u) << ((k - 8u) * 3u));
    }
  }

  // BC5 block = R half (bytes 0..7) || G half (bytes 8..15) = 4 u32s.
  let o = bi * 4u;
  dst[o] = n0.x | (n1.x << 8u) | (iAx << 16u);
  dst[o + 1u] = (iAx >> 16u) | (iBx << 8u);
  dst[o + 2u] = n0.y | (n1.y << 8u) | (iAy << 16u);
  dst[o + 3u] = (iAy >> 16u) | (iBy << 8u);
}
