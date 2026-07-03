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
//   • The 16 texel reads are 8 textureGather fetches (4 quads × R,G) for
//     interior blocks — byte-identical output to per-texel loads, −3.6%
//     GPU on 4096² (/ab, 2026-07). Blocks straddling the source edge of a
//     non-multiple-of-4 image use clamped per-texel loads instead: the
//     upload pads the texture with ZEROS, so a normalised-coordinate
//     gather there would read padding (or mis-scale against the padded
//     size) instead of replicating the last real texel.
//   • The LSQ refit is accepted or rejected CLOSED-FORM, with no trial
//     projection pass. The normal-equation sums are accumulated in RESIDUAL
//     coordinates (r = v − seed prediction, all small numbers, so the f16
//     sums don't cancel): the refit solve is e = seed + M⁻¹(sAR,sBR), and
//     the error of the re-quantised refit endpoints ON THE SEED'S INDICES is
//       E(δ) = sErr − 2(δ0·sAR + δ1·sBR) + δ0²sAA + 2δ0δ1·sAB + δ1²sBB
//     with δ = quantised endpoint − seed endpoint. E < sErr accepts (the
//     code compares the delta form E − sErr < 0, so sErr itself is only
//     accumulated for the derivation's sake — see below). All four
//     floor/ceil roundings of the fractional solve are priced, since the
//     integer optimum of a correlated 2-D quadratic isn't always the
//     component-wise nearest rounding (+0.01 dB, free).
//   • Accepted refits SHIP THE SEED INDICES — there is no reprojection
//     pass, so E(δ) is exactly the shipped error and the accept test is
//     exact. Re-optimising the indices against the refit endpoints was
//     measured at +14% GPU wall for +0.08 dB (synthetic normal card,
//     53.10 → 53.18) to +0.19 dB (roughness/normal photo set) — the wrong
//     side of the trade for a throughput encoder (/ab + PSNR A/B,
//     2026-07). A margin-gated reprojection was also tried and is a dead
//     end: gain-vs-margin has no exploitable knee (θ=0.15 recovers 0% of
//     the cost on noisy content, θ=0.4 costs the same quality as dropping
//     the pass for half the saving).
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
@group(0) @binding(3) var smp: sampler;

const IDX_LUT: u32 = 0x3F58D0u;

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);

  // Load 4×4 R/G pairs (x = R, y = G throughout), min/max fused in.
  // INTERIOR blocks — every block when the source is a multiple of 4, so
  // the branch is wavefront-uniform on benchmark-shaped content — read via
  // 8 gathers (4 quads × R,G); the gather point (base+quad+1) normalised by
  // the PHYSICAL (padded) texture size sits exactly between the quad's
  // texel centers, and interior quads never touch the zero-initialised
  // padding strip. Blocks straddling the source edge of a
  // non-multiple-of-4 image fall back to per-texel loads clamped to the
  // last real texel (gather cannot replicate an edge texel mid-quad).
  // Gather components: w=(0,0) z=(1,0) x=(0,1) y=(1,1) within each quad.
  var v: array<h2, 16>;
  var vmin = h2(255.0);
  var vmax = h2(0.0);
  if (u32(base.x) + 4u <= params.width && u32(base.y) + 4u <= params.height) {
    let inv_size = vec2<f32>(1.0, 1.0) / vec2<f32>(textureDimensions(src_tex));
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
      let qo = vec2<u32>((q & 1u) * 2u, (q >> 1u) * 2u);
      let cc = (vec2<f32>(base) + vec2<f32>(qo) + vec2<f32>(1.0, 1.0)) * inv_size;
      let r4 = textureGather(0, src_tex, smp, cc) * 255.0;
      let g4 = textureGather(1, src_tex, smp, cc) * 255.0;
      let i = qo.y * 4u + qo.x;
      let vw = h2(h(r4.w), h(g4.w));
      let vz = h2(h(r4.z), h(g4.z));
      let vx = h2(h(r4.x), h(g4.x));
      let vy = h2(h(r4.y), h(g4.y));
      v[i] = vw; v[i + 1u] = vz; v[i + 4u] = vx; v[i + 5u] = vy;
      vmin = min(min(vmin, min(vw, vz)), min(vx, vy));
      vmax = max(max(vmax, max(vw, vz)), max(vx, vy));
    }
  } else {
    let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
      let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0), mx);
      let c = textureLoad(src_tex, p, 0);
      let val = h2(h(c.r * 255.0), h(c.g * 255.0));
      v[i] = val; vmin = min(vmin, val); vmax = max(vmax, val);
    }
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
  for (var c: u32 = 0u; c < 2u; c = c + 1u) {
    if (lmin[c] < lmax[c] && abs(det[c]) > h(0.1)) {
      let e0 = clamp(r0f[c] + d0[c] / det[c], h(0.0), h(255.0));
      let e1 = clamp(r1f[c] + d1[c] / det[c], h(0.0), h(255.0));
      // The integer optimum of the E() quadratic isn't always the
      // component-wise rounding of the fractional solve, so price all four
      // floor/ceil combinations — closed-form, no per-pixel work — and
      // keep the best that stays in 6-interp mode (q0 > q1 strictly) and
      // beats the seed (E(seed) − sErr = 0).
      var bestE = h(0.0);
      for (var m: u32 = 0u; m < 4u; m = m + 1u) {
        let q0f = clamp(floor(e0) + h(f32(m & 1u)), h(0.0), h(255.0));
        let q1f = clamp(floor(e1) + h(f32(m >> 1u)), h(0.0), h(255.0));
        let q0 = u32(q0f);
        let q1 = u32(q1f);
        if (q0 > q1 && !(q0 == r0[c] && q1 == r1[c])) {
          let dd0 = (q0f - r0f[c]) * h(1.0 / 16.0);
          let dd1 = (q1f - r1f[c]) * h(1.0 / 16.0);
          let eNew = -h(2.0) * (dd0 * sAR[c] + dd1 * sBR[c])
            + dd0 * dd0 * sAA[c] + h(2.0) * dd0 * dd1 * sAB[c] + dd1 * dd1 * sBB[c];
          if (eNew < bestE) {
            bestE = eNew;
            n0[c] = q0; n1[c] = q1;
          }
        }
      }
    }
  }

  // BC5 block = R half (bytes 0..7) || G half (bytes 8..15) = 4 u32s.
  let o = bi * 4u;
  dst[o] = n0.x | (n1.x << 8u) | (iAx << 16u);
  dst[o + 1u] = (iAx >> 16u) | (iBx << 8u);
  dst[o + 2u] = n0.y | (n1.y << 8u) | (iAy << 16u);
  dst[o + 3u] = (iAy >> 16u) | (iBy << 8u);
}
