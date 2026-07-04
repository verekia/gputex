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
//   • BOTH channels ride the same fused passes as vec2<f16> lanes — each
//     loop computes projections and moments for R and G at once instead of
//     two scalar encode_bc4 calls.
//   • The 16 texel reads are 8 textureGather fetches (4 quads × R,G) for
//     interior blocks — byte-identical output to per-texel loads, −3.6%
//     GPU on 4096² (/ab, 2026-07). Blocks straddling the source edge of a
//     non-multiple-of-4 image use clamped per-texel loads instead: the
//     upload pads the texture with ZEROS, so a normalised-coordinate
//     gather there would read padding (or mis-scale against the padded
//     size) instead of replicating the last real texel.
//   • Pass 1 accumulates MOMENTS, not normal-equation sums. With b = L/7
//     and the level-space residual ρ = t − L (t = 7(v−r0)/(r1−r0), so the
//     value-space residual is r = ρ·dir/7), every LSQ sum is an O(1)
//     per-block function of four accumulators:
//       sBB = ΣL²/49       sAB = ΣL/7 − ΣL²/49    sAA = 16 − 2ΣL/7 + ΣL²/49
//       sBR = ΣLρ·dir/49   sAR = (Σρ − ΣLρ/7)·dir/7
//     Per-pixel work drops from 5 product-accumulates + 3 temporaries to 4
//     cheap accumulates (ΣL, ΣL², Σρ, ΣLρ), and the f16 range analysis
//     becomes trivial: ΣL ≤ 112 and ΣL² ≤ 784 are exact f16 integers,
//     |ρ| ≤ ~0.5 keeps Σρ/ΣLρ tiny. The per-BLOCK refit math (including
//     the solve and E(δ) pricing) runs in f32 — free at block granularity,
//     and it retires the sum-cancellation worries the old residual-sum
//     scheme was built around.
//   • The rank guard is EXACT: all pixels on one level ⟺ 16·ΣL² == (ΣL)²
//     (integers, so the comparison is precise in f32) — no lmin/lmax
//     tracking in the loop.
//   • The refit is accepted or rejected CLOSED-FORM, with no trial
//     projection pass: the solve is e = seed + M⁻¹(sAR,sBR), and the error
//     of re-quantised endpoints ON THE CURRENT INDICES is
//       E(δ) = err − 2(δ0·sAR + δ1·sBR) + δ0²sAA + 2δ0δ1·sAB + δ1²sBB
//     with δ = quantised endpoint − base endpoint, compared as the delta
//     form E − err < 0. Only the NEAREST rounding of the fractional solve
//     is priced: pricing all four floor/ceil combinations (the correlated
//     2-D quadratic's integer optimum isn't always the nearest rounding)
//     measured ≤0.015 dB on every content class but ~8% GPU on smooth
//     content, where Apple's lossless texture compression collapses the
//     read cost and leaves the kernel ALU-bound (/ab 2026-07, rock 4K
//     displacement: floor 0.61× of the rgba8-noise floor).
//   • Pass 2 packs the indices ONCE, against the FINAL endpoints — full
//     reprojection quality. The previous single-pass scheme shipped the
//     SEED indices with accepted refit endpoints, giving up 0.08..0.19 dB
//     because a third reprojection loop cost +14% GPU; with the moment
//     form paying for the second loop, reprojection now measures at PARITY
//     with that scheme (/ab 2026-07, interleaved: proc 2048²/4096² and the
//     rock 4K normal map, all within ±1%, vs read floor ~2% below).
//     Measured and NOT taken: a second refit round off pass-2 moments
//     (+14%, register cliff, ≈0 gain on smooth content — round 2 only pays
//     on noise); full normal-equation sums in both passes (+49%).
//   • 3-bit indices are packed BRANCH-FREE: pixels 0..7 accumulate into a
//     24-bit word, pixels 8..15 into another, recombined with constant
//     shifts into the 48-bit field (w0 gets field bits 0..15 above the two
//     endpoint bytes, w1 gets field bits 16..47) — no per-pixel straddle
//     branches.
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

// Closed-form accept-if-better endpoint refinement for one channel (f32:
// per-block O(1) work, so precision is free here). Takes the LSQ sums for
// the current indices, the current integer endpoints b0 > b1, and the rank
// guard; prices the nearest rounding of the fractional solve via E(δ) and
// returns it when it strictly improves and stays in 6-interp mode
// (q0 > q1), or (b0,b1) unchanged. Endpoints clamp to [0,255], NOT the
// block's value range: for a scalar channel, endpoints beyond the data
// range are often genuinely optimal and there is no colour axis to bend
// (the bbox clamp the colour formats need costs ~0.3 dB here).
fn refine(sAA: f32, sBB: f32, sAB: f32, sAR: f32, sBR: f32, b0: u32, b1: u32, spread: bool) -> vec2<u32> {
  var out = vec2<u32>(b0, b1);
  let det = sAA * sBB - sAB * sAB;
  if (!spread || abs(det) <= 1e-3) { return out; }
  let b0f = f32(b0);
  let b1f = f32(b1);
  let e0 = clamp(b0f + (sBB * sAR - sAB * sBR) / det, 0.0, 255.0);
  let e1 = clamp(b1f + (sAA * sBR - sAB * sAR) / det, 0.0, 255.0);
  let q0f = floor(e0 + 0.5);
  let q1f = floor(e1 + 0.5);
  let q0 = u32(q0f);
  let q1 = u32(q1f);
  if (q0 > q1 && !(q0 == b0 && q1 == b1)) {
    let dd0 = q0f - b0f;
    let dd1 = q1f - b1f;
    let eNew = -2.0 * (dd0 * sAR + dd1 * sBR)
      + dd0 * dd0 * sAA + 2.0 * dd0 * dd1 * sAB + dd1 * dd1 * sBB;
    if (eNew < 0.0) {
      out = vec2<u32>(q0, q1);
    }
  }
  return out;
}

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

  // Pass 1, both channels — MOMENTS only. t = 7(v−r0)/(r1−r0) ∈ [0,7]
  // (the seed covers the data), L = round(t), ρ = t − L.
  var sL = h2(0.0); var sLL = h2(0.0); var pR = h2(0.0); var pLR = h2(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let t = (v[k] - r0f) * scale;
    let L = clamp(floor(t + h2(0.5)), h2(0.0), h2(7.0));
    let rho = t - L;
    sL = sL + L; sLL = sLL + L * L;
    pR = pR + rho; pLR = pLR + L * rho;
  }

  // Per-block refit in f32 off the moments (see header for the identities).
  let sLf = vec2<f32>(sL);
  let sLLf = vec2<f32>(sLL);
  let dirf = vec2<f32>(r1) - vec2<f32>(r0);
  let sBB = sLLf * (1.0 / 49.0);
  let sAB = sLf * (1.0 / 7.0) - sBB;
  let sAA = vec2<f32>(16.0) - 2.0 * sLf * (1.0 / 7.0) + sBB;
  let sBR = vec2<f32>(pLR) * dirf * (1.0 / 49.0);
  let sAR = (vec2<f32>(pR) - vec2<f32>(pLR) * (1.0 / 7.0)) * dirf * (1.0 / 7.0);
  let spread = 16.0 * sLLf != sLf * sLf;

  let fx = refine(sAA.x, sBB.x, sAB.x, sAR.x, sBR.x, r0.x, r1.x, spread.x);
  let fy = refine(sAA.y, sBB.y, sAB.y, sAR.y, sBR.y, r0.y, r1.y, spread.y);
  let n0 = vec2<u32>(fx.x, fy.x);
  let n1 = vec2<u32>(fx.y, fy.y);

  // Pass 2, both channels — pack the shipped indices against the FINAL
  // endpoints (rejected channels re-derive their seed assignment). iA
  // holds pixels 0..7 (3 bits each), iB pixels 8..15.
  let n0f = h2(vec2<f32>(n0));
  let n1f = h2(vec2<f32>(n1));
  let scale2 = h2(7.0) / (n1f - n0f);
  var iAx = 0u; var iBx = 0u; var iAy = 0u; var iBy = 0u;
  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let L = clamp(floor((v[k] - n0f) * scale2 + h2(0.5)), h2(0.0), h2(7.0));
    iAx = iAx | (((IDX_LUT >> (u32(L.x) * 3u)) & 7u) << (k * 3u));
    iAy = iAy | (((IDX_LUT >> (u32(L.y) * 3u)) & 7u) << (k * 3u));
  }
  for (var k: u32 = 8u; k < 16u; k = k + 1u) {
    let L = clamp(floor((v[k] - n0f) * scale2 + h2(0.5)), h2(0.0), h2(7.0));
    iBx = iBx | (((IDX_LUT >> (u32(L.x) * 3u)) & 7u) << ((k - 8u) * 3u));
    iBy = iBy | (((IDX_LUT >> (u32(L.y) * 3u)) & 7u) << ((k - 8u) * 3u));
  }

  // BC5 block = R half (bytes 0..7) || G half (bytes 8..15) = 4 u32s.
  let o = bi * 4u;
  dst[o] = n0.x | (n1.x << 8u) | (iAx << 16u);
  dst[o + 1u] = (iAx >> 16u) | (iBx << 8u);
  dst[o + 2u] = n0.y | (n1.y << 8u) | (iAy << 16u);
  dst[o + 3u] = (iAy >> 16u) | (iBy << 8u);
}
