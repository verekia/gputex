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
// Same algorithm as bc5_fast_f16.wgsl (see that file for the full notes):
//   • both channels processed as vec2 lanes of the fused passes;
//   • pass 1 accumulates MOMENTS (ΣL, ΣL², Σρ, ΣLρ) from which every LSQ
//     normal-equation sum is an O(1) per-block expression; the rank guard
//     is the exact 16·ΣL² == (ΣL)² test;
//   • the closed-form refit prices the nearest rounding of the solve
//     through E(δ) = err − 2(δ0·sAR + δ1·sBR) + δ0²sAA + 2δ0δ1·sAB
//     + δ1²sBB, accept-if-better;
//   • pass 2 packs the indices ONCE, against the FINAL endpoints — full
//     reprojection quality at parity cost;
//   • the 16 texel reads are 8 textureGather fetches (4 quads × R,G)
//     through a clamp-to-edge sampler, byte-identical to per-texel loads;
//   • 3-bit indices accumulate branch-free into two 24-bit words (pixels
//     0..7 and 8..15) recombined with constant shifts — no per-pixel
//     straddle branches.
// Values are kept in the [0,255] f32 domain throughout.
//
// Level → BC4 index LUT (0,2,3,4,5,6,7,1) packed as 3-bit entries in
// 0x3F58D0.

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var smp: sampler;

const IDX_LUT: u32 = 0x3F58D0u;

// Closed-form accept-if-better endpoint refinement for one channel — same
// as the f16 module's `refine` (both run this per-block step in f32).
// Endpoints clamp to [0,255], NOT the block's value range: for a scalar
// channel, endpoints beyond the data range are often genuinely optimal and
// there is no colour axis to bend.
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
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) {
    return;
  }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);

  // Load 4×4 R/G pairs (x = R, y = G throughout), min/max fused in.
  // Interior blocks read via 8 gathers normalised by the PHYSICAL (padded)
  // texture size; blocks straddling the source edge of a non-multiple-of-4
  // image fall back to per-texel loads clamped to the last real texel (the
  // padding strip is zero-initialised — see bc5_fast_f16.wgsl).
  // Gather components: w=(0,0) z=(1,0) x=(0,1) y=(1,1) within each quad.
  var v: array<vec2<f32>, 16>;
  var vmin = vec2<f32>(255.0);
  var vmax = vec2<f32>(0.0);
  if (u32(base.x) + 4u <= params.width && u32(base.y) + 4u <= params.height) {
    let inv_size = vec2<f32>(1.0, 1.0) / vec2<f32>(textureDimensions(src_tex));
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
      let qo = vec2<u32>((q & 1u) * 2u, (q >> 1u) * 2u);
      let cc = (vec2<f32>(base) + vec2<f32>(qo) + vec2<f32>(1.0, 1.0)) * inv_size;
      let r4 = textureGather(0, src_tex, smp, cc) * 255.0;
      let g4 = textureGather(1, src_tex, smp, cc) * 255.0;
      let i = qo.y * 4u + qo.x;
      let vw = vec2<f32>(r4.w, g4.w);
      let vz = vec2<f32>(r4.z, g4.z);
      let vx = vec2<f32>(r4.x, g4.x);
      let vy = vec2<f32>(r4.y, g4.y);
      v[i] = vw; v[i + 1u] = vz; v[i + 4u] = vx; v[i + 5u] = vy;
      vmin = min(min(vmin, min(vw, vz)), min(vx, vy));
      vmax = max(max(vmax, max(vw, vz)), max(vx, vy));
    }
  } else {
    let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
      let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0), mx);
      let c = textureLoad(src_tex, p, 0);
      let val = vec2<f32>(c.r, c.g) * 255.0;
      v[i] = val; vmin = min(vmin, val); vmax = max(vmax, val);
    }
  }

  // Seed endpoints at the exact per-channel extremes (round-to-nearest, the
  // same rule the CPU reference uses). Flat blocks get nudged apart to keep
  // the 6-interp mode (r0 > r1 strictly).
  var r0 = vec2<u32>(clamp(floor(vmax + 0.5), vec2<f32>(0.0), vec2<f32>(255.0)));
  var r1 = vec2<u32>(clamp(floor(vmin + 0.5), vec2<f32>(0.0), vec2<f32>(255.0)));
  if (r0.x == r1.x) { if (r1.x > 0u) { r1.x = r1.x - 1u; } else { r0.x = r0.x + 1u; } }
  if (r0.y == r1.y) { if (r1.y > 0u) { r1.y = r1.y - 1u; } else { r0.y = r0.y + 1u; } }

  let r0f = vec2<f32>(r0);
  let r1f = vec2<f32>(r1);
  let dir = r1f - r0f;
  let scale = vec2<f32>(7.0) / dir;

  // Pass 1, both channels — MOMENTS only. t = 7(v−r0)/(r1−r0) ∈ [0,7]
  // (the seed covers the data), L = round(t), ρ = t − L.
  var sL = vec2<f32>(0.0); var sLL = vec2<f32>(0.0);
  var pR = vec2<f32>(0.0); var pLR = vec2<f32>(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let t = (v[k] - r0f) * scale;
    let L = clamp(floor(t + 0.5), vec2<f32>(0.0), vec2<f32>(7.0));
    let rho = t - L;
    sL = sL + L; sLL = sLL + L * L;
    pR = pR + rho; pLR = pLR + L * rho;
  }

  // Per-block refit off the moments (see bc5_fast_f16.wgsl for the
  // identities).
  let sBB = sLL * (1.0 / 49.0);
  let sAB = sL * (1.0 / 7.0) - sBB;
  let sAA = vec2<f32>(16.0) - 2.0 * sL * (1.0 / 7.0) + sBB;
  let sBR = pLR * dir * (1.0 / 49.0);
  let sAR = (pR - pLR * (1.0 / 7.0)) * dir * (1.0 / 7.0);
  let spread = 16.0 * sLL != sL * sL;

  let fx = refine(sAA.x, sBB.x, sAB.x, sAR.x, sBR.x, r0.x, r1.x, spread.x);
  let fy = refine(sAA.y, sBB.y, sAB.y, sAR.y, sBR.y, r0.y, r1.y, spread.y);
  let n0 = vec2<u32>(fx.x, fy.x);
  let n1 = vec2<u32>(fx.y, fy.y);

  // Pass 2, both channels — pack the shipped indices against the FINAL
  // endpoints (rejected channels re-derive their seed assignment). iA
  // holds pixels 0..7 (3 bits each), iB pixels 8..15.
  let n0f = vec2<f32>(n0);
  let n1f = vec2<f32>(n1);
  let scale2 = vec2<f32>(7.0) / (n1f - n0f);
  var iAx = 0u; var iBx = 0u; var iAy = 0u; var iBy = 0u;
  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let L = clamp(floor((v[k] - n0f) * scale2 + 0.5), vec2<f32>(0.0), vec2<f32>(7.0));
    iAx = iAx | (((IDX_LUT >> (u32(L.x) * 3u)) & 7u) << (k * 3u));
    iAy = iAy | (((IDX_LUT >> (u32(L.y) * 3u)) & 7u) << (k * 3u));
  }
  for (var k: u32 = 8u; k < 16u; k = k + 1u) {
    let L = clamp(floor((v[k] - n0f) * scale2 + 0.5), vec2<f32>(0.0), vec2<f32>(7.0));
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
