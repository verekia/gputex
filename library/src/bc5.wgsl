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
//   • both channels processed as vec2 lanes of one fused pass — projection
//     assignment, branch-free packed indices, and LSQ sums in residual
//     coordinates;
//   • the refit is accepted CLOSED-FORM from the sums (no trial projection
//     pass): E(δ) = sErr − 2(δ0·sAR + δ1·sBR) + δ0²sAA + 2δ0δ1·sAB + δ1²sBB
//     for the re-quantised endpoint deltas δ, compared against the seed's
//     error; accepted refits ship the SEED indices (no reprojection pass —
//     see the f16 module's header for the measured trade);
//   • the 16 texel reads are 8 textureGather fetches (4 quads × R,G)
//     through a clamp-to-edge sampler, byte-identical to per-texel loads;
//   • 3-bit indices accumulate branch-free into two 24-bit words (pixels
//     0..7 and 8..15) recombined with constant shifts — no per-pixel
//     straddle branches.
// Values are kept in the [0,255] f32 domain; the only difference from the
// f16 module is that no /16 range scaling is needed.
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

  // ONE fused pass, both channels: projection assignment, branch-free packed
  // indices, and the least-squares sums in residual coordinates.
  var sAA = vec2<f32>(0.0); var sBB = vec2<f32>(0.0); var sAB = vec2<f32>(0.0);
  var sAR = vec2<f32>(0.0); var sBR = vec2<f32>(0.0); var sErr = vec2<f32>(0.0);
  var lmin = vec2<f32>(7.0); var lmax = vec2<f32>(0.0);
  var iAx = 0u; var iBx = 0u; var iAy = 0u; var iBy = 0u;
  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let vr = v[k] - r0f;
    let L = clamp(floor(vr * scale + 0.5), vec2<f32>(0.0), vec2<f32>(7.0));
    lmin = min(lmin, L); lmax = max(lmax, L);
    let b = L * (1.0 / 7.0);
    let a = 1.0 - b;
    let r = vr - b * dir;
    sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b;
    sAR = sAR + a * r; sBR = sBR + b * r; sErr = sErr + r * r;
    iAx = iAx | (((IDX_LUT >> (u32(L.x) * 3u)) & 7u) << (k * 3u));
    iAy = iAy | (((IDX_LUT >> (u32(L.y) * 3u)) & 7u) << (k * 3u));
  }
  for (var k: u32 = 8u; k < 16u; k = k + 1u) {
    let vr = v[k] - r0f;
    let L = clamp(floor(vr * scale + 0.5), vec2<f32>(0.0), vec2<f32>(7.0));
    lmin = min(lmin, L); lmax = max(lmax, L);
    let b = L * (1.0 / 7.0);
    let a = 1.0 - b;
    let r = vr - b * dir;
    sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b;
    sAR = sAR + a * r; sBR = sBR + b * r; sErr = sErr + r * r;
    iBx = iBx | (((IDX_LUT >> (u32(L.x) * 3u)) & 7u) << ((k - 8u) * 3u));
    iBy = iBy | (((IDX_LUT >> (u32(L.y) * 3u)) & 7u) << ((k - 8u) * 3u));
  }

  // Closed-form LSQ refit per channel. Rank-1 guard: with every pixel on ONE
  // level the system is singular (det is float rounding noise); with ≥2
  // distinct levels det = Σ_i<j (b_j − b_i)² ≥ 15/49 ≈ 0.306. Endpoints
  // clamp to [0,255], NOT the block's value range: for a scalar channel,
  // endpoints beyond the data range are often genuinely optimal and there is
  // no colour axis to bend.
  let det = sAA * sBB - sAB * sAB;
  let d0 = sBB * sAR - sAB * sBR;
  let d1 = sAA * sBR - sAB * sAR;
  var n0 = r0; var n1 = r1;
  for (var c: u32 = 0u; c < 2u; c = c + 1u) {
    if (lmin[c] < lmax[c] && abs(det[c]) > 1e-3) {
      let e0 = clamp(r0f[c] + d0[c] / det[c], 0.0, 255.0);
      let e1 = clamp(r1f[c] + d1[c] / det[c], 0.0, 255.0);
      // Price all four floor/ceil roundings of the fractional solve
      // closed-form (see bc5_fast_f16.wgsl); keep the best that stays in
      // 6-interp mode and beats the seed.
      var bestE = 0.0;
      for (var m: u32 = 0u; m < 4u; m = m + 1u) {
        let q0f = clamp(floor(e0) + f32(m & 1u), 0.0, 255.0);
        let q1f = clamp(floor(e1) + f32(m >> 1u), 0.0, 255.0);
        let q0 = u32(q0f);
        let q1 = u32(q1f);
        if (q0 > q1 && !(q0 == r0[c] && q1 == r1[c])) {
          let dd0 = q0f - r0f[c];
          let dd1 = q1f - r1f[c];
          let eNew = -2.0 * (dd0 * sAR[c] + dd1 * sBR[c])
            + dd0 * dd0 * sAA[c] + 2.0 * dd0 * dd1 * sAB[c] + dd1 * dd1 * sBB[c];
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
