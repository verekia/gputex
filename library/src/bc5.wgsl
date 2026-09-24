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
//   • both channels processed in the same fused passes, texels held as
//     quad-major vec4s per channel (the gather layout);
//   • pass 1 accumulates MOMENTS (ΣL, ΣL², Σd, ΣL·d with d = v − r0) from
//     which every LSQ normal-equation sum is an O(1) per-block expression;
//     the seed covers the data, so pass 1 needs no clamp; the rank guard
//     is the exact 16·ΣL² == (ΣL)² test;
//   • the closed-form refit prices the nearest rounding of the solve
//     through E(δ) = err − 2(δ0·sAR + δ1·sBR) + δ0²sAA + 2δ0δ1·sAB
//     + δ1²sBB, accept-if-better, both channels as branch-free vec2 lanes;
//   • pass 2 derives the levels ONCE, against the FINAL endpoints — full
//     reprojection quality;
//   • the 16 texel reads are 8 textureGather fetches (4 quads × R,G)
//     through a clamp-to-edge sampler, byte-identical to per-texel loads;
//   • 3-bit levels pack as Σ L·8^k in f32 (exact below 2^24), then one
//     SWAR level → BC4 index map (0→0, 7→1, L→L+1) per 24-bit word.
// Values are kept in the [0,255] f32 domain throughout.

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
  y0:       u32, // first block row of this dispatch (row-band encodes)
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var smp: sampler;

// 8 packed 3-bit levels → BC4 indices (0→0, 7→1, L→L+1 otherwise).
fn lvl_to_idx(x: u32) -> u32 {
  let y = ((x & 0x6DB6DBu) + 0x249249u) ^ (x & 0x924924u);
  return y ^ (~((y >> 1u) | (y >> 2u)) & 0x249249u);
}

// Per-quad pixel weights 8^k (gather order x,y,z,w = (0,1),(1,1),(1,0),(0,0)).
const W0 = vec4<f32>(4096.0, 32768.0, 8.0, 1.0);
const W1 = vec4<f32>(262144.0, 2097152.0, 512.0, 64.0);

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid_raw: vec3<u32>) {
  // Row-band encodes dispatch a slice of the block grid starting at row y0.
  let gid = vec3<u32>(gid_raw.x, gid_raw.y + params.y0, gid_raw.z);
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) {
    return;
  }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);

  // Load 4×4 R and G as quad-major vec4s in gather order (w=(0,0) z=(1,0)
  // x=(0,1) y=(1,1) of quad q = (x ≥ 2) + 2·(y ≥ 2)). Interior blocks read
  // via 8 gathers normalised by the PHYSICAL (padded) texture size; blocks
  // straddling the source edge of a non-multiple-of-4 image fall back to
  // per-texel loads clamped to the last real texel (the padding strip is
  // zero-initialised — see bc5_fast_f16.wgsl).
  var vr: array<vec4<f32>, 4>;
  var vg: array<vec4<f32>, 4>;
  if (u32(base.x) + 4u <= params.width && u32(base.y) + 4u <= params.height) {
    let inv_size = vec2<f32>(1.0, 1.0) / vec2<f32>(textureDimensions(src_tex));
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
      let qo = vec2<u32>((q & 1u) * 2u, (q >> 1u) * 2u);
      let cc = (vec2<f32>(base) + vec2<f32>(qo) + vec2<f32>(1.0, 1.0)) * inv_size;
      vr[q] = textureGather(0, src_tex, smp, cc) * 255.0;
      vg[q] = textureGather(1, src_tex, smp, cc) * 255.0;
    }
  } else {
    let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
      let qo = base + vec2<i32>(i32(q & 1u) * 2, i32(q >> 1u) * 2);
      let cx = textureLoad(src_tex, clamp(qo + vec2<i32>(0, 1), vec2<i32>(0), mx), 0);
      let cy = textureLoad(src_tex, clamp(qo + vec2<i32>(1, 1), vec2<i32>(0), mx), 0);
      let cz = textureLoad(src_tex, clamp(qo + vec2<i32>(1, 0), vec2<i32>(0), mx), 0);
      let cw = textureLoad(src_tex, clamp(qo, vec2<i32>(0), mx), 0);
      vr[q] = vec4<f32>(cx.r, cy.r, cz.r, cw.r) * 255.0;
      vg[q] = vec4<f32>(cx.g, cy.g, cz.g, cw.g) * 255.0;
    }
  }
  let mnr = min(min(vr[0], vr[1]), min(vr[2], vr[3]));
  let mxr = max(max(vr[0], vr[1]), max(vr[2], vr[3]));
  let mng = min(min(vg[0], vg[1]), min(vg[2], vg[3]));
  let mxg = max(max(vg[0], vg[1]), max(vg[2], vg[3]));
  let vmin = vec2<f32>(min(min(mnr.x, mnr.y), min(mnr.z, mnr.w)), min(min(mng.x, mng.y), min(mng.z, mng.w)));
  let vmax = vec2<f32>(max(max(mxr.x, mxr.y), max(mxr.z, mxr.w)), max(max(mxg.x, mxg.y), max(mxg.z, mxg.w)));

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

  // Pass 1, both channels — MOMENTS only. t = d·scale ∈ [0,7] (the seed
  // covers the data, so no clamp), L = round(t).
  var sL = vec2<f32>(0.0); var sLL = vec2<f32>(0.0);
  var sd = vec2<f32>(0.0); var sLd = vec2<f32>(0.0);
  for (var q: u32 = 0u; q < 4u; q = q + 1u) {
    let dr = vr[q] - r0f.x;
    let dg = vg[q] - r0f.y;
    let Lr = floor(dr * scale.x + 0.5);
    let Lg = floor(dg * scale.y + 0.5);
    sL = sL + vec2<f32>(dot(Lr, vec4<f32>(1.0)), dot(Lg, vec4<f32>(1.0)));
    sLL = sLL + vec2<f32>(dot(Lr, Lr), dot(Lg, Lg));
    sd = sd + vec2<f32>(dot(dr, vec4<f32>(1.0)), dot(dg, vec4<f32>(1.0)));
    sLd = sLd + vec2<f32>(dot(Lr, dr), dot(Lg, dg));
  }

  // Per-block refit off the moments (see bc5_fast_f16.wgsl for the
  // identities): Σρ = s·Σd − ΣL, ΣLρ = s·ΣLd − ΣL².
  let pR = scale * sd - sL;
  let pLR = scale * sLd - sLL;
  let sBB = sLL * (1.0 / 49.0);
  let sAB = sL * (1.0 / 7.0) - sBB;
  let sAA = vec2<f32>(16.0) - 2.0 * sL * (1.0 / 7.0) + sBB;
  let sBR = pLR * dir * (1.0 / 49.0);
  let sAR = (pR - pLR * (1.0 / 7.0)) * dir * (1.0 / 7.0);
  let spread = 16.0 * sLL != sL * sL;

  // Both channels at once, branch-free: nearest rounding of the LSQ
  // solve, accepted when it stays in 6-interp mode, moves, and prices
  // strictly better on the current indices. Endpoints clamp to [0,255],
  // NOT the block's value range: for a scalar channel, endpoints beyond
  // the data range are often genuinely optimal and there is no colour
  // axis to bend.
  let det = sAA * sBB - sAB * sAB;
  let idet = 1.0 / det;
  let q0f = floor(clamp(r0f + (sBB * sAR - sAB * sBR) * idet, vec2<f32>(0.0), vec2<f32>(255.0)) + 0.5);
  let q1f = floor(clamp(r1f + (sAA * sBR - sAB * sAR) * idet, vec2<f32>(0.0), vec2<f32>(255.0)) + 0.5);
  let dd0 = q0f - r0f;
  let dd1 = q1f - r1f;
  let eNew = -2.0 * (dd0 * sAR + dd1 * sBR) + dd0 * dd0 * sAA + 2.0 * dd0 * dd1 * sAB + dd1 * dd1 * sBB;
  let acc = spread & (abs(det) > vec2<f32>(1e-3)) & (q0f > q1f) & (eNew < vec2<f32>(0.0));
  let n0 = select(r0, vec2<u32>(q0f), acc);
  let n1 = select(r1, vec2<u32>(q1f), acc);

  // Pass 2, both channels — levels against the FINAL endpoints (rejected
  // channels re-derive their seed assignment), packed as Σ L·8^k:
  // iA = pixels 0..7, iB = pixels 8..15.
  let n0f = vec2<f32>(n0);
  let sc2 = vec2<f32>(7.0) / (vec2<f32>(n1) - n0f);
  var Lr: array<vec4<f32>, 4>;
  var Lg: array<vec4<f32>, 4>;
  for (var q: u32 = 0u; q < 4u; q = q + 1u) {
    Lr[q] = clamp(floor((vr[q] - n0f.x) * sc2.x + 0.5), vec4<f32>(0.0), vec4<f32>(7.0));
    Lg[q] = clamp(floor((vg[q] - n0f.y) * sc2.y + 0.5), vec4<f32>(0.0), vec4<f32>(7.0));
  }
  let iAx = lvl_to_idx(u32(dot(Lr[0], W0) + dot(Lr[1], W1)));
  let iBx = lvl_to_idx(u32(dot(Lr[2], W0) + dot(Lr[3], W1)));
  let iAy = lvl_to_idx(u32(dot(Lg[0], W0) + dot(Lg[1], W1)));
  let iBy = lvl_to_idx(u32(dot(Lg[2], W0) + dot(Lg[3], W1)));

  // BC5 block = R half (bytes 0..7) || G half (bytes 8..15) = 4 u32s.
  let o = bi * 4u;
  dst[o] = n0.x | (n1.x << 8u) | (iAx << 16u);
  dst[o + 1u] = (iAx >> 16u) | (iBx << 8u);
  dst[o + 2u] = n0.y | (n1.y << 8u) | (iAy << 16u);
  dst[o + 3u] = (iAy >> 16u) | (iBy << 8u);
}
