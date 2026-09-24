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
//   • BOTH channels ride the same fused passes — each loop computes
//     projections and moments for R and G at once instead of two scalar
//     encode_bc4 calls.
//   • The 16 texel reads are 8 textureGather fetches (4 quads × R,G) for
//     interior blocks — byte-identical output to per-texel loads, −3.6%
//     GPU on 4096² (/ab, 2026-07). Blocks straddling the source edge of a
//     non-multiple-of-4 image use clamped per-texel loads instead: the
//     upload pads the texture with ZEROS, so a normalised-coordinate
//     gather there would read padding (or mis-scale against the padded
//     size) instead of replicating the last real texel.
//   • SEED: endpoints at the per-channel extremes, but pass-1 levels are
//     assigned against that range INSET by ~5.5/256 of the span on both
//     ends (level scale ×7.3125/7, offset ½ − 0.15625 = 11/32 — exact dyadic
//     constants so every backend folds them identically), so each extreme level
//     gathers the pixels NEAR the extremes instead of only the extreme
//     pixel itself — the refit then lands much closer to the optimum
//     (an exhaustive search over all endpoint pairs showed the plain
//     bbox seed leaving 0.7–6 dB on the table). Swept 0..20/256; 5.5 wins
//     under both the /7 spec decode and Apple's hardware decode (below),
//     and per-block adaptive insets (variance, extreme gaps) all lost.
//     Spans ≤ 7 (incl. flat blocks) seed a 7-wide window instead, whose
//     levels land on every integer the block holds: lossless.
//   • Pass 1 accumulates MOMENTS ΣL, ΣL², Σd, ΣL·d per channel (d = v − r0,
//     exact integers). ΣL ≤ 112 and ΣL² ≤ 784 are exact f16 integers; ΣL·d
//     (≤ 28560) and Σd accumulate in f32 so they stay exact too. The seed
//     covers the data, so t ∈ [0,7.3125] and L = floor(t + 11/32) ∈ [0,7] needs
//     no clamp.
//   • REFIT = the least-squares line v ≈ r0 + α + β·L through those levels,
//     straight off the moments: β = (16ΣLd − ΣL·Σd)/(16ΣL² − (ΣL)²), α =
//     (Σd − βΣL)/16 — ~12 ops per channel. den = 0 ⟺ every pixel on one
//     level (exact integer test) keeps the seed. Accepted whenever it stays
//     in 6-interp mode: on the inset partition, pricing it against the seed
//     (the previous E(δ) closed form) changed nothing, and dropping that
//     pricing is what pays for the offset round below.
//   • Pass 2 derives the shipped levels ONCE, against the refit endpoints —
//     full reprojection quality — as a LOOP over quads (the unrolled form
//     with all eight level vectors live measured ~6% slower once ΣL was
//     added).
//   • OFFSET ROUND: both endpoints shift by round(mean residual) of the
//     shipped levels (Σv is exact from pass 1, so only ΣL is new). A
//     whole-level shift moves every palette entry equally, so the error on
//     these indices can only drop, under ANY decoder's weights. Buys half
//     of a full second refit round (+0.05 dB) for ~1/4 of its cost; the
//     full round (ΣL², ΣL·v in pass 2 + a second solve) measured +0.1 dB
//     more but +18% GPU at 1K — rejected.
//   • Apple GPUs (M3 measured) decode BC4/BC5 with BC7-style 6-bit weights
//     (0,9,18,27,37,46,55,64)/64, not exact sevenths — up to ±0.0067·span
//     off the spec palette. The encoder targets the spec (/7) palette;
//     /eval's hardware-decoded PSNR sits ~0.07 dB under the CPU-decoded one.
//   • Texels live as quad-major vec4<f16> per channel (the gather layout),
//     so min/max reduce as vectors and both passes run 4-wide; edge blocks
//     load into the same layout.
//   • 3-bit indices are packed as FLOAT: each group of 8 pixels'
//     levels accumulates as Σ L·8^k in f32 (≤ 2^24 − 1, exact) — one fma
//     per texel — and the level → BC4 index map (0→0, 7→1, L→L+1) is
//     applied to the whole 24-bit word with SWAR bit tricks.
//   The moment/float-packing structure measured −22% GPU vs the kernel
//   before it (rg8 source; 2K/4K normal maps sit at the read floor on rgba8
//   either way) — the 1K case is ALU-bound, not read-bound (/eval 2026-09).
//   The inset seed + regression refit + offset round then lowered MSE by
//   3.8–9% on every real texture (normals 3.8–6.4%, displacement 9–60%,
//   hardware decode) at equal GPU time (/eval 2026-09-24).
//
// The host selects this module only when the device reports shader-f16,
// falling back to bc5.wgsl otherwise.
enable f16;
alias h = f16;
alias h4 = vec4<f16>;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, y0: u32, };
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
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);

  // Load 4×4 R and G as quad-major vec4s in gather order: component
  // w=(0,0) z=(1,0) x=(0,1) y=(1,1) of quad q = (x ≥ 2) + 2·(y ≥ 2).
  // INTERIOR blocks — every block when the source is a multiple of 4, so
  // the branch is wavefront-uniform on benchmark-shaped content — read via
  // 8 gathers; the gather point (base+quad+1) normalised by the PHYSICAL
  // (padded) texture size sits exactly between the quad's texel centers,
  // and interior quads never touch the zero-initialised padding strip.
  // Blocks straddling the source edge of a non-multiple-of-4 image fall
  // back to per-texel loads clamped to the last real texel (gather cannot
  // replicate an edge texel mid-quad). ×255 then f16 lands every value on
  // an exact integer.
  var vr: array<h4, 4>;
  var vg: array<h4, 4>;
  if (u32(base.x) + 4u <= params.width && u32(base.y) + 4u <= params.height) {
    let inv_size = vec2<f32>(1.0, 1.0) / vec2<f32>(textureDimensions(src_tex));
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
      let qo = vec2<u32>((q & 1u) * 2u, (q >> 1u) * 2u);
      let cc = (vec2<f32>(base) + vec2<f32>(qo) + vec2<f32>(1.0, 1.0)) * inv_size;
      vr[q] = h4(textureGather(0, src_tex, smp, cc) * 255.0);
      vg[q] = h4(textureGather(1, src_tex, smp, cc) * 255.0);
    }
  } else {
    let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
      let qo = base + vec2<i32>(i32(q & 1u) * 2, i32(q >> 1u) * 2);
      let cx = textureLoad(src_tex, clamp(qo + vec2<i32>(0, 1), vec2<i32>(0), mx), 0);
      let cy = textureLoad(src_tex, clamp(qo + vec2<i32>(1, 1), vec2<i32>(0), mx), 0);
      let cz = textureLoad(src_tex, clamp(qo + vec2<i32>(1, 0), vec2<i32>(0), mx), 0);
      let cw = textureLoad(src_tex, clamp(qo, vec2<i32>(0), mx), 0);
      vr[q] = h4(vec4<f32>(cx.r, cy.r, cz.r, cw.r) * 255.0);
      vg[q] = h4(vec4<f32>(cx.g, cy.g, cz.g, cw.g) * 255.0);
    }
  }
  let mnr = min(min(vr[0], vr[1]), min(vr[2], vr[3]));
  let mxr = max(max(vr[0], vr[1]), max(vr[2], vr[3]));
  let mng = min(min(vg[0], vg[1]), min(vg[2], vg[3]));
  let mxg = max(max(vg[0], vg[1]), max(vg[2], vg[3]));
  let vmin = vec2<h>(min(min(mnr.x, mnr.y), min(mnr.z, mnr.w)), min(min(mng.x, mng.y), min(mng.z, mng.w)));
  let vmax = vec2<h>(max(max(mxr.x, mxr.y), max(mxr.z, mxr.w)), max(max(mxg.x, mxg.y), max(mxg.z, mxg.w)));

  // Seed endpoints at the exact per-channel extremes; spans ≤ 7 (incl.
  // flat blocks) seed a 7-wide window instead, whose levels land on every
  // integer the block holds — lossless, and the refit keeps it.
  let small = vmax - vmin <= vec2<h>(7.0);
  let r1 = vec2<u32>(select(vmin, min(vmin, vec2<h>(248.0)), small));
  let r0 = select(vec2<u32>(vmax), r1 + 7u, small);

  let r0h = vec2<h>(vec2<f32>(r0));
  let dirf = vec2<f32>(r1) - vec2<f32>(r0);
  // Pass-1 levels come from the seed range INSET by ~5.5/256 of the span
  // on both ends (see header).
  let scale = vec2<h>(vec2<f32>(7.3125) / dirf);

  // Pass 1 — t = d·scale ∈ [0,7.3125] by construction (seed covers the
  // data), so L = floor(t + 11/32) ∈ [0,7] needs no clamp.
  var sLr = h(0.0); var sLLr = h(0.0); var sdr = 0.0; var sLdr = 0.0;
  var sLg = h(0.0); var sLLg = h(0.0); var sdg = 0.0; var sLdg = 0.0;
  for (var q: u32 = 0u; q < 4u; q = q + 1u) {
    let dr = vr[q] - r0h.x;
    let dg = vg[q] - r0h.y;
    let Lr = floor(dr * scale.x + h(0.34375));
    let Lg = floor(dg * scale.y + h(0.34375));
    sLr = sLr + dot(Lr, h4(1.0)); sLLr = sLLr + dot(Lr, Lr);
    sLg = sLg + dot(Lg, h4(1.0)); sLLg = sLLg + dot(Lg, Lg);
    sdr = sdr + f32(dot(dr, h4(1.0)));
    sdg = sdg + f32(dot(dg, h4(1.0)));
    sLdr = sLdr + dot(vec4<f32>(Lr), vec4<f32>(dr));
    sLdg = sLdg + dot(vec4<f32>(Lg), vec4<f32>(dg));
  }

  // Per-block refit in f32: least-squares line v ≈ r0 + α + β·L through
  // the pass-1 levels, straight off the (exact-integer) moments. den = 0
  // ⟺ every pixel on one level (rank-deficient) — keep the seed then.
  let sLf = vec2<f32>(f32(sLr), f32(sLg));
  let sLLf = vec2<f32>(f32(sLLr), f32(sLLg));
  let sdf = vec2<f32>(sdr, sdg);
  let den = 16.0 * sLLf - sLf * sLf;
  let beta = (16.0 * vec2<f32>(sLdr, sLdg) - sLf * sdf) / den;
  let e0 = vec2<f32>(r0) + (sdf - beta * sLf) * (1.0 / 16.0);
  let q0f = floor(clamp(e0, vec2<f32>(0.0), vec2<f32>(255.0)) + 0.5);
  let q1f = floor(clamp(e0 + 7.0 * beta, vec2<f32>(0.0), vec2<f32>(255.0)) + 0.5);
  let acc = (den > vec2<f32>(0.0)) & (q0f > q1f);
  let n0 = select(r0, vec2<u32>(q0f), acc);
  let n1 = select(r1, vec2<u32>(q1f), acc);

  // Pass 2 — levels against the FINAL endpoints (rejected channels
  // re-derive their seed assignment), accumulated as 3-bit fields in f32
  // (Σ L·8^k ≤ 2^24 − 1, exact): iA = pixels 0..7, iB = pixels 8..15.
  let n0h = vec2<h>(vec2<f32>(n0));
  let sc2 = vec2<h>(vec2<f32>(7.0) / (vec2<f32>(n1) - vec2<f32>(n0)));
  var pk = vec4<f32>(0.0);   // (Ax, Bx, Ay, By) level words
  var sLq = vec2<h>(0.0);
  for (var q: u32 = 0u; q < 4u; q = q + 1u) {
    let Lr = clamp(floor((vr[q] - n0h.x) * sc2.x + h(0.5)), h4(0.0), h4(7.0));
    let Lg = clamp(floor((vg[q] - n0h.y) * sc2.y + h(0.5)), h4(0.0), h4(7.0));
    let w = select(W0, W1, (q & 1u) == 1u);
    let hi = q >= 2u;
    let pr = dot(vec4<f32>(Lr), w);
    let pg = dot(vec4<f32>(Lg), w);
    pk = pk + vec4<f32>(select(pr, 0.0, hi), select(0.0, pr, hi), select(pg, 0.0, hi), select(0.0, pg, hi));
    sLq = sLq + vec2<h>(dot(Lr, h4(1.0)), dot(Lg, h4(1.0)));
  }
  let n0f = vec2<f32>(n0);
  let sL2 = vec2<f32>(sLq);
  let res = sdf + 16.0 * (vec2<f32>(r0) - n0f) - (vec2<f32>(n1) - n0f) * sL2 * (1.0 / 7.0);
  let sh = clamp(floor(res * (1.0 / 16.0) + 0.5), -vec2<f32>(n1), vec2<f32>(255.0) - n0f);
  let m0 = vec2<u32>(n0f + sh);
  let m1 = vec2<u32>(vec2<f32>(n1) + sh);
  let iAx = lvl_to_idx(u32(pk.x));
  let iBx = lvl_to_idx(u32(pk.y));
  let iAy = lvl_to_idx(u32(pk.z));
  let iBy = lvl_to_idx(u32(pk.w));

  // BC5 block = R half (bytes 0..7) || G half (bytes 8..15) = 4 u32s.
  let o = bi * 4u;
  dst[o] = m0.x | (m1.x << 8u) | (iAx << 16u);
  dst[o + 1u] = (iAx >> 16u) | (iBx << 8u);
  dst[o + 2u] = m0.y | (m1.y << 8u) | (iAy << 16u);
  dst[o + 3u] = (iAy >> 16u) | (iBy << 8u);
}
