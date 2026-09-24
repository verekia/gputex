#version 300 es
// BC5 (RGTC2) fragment-shader encoder — WebGL2 port of bc5.wgsl.
//
// One fragment per 4×4 block → 16-byte BC5 block as 4 × u32 in outColor.
// BC5 = two BC4 halves (R then G), both channels processed together. Same
// algorithm and arithmetic order as bc5.wgsl (see bc5_fast_f16.wgsl for the
// full notes and measurements):
//   • texels held as quad-major vec4s per channel, in textureGather order
//     (x=(0,1) y=(1,1) z=(1,0) w=(0,0) within each 2×2 quad);
//   • seed endpoints at the per-channel extremes (spans ≤ 7: a 7-wide
//     window, lossless); pass-1 levels come from the seed range inset by
//     ~5.5/256 of the span (scale ×7.3125/7, offset 11/32), accumulating
//     the MOMENTS ΣL, ΣL², Σd, ΣL·d (d = v − r0; the seed covers the
//     data, so no clamp);
//   • one least-squares line through the pass-1 levels straight off the
//     moments (den = 16ΣL² − (ΣL)² = 0 keeps the seed);
//   • pass 2 derives the shipped levels against the refit endpoints,
//     packed as float fields Σ L·8^k (exact below 2^24), then one SWAR
//     level → BC4 index map (0→0, 7→1, L→L+1) per 24-bit word;
//   • offset round: both endpoints shift by the rounded mean residual of
//     the shipped levels (never worse on those indices).
// Always emits 6-interpolation mode (red0 > red1).

precision highp float;
precision highp int;

uniform sampler2D uSrc;
uniform ivec2 uSrcSize;
uniform int uFlipY;

layout(location = 0) out uvec4 outColor;

// 8 packed 3-bit levels → BC4 indices (0→0, 7→1, L→L+1 otherwise).
uint lvlToIdx(uint x) {
  uint y = ((x & 0x6DB6DBu) + 0x249249u) ^ (x & 0x924924u);
  return y ^ (~((y >> 1u) | (y >> 2u)) & 0x249249u);
}

// Per-quad pixel weights 8^k (gather order x,y,z,w = (0,1),(1,1),(1,0),(0,0)).
const vec4 W0 = vec4(4096.0, 32768.0, 8.0, 1.0);
const vec4 W1 = vec4(262144.0, 2097152.0, 512.0, 64.0);

vec4 fetchRG(ivec2 p, ivec2 maxXY) {
  ivec2 c = clamp(p, ivec2(0), maxXY);
  int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - c.y) : c.y;
  return texelFetch(uSrc, ivec2(c.x, sy), 0);
}

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 4;
  ivec2 maxXY = uSrcSize - ivec2(1);

  vec4 vr[4];
  vec4 vg[4];
  for (int q = 0; q < 4; q++) {
    ivec2 qo = base + ivec2((q & 1) * 2, (q >> 1) * 2);
    vec4 cx = fetchRG(qo + ivec2(0, 1), maxXY);
    vec4 cy = fetchRG(qo + ivec2(1, 1), maxXY);
    vec4 cz = fetchRG(qo + ivec2(1, 0), maxXY);
    vec4 cw = fetchRG(qo, maxXY);
    vr[q] = vec4(cx.r, cy.r, cz.r, cw.r) * 255.0;
    vg[q] = vec4(cx.g, cy.g, cz.g, cw.g) * 255.0;
  }
  vec4 mnr = min(min(vr[0], vr[1]), min(vr[2], vr[3]));
  vec4 mxr = max(max(vr[0], vr[1]), max(vr[2], vr[3]));
  vec4 mng = min(min(vg[0], vg[1]), min(vg[2], vg[3]));
  vec4 mxg = max(max(vg[0], vg[1]), max(vg[2], vg[3]));
  vec2 vmin = vec2(min(min(mnr.x, mnr.y), min(mnr.z, mnr.w)), min(min(mng.x, mng.y), min(mng.z, mng.w)));
  vec2 vmax = vec2(max(max(mxr.x, mxr.y), max(mxr.z, mxr.w)), max(max(mxg.x, mxg.y), max(mxg.z, mxg.w)));

  // Seed endpoints at the per-channel extremes (round-to-nearest); spans
  // ≤ 7 (incl. flat blocks) seed a 7-wide window instead — lossless.
  vec2 vhi = clamp(floor(vmax + 0.5), vec2(0.0), vec2(255.0));
  vec2 vlo = clamp(floor(vmin + 0.5), vec2(0.0), vec2(255.0));
  bvec2 small = lessThanEqual(vhi - vlo, vec2(7.0));
  vec2 r1f = vec2(small.x ? min(vlo.x, 248.0) : vlo.x, small.y ? min(vlo.y, 248.0) : vlo.y);
  vec2 r0f = vec2(small.x ? r1f.x + 7.0 : vhi.x, small.y ? r1f.y + 7.0 : vhi.y);
  // Pass-1 levels from the seed range inset by ~5.5/256 of the span.
  vec2 scale = vec2(7.3125) / (r1f - r0f);

  // Pass 1 — moments only.
  vec2 sL = vec2(0.0);
  vec2 sLL = vec2(0.0);
  vec2 sd = vec2(0.0);
  vec2 sLd = vec2(0.0);
  for (int q = 0; q < 4; q++) {
    vec4 dr = vr[q] - r0f.x;
    vec4 dg = vg[q] - r0f.y;
    vec4 Lr = floor(dr * scale.x + 0.34375);
    vec4 Lg = floor(dg * scale.y + 0.34375);
    sL += vec2(dot(Lr, vec4(1.0)), dot(Lg, vec4(1.0)));
    sLL += vec2(dot(Lr, Lr), dot(Lg, Lg));
    sd += vec2(dot(dr, vec4(1.0)), dot(dg, vec4(1.0)));
    sLd += vec2(dot(Lr, dr), dot(Lg, dg));
  }

  // Refit: least-squares line v ≈ r0 + α + β·L through the pass-1 levels.
  // Endpoints clamp to [0,255], NOT the block's value range.
  vec2 den = 16.0 * sLL - sL * sL;
  vec2 beta = (16.0 * sLd - sL * sd) / den;
  vec2 e0 = r0f + (sd - beta * sL) * (1.0 / 16.0);
  vec2 q0f = floor(clamp(e0, vec2(0.0), vec2(255.0)) + 0.5);
  vec2 q1f = floor(clamp(e0 + 7.0 * beta, vec2(0.0), vec2(255.0)) + 0.5);
  bvec2 acc = bvec2(den.x > 0.0 && q0f.x > q1f.x, den.y > 0.0 && q0f.y > q1f.y);
  vec2 n0f = vec2(acc.x ? q0f.x : r0f.x, acc.y ? q0f.y : r0f.y);
  vec2 n1f = vec2(acc.x ? q1f.x : r1f.x, acc.y ? q1f.y : r1f.y);

  // Pass 2 — levels against the refit endpoints, packed as Σ L·8^k
  // (A = pixels 0..7, B = pixels 8..15), plus ΣL for the offset round.
  vec2 sc2 = vec2(7.0) / (n1f - n0f);
  vec4 pk = vec4(0.0); // (Ax, Bx, Ay, By)
  vec2 sL2 = vec2(0.0);
  for (int q = 0; q < 4; q++) {
    vec4 Lr = clamp(floor((vr[q] - n0f.x) * sc2.x + 0.5), vec4(0.0), vec4(7.0));
    vec4 Lg = clamp(floor((vg[q] - n0f.y) * sc2.y + 0.5), vec4(0.0), vec4(7.0));
    vec4 w = (q & 1) == 1 ? W1 : W0;
    bool hi = q >= 2;
    float pr = dot(Lr, w);
    float pg = dot(Lg, w);
    pk += vec4(hi ? 0.0 : pr, hi ? pr : 0.0, hi ? 0.0 : pg, hi ? pg : 0.0);
    sL2 += vec2(dot(Lr, vec4(1.0)), dot(Lg, vec4(1.0)));
  }

  // Offset round: shift both endpoints by the rounded mean residual of the
  // shipped levels.
  vec2 res = sd + 16.0 * (r0f - n0f) - (n1f - n0f) * sL2 * (1.0 / 7.0);
  vec2 sh = clamp(floor(res * (1.0 / 16.0) + 0.5), -n1f, vec2(255.0) - n0f);
  uvec2 m0 = uvec2(n0f + sh);
  uvec2 m1 = uvec2(n1f + sh);
  uint iAx = lvlToIdx(uint(pk.x));
  uint iBx = lvlToIdx(uint(pk.y));
  uint iAy = lvlToIdx(uint(pk.z));
  uint iBy = lvlToIdx(uint(pk.w));

  // BC5 block = R half (bytes 0..7) || G half (bytes 8..15) = 4 u32s.
  outColor = uvec4(
    m0.x | (m1.x << 8u) | (iAx << 16u),
    (iAx >> 16u) | (iBx << 8u),
    m0.y | (m1.y << 8u) | (iAy << 16u),
    (iAy >> 16u) | (iBy << 8u)
  );
}
