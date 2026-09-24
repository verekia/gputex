#version 300 es
// BC5 (RGTC2) fragment-shader encoder — WebGL2 port of bc5.wgsl.
//
// One fragment per 4×4 block → 16-byte BC5 block as 4 × u32 in outColor.
// BC5 = two BC4 halves (R then G), both channels processed together. Same
// algorithm and arithmetic order as bc5.wgsl (see bc5_fast_f16.wgsl for the
// full notes and measurements):
//   • texels held as quad-major vec4s per channel, in textureGather order
//     (x=(0,1) y=(1,1) z=(1,0) w=(0,0) within each 2×2 quad);
//   • seed endpoints at the per-channel extremes; pass 1 accumulates the
//     MOMENTS ΣL, ΣL², Σd, ΣL·d (d = v − r0, L = the seed level — the seed
//     covers the data, so no clamp), from which every least-squares sum is
//     O(1) per block; exact rank guard 16·ΣL² == (ΣL)²;
//   • one closed-form refit accepted when it prices better on the seed
//     levels (nearest rounding of the solve, 6-interp mode kept);
//   • pass 2 derives the shipped levels against the FINAL endpoints, packed
//     as float fields Σ L·8^k (exact below 2^24), then one SWAR level → BC4
//     index map (0→0, 7→1, L→L+1) per 24-bit word.
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

// Closed-form accept-if-better refit for one channel off the pass-1
// moments; returns the final (r0, r1).
uvec2 refit(float sL, float sLL, float sd, float sLd, uint r0, uint r1) {
  float r0f = float(r0);
  float r1f = float(r1);
  float dir = r1f - r0f;
  float scale = 7.0 / dir;
  // Σρ = s·Σd − ΣL, ΣLρ = s·ΣLd − ΣL² (ρ = level-space residual).
  float pR = scale * sd - sL;
  float pLR = scale * sLd - sLL;
  float sBB = sLL * (1.0 / 49.0);
  float sAB = sL * (1.0 / 7.0) - sBB;
  float sAA = 16.0 - 2.0 * sL * (1.0 / 7.0) + sBB;
  float sBR = pLR * dir * (1.0 / 49.0);
  float sAR = (pR - pLR * (1.0 / 7.0)) * dir * (1.0 / 7.0);
  bool spread = 16.0 * sLL != sL * sL;
  float det = sAA * sBB - sAB * sAB;
  float idet = 1.0 / det;
  // Endpoints clamp to [0,255], NOT the block's value range: for a scalar
  // channel, endpoints beyond the data range are often genuinely optimal.
  float q0f = floor(clamp(r0f + (sBB * sAR - sAB * sBR) * idet, 0.0, 255.0) + 0.5);
  float q1f = floor(clamp(r1f + (sAA * sBR - sAB * sAR) * idet, 0.0, 255.0) + 0.5);
  float dd0 = q0f - r0f;
  float dd1 = q1f - r1f;
  float eNew = -2.0 * (dd0 * sAR + dd1 * sBR) + dd0 * dd0 * sAA + 2.0 * dd0 * dd1 * sAB + dd1 * dd1 * sBB;
  bool acc = spread && abs(det) > 1e-3 && q0f > q1f && eNew < 0.0;
  return acc ? uvec2(uint(q0f), uint(q1f)) : uvec2(r0, r1);
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

  // Seed endpoints at the exact per-channel extremes (round-to-nearest).
  // Flat blocks get nudged apart to keep the 6-interp mode (r0 > r1).
  uvec2 r0 = uvec2(clamp(floor(vmax + 0.5), vec2(0.0), vec2(255.0)));
  uvec2 r1 = uvec2(clamp(floor(vmin + 0.5), vec2(0.0), vec2(255.0)));
  if (r0.x == r1.x) { if (r1.x > 0u) { r1.x = r1.x - 1u; } else { r0.x = r0.x + 1u; } }
  if (r0.y == r1.y) { if (r1.y > 0u) { r1.y = r1.y - 1u; } else { r0.y = r0.y + 1u; } }

  vec2 r0f = vec2(r0);
  vec2 scale = vec2(7.0) / (vec2(r1) - r0f);

  // Pass 1 — moments only.
  vec2 sL = vec2(0.0);
  vec2 sLL = vec2(0.0);
  vec2 sd = vec2(0.0);
  vec2 sLd = vec2(0.0);
  for (int q = 0; q < 4; q++) {
    vec4 dr = vr[q] - r0f.x;
    vec4 dg = vg[q] - r0f.y;
    vec4 Lr = floor(dr * scale.x + 0.5);
    vec4 Lg = floor(dg * scale.y + 0.5);
    sL += vec2(dot(Lr, vec4(1.0)), dot(Lg, vec4(1.0)));
    sLL += vec2(dot(Lr, Lr), dot(Lg, Lg));
    sd += vec2(dot(dr, vec4(1.0)), dot(dg, vec4(1.0)));
    sLd += vec2(dot(Lr, dr), dot(Lg, dg));
  }

  uvec2 fr = refit(sL.x, sLL.x, sd.x, sLd.x, r0.x, r1.x);
  uvec2 fg = refit(sL.y, sLL.y, sd.y, sLd.y, r0.y, r1.y);
  uvec2 n0 = uvec2(fr.x, fg.x);
  uvec2 n1 = uvec2(fr.y, fg.y);

  // Pass 2 — levels against the FINAL endpoints, packed as Σ L·8^k:
  // iA = pixels 0..7, iB = pixels 8..15.
  vec2 n0f = vec2(n0);
  vec2 sc2 = vec2(7.0) / (vec2(n1) - n0f);
  vec4 Lr[4];
  vec4 Lg[4];
  for (int q = 0; q < 4; q++) {
    Lr[q] = clamp(floor((vr[q] - n0f.x) * sc2.x + 0.5), vec4(0.0), vec4(7.0));
    Lg[q] = clamp(floor((vg[q] - n0f.y) * sc2.y + 0.5), vec4(0.0), vec4(7.0));
  }
  uint iAx = lvlToIdx(uint(dot(Lr[0], W0) + dot(Lr[1], W1)));
  uint iBx = lvlToIdx(uint(dot(Lr[2], W0) + dot(Lr[3], W1)));
  uint iAy = lvlToIdx(uint(dot(Lg[0], W0) + dot(Lg[1], W1)));
  uint iBy = lvlToIdx(uint(dot(Lg[2], W0) + dot(Lg[3], W1)));

  // BC5 block = R half (bytes 0..7) || G half (bytes 8..15) = 4 u32s.
  outColor = uvec4(
    n0.x | (n1.x << 8u) | (iAx << 16u),
    (iAx >> 16u) | (iBx << 8u),
    n0.y | (n1.y << 8u) | (iAy << 16u),
    (iAy >> 16u) | (iBy << 8u)
  );
}
