#version 300 es
// BC4 (RGTC1) fragment-shader encoder — WebGL2 port of bc4.wgsl.
//
// One fragment per 4×4 block → 8-byte BC4 block as 2 × u32 in outColor.rg
// (outColor.ba unused), from the source's R channel. Same algorithm and
// arithmetic order as bc4.wgsl — the single-channel half of bc5.frag.glsl
// (see bc5_fast_f16.wgsl for the full notes and measurements):
//   • texels held as quad-major vec4s in textureGather order
//     (x=(0,1) y=(1,1) z=(1,0) w=(0,0) within each 2×2 quad);
//   • seed endpoints at the extremes (spans ≤ 7: a 7-wide window,
//     lossless); pass-1 levels come from the seed range inset by ~5.5/256
//     of the span, accumulating the moments ΣL, ΣL², Σd, ΣL·d;
//   • one least-squares line through the pass-1 levels (den = 0 keeps the
//     seed);
//   • pass 2 derives the shipped levels against the refit endpoints,
//     packed as float fields Σ L·8^k, then one SWAR level → BC4 index map;
//   • offset round: both endpoints shift by the rounded mean residual.
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

float fetchR(ivec2 p, ivec2 maxXY) {
  ivec2 c = clamp(p, ivec2(0), maxXY);
  int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - c.y) : c.y;
  return texelFetch(uSrc, ivec2(c.x, sy), 0).r;
}

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 4;
  ivec2 maxXY = uSrcSize - ivec2(1);

  vec4 v[4];
  for (int q = 0; q < 4; q++) {
    ivec2 qo = base + ivec2((q & 1) * 2, (q >> 1) * 2);
    v[q] = vec4(
      fetchR(qo + ivec2(0, 1), maxXY),
      fetchR(qo + ivec2(1, 1), maxXY),
      fetchR(qo + ivec2(1, 0), maxXY),
      fetchR(qo, maxXY)
    ) * 255.0;
  }
  vec4 mn = min(min(v[0], v[1]), min(v[2], v[3]));
  vec4 mxv = max(max(v[0], v[1]), max(v[2], v[3]));
  float vmin = min(min(mn.x, mn.y), min(mn.z, mn.w));
  float vmax = max(max(mxv.x, mxv.y), max(mxv.z, mxv.w));

  // Seed endpoints at the extremes (round-to-nearest); spans ≤ 7 (incl.
  // flat blocks) seed a 7-wide window instead — lossless.
  float vhi = clamp(floor(vmax + 0.5), 0.0, 255.0);
  float vlo = clamp(floor(vmin + 0.5), 0.0, 255.0);
  bool small = vhi - vlo <= 7.0;
  float r1f = small ? min(vlo, 248.0) : vlo;
  float r0f = small ? r1f + 7.0 : vhi;
  // Pass-1 levels from the seed range inset by ~5.5/256 of the span.
  float scale = 7.3125 / (r1f - r0f);

  // Pass 1 — moments only.
  float sL = 0.0;
  float sLL = 0.0;
  float sd = 0.0;
  float sLd = 0.0;
  for (int q = 0; q < 4; q++) {
    vec4 d = v[q] - r0f;
    vec4 L = floor(d * scale + 0.34375);
    sL += dot(L, vec4(1.0));
    sLL += dot(L, L);
    sd += dot(d, vec4(1.0));
    sLd += dot(L, d);
  }

  // Refit: least-squares line v ≈ r0 + α + β·L through the pass-1 levels.
  // Endpoints clamp to [0,255], NOT the block's value range.
  float den = 16.0 * sLL - sL * sL;
  float beta = (16.0 * sLd - sL * sd) / den;
  float e0 = r0f + (sd - beta * sL) * (1.0 / 16.0);
  float q0f = floor(clamp(e0, 0.0, 255.0) + 0.5);
  float q1f = floor(clamp(e0 + 7.0 * beta, 0.0, 255.0) + 0.5);
  bool acc = den > 0.0 && q0f > q1f;
  float n0f = acc ? q0f : r0f;
  float n1f = acc ? q1f : r1f;

  // Pass 2 — levels against the refit endpoints, packed as Σ L·8^k
  // (A = pixels 0..7, B = pixels 8..15), plus ΣL for the offset round.
  float sc2 = 7.0 / (n1f - n0f);
  vec2 pk = vec2(0.0);
  float sL2 = 0.0;
  for (int q = 0; q < 4; q++) {
    vec4 L = clamp(floor((v[q] - n0f) * sc2 + 0.5), vec4(0.0), vec4(7.0));
    vec4 w = (q & 1) == 1 ? W1 : W0;
    bool hi = q >= 2;
    float p = dot(L, w);
    pk += vec2(hi ? 0.0 : p, hi ? p : 0.0);
    sL2 += dot(L, vec4(1.0));
  }

  // Offset round: shift both endpoints by the rounded mean residual of the
  // shipped levels.
  float res = sd + 16.0 * (r0f - n0f) - (n1f - n0f) * sL2 * (1.0 / 7.0);
  float sh = clamp(floor(res * (1.0 / 16.0) + 0.5), -n1f, 255.0 - n0f);
  uint m0 = uint(n0f + sh);
  uint m1 = uint(n1f + sh);
  uint iA = lvlToIdx(uint(pk.x));
  uint iB = lvlToIdx(uint(pk.y));

  // BC4 block = 2 endpoint bytes + 48 index bits = 2 u32s.
  outColor = uvec4(m0 | (m1 << 8u) | (iA << 16u), (iA >> 16u) | (iB << 8u), 0u, 0u);
}
