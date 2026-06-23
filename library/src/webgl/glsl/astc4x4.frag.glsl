#version 300 es
// ASTC 4×4 LDR fragment-shader encoder — WebGL2 port of astc4x4.wgsl (fast).
//
// One fragment per 4×4 block → 16-byte block as 4 × u32 in outColor. Restricted
// subset: single partition, no dual-plane, CEM 12 (LDR RGBA direct), 4×4 weight
// grid with 2-bit weights (QUANT_4), 8-bit endpoints (QUANT_256). Fast path:
// bbox seed → one LSQ refit fused into a projection weight assignment (4 colinear
// levels). Mirrors the `QUALITY_HIGH == 0` branch of astc4x4.wgsl; see that file
// for the 128-bit block layout.
//
// Determinism note: floor(x + 0.5) replaces WGSL round() for the refit endpoints
// (sub-LSB difference at exact .5 ties only).

precision highp float;
precision highp int;

uniform sampler2D uSrc;
uniform ivec2 uSrcSize;
uniform int uFlipY;

layout(location = 0) out uvec4 outColor;

ivec4 gPixels[16];
uint gIdx[16];

struct Fit { ivec4 e0; ivec4 e1; bool valid; };

ivec4 to8(vec4 v) {
  return ivec4(clamp(floor(v * 255.0 + 0.5), vec4(0.0), vec4(255.0)));
}

// Projection weight assignment over 4 levels (QUANT_4 ≈ thirds), with the LSQ
// normal-equation sums accumulated in the same pass for a fused refit.
Fit projAssign(ivec4 pe0, ivec4 pe1, bool fit) {
  Fit res;
  res.e0 = ivec4(0);
  res.e1 = ivec4(0);
  res.valid = false;
  ivec4 dir = pe1 - pe0;
  int dd = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + dir.w * dir.w;
  if (dd == 0) {
    for (int k = 0; k < 16; k++) { gIdx[k] = 0u; }
    return res;
  }
  float inv = 3.0 / float(dd);
  float sAA = 0.0, sBB = 0.0, sAB = 0.0;
  vec4 sAV = vec4(0.0), sBV = vec4(0.0);
  for (int k = 0; k < 16; k++) {
    ivec4 q = gPixels[k] - pe0;
    float proj = float(q.x * dir.x + q.y * dir.y + q.z * dir.z + q.w * dir.w) * inv;
    float s = clamp(floor(proj + 0.5), 0.0, 3.0);
    gIdx[k] = uint(s);
    if (fit) {
      vec4 v = vec4(gPixels[k]);
      float b = s / 3.0;
      float a = 1.0 - b;
      sAA += a * a; sBB += b * b; sAB += a * b; sAV += a * v; sBV += b * v;
    }
  }
  if (!fit) { return res; }
  float det = sAA * sBB - sAB * sAB;
  if (abs(det) < 1e-9) { return res; }
  res.e0 = ivec4(clamp(floor((sBB * sAV - sAB * sBV) / det + 0.5), vec4(0.0), vec4(255.0)));
  res.e1 = ivec4(clamp(floor((sAA * sBV - sAB * sAV) / det + 0.5), vec4(0.0), vec4(255.0)));
  res.valid = true;
  return res;
}

void writeBits(inout uint block[4], uint pos, uint nbits, uint value) {
  uint v = value & ((1u << nbits) - 1u);
  uint wordLo = pos / 32u;
  uint bitLo = pos % 32u;
  uint bitsInLo = min(nbits, 32u - bitLo);
  uint maskLo = ((1u << bitsInLo) - 1u) << bitLo;
  block[wordLo] = (block[wordLo] & ~maskLo) | ((v << bitLo) & maskLo);
  if (bitsInLo < nbits) {
    uint bitsInHi = nbits - bitsInLo;
    uint maskHi = (1u << bitsInHi) - 1u;
    uint valHi = v >> bitsInLo;
    block[wordLo + 1u] = (block[wordLo + 1u] & ~maskHi) | (valHi & maskHi);
  }
}

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 4;
  ivec2 maxXY = uSrcSize - ivec2(1);

  ivec4 lo = ivec4(255);
  ivec4 hi = ivec4(0);
  for (int i = 0; i < 16; i++) {
    ivec2 p = clamp(base + ivec2(i & 3, i >> 2), ivec2(0), maxXY);
    int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - p.y) : p.y;
    ivec4 px = to8(texelFetch(uSrc, ivec2(p.x, sy), 0));
    gPixels[i] = px;
    lo = min(lo, px);
    hi = max(hi, px);
  }

  ivec4 e0 = lo;
  ivec4 e1 = hi;
  Fit r = projAssign(e0, e1, true);
  if (r.valid) {
    e0 = r.e0;
    e1 = r.e1;
    projAssign(e0, e1, false);
  }

  // Endpoint ordering so the decoder doesn't apply blue contraction.
  if (e0.x + e0.y + e0.z > e1.x + e1.y + e1.z) {
    ivec4 t = e0; e0 = e1; e1 = t;
    for (int k = 0; k < 16; k++) { gIdx[k] = 3u - gIdx[k]; }
  }

  uint block[4];
  block[0] = 0u; block[1] = 0u; block[2] = 0u; block[3] = 0u;
  writeBits(block, 0u, 11u, 0x042u);
  writeBits(block, 11u, 2u, 0u);
  writeBits(block, 13u, 4u, 12u);
  writeBits(block, 17u + 0u * 8u, 8u, uint(e0.x));
  writeBits(block, 17u + 1u * 8u, 8u, uint(e1.x));
  writeBits(block, 17u + 2u * 8u, 8u, uint(e0.y));
  writeBits(block, 17u + 3u * 8u, 8u, uint(e1.y));
  writeBits(block, 17u + 4u * 8u, 8u, uint(e0.z));
  writeBits(block, 17u + 5u * 8u, 8u, uint(e1.z));
  writeBits(block, 17u + 6u * 8u, 8u, uint(e0.w));
  writeBits(block, 17u + 7u * 8u, 8u, uint(e1.w));

  uint w3 = 0u;
  for (int k = 0; k < 16; k++) {
    uint w = gIdx[k] & 0x3u;
    w3 = w3 | ((w & 1u) << (31u - 2u * uint(k))) | (((w >> 1u) & 1u) << (30u - 2u * uint(k)));
  }
  block[3] = w3;

  outColor = uvec4(block[0], block[1], block[2], block[3]);
}
