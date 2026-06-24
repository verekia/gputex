#version 300 es
// BC5 (RGTC2) fragment-shader encoder — WebGL2 port of bc5.wgsl (fast path).
//
// One fragment per 4×4 block → 16-byte BC5 block as 4 × u32 in outColor.
// BC5 = two BC4 halves (R then G). This is the *fast* path only: bbox
// endpoints + a single full-L2 index assignment per channel, no LSQ refit
// (the WGSL `QUALITY_HIGH` branch). Always emits 6-interpolation mode
// (red0 > red1). See bc5.wgsl for the full derivation.

precision highp float;
precision highp int;

uniform sampler2D uSrc;
uniform ivec2 uSrcSize;
uniform int uFlipY;

layout(location = 0) out uvec4 outColor;

// 6-interpolation-mode palette weights: pal[j] = W0_6[j]*r0 + W1_6[j]*r1.
const float W0_6[8] = float[8](1.0, 0.0, 6.0 / 7.0, 5.0 / 7.0, 4.0 / 7.0, 3.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0);
const float W1_6[8] = float[8](0.0, 1.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0);

uint quantize8(float v) {
  return uint(clamp(floor(v * 255.0 + 0.5), 0.0, 255.0));
}

// Encode 16 single-channel values into an 8-byte BC4 block (two little-endian
// u32s). Mirrors encode_bc4() in bc5.wgsl with the refit pass omitted.
uvec2 encodeBC4(float values[16]) {
  float vmin = 1.0;
  float vmax = 0.0;
  for (int k = 0; k < 16; k++) {
    vmin = min(vmin, values[k]);
    vmax = max(vmax, values[k]);
  }
  uint r0 = quantize8(vmax);
  uint r1 = quantize8(vmin);
  if (r0 == r1) {
    if (r1 > 0u) { r1 = r1 - 1u; } else { r0 = r0 + 1u; }
  }

  float pal[8];
  float r0f = float(r0) / 255.0;
  float r1f = float(r1) / 255.0;
  for (int j = 0; j < 8; j++) {
    pal[j] = W0_6[j] * r0f + W1_6[j] * r1f;
  }

  uint indices[16];
  for (int k = 0; k < 16; k++) {
    float v = values[k];
    uint bestJ = 0u;
    float bestD = 1e20;
    for (int j = 0; j < 8; j++) {
      float d = pal[j] - v;
      float d2 = d * d;
      if (d2 < bestD) { bestD = d2; bestJ = uint(j); }
    }
    indices[k] = bestJ;
  }

  // Pack the 48-bit index field (bytes 2..7) split across two u32 halves.
  uint idxLo = 0u;
  uint idxHi = 0u;
  for (int k = 0; k < 16; k++) {
    uint bit = 3u * uint(k);
    uint v = indices[k] & 7u;
    if (bit + 3u <= 32u) {
      idxLo = idxLo | (v << bit);
    } else if (bit >= 32u) {
      idxHi = idxHi | (v << (bit - 32u));
    } else {
      idxLo = idxLo | (v << bit);
      idxHi = idxHi | (v >> (32u - bit));
    }
  }

  uint outLo = r0 | (r1 << 8) | ((idxLo & 0xFFFFu) << 16);
  uint outHi = (idxLo >> 16) | (idxHi << 16);
  return uvec2(outLo, outHi);
}

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 4;
  ivec2 maxXY = uSrcSize - ivec2(1);

  float rValues[16];
  float gValues[16];
  for (int i = 0; i < 16; i++) {
    ivec2 p = clamp(base + ivec2(i & 3, i >> 2), ivec2(0), maxXY);
    int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - p.y) : p.y;
    vec4 c = texelFetch(uSrc, ivec2(p.x, sy), 0);
    rValues[i] = c.r;
    gValues[i] = c.g;
  }

  uvec2 rBlock = encodeBC4(rValues);
  uvec2 gBlock = encodeBC4(gValues);
  outColor = uvec4(rBlock.x, rBlock.y, gBlock.x, gBlock.y);
}
