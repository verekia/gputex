#version 300 es
// BC5 (RGTC2) fragment-shader encoder — WebGL2 port of bc5.wgsl (fast path).
//
// One fragment per 4×4 block → 16-byte BC5 block as 4 × u32 in outColor.
// BC5 = two BC4 halves (R then G). This is the *fast* path only: bbox
// endpoints + a full-L2 index assignment per channel with the least-squares
// refit sums accumulated in the same pass, then one refit accepted only when
// it lowers the block's error (mirrors bc5.wgsl's fast branch — worth
// ~1.3 dB on the normal-map card). Always emits 6-interpolation mode
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

// Nearest-palette assignment with the LSQ normal-equation sums and total
// squared error accumulated in the same pass. Sums are only consumed by the
// caller's refit; err drives the accept-if-better test.
struct Assign { float err; float sAA; float sBB; float sAB; float sAV; float sBV; };
Assign assignAll(float values[16], float pal[8], out uint indices[16]) {
  Assign r = Assign(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
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
    r.err += bestD;
    float a = W0_6[int(bestJ)];
    float b = W1_6[int(bestJ)];
    r.sAA += a * a; r.sBB += b * b; r.sAB += a * b; r.sAV += a * v; r.sBV += b * v;
  }
  return r;
}

// Encode 16 single-channel values into an 8-byte BC4 block (two little-endian
// u32s). Mirrors encode_bc4() in bc5.wgsl's fast branch: bbox seed, fused
// assignment + LSQ sums, refit accepted only if the error drops.
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
  Assign seed = assignAll(values, pal, indices);

  // One least-squares refit, accepted only if the requantised endpoints lower
  // the block error. Clamp to the block's value range (a strict-SSE win vs
  // clamping to [0,1], same as the other formats' fast paths); keep 6-interp
  // mode (r0 > r1 strictly).
  float det = seed.sAA * seed.sBB - seed.sAB * seed.sAB;
  if (abs(det) > 1e-9) {
    float e0 = clamp((seed.sBB * seed.sAV - seed.sAB * seed.sBV) / det, vmin, vmax);
    float e1 = clamp((seed.sAA * seed.sBV - seed.sAB * seed.sAV) / det, vmin, vmax);
    uint n0 = quantize8(e0);
    uint n1 = quantize8(e1);
    if (n0 > n1 && !(n0 == r0 && n1 == r1)) {
      float pal2[8];
      float n0f = float(n0) / 255.0;
      float n1f = float(n1) / 255.0;
      for (int j = 0; j < 8; j++) {
        pal2[j] = W0_6[j] * n0f + W1_6[j] * n1f;
      }
      uint idx2[16];
      Assign refit = assignAll(values, pal2, idx2);
      if (refit.err < seed.err) {
        r0 = n0; r1 = n1;
        for (int k = 0; k < 16; k++) indices[k] = idx2[k];
      }
    }
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
