#version 300 es
// BC1 (DXT1) fragment-shader encoder — WebGL2 port of bc1.wgsl (fast path).
//
// One fragment per 4×4 block. Output is the 8-byte BC1 block as 2 × u32 in
// outColor.rg (outColor.ba unused); the encoder reads back RGBA32UI and keeps
// the low two words per block. This is the *fast* path only (the WGSL
// `QUALITY_HIGH == 0` branch): bbox endpoints, 1/16 inset, RGB565 quantisation,
// forced 4-colour mode, full 4-entry L2 index search, then a single
// least-squares endpoint refit accepted only when it lowers the block's error.
// See bc1.wgsl for the full derivation.

precision highp float;
precision highp int;

uniform sampler2D uSrc;
uniform ivec2 uSrcSize; // original (unpadded) width, height
uniform int uFlipY;     // 1 = sample bottom-up (matches Three.js flipY)

layout(location = 0) out uvec4 outColor;

// 4-colour-mode interpolation weights: pal[j] = WA[j]*c0 + WB[j]*c1.
const float WA[4] = float[4](1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0);
const float WB[4] = float[4](0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0);

uint to565(vec3 c) {
  uint r = uint(clamp(floor(c.r * 31.0 + 0.5), 0.0, 31.0));
  uint g = uint(clamp(floor(c.g * 63.0 + 0.5), 0.0, 63.0));
  uint b = uint(clamp(floor(c.b * 31.0 + 0.5), 0.0, 31.0));
  return (r << 11) | (g << 5) | b;
}

vec3 from565(uint c) {
  float r = float((c >> 11) & 31u);
  float g = float((c >> 5) & 63u);
  float b = float(c & 31u);
  // 5/6-bit → 8-bit. floor((x*527+23)/64) == (x<<3)|(x>>2): exact hardware
  // bit-replication (white → 255), so index selection matches the GPU decode.
  float r8 = floor((r * 527.0 + 23.0) / 64.0);
  float g8 = floor((g * 259.0 + 33.0) / 64.0);
  float b8 = floor((b * 527.0 + 23.0) / 64.0);
  return vec3(r8, g8, b8) / 255.0;
}

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 4;
  ivec2 maxXY = uSrcSize - ivec2(1);

  vec3 pixels[16];
  vec3 bbMin = vec3(1.0);
  vec3 bbMax = vec3(0.0);
  for (int i = 0; i < 16; i++) {
    ivec2 p = clamp(base + ivec2(i & 3, i >> 2), ivec2(0), maxXY);
    int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - p.y) : p.y;
    vec3 c = texelFetch(uSrc, ivec2(p.x, sy), 0).rgb;
    pixels[i] = c;
    bbMin = min(bbMin, c);
    bbMax = max(bbMax, c);
  }

  // Inset the bbox by ~half an RGB565 cell (1/16) to tighten the quantised
  // 4-colour palette around the real data range.
  vec3 inset = (bbMax - bbMin) / 16.0;
  vec3 hi = clamp(bbMax - inset, vec3(0.0), vec3(1.0));
  vec3 lo = clamp(bbMin + inset, vec3(0.0), vec3(1.0));

  uint c0 = to565(hi);
  uint c1 = to565(lo);
  // 4-colour mode requires color0 > color1.
  if (c0 == c1) {
    if (c1 > 0u) { c1 = c1 - 1u; } else { c0 = c0 + 1u; }
  } else if (c0 < c1) {
    uint tmp = c0; c0 = c1; c1 = tmp;
  }

  // Build the palette in decoded space, assign each pixel its nearest entry.
  vec3 pal[4];
  vec3 p0 = from565(c0);
  vec3 p1 = from565(c1);
  for (int j = 0; j < 4; j++) pal[j] = WA[j] * p0 + WB[j] * p1;

  uint idx[16];
  float err = 0.0;
  for (int k = 0; k < 16; k++) {
    vec3 c = pixels[k];
    uint bestJ = 0u;
    float bestD = 1e30;
    for (int j = 0; j < 4; j++) {
      vec3 d = pal[j] - c;
      float d2 = dot(d, d);
      if (d2 < bestD) { bestD = d2; bestJ = uint(j); }
    }
    idx[k] = bestJ;
    err += bestD;
  }

  // One least-squares refit: re-solve the endpoints for the current indices,
  // re-quantise, re-assign; keep it only if the squared error drops.
  float sAA = 0.0, sBB = 0.0, sAB = 0.0;
  vec3 sAV = vec3(0.0), sBV = vec3(0.0);
  for (int k = 0; k < 16; k++) {
    float a = WA[int(idx[k])];
    float b = WB[int(idx[k])];
    vec3 v = pixels[k];
    sAA += a * a; sBB += b * b; sAB += a * b; sAV += a * v; sBV += b * v;
  }
  float det = sAA * sBB - sAB * sAB;
  if (abs(det) > 1e-9) {
    vec3 e0 = clamp((sBB * sAV - sAB * sBV) / det, vec3(0.0), vec3(1.0));
    vec3 e1 = clamp((sAA * sBV - sAB * sAV) / det, vec3(0.0), vec3(1.0));
    uint nc0 = to565(e0);
    uint nc1 = to565(e1);
    if (nc0 < nc1) { uint t = nc0; nc0 = nc1; nc1 = t; }
    if (nc0 != nc1 && !(nc0 == c0 && nc1 == c1)) {
      vec3 q0 = from565(nc0);
      vec3 q1 = from565(nc1);
      vec3 pal2[4];
      for (int j = 0; j < 4; j++) pal2[j] = WA[j] * q0 + WB[j] * q1;
      uint idx2[16];
      float nerr = 0.0;
      for (int k = 0; k < 16; k++) {
        vec3 c = pixels[k];
        uint bestJ = 0u;
        float bestD = 1e30;
        for (int j = 0; j < 4; j++) {
          vec3 d = pal2[j] - c;
          float d2 = dot(d, d);
          if (d2 < bestD) { bestD = d2; bestJ = uint(j); }
        }
        idx2[k] = bestJ;
        nerr += bestD;
      }
      if (nerr < err) {
        c0 = nc0; c1 = nc1;
        for (int k = 0; k < 16; k++) idx[k] = idx2[k];
      }
    }
  }

  uint indices = 0u;
  for (int k = 0; k < 16; k++) indices |= (idx[k] & 3u) << (uint(k) * 2u);

  outColor = uvec4(c0 | (c1 << 16), indices, 0u, 0u);
}
