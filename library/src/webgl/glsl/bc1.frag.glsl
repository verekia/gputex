#version 300 es
// BC1 (DXT1) fragment-shader encoder — WebGL2 port of bc1.wgsl.
//
// One fragment per 4×4 block. Output is the 8-byte BC1 block as 2 × u32 in
// outColor.rg (outColor.ba unused); the encoder reads back RGBA32UI and keeps
// the low two words per block. Algorithm mirrors bc1.wgsl line-for-line:
// bbox endpoints, 1/16 inset, RGB565 quantisation, forced 4-colour mode, full
// L2 index search. See bc1.wgsl for the detailed rationale.

precision highp float;
precision highp int;

uniform sampler2D uSrc;
uniform ivec2 uSrcSize; // original (unpadded) width, height
uniform int uFlipY;     // 1 = sample bottom-up (matches Three.js flipY)

layout(location = 0) out uvec4 outColor;

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
  float r8 = (r * 527.0 + 23.0) / 256.0;
  float g8 = (g * 259.0 + 33.0) / 256.0;
  float b8 = (b * 527.0 + 23.0) / 256.0;
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

  // Inset the bounding box by ~half an RGB565 cell (1/16) to tighten the
  // quantised palette around the real data range.
  vec3 inset = (bbMax - bbMin) / 16.0;
  vec3 hi = clamp(bbMax - inset, vec3(0.0), vec3(1.0));
  vec3 lo = clamp(bbMin + inset, vec3(0.0), vec3(1.0));

  uint c0 = to565(hi);
  uint c1 = to565(lo);

  // 4-colour mode requires c0 > c1.
  if (c0 == c1) {
    if (c1 > 0u) { c1 = c1 - 1u; } else { c0 = c0 + 1u; }
  } else if (c0 < c1) {
    uint tmp = c0; c0 = c1; c1 = tmp;
  }

  vec3 p0 = from565(c0);
  vec3 p1 = from565(c1);
  vec3 p2 = (2.0 * p0 + p1) * (1.0 / 3.0);
  vec3 p3 = (p0 + 2.0 * p1) * (1.0 / 3.0);

  uint indices = 0u;
  for (int i = 0; i < 16; i++) {
    vec3 c = pixels[i];
    float d0 = dot(c - p0, c - p0);
    float d1 = dot(c - p1, c - p1);
    float d2 = dot(c - p2, c - p2);
    float d3 = dot(c - p3, c - p3);

    float bestD = d0;
    uint bestI = 0u;
    if (d1 < bestD) { bestD = d1; bestI = 1u; }
    if (d2 < bestD) { bestD = d2; bestI = 2u; }
    if (d3 < bestD) { bestD = d3; bestI = 3u; }

    indices = indices | (bestI << (i * 2));
  }

  outColor = uvec4(c0 | (c1 << 16), indices, 0u, 0u);
}
