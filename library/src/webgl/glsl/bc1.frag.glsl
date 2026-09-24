#version 300 es
// BC1 (DXT1) fragment-shader encoder — WebGL2 port of bc1.wgsl.
//
// One fragment per 4×4 block. Output is the 8-byte BC1 block as 2 × u32 in
// outColor.rg (outColor.ba unused); the encoder reads back RGBA32UI and keeps
// the low two words per block. Same algorithm and arithmetic as bc1.wgsl
// (see bc1_fast_f16.wgsl for the measurements): near-flat blocks take a
// solid colour — per channel the endpoint pair whose ⅔/⅓ interpolant lands
// nearest the block mean; other blocks get a principal-axis endpoint seed
// (covariance power-iteration; inset bbox on degenerate blocks), inset by
// ~half a 565 cell along the axis. A projection pass assigns every pixel the
// rounded projection onto the decoded-endpoint line (the 4 palette entries
// are colinear and evenly spaced, so that is the nearest entry) while
// accumulating the block error and projection moments, followed by up to
// TWO least-squares refit rounds solved from those moments (re-quantise,
// reproject, accept only on lower block error). Always 4-colour mode.

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

// 5/6-bit → 8-bit: (x*527+23)>>6 (6-bit: 259/33) — round-to-nearest scaling,
// matching bc1_ref.ts and the hardware decoders (NOT plain bit-replication,
// which differs for some codes, e.g. 5-bit 3 → 25 vs 24).
vec3 from565(uint c) {
  uint r = (c >> 11) & 31u;
  uint g = (c >> 5) & 63u;
  uint b = c & 31u;
  uint r8 = (r * 527u + 23u) >> 6;
  uint g8 = (g * 259u + 33u) >> 6;
  uint b8 = (b * 527u + 23u) >> 6;
  return vec3(float(r8), float(g8), float(b8)) / 255.0;
}

// Force 4-colour mode: c0 > c1 strictly.
uvec2 order565(uint a, uint b) {
  uint c0 = a;
  uint c1 = b;
  if (c0 == c1) {
    if (c1 > 0u) { c1 = c1 - 1u; } else { c0 = c0 + 1u; }
  } else if (c0 < c1) {
    uint t = c0; c0 = c1; c1 = t;
  }
  return uvec2(c0, c1);
}

// Per-invocation scratch (GLSL would copy array parameters by value).
vec3 gPixels[16];

// One projection pass against the decoded endpoints of (c0,c1): levels
// L = 0..3 along p0→p1, the packed indices, the block's squared error, and
// the projection MOMENTS a refit needs: ΣL, ΣL², Σu, ΣL·u (u = v − p0).
// Level → BC1 index: 0→0 (c0), 1→2, 2→3, 3→1 (c1): packed LUT (0x78 >> 2L) & 3.
struct Moments { float sL; float sLL; vec3 sU; vec3 sLu; uint indices; float err; };
Moments moments(uint c0, uint c1) {
  vec3 p0 = from565(c0);
  vec3 dir = from565(c1) - p0;
  float inv = 3.0 / dot(dir, dir);
  Moments m = Moments(0.0, 0.0, vec3(0.0), vec3(0.0), 0u, 0.0);
  for (int k = 0; k < 16; k++) {
    vec3 u = gPixels[k] - p0;
    float L = clamp(floor(dot(u, dir) * inv + 0.5), 0.0, 3.0);
    m.sL += L;
    m.sLL += L * L;
    m.sU += u;
    m.sLu += L * u;
    m.indices |= ((0x78u >> (uint(L) * 2u)) & 3u) << (uint(k) * 2u);
    vec3 e = u - L * (1.0 / 3.0) * dir;
    m.err += dot(e, e);
  }
  return m;
}

// One least-squares refit from moments (b = L/3, a = 1 − b):
//   sBB = ΣL²/9   sAB = ΣL/3 − ΣL²/9   sAA = 16 − 2ΣL/3 + ΣL²/9
//   Σb·u = ΣL·u/3   Σa·u = Σu − Σb·u
// clamped to [limLo, limHi], re-quantised and ordered. Returns (c0, c1)
// unchanged when every pixel sits on ONE level (16·ΣL² == (ΣL)², singular).
uvec2 solve(Moments m, uint c0, uint c1, vec3 limLo, vec3 limHi) {
  if (16.0 * m.sLL == m.sL * m.sL) { return uvec2(c0, c1); }
  float sBB = m.sLL * (1.0 / 9.0);
  float sAB = m.sL * (1.0 / 3.0) - sBB;
  float sAA = 16.0 - m.sL * (2.0 / 3.0) + sBB;
  float det = sAA * sBB - sAB * sAB;
  vec3 p0 = from565(c0);
  vec3 sBu = m.sLu * (1.0 / 3.0);
  vec3 sAu = m.sU - sBu;
  vec3 e0 = clamp(p0 + (sBB * sAu - sAB * sBu) / det, limLo, limHi);
  vec3 e1 = clamp(p0 + (sAA * sBu - sAB * sAu) / det, limLo, limHi);
  return order565(to565(e0), to565(e1));
}

// Solid-colour channel code: the pair (a, b) of `bits`-bit codes whose ⅔/⅓
// interpolant (2·dec(a) + dec(b))/3 — palette index 2 — lands nearest v
// (8-bit units).
uvec2 solidPair(float v, uint bits) {
  uint maxc = (1u << bits) - 1u;
  uint q = min(uint(v * float(maxc) / 255.0), maxc - 1u);
  float x;
  float y;
  if (bits == 5u) {
    x = float((q * 527u + 23u) >> 6);
    y = float(((q + 1u) * 527u + 23u) >> 6);
  } else {
    x = float((q * 259u + 33u) >> 6);
    y = float(((q + 1u) * 259u + 33u) >> 6);
  }
  uvec2 best = uvec2(q, q);
  float be = abs(x - v);
  float m1 = (2.0 * x + y) / 3.0;
  if (abs(m1 - v) < be) { be = abs(m1 - v); best = uvec2(q, q + 1u); }
  float m2 = (x + 2.0 * y) / 3.0;
  if (abs(m2 - v) < be) { be = abs(m2 - v); best = uvec2(q + 1u, q); }
  if (abs(y - v) < be) { best = uvec2(q + 1u, q + 1u); }
  return best;
}

// Principal colour axis via covariance power-iteration, seeded with the bbox
// diagonal. Returns a unit axis, or vec3(0) for a degenerate (constant)
// block. The bbox diagonal alone is sign-blind and points across
// anti-correlated data (normal maps, hue edges) instead of along it.
vec3 principalAxis(vec3 mean, vec3 seed) {
  vec3 c0v = vec3(0.0);
  vec3 c1v = vec3(0.0);
  vec3 c2v = vec3(0.0);
  for (int k = 0; k < 16; k++) {
    vec3 d = gPixels[k] - mean;
    c0v += d.x * d;
    c1v += d.y * d;
    c2v += d.z * d;
  }
  vec3 v = seed;
  float len = length(v);
  if (len < 1e-9) { return vec3(0.0); }
  v = v / len;
  for (int it = 0; it < 8; it++) {
    vec3 nv = vec3(dot(c0v, v), dot(c1v, v), dot(c2v, v));
    len = length(nv);
    if (len < 1e-12) { return vec3(0.0); }
    v = nv / len;
  }
  return v;
}

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 4;
  ivec2 maxXY = uSrcSize - ivec2(1);

  vec3 bbMin = vec3(1.0);
  vec3 bbMax = vec3(0.0);
  vec3 mean = vec3(0.0);
  float gd = 0.0;
  for (int i = 0; i < 16; i++) {
    ivec2 p = clamp(base + ivec2(i & 3, i >> 2), ivec2(0), maxXY);
    int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - p.y) : p.y;
    vec3 c = texelFetch(uSrc, ivec2(p.x, sy), 0).rgb;
    gPixels[i] = c;
    bbMin = min(bbMin, c);
    bbMax = max(bbMax, c);
    mean += c;
    gd = max(gd, max(abs(c.x - c.y), abs(c.x - c.z)));
  }
  mean = mean * (1.0 / 16.0);
  // Exactly-gray blocks free the refit from the bbox clamp (no hue to
  // protect; smooth gradients want endpoints outside the data range).
  bool gray = gd == 0.0;
  vec3 limLo = gray ? vec3(0.0) : bbMin;
  vec3 limHi = gray ? vec3(1.0) : bbMax;

  // Near-flat blocks (every channel within 3 levels): solid colour at the
  // block mean; they share the index pass below and skip the seed + refits.
  vec3 span = bbMax - bbMin;
  bool flatBlock = max(max(span.x, span.y), span.z) <= 3.0 / 255.0;
  uint c0;
  uint c1;
  if (flatBlock) {
    vec3 m8 = mean * 255.0;
    uvec2 pr = solidPair(m8.x, 5u);
    uvec2 pg = solidPair(m8.y, 6u);
    uvec2 pb = solidPair(m8.z, 5u);
    uint s0 = (pr.x << 11) | (pg.x << 5) | pb.x;
    uint s1 = (pr.y << 11) | (pg.y << 5) | pb.y;
    c0 = max(s0, s1);
    c1 = min(s0, s1);
  } else {
    // Principal-axis seed at the exact projection extents, inset by ~half a
    // 565 cell along the axis. Degenerate blocks keep the inset-bbox seed.
    vec3 seedHi;
    vec3 seedLo;
    vec3 axis = principalAxis(mean, bbMax - bbMin);
    if (dot(axis, axis) > 0.0) {
      float tMin = 1e30;
      float tMax = -1e30;
      for (int k = 0; k < 16; k++) {
        float t = dot(gPixels[k] - mean, axis);
        tMin = min(tMin, t);
        tMax = max(tMax, t);
      }
      float pad = (tMax - tMin) / 16.0;
      seedHi = clamp(mean + (tMax - pad) * axis, vec3(0.0), vec3(1.0));
      seedLo = clamp(mean + (tMin + pad) * axis, vec3(0.0), vec3(1.0));
    } else {
      vec3 inset = (bbMax - bbMin) / 16.0;
      seedHi = clamp(bbMax - inset, vec3(0.0), vec3(1.0));
      seedLo = clamp(bbMin + inset, vec3(0.0), vec3(1.0));
    }
    uvec2 seed = order565(to565(seedHi), to565(seedLo));
    c0 = seed.x;
    c1 = seed.y;
  }

  // Projection pass on the seed, then up to two least-squares refit rounds,
  // each re-projected and accepted only if the block error drops. Equal
  // flat codes encode the colour itself: index 0.
  uint indices = 0u;
  if (c0 != c1) {
    Moments cur = moments(c0, c1);
    int rounds = flatBlock ? 0 : 2;
    for (int it = 0; it < rounds; it++) {
      uvec2 cand = solve(cur, c0, c1, limLo, limHi);
      if (cand.x == c0 && cand.y == c1) { break; }
      Moments nxt = moments(cand.x, cand.y);
      if (nxt.err >= cur.err) { break; }
      c0 = cand.x;
      c1 = cand.y;
      cur = nxt;
    }
    indices = cur.indices;
  }

  outColor = uvec4(c0 | (c1 << 16), indices, 0u, 0u);
}
