#version 300 es
// BC1 (DXT1) fragment-shader encoder — WebGL2 port of bc1.wgsl (fast path).
//
// One fragment per 4×4 block. Output is the 8-byte BC1 block as 2 × u32 in
// outColor.rg (outColor.ba unused); the encoder reads back RGBA32UI and keeps
// the low two words per block. This is the *fast* path only (the WGSL
// `QUALITY_HIGH == 0` branch): principal-axis endpoint seed (covariance
// power-iteration; inset bbox on degenerate blocks), RGB565 quantisation,
// forced 4-colour mode, full 4-entry L2 index search, then up to TWO
// least-squares endpoint refit rounds, each accepted only when it lowers the
// block's error. See bc1.wgsl for the full derivation.

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

// Per-invocation scratch (mirrors the WGSL function-scope arrays passed by
// ptr; GLSL would copy array parameters by value).
vec3 gPixels[16];
uint gIdx[16];

// Nearest-palette assignment of gPixels for the decoded palette of (c0,c1),
// with the block's squared error and the LSQ normal-equation sums of the
// resulting assignment accumulated in the same pass — so an accepted refit
// can seed the next round.
struct Assign { float err; float sAA; float sBB; float sAB; vec3 sAV; vec3 sBV; };
Assign assignStats(uint c0, uint c1, out uint indices[16]) {
  vec3 p0 = from565(c0);
  vec3 p1 = from565(c1);
  vec3 pal[4];
  for (int j = 0; j < 4; j++) pal[j] = WA[j] * p0 + WB[j] * p1;
  Assign r = Assign(0.0, 0.0, 0.0, 0.0, vec3(0.0), vec3(0.0));
  for (int k = 0; k < 16; k++) {
    vec3 c = gPixels[k];
    uint bestJ = 0u;
    float bestD = 1e30;
    for (int j = 0; j < 4; j++) {
      vec3 d = pal[j] - c;
      float d2 = dot(d, d);
      if (d2 < bestD) { bestD = d2; bestJ = uint(j); }
    }
    indices[k] = bestJ;
    r.err += bestD;
    float a = WA[int(bestJ)];
    float b = WB[int(bestJ)];
    r.sAA += a * a; r.sBB += b * b; r.sAB += a * b; r.sAV += a * c; r.sBV += b * c;
  }
  return r;
}

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 4;
  ivec2 maxXY = uSrcSize - ivec2(1);

  vec3 bbMin = vec3(1.0);
  vec3 bbMax = vec3(0.0);
  vec3 mean = vec3(0.0);
  for (int i = 0; i < 16; i++) {
    ivec2 p = clamp(base + ivec2(i & 3, i >> 2), ivec2(0), maxXY);
    int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - p.y) : p.y;
    vec3 c = texelFetch(uSrc, ivec2(p.x, sy), 0).rgb;
    gPixels[i] = c;
    bbMin = min(bbMin, c);
    bbMax = max(bbMax, c);
    mean += c;
  }
  mean /= 16.0;

  // Seed endpoints from the block's principal colour axis (covariance
  // power-iteration, seeded with the bbox diagonal — mirrors bc1.wgsl). The
  // bbox diagonal is sign-blind: on anti-correlated channels (normal maps,
  // hue edges) it points across the data instead of along it, and the LSQ
  // refit below — which fits endpoints GIVEN the indices — can't recover.
  // Degenerate (near-flat) blocks keep the inset-bbox seed. Both seeds inset
  // by ~half an RGB565 cell (1/16) to tighten the quantised palette.
  vec3 c0v = vec3(0.0);
  vec3 c1v = vec3(0.0);
  vec3 c2v = vec3(0.0);
  for (int k = 0; k < 16; k++) {
    vec3 d = gPixels[k] - mean;
    c0v += d.x * d;
    c1v += d.y * d;
    c2v += d.z * d;
  }
  vec3 hi;
  vec3 lo;
  vec3 axis = bbMax - bbMin;
  float alen = length(axis);
  bool axisOk = alen > 1e-9;
  if (axisOk) {
    axis /= alen;
    for (int it = 0; it < 8; it++) {
      vec3 nv = vec3(dot(c0v, axis), dot(c1v, axis), dot(c2v, axis));
      float nlen = length(nv);
      if (nlen < 1e-12) { axisOk = false; break; }
      axis = nv / nlen;
    }
  }
  if (axisOk) {
    float tMin = 1e30;
    float tMax = -1e30;
    for (int k = 0; k < 16; k++) {
      float t = dot(gPixels[k] - mean, axis);
      tMin = min(tMin, t);
      tMax = max(tMax, t);
    }
    float pad = (tMax - tMin) / 16.0;
    hi = clamp(mean + (tMax - pad) * axis, vec3(0.0), vec3(1.0));
    lo = clamp(mean + (tMin + pad) * axis, vec3(0.0), vec3(1.0));
  } else {
    vec3 inset = (bbMax - bbMin) / 16.0;
    hi = clamp(bbMax - inset, vec3(0.0), vec3(1.0));
    lo = clamp(bbMin + inset, vec3(0.0), vec3(1.0));
  }

  uint c0 = to565(hi);
  uint c1 = to565(lo);
  // 4-colour mode requires color0 > color1.
  if (c0 == c1) {
    if (c1 > 0u) { c1 = c1 - 1u; } else { c0 = c0 + 1u; }
  } else if (c0 < c1) {
    uint tmp = c0; c0 = c1; c1 = tmp;
  }

  // Seed assignment, then up to TWO least-squares refit rounds (mirroring
  // bc1.wgsl's fast path), each accepted only if the block's squared error
  // drops — the refit minimises a continuous objective and can lose after
  // 565 quantisation. Every assignment pass re-accumulates the sums, so an
  // accepted round seeds the next.
  Assign cur = assignStats(c0, c1, gIdx);
  for (int it = 0; it < 2; it++) {
    float det = cur.sAA * cur.sBB - cur.sAB * cur.sAB;
    if (abs(det) <= 1e-9) { break; }
    // Clamp the refit to the block bbox (not [0,1]): on multi-cluster blocks
    // the unconstrained LSQ solve extrapolates far outside the block's
    // colours and the per-channel clamp then bends the hue — fringe pixels
    // decode to colours that exist nowhere in the block. Constraining to the
    // bbox also measures better in plain SSE (+1.6 dB on the colour test
    // card), so the accept-if-better guard below keeps more refits.
    vec3 e0 = clamp((cur.sBB * cur.sAV - cur.sAB * cur.sBV) / det, bbMin, bbMax);
    vec3 e1 = clamp((cur.sAA * cur.sBV - cur.sAB * cur.sAV) / det, bbMin, bbMax);
    uint nc0 = to565(e0);
    uint nc1 = to565(e1);
    if (nc0 < nc1) { uint t = nc0; nc0 = nc1; nc1 = t; }
    if (nc0 == nc1 || (nc0 == c0 && nc1 == c1)) { break; }
    uint idx2[16];
    Assign nxt = assignStats(nc0, nc1, idx2);
    if (nxt.err >= cur.err) { break; }
    c0 = nc0; c1 = nc1;
    cur = nxt;
    for (int k = 0; k < 16; k++) gIdx[k] = idx2[k];
  }

  uint indices = 0u;
  for (int k = 0; k < 16; k++) indices |= (gIdx[k] & 3u) << (uint(k) * 2u);

  outColor = uvec4(c0 | (c1 << 16), indices, 0u, 0u);
}
