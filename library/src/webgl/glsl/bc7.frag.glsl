#version 300 es
// BC7 (BPTC) mode-6 fragment-shader encoder — WebGL2 port of bc7.wgsl (fast).
//
// One fragment per 4×4 block → 16-byte block as 4 × u32 in outColor. Fast path
// only: principal-axis seed (covariance power-iteration; bbox on degenerate
// blocks) at the exact projection extents, quantised directly (no LSQ refit —
// mode 6's 16-level palette leaves it under 0.15 dB) → one projection-based
// index-assignment pass (palette is colinear, so the nearest entry is found
// by projecting onto the endpoint line — O(1) per pixel). Mirrors the
// `QUALITY_HIGH == 0` branch of bc7.wgsl; see that file for the mode-6 bit
// layout and rationale.
//
// Determinism note: the WGSL refit uses round() (half-to-even); here we use
// floor(x + 0.5) for portability. The two differ only at exact .5 ties, a
// sub-LSB endpoint nudge that is visually identical.

precision highp float;
precision highp int;

uniform sampler2D uSrc;
uniform ivec2 uSrcSize;
uniform int uFlipY;

layout(location = 0) out uvec4 outColor;

// Per-invocation scratch (mirrors the WGSL function-scope arrays passed by ptr).
ivec4 gPixels[16];
uint gIdx[16];

struct QuantPair { ivec4 seven; ivec4 eight; };
struct Ep { ivec4 seven; ivec4 eight; uint p; };
struct Fit { ivec4 e0; ivec4 e1; bool valid; };

ivec4 to8(vec4 v) {
  return ivec4(clamp(floor(v * 255.0 + 0.5), vec4(0.0), vec4(255.0)));
}

int dist2(ivec4 a, ivec4 b) {
  ivec4 d = a - b;
  ivec4 e = d * d;
  return e.x + e.y + e.z + e.w;
}

// Quantize an 8-bit ideal endpoint to (7-bit value, reconstructed 8-bit) under
// a fixed p-bit, all four channels at once.
QuantPair quantizeEndpoint(ivec4 ideal8, uint p) {
  ivec4 q = ivec4(clamp(floor((vec4(ideal8) - float(p)) / 2.0 + 0.5), vec4(0.0), vec4(127.0)));
  // eff = (q << 1) | p. q*2 is even and p ∈ {0,1}, so q*2 + p is identical and
  // avoids any vector-shift-by-scalar portability question.
  ivec4 eff = q * 2 + ivec4(int(p));
  return QuantPair(q, eff);
}

// Endpoint with its chosen p-bit, picked by minimum quantisation error.
Ep pickEp(ivec4 ideal) {
  QuantPair a = quantizeEndpoint(ideal, 0u);
  QuantPair b = quantizeEndpoint(ideal, 1u);
  if (dist2(b.eight, ideal) < dist2(a.eight, ideal)) {
    return Ep(b.seven, b.eight, 1u);
  }
  return Ep(a.seven, a.eight, 0u);
}

// Principal colour axis via power-iteration over precomputed, mean-corrected
// covariance rows (the moments are accumulated for free in the pixel-load
// loop), seeded with the bbox diagonal. Returns a unit axis, or vec4(0.0)
// for a degenerate (constant) block. The bbox diagonal alone is sign-blind
// and points across anti-correlated data (normal maps, hue edges) instead of
// along it.
vec4 principalAxis(vec4 c0v, vec4 c1v, vec4 c2v, vec4 c3v, vec4 seed) {
  vec4 v = seed;
  float len = length(v);
  if (len < 1e-9) { return vec4(0.0); }
  v /= len;
  for (int it = 0; it < 8; it++) {
    vec4 nv = vec4(dot(c0v, v), dot(c1v, v), dot(c2v, v), dot(c3v, v));
    len = length(nv);
    if (len < 1e-12) { return vec4(0.0); }
    v = nv / len;
  }
  return v;
}

// Projection index assignment over gPixels → gIdx. When `fit`, accumulate the
// LSQ normal-equation sums in the same pass and return refitted endpoints.
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
  float inv = 15.0 / float(dd);
  float sAA = 0.0, sBB = 0.0, sAB = 0.0;
  vec4 sAV = vec4(0.0), sBV = vec4(0.0);
  for (int k = 0; k < 16; k++) {
    ivec4 q = gPixels[k] - pe0;
    float proj = float(q.x * dir.x + q.y * dir.y + q.z * dir.z + q.w * dir.w) * inv;
    float s = clamp(floor(proj + 0.5), 0.0, 15.0);
    gIdx[k] = uint(s);
    if (fit) {
      vec4 v = vec4(gPixels[k]);
      float b = s / 15.0;
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

  // Load pass with the covariance moments FUSED in: d = px − pixel0
  // (first-pixel-relative, so the sums scale with the block's span).
  ivec4 lo = ivec4(255);
  ivec4 hi = ivec4(0);
  vec4 p0f = vec4(0.0);
  vec4 sd = vec4(0.0);
  vec4 c0v = vec4(0.0);
  vec4 c1v = vec4(0.0);
  vec4 c2v = vec4(0.0);
  vec4 c3v = vec4(0.0);
  for (int i = 0; i < 16; i++) {
    ivec2 p = clamp(base + ivec2(i & 3, i >> 2), ivec2(0), maxXY);
    int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - p.y) : p.y;
    ivec4 px = to8(texelFetch(uSrc, ivec2(p.x, sy), 0));
    gPixels[i] = px;
    lo = min(lo, px);
    hi = max(hi, px);
    if (i == 0) { p0f = vec4(px); }
    vec4 d = vec4(px) - p0f;
    sd += d;
    c0v += d.x * d;
    c1v += d.y * d;
    c2v += d.z * d;
    c3v += d.w * d;
  }
  vec4 mean = p0f + sd / 16.0;
  // Mean-correct the fused moments: C = Σddᵀ − (Σd)(Σd)ᵀ/16.
  vec4 sd16 = sd / 16.0;
  c0v -= sd.x * sd16;
  c1v -= sd.y * sd16;
  c2v -= sd.z * sd16;
  c3v -= sd.w * sd16;

  ivec4 seed0 = lo;
  ivec4 seed1 = hi;
  vec4 axis = principalAxis(c0v, c1v, c2v, c3v, vec4(hi - lo));
  if (dot(axis, axis) > 0.0) {
    // Exact projection extents along the axis. (A Rayleigh-quotient span
    // estimate was tried in place of this pass — it saves 16 dots but costs
    // 0.1–0.8 dB and 4–10× on the worst-easy-block gate: σ misjudges
    // two-cluster and outlier blocks. The pass stays.)
    float tMin = 1e30;
    float tMax = -1e30;
    for (int k = 0; k < 16; k++) {
      float t = dot(vec4(gPixels[k]) - mean, axis);
      tMin = min(tMin, t);
      tMax = max(tMax, t);
    }
    seed0 = ivec4(clamp(floor(mean + tMin * axis + 0.5), vec4(0.0), vec4(255.0)));
    seed1 = ivec4(clamp(floor(mean + tMax * axis + 0.5), vec4(0.0), vec4(255.0)));
  }

  // Quantise the PCA-extents seed directly and assign indices in one
  // projection pass — no LSQ refit: with the seed already on the principal
  // axis, mode 6's 16-level palette leaves the refit under 0.15 dB (the
  // coarse 4-level BC1/ASTC fast paths DO keep theirs).
  Ep ep0 = pickEp(seed0);
  Ep ep1 = pickEp(seed1);
  projAssign(ep0.eight, ep1.eight, false);
  ivec4 e0_7 = ep0.seven;
  ivec4 e1_7 = ep1.seven;
  uint p0 = ep0.p;
  uint p1 = ep1.p;

  // Anchor rule — pixel 0's index MSB must be 0; otherwise swap endpoints and
  // reflect every index (decoded image unchanged).
  if ((gIdx[0] & 0x8u) != 0u) {
    ivec4 t = e0_7; e0_7 = e1_7; e1_7 = t;
    uint tp = p0; p0 = p1; p1 = tp;
    for (int k = 0; k < 16; k++) { gIdx[k] = 15u - gIdx[k]; }
  }

  uint block[4];
  block[0] = 0u; block[1] = 0u; block[2] = 0u; block[3] = 0u;
  uint pos = 0u;
  writeBits(block, pos, 7u, 0x40u); pos += 7u;
  writeBits(block, pos, 7u, uint(e0_7.x)); pos += 7u;
  writeBits(block, pos, 7u, uint(e1_7.x)); pos += 7u;
  writeBits(block, pos, 7u, uint(e0_7.y)); pos += 7u;
  writeBits(block, pos, 7u, uint(e1_7.y)); pos += 7u;
  writeBits(block, pos, 7u, uint(e0_7.z)); pos += 7u;
  writeBits(block, pos, 7u, uint(e1_7.z)); pos += 7u;
  writeBits(block, pos, 7u, uint(e0_7.w)); pos += 7u;
  writeBits(block, pos, 7u, uint(e1_7.w)); pos += 7u;
  writeBits(block, pos, 1u, p0); pos += 1u;
  writeBits(block, pos, 1u, p1); pos += 1u;
  writeBits(block, pos, 3u, gIdx[0] & 0x7u); pos += 3u;
  for (int k = 1; k < 16; k++) {
    writeBits(block, pos, 4u, gIdx[k] & 0xFu);
    pos += 4u;
  }

  outColor = uvec4(block[0], block[1], block[2], block[3]);
}
