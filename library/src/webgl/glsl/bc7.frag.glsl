#version 300 es
// BC7 (BPTC) mode-6 fragment-shader encoder — WebGL2 port of bc7.wgsl (fast).
//
// One fragment per 4×4 block → 16-byte block as 4 × u32 in outColor. Fast path
// only: principal-axis seed (covariance power-iteration; bbox on degenerate
// blocks) at the exact projection extents, quantised directly (no LSQ refit —
// mode 6's 16-level palette leaves it under 0.15 dB) → one projection-based
// index-assignment pass (palette is colinear, so the nearest entry is found
// by projecting onto the endpoint line — O(1) per pixel). Gray + opaque
// blocks take an integer 1-D tail: lossless for spans ≤ 15 with odd
// endpoints (alpha exactly 255), alpha-aware scalar LSQ refit above. Same
// algorithm as bc7.wgsl; see that file (and bc7_fast_f16.wgsl) for the
// mode-6 bit layout and rationale.
//
// Same arithmetic as bc7.wgsl (roundEven == WGSL round()), so the two
// produce the same blocks up to driver float-contraction differences.

precision highp float;
precision highp int;

uniform sampler2D uSrc;
uniform ivec2 uSrcSize;
uniform int uFlipY;

layout(location = 0) out uvec4 outColor;

// Per-invocation scratch (mirrors the WGSL function-scope array).
ivec4 gPixels[16];

struct QuantPair { ivec4 seven; ivec4 eight; };
struct Ep { ivec4 seven; ivec4 eight; uint p; };

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

// Gray + opaque block (every texel R == G == B, A == 255) from gPixels[].x,
// packed straight into the 4 block words. 8-bit endpoints E = 2q + p; RGB
// share q, alpha is 254 + p.
uvec4 encodeGray(float vmin, float vmax) {
  float e0 = vmin;
  float e1 = vmax;
  if (vmax - vmin <= 15.0) {
    // LOSSLESS: integer endpoints ≤ 15 apart cover every integer between
    // them and round(15·(v − e0)/d) selects it; even endpoints step outward
    // while the span stays ≤ 15 so both p-bits are 1 (alpha 255).
    if (fract(e0 * 0.5) == 0.0 && e0 > 0.0 && e1 - e0 < 15.0) { e0 -= 1.0; }
    if (fract(e1 * 0.5) == 0.0 && e1 < 255.0 && e1 - e0 < 15.0) { e1 += 1.0; }
  } else {
    // Moment-form scalar LSQ refit on the seed levels, all four p-bit
    // combinations priced including the alpha term, accept-if-better.
    float k1 = 15.0 / (vmax - vmin);
    float k0 = 0.5 - vmin * k1;
    float sL = 0.0;
    float sLL = 0.0;
    float sv = 0.0;
    float sLv = 0.0;
    for (int k = 0; k < 16; k++) {
      float v = float(gPixels[k].x);
      float L = floor(v * k1 + k0);
      sL += L;
      sLL += L * L;
      sv += v;
      sLv += L * v;
    }
    float C = sLL * (1.0 / 225.0);
    float B = sL * (1.0 / 15.0) - C;
    float A = 16.0 - sL * (2.0 / 15.0) + C;
    float Y = sLv * (1.0 / 15.0);
    float X = sv - Y;
    float det = A * C - B * B;
    if (det > 1e-3) {
      float s0 = clamp((C * X - B * Y) / det, 0.0, 255.0);
      float s1 = clamp((A * Y - B * X) / det, 0.0, 255.0);
      // price(e0, e1, p0, p1) up to the block constant: RGB ×3 + alpha.
      float ps0 = vmin - 2.0 * floor(vmin * 0.5);
      float ps1 = vmax - 2.0 * floor(vmax * 0.5);
      float best = 3.0 * (A * vmin * vmin + 2.0 * B * vmin * vmax + C * vmax * vmax - 2.0 * (X * vmin + Y * vmax))
        + A * (1.0 - ps0) + 2.0 * B * (1.0 - ps0) * (1.0 - ps1) + C * (1.0 - ps1);
      for (int pc = 0; pc < 4; pc++) {
        float p0 = float(pc & 1);
        float p1 = float(pc >> 1);
        float c0 = 2.0 * clamp(floor((s0 - p0) * 0.5 + 0.5), 0.0, 127.0) + p0;
        float c1 = 2.0 * clamp(floor((s1 - p1) * 0.5 + 0.5), 0.0, 127.0) + p1;
        float pr = 3.0 * (A * c0 * c0 + 2.0 * B * c0 * c1 + C * c1 * c1 - 2.0 * (X * c0 + Y * c1))
          + A * (1.0 - p0) + 2.0 * B * (1.0 - p0) * (1.0 - p1) + C * (1.0 - p1);
        if (pr < best) {
          best = pr;
          e0 = c0;
          e1 = c1;
        }
      }
    }
  }
  uint glo = 0u;
  uint ghi = 0u;
  if (e1 != e0) {
    float k1 = 15.0 / (e1 - e0);
    float k0 = 0.5 - e0 * k1;
    for (int k = 0; k < 8; k++) {
      float sg = clamp(floor(float(gPixels[k].x) * k1 + k0), 0.0, 15.0);
      glo |= uint(sg) << uint(k * 4);
    }
    for (int k = 8; k < 16; k++) {
      float sg = clamp(floor(float(gPixels[k].x) * k1 + k0), 0.0, 15.0);
      ghi |= uint(sg) << uint((k - 8) * 4);
    }
  }
  uint u0 = uint(e0);
  uint u1 = uint(e1);
  // Anchor rule: pixel 0's index MSB must be 0 — swap + reflect (bitwise NOT).
  if ((glo & 0x8u) != 0u) {
    uint t = u0; u0 = u1; u1 = t;
    glo = ~glo; ghi = ~ghi;
  }
  uint q0 = u0 >> 1u;
  uint q1 = u1 >> 1u;
  return uvec4(
    0x40u | (q0 << 7u) | (q1 << 14u) | (q0 << 21u) | (q1 << 28u),
    (q1 >> 4u) | (q0 << 3u) | (q1 << 10u) | (127u << 17u) | (127u << 24u) | ((u0 & 1u) << 31u),
    (u1 & 1u) | ((glo & 0x7u) << 1u) | (glo & 0xFFFFFFF0u),
    ghi
  );
}

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 4;
  ivec2 maxXY = uSrcSize - ivec2(1);

  // Load pass: texels, bbox and the gray test.
  ivec4 lo = ivec4(255);
  ivec4 hi = ivec4(0);
  int gd = 0;
  for (int i = 0; i < 16; i++) {
    ivec2 p = clamp(base + ivec2(i & 3, i >> 2), ivec2(0), maxXY);
    int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - p.y) : p.y;
    ivec4 px = to8(texelFetch(uSrc, ivec2(p.x, sy), 0));
    gPixels[i] = px;
    lo = min(lo, px);
    hi = max(hi, px);
    gd = max(gd, max(abs(px.x - px.y), abs(px.x - px.z)));
  }

  // Gray + opaque blocks: 1-D, own tail (see bc7_fast_f16.wgsl) — lossless
  // for spans ≤ 15 with odd endpoints (alpha exactly 255), closed-form
  // scalar LSQ refit with alpha-aware p-bit pricing above that.
  if (lo.w == 255 && gd == 0) {
    outColor = encodeGray(float(lo.x), float(hi.x));
    return;
  }

  // Covariance moments: d = px − pixel0 (first-pixel-relative, so the sums
  // scale with the block's span).
  vec4 p0f = vec4(gPixels[0]);
  vec4 sd = vec4(0.0);
  vec4 c0v = vec4(0.0);
  vec4 c1v = vec4(0.0);
  vec4 c2v = vec4(0.0);
  vec4 c3v = vec4(0.0);
  for (int i = 1; i < 16; i++) {
    vec4 d = vec4(gPixels[i]) - p0f;
    sd += d;
    c0v += d.x * d;
    c1v += d.y * d;
    c2v += d.z * d;
    c3v += d.w * d;
  }
  vec4 mean = p0f + sd / 16.0;
  // Mean-correct the moments: C = Σddᵀ − (Σd)(Σd)ᵀ/16.
  vec4 sd16 = sd / 16.0;
  c0v -= sd.x * sd16;
  c1v -= sd.y * sd16;
  c2v -= sd.z * sd16;
  c3v -= sd.w * sd16;

  ivec4 seed0 = lo;
  ivec4 seed1 = hi;
  {
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
      seed0 = ivec4(clamp(roundEven(mean + tMin * axis), vec4(0.0), vec4(255.0)));
      seed1 = ivec4(clamp(roundEven(mean + tMax * axis), vec4(0.0), vec4(255.0)));
    }
  }

  // Quantise the PCA-extents seed directly and assign indices in one
  // projection pass — no LSQ refit: with the seed already on the principal
  // axis, mode 6's 16-level palette leaves the refit under 0.15 dB (the
  // coarse 4-level BC1/ASTC paths DO keep theirs). The 16 4-bit indices pack
  // on the fly into two nibble words (pixel k → bits 4k..4k+3).
  uint ilo = 0u;
  uint ihi = 0u;
  Ep ep0 = pickEp(seed0);
  Ep ep1 = pickEp(seed1);
  vec4 dir = vec4(ep1.eight - ep0.eight);
  float dd = dot(dir, dir);
  if (dd > 0.0) {
    vec4 e0f = vec4(ep0.eight);
    float inv = 15.0 / dd;
    for (int k = 0; k < 8; k++) {
      float sk = clamp(floor(dot(vec4(gPixels[k]) - e0f, dir) * inv + 0.5), 0.0, 15.0);
      ilo |= uint(sk) << uint(k * 4);
    }
    for (int k = 8; k < 16; k++) {
      float sk = clamp(floor(dot(vec4(gPixels[k]) - e0f, dir) * inv + 0.5), 0.0, 15.0);
      ihi |= uint(sk) << uint((k - 8) * 4);
    }
  }
  ivec4 e0_7 = ep0.seven;
  ivec4 e1_7 = ep1.seven;
  uint p0 = ep0.p;
  uint p1 = ep1.p;

  // Anchor rule — pixel 0's index MSB must be 0. Swapping endpoints reflects
  // every index (i → 15−i), which on packed nibbles is a bitwise NOT.
  if ((ilo & 0x8u) != 0u) {
    ivec4 t = e0_7; e0_7 = e1_7; e1_7 = t;
    uint tp = p0; p0 = p1; p1 = tp;
    ilo = ~ilo; ihi = ~ihi;
  }

  // Straight-line mode-6 packing (a generic bit writer's dynamic word
  // indexing keeps the output array out of registers).
  uvec4 e0 = uvec4(e0_7);
  uvec4 e1 = uvec4(e1_7);
  outColor = uvec4(
    0x40u | (e0.x << 7u) | (e1.x << 14u) | (e0.y << 21u) | (e1.y << 28u),
    (e1.y >> 4u) | (e0.z << 3u) | (e1.z << 10u) | (e0.w << 17u) | (e1.w << 24u) | (p0 << 31u),
    p1 | ((ilo & 0x7u) << 1u) | (ilo & 0xFFFFFFF0u),
    ihi
  );
}
