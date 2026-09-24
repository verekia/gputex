#version 300 es
// ASTC 4×4 LDR fragment-shader encoder — WebGL2 port of astc4x4.wgsl.
//
// One fragment per 4×4 block → 16-byte block as 4 × u32 in outColor.
// Restricted subset: single partition, no dual-plane, with the block class
// picked per block from the content (see astc4x4_ref.ts for layouts, block
// modes, the QUANT_192 trit ISE and the weight-stream bit order):
//   gray + opaque → CEM 0  (luminance), 5-bit weights, mode 0x253
//   opaque        → CEM 8  (RGB), 3-channel PCA extents, two bit budgets:
//                   span > 12 → QUANT_192 endpoints (trit ISE) + 4-bit
//                   weights, mode 0x242; span ≤ 12 → 8-bit endpoints +
//                   3-bit weights, mode 0x053
//   translucent   → CEM 12 (RGBA), 2-bit weights, mode 0x042 — PCA seed →
//                   fused projection + least-squares refit, the fit pass's
//                   weights shipped
// Same algorithm and arithmetic as astc4x4.wgsl (roundEven == WGSL round());
// see astc4x4_fast_f16.wgsl for the design measurements.

precision highp float;
precision highp int;

uniform sampler2D uSrc;
uniform ivec2 uSrcSize;
uniform int uFlipY;

layout(location = 0) out uvec4 outColor;

ivec4 gPixels[16];

ivec4 to8(vec4 v) {
  return ivec4(clamp(floor(v * 255.0 + 0.5), vec4(0.0), vec4(255.0)));
}

// GLSL ES 3.00 has no bitfieldReverse; classic 5-step swap. Weight-stream
// bit q lives at block bit 127 − q, so a stream word assembled LSB-first
// maps onto a block word with one reversal (see astc4x4.wgsl).
uint rev32(uint x) {
  uint v = x;
  v = ((v & 0x55555555u) << 1) | ((v >> 1) & 0x55555555u);
  v = ((v & 0x33333333u) << 2) | ((v >> 2) & 0x33333333u);
  v = ((v & 0x0F0F0F0Fu) << 4) | ((v >> 4) & 0x0F0F0F0Fu);
  v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
  return (v << 16) | (v >> 16);
}

// Principal colour axis via covariance power-iteration (RGBA, 8-bit integer
// pixel domain), seeded with the bbox diagonal. Returns a unit axis, or
// vec4(0.0) for a degenerate (constant) block. The bbox diagonal alone is
// sign-blind and points across anti-correlated data (normal maps, hue
// edges) instead of along it.
vec4 principalAxis4(vec4 mean, vec4 seed, int iters) {
  vec4 c0v = vec4(0.0);
  vec4 c1v = vec4(0.0);
  vec4 c2v = vec4(0.0);
  vec4 c3v = vec4(0.0);
  for (int k = 0; k < 16; k++) {
    vec4 d = vec4(gPixels[k]) - mean;
    c0v += d.x * d;
    c1v += d.y * d;
    c2v += d.z * d;
    c3v += d.w * d;
  }
  vec4 v = seed;
  float len = length(v);
  if (len < 1e-9) { return vec4(0.0); }
  v = v / len;
  for (int it = 0; it < iters; it++) {
    vec4 nv = vec4(dot(c0v, v), dot(c1v, v), dot(c2v, v), dot(c3v, v));
    len = length(nv);
    if (len < 1e-12) { return vec4(0.0); }
    v = nv / len;
  }
  return v;
}

// Principal RGB axis for opaque blocks (alpha is constant there, so the
// 4th covariance lane is dead weight). Same iteration as principalAxis4.
vec3 principalAxis3(vec3 mean, vec3 seed) {
  vec3 c0v = vec3(0.0);
  vec3 c1v = vec3(0.0);
  vec3 c2v = vec3(0.0);
  for (int k = 0; k < 16; k++) {
    vec3 d = vec3(gPixels[k].xyz) - mean;
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

// One pass over the block: project every pixel onto the e0→e1 line (4
// colinear levels, so the nearest entry is the rounded projection), pack
// the 2-bit weights, and solve the least-squares refit from the
// normal-equation sums accumulated in the same pass.
struct Fit { ivec4 e0; ivec4 e1; bool valid; uint wstream; };
Fit projFit(ivec4 e0, ivec4 e1) {
  Fit r;
  r.e0 = ivec4(0);
  r.e1 = ivec4(0);
  r.valid = false;
  r.wstream = 0u;
  vec4 dir = vec4(e1 - e0);
  float dd = dot(dir, dir);
  if (dd == 0.0) { return r; }
  vec4 e0f = vec4(e0);
  float inv = 3.0 / dd;
  float sAA = 0.0;
  float sBB = 0.0;
  float sAB = 0.0;
  vec4 sAV = vec4(0.0);
  vec4 sBV = vec4(0.0);
  float sMin = 3.0;
  float sMax = 0.0;
  for (int k = 0; k < 16; k++) {
    vec4 v = vec4(gPixels[k]);
    float s = clamp(floor(dot(v - e0f, dir) * inv + 0.5), 0.0, 3.0);
    r.wstream |= uint(s) << (2u * uint(k));
    sMin = min(sMin, s);
    sMax = max(sMax, s);
    float b = s * (1.0 / 3.0);
    float a = 1.0 - b;
    sAA += a * a; sBB += b * b; sAB += a * b;
    sAV += a * v; sBV += b * v;
  }
  // Rank-1 guard: one level ⇒ singular system (the solve would be noise).
  if (sMin == sMax) { return r; }
  float det = sAA * sBB - sAB * sAB;
  if (abs(det) < 1e-3) { return r; }
  r.e0 = ivec4(clamp(roundEven((sBB * sAV - sAB * sBV) / det), vec4(0.0), vec4(255.0)));
  r.e1 = ivec4(clamp(roundEven((sAA * sBV - sAB * sAV) / det), vec4(0.0), vec4(255.0)));
  r.valid = true;
  return r;
}

// ISE trit-block encoder: 5 trits → the 8-bit T field (the inverse of the
// spec's trit-block decode; see astc4x4_ref.ts).
uint tritEnc(uint t0, uint t1, uint t2, uint t3, uint t4) {
  uint c = (t2 == 2u && t1 == 2u) ? (12u | t0)
    : (t2 == 2u ? ((t1 << 4u) | (t0 << 2u) | 3u) : ((t2 << 4u) | (t1 << 2u) | t0));
  return (t3 == 2u && t4 == 2u) ? (((c >> 2u) << 5u) | 28u | (c & 3u))
    : (t4 == 2u ? ((t3 << 7u) | 96u | c) : ((t4 << 7u) | (t3 << 5u) | c));
}

// Nearest QUANT_192 endpoint to x ∈ [0,255]; returns (ISE value =
// trit·64 + bits, unquantised level). See astc4x4_fast_f16.wgsl.
uvec2 q192(float x) {
  uint v = uint(clamp(floor(x + 0.5), 0.0, 255.0));
  bool up = v > 127u;
  uint u = up ? 255u - v : v;
  if ((u & 3u) == 3u) {
    float xu = up ? 255.0 - x : x;
    u = (xu > float(u) && u < 127u) ? u + 1u : u - 1u;
  }
  return uvec2(((u & 3u) << 6u) | ((u >> 2u) << 1u) | (up ? 1u : 0u), up ? 255u - u : u);
}

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 4;
  ivec2 maxXY = uSrcSize - ivec2(1);

  ivec4 lo = ivec4(255);
  ivec4 hi = ivec4(0);
  ivec4 isum = ivec4(0);
  int gd = 0; // max |R−G|, |R−B| over the block; 0 ⇔ exactly grayscale
  for (int i = 0; i < 16; i++) {
    ivec2 p = clamp(base + ivec2(i & 3, i >> 2), ivec2(0), maxXY);
    int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - p.y) : p.y;
    ivec4 px = to8(texelFetch(uSrc, ivec2(p.x, sy), 0));
    gPixels[i] = px;
    lo = min(lo, px);
    hi = max(hi, px);
    isum += px;
    gd = max(gd, max(abs(px.x - px.y), abs(px.x - px.z)));
  }
  bool opaque = lo.w == 255;

  uint w0 = 0u; uint w1 = 0u; uint w2 = 0u; uint w3 = 0u;

  if (opaque && gd == 0) {
    // ---------------- Luminance path: CEM 0, 5-bit weights ----------------
    // Endpoints at the exact extremes; 32 palette levels make an LSQ refit
    // unnecessary.
    uint L0 = uint(lo.x);
    uint L1 = uint(hi.x);
    uint s0 = 0u; uint s1 = 0u; uint s2 = 0u;
    if (L1 > L0) {
      float sc = 64.0 / float(hi.x - lo.x);
      // Exact nearest entry of the QUANT_32 grid: unq = 2w for w ≤ 15,
      // 2w + 2 for w ≥ 16 (4-wide gap at the middle, so uniform rounding
      // is wrong there). Best candidate of each half, keep the closer.
      for (int k = 0; k < 16; k++) {
        float u = clamp(float(gPixels[k].x - lo.x) * sc, 0.0, 64.0);
        float wlo = clamp(floor(u * 0.5 + 0.5), 0.0, 15.0);
        float whi = clamp(floor((u - 2.0) * 0.5 + 0.5), 16.0, 31.0);
        bool pick = abs(u - wlo * 2.0) <= abs(u - (whi * 2.0 + 2.0));
        uint w = uint(pick ? wlo : whi);
        // Stream bit q = 5k + j; straddles handled with constant shifts.
        uint off = 5u * uint(k);
        if (off < 28u) { s0 |= (w << off); }
        else if (off == 30u) { s0 |= (w << 30u); s1 |= (w >> 2u); }
        else if (off < 60u) { s1 |= (w << (off - 32u)); }
        else if (off == 60u) { s1 |= (w << 28u); s2 |= (w >> 4u); }
        else { s2 |= (w << (off - 64u)); }
      }
    }
    // Mode 0x253, partitions−1 = 0, CEM 0, L0 @17, L1 @25 (top bit spills
    // into word 1 bit 0); stream words map onto block words via rev32.
    w0 = 0x253u | (L0 << 17u) | (L1 << 25u);
    w1 = (L1 >> 7u) | rev32(s2);
    w2 = rev32(s1);
    w3 = rev32(s0);
  } else if (opaque) {
    // ------------- Opaque colour: CEM 8, two bit budgets -------------------
    // span > 12 → QUANT_192 endpoints + 4-bit weights (mode 0x242), else
    // exact 8-bit endpoints + 3-bit weights (mode 0x053); endpoints are the
    // bbox-clamped PCA extents.
    vec3 mean3 = vec3(isum.xyz) * (1.0 / 16.0);
    vec3 lo3 = vec3(lo.xyz);
    vec3 hi3 = vec3(hi.xyz);
    vec3 x0 = lo3;
    vec3 x1 = hi3;
    vec3 axis = principalAxis3(mean3, hi3 - lo3);
    if (dot(axis, axis) > 0.0) {
      float tMin = 1e30;
      float tMax = -1e30;
      for (int k = 0; k < 16; k++) {
        float t = dot(vec3(gPixels[k].xyz) - mean3, axis);
        tMin = min(tMin, t);
        tMax = max(tMax, t);
      }
      x0 = clamp(mean3 + tMin * axis, lo3, hi3);
      x1 = clamp(mean3 + tMax * axis, lo3, hi3);
    }
    ivec3 span3 = hi.xyz - lo.xyz;
    bool small = max(max(span3.x, span3.y), span3.z) <= 12;
    uvec2 r0; uvec2 g0; uvec2 b0;
    uvec2 r1; uvec2 g1; uvec2 b1;
    if (small) {
      uvec3 q0 = uvec3(clamp(floor(x0 + 0.5), vec3(0.0), vec3(255.0)));
      uvec3 q1 = uvec3(clamp(floor(x1 + 0.5), vec3(0.0), vec3(255.0)));
      r0 = uvec2(q0.x); g0 = uvec2(q0.y); b0 = uvec2(q0.z);
      r1 = uvec2(q1.x); g1 = uvec2(q1.y); b1 = uvec2(q1.z);
    } else {
      r0 = q192(x0.x); g0 = q192(x0.y); b0 = q192(x0.z);
      r1 = q192(x1.x); g1 = q192(x1.y); b1 = q192(x1.z);
    }
    // Blue-contraction ordering on the unquantised levels.
    if (r0.y + g0.y + b0.y > r1.y + g1.y + b1.y) {
      uvec2 tr = r0; r0 = r1; r1 = tr;
      uvec2 tg = g0; g0 = g1; g1 = tg;
      uvec2 tb = b0; b0 = b1; b1 = tb;
    }
    vec3 d0 = vec3(uvec3(r0.y, g0.y, b0.y));
    vec3 d1 = vec3(uvec3(r1.y, g1.y, b1.y));
    // One weight loop for both budgets; weights as 4-bit nibbles.
    float lmax = small ? 7.0 : 15.0;
    vec3 dir = d1 - d0;
    float dd = dot(dir, dir);
    uint s0 = 0u;
    uint s1 = 0u;
    if (dd > 0.0) {
      float inv = lmax / dd;
      for (int k = 0; k < 8; k++) {
        uint w = uint(clamp(floor(dot(vec3(gPixels[k].xyz) - d0, dir) * inv + 0.5), 0.0, lmax));
        s0 |= w << (4u * uint(k));
      }
      for (int k = 8; k < 16; k++) {
        uint w = uint(clamp(floor(dot(vec3(gPixels[k].xyz) - d0, dir) * inv + 0.5), 0.0, lmax));
        s1 |= w << (4u * uint(k - 8));
      }
    }
    if (small) {
      // Mode 0x053: plain 8-bit endpoints; the 3-bit weight stream is the
      // nibbles compacted (8 nibbles → 24 bits).
      uint c0 = (s0 & 0x07070707u) | ((s0 & 0x70707070u) >> 1u);
      c0 = (c0 & 0x003F003Fu) | ((c0 & 0x3F003F00u) >> 2u);
      c0 = (c0 & 0x00000FFFu) | ((c0 & 0x0FFF0000u) >> 4u);
      uint c1 = (s1 & 0x07070707u) | ((s1 & 0x70707070u) >> 1u);
      c1 = (c1 & 0x003F003Fu) | ((c1 & 0x3F003F00u) >> 2u);
      c1 = (c1 & 0x00000FFFu) | ((c1 & 0x0FFF0000u) >> 4u);
      w0 = 0x053u | (8u << 13u) | (r0.x << 17u) | (r1.x << 25u);
      w1 = (r1.x >> 7u) | (g0.x << 1u) | (g1.x << 9u) | (b0.x << 17u) | (b1.x << 25u);
      w2 = (b1.x >> 7u) | rev32(c1 >> 8u);
      w3 = rev32(c0 | (c1 << 24u));
    } else {
      // Mode 0x242: trit-ISE QUANT_192 endpoints (v0..v5 = R0 R1 G0 G1 B0
      // B1, 46 bits from bit 17: group 1 = v0..v4 with trit field T, group
      // 2 = v5 with its lone trit as 2 bits), 4-bit weights.
      uint tg = tritEnc(r0.x >> 6u, r1.x >> 6u, g0.x >> 6u, g1.x >> 6u, b0.x >> 6u);
      w0 = 0x242u | (8u << 13u) | ((r0.x & 63u) << 17u) | ((tg & 3u) << 23u) | ((r1.x & 63u) << 25u) | (((tg >> 2u) & 1u) << 31u);
      w1 = ((tg >> 3u) & 1u) | ((g0.x & 63u) << 1u) | (((tg >> 4u) & 1u) << 7u) | ((g1.x & 63u) << 8u)
        | (((tg >> 5u) & 3u) << 14u) | ((b0.x & 63u) << 16u) | ((tg >> 7u) << 22u) | ((b1.x & 63u) << 23u)
        | ((b1.x >> 6u) << 29u);
      w2 = rev32(s1);
      w3 = rev32(s0);
    }
  } else {
    // ------------- Translucent: CEM 12, 2-bit weights, PCA seed + refit ----
    vec4 mean = vec4(isum) * (1.0 / 16.0);
    // Fused LSQ fit seeded from the block's principal RGBA axis (4 power
    // iterations — the refit absorbs residual axis error) at the exact
    // projection extents, clamped to the block bbox (the unconstrained solve
    // extrapolates outside multi-cluster blocks and would bend the hue).
    ivec4 seed0 = lo;
    ivec4 seed1 = hi;
    vec4 axis = principalAxis4(mean, vec4(hi - lo), 4);
    if (dot(axis, axis) > 0.0) {
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
    ivec4 e0 = lo;
    ivec4 e1 = hi;
    uint fitStream = 0u;
    bool haveFitWeights = false;
    Fit r = projFit(seed0, seed1);
    if (r.valid) {
      e0 = clamp(r.e0, lo, hi);
      e1 = clamp(r.e1, lo, hi);
      fitStream = r.wstream;
      haveFitWeights = true;
    }
    bool swapped = false;
    if (e0.x + e0.y + e0.z > e1.x + e1.y + e1.z) {
      ivec4 t = e0; e0 = e1; e1 = t;
      swapped = true;
    }
    uvec4 E0 = uvec4(e0);
    uvec4 E1 = uvec4(e1);
    // Valid fits ship the fit-pass weights; the blue-contraction swap is a
    // full reflection w → 3−w = bitwise NOT of the packed stream.
    uint s0 = 0u;
    if (haveFitWeights) {
      s0 = swapped ? ~fitStream : fitStream;
    } else {
      vec4 dir = vec4(e1 - e0);
      float dd = dot(dir, dir);
      vec4 e0f = vec4(e0);
      if (dd > 0.0) {
        float inv = 3.0 / dd;
        for (int k = 0; k < 16; k++) {
          uint w = uint(clamp(floor(dot(vec4(gPixels[k]) - e0f, dir) * inv + 0.5), 0.0, 3.0));
          s0 |= w << (2u * uint(k));
        }
      }
    }
    // Mode 0x042, CEM 12 @13, endpoints R0 R1 G0 G1 B0 B1 A0 A1 from 17.
    w0 = 0x042u | (12u << 13u) | (E0.x << 17u) | (E1.x << 25u);
    w1 = (E1.x >> 7u) | (E0.y << 1u) | (E1.y << 9u) | (E0.z << 17u) | (E1.z << 25u);
    w2 = (E1.z >> 7u) | (E0.w << 1u) | (E1.w << 9u);
    w3 = rev32(s0);
  }

  outColor = uvec4(w0, w1, w2, w3);
}
