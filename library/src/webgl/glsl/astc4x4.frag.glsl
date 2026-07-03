#version 300 es
// ASTC 4×4 LDR fragment-shader encoder — WebGL2 port of astc4x4.wgsl (fast).
//
// One fragment per 4×4 block → 16-byte block as 4 × u32 in outColor.
// Restricted subset: single partition, no dual-plane, 8-bit endpoints, with
// the block class picked per block from the content (see astc4x4_ref.ts for
// layouts and block-mode derivations):
//   gray + opaque → CEM 0  (luminance), 5-bit weights, mode 0x253
//   opaque        → CEM 8  (RGB),       3-bit weights, mode 0x053
//   translucent   → CEM 12 (RGBA),      2-bit weights, mode 0x042
// Colour paths: principal-axis seed (covariance power-iteration; bbox on
// degenerate blocks) → one LSQ refit fused into a projection weight
// assignment. Mirrors astc4x4.wgsl.
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

// Principal colour axis of gPixels via covariance power-iteration, seeded
// with the bbox diagonal. Returns a unit axis, or vec4(0.0) for a degenerate
// (constant) block. The bbox diagonal alone is sign-blind and points across
// anti-correlated data (normal maps, hue edges) instead of along it.
vec4 principalAxis(vec4 mean, vec4 seed, int iters) {
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
  v /= len;
  for (int it = 0; it < iters; it++) {
    vec4 nv = vec4(dot(c0v, v), dot(c1v, v), dot(c2v, v), dot(c3v, v));
    len = length(nv);
    if (len < 1e-12) { return vec4(0.0); }
    v = nv / len;
  }
  return v;
}

// Projection weight assignment over lmax + 1 colinear levels, with the LSQ
// normal-equation sums accumulated in the same pass for a fused refit.
Fit projAssign(ivec4 pe0, ivec4 pe1, float lmax, bool fit) {
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
  float inv = lmax / float(dd);
  float sAA = 0.0, sBB = 0.0, sAB = 0.0;
  vec4 sAV = vec4(0.0), sBV = vec4(0.0);
  for (int k = 0; k < 16; k++) {
    ivec4 q = gPixels[k] - pe0;
    float proj = float(q.x * dir.x + q.y * dir.y + q.z * dir.z + q.w * dir.w) * inv;
    float s = clamp(floor(proj + 0.5), 0.0, lmax);
    gIdx[k] = uint(s);
    if (fit) {
      vec4 v = vec4(gPixels[k]);
      float b = s / lmax;
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
  } else {
    // ------------- Colour paths: shared PCA seed ---------------------------
    vec4 mean = vec4(isum) / 16.0;
    float lmax = opaque ? 7.0 : 3.0;
    uint wmax = opaque ? 7u : 3u;

    ivec4 e0 = lo;
    ivec4 e1 = hi;
    // 8 iterations for opaque blocks, 4 for translucent (their refit
    // absorbs residual axis error — see astc4x4_fast_f16.wgsl).
    vec4 axis = principalAxis(mean, vec4(hi - lo), opaque ? 8 : 4);
    if (dot(axis, axis) > 0.0) {
      float tMin = 1e30;
      float tMax = -1e30;
      for (int k = 0; k < 16; k++) {
        float t = dot(vec4(gPixels[k]) - mean, axis);
        tMin = min(tMin, t);
        tMax = max(tMax, t);
      }
      e0 = ivec4(clamp(floor(mean + tMin * axis + 0.5), vec4(0.0), vec4(255.0)));
      e1 = ivec4(clamp(floor(mean + tMax * axis + 0.5), vec4(0.0), vec4(255.0)));
    }
    if (opaque) {
      // CEM 8 ships the quantised PCA extents directly (no LSQ fit — see
      // astc4x4_fast_f16.wgsl for the measured trade), bbox-clamped like
      // the fit output; one assignment pass fills gIdx for the packer.
      e0 = clamp(e0, lo, hi);
      e1 = clamp(e1, lo, hi);
      projAssign(e0, e1, lmax, false);
    } else {
      Fit r = projAssign(e0, e1, lmax, true);
      if (r.valid) {
        // Clamp the refit to the block bbox: on multi-cluster blocks the
        // unconstrained LSQ solve extrapolates far outside the block's
        // colours and the per-channel [0,255] clamp then bends the hue —
        // fringe pixels decode to colours that exist nowhere in the block.
        // gIdx keeps the fit-pass weights (assigned against the seed line)
        // rather than reassigning against the refit endpoints — see
        // astc4x4_fast_f16.wgsl for the measured trade.
        e0 = clamp(r.e0, lo, hi);
        e1 = clamp(r.e1, lo, hi);
      }
    }

    // Endpoint ordering so the decoder doesn't apply blue contraction.
    if (e0.x + e0.y + e0.z > e1.x + e1.y + e1.z) {
      ivec4 t = e0; e0 = e1; e1 = t;
      for (int k = 0; k < 16; k++) { gIdx[k] = wmax - gIdx[k]; }
    }

    if (opaque) {
      // CEM 8: 3-bit weights, stream bit q = 3k.
      uint s0 = 0u; uint s1 = 0u;
      for (int k = 0; k < 16; k++) {
        uint w = gIdx[k];
        uint off = 3u * uint(k);
        if (off < 30u) { s0 |= (w << off); }
        else if (off == 30u) { s0 |= (w << 30u); s1 |= (w >> 2u); }
        else { s1 |= (w << (off - 32u)); }
      }
      // Mode 0x053, CEM 8 @13, endpoints R0 R1 G0 G1 B0 B1 from bit 17.
      w0 = 0x053u | (8u << 13u) | (uint(e0.x) << 17u) | (uint(e1.x) << 25u);
      w1 = (uint(e1.x) >> 7u) | (uint(e0.y) << 1u) | (uint(e1.y) << 9u)
         | (uint(e0.z) << 17u) | (uint(e1.z) << 25u);
      w2 = (uint(e1.z) >> 7u) | rev32(s1);
      w3 = rev32(s0);
    } else {
      // CEM 12: 2-bit weights, stream bit q = 2k (single stream word).
      uint s0 = 0u;
      for (int k = 0; k < 16; k++) {
        s0 |= (gIdx[k] << (2u * uint(k)));
      }
      // Mode 0x042, CEM 12 @13, endpoints R0 R1 G0 G1 B0 B1 A0 A1 from 17.
      w0 = 0x042u | (12u << 13u) | (uint(e0.x) << 17u) | (uint(e1.x) << 25u);
      w1 = (uint(e1.x) >> 7u) | (uint(e0.y) << 1u) | (uint(e1.y) << 9u)
         | (uint(e0.z) << 17u) | (uint(e1.z) << 25u);
      w2 = (uint(e1.z) >> 7u) | (uint(e0.w) << 1u) | (uint(e1.w) << 9u);
      w3 = rev32(s0);
    }
  }

  outColor = uvec4(w0, w1, w2, w3);
}
