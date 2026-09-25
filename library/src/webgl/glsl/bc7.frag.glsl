#version 300 es
// BC7 (BPTC) mode-6 / mode-4 fragment-shader encoder — WebGL2 port of
// bc7.wgsl.
//
// One fragment per 4×4 block → 16-byte block as 4 × u32 in outColor. Per
// block the covariance picks mode 6 (one RGBA line, 16 levels) or mode 4
// (one channel split off into its own scalar plane, the other three on a
// line; 2-bit + 3-bit index sets), both then encoded by the same extents +
// stored-projection index passes. Gray + opaque blocks take an integer 1-D
// tail: lossless for spans ≤ 15 with odd endpoints (alpha exactly 255),
// alpha-aware scalar LSQ refit above. See bc7_fast_f16.wgsl for the design,
// the mode-decision model, the bit layouts and the measurements.
//
// Same arithmetic, in the same order, as bc7.wgsl — the two produce the
// same blocks.

precision highp float;
precision highp int;

uniform sampler2D uSrc;
uniform ivec2 uSrcSize;
uniform int uFlipY;

layout(location = 0) out uvec4 outColor;

// Mode-4 endpoint-precision charge (8-bit² covariance units).
const float N4 = 38.1;

// Per-invocation scratch (mirrors the WGSL function-scope arrays).
vec4 gPixels[16];
float gT[16];

// Nibble-slot compaction for the mode-4 index fields.
uint compact2(uint x) {
  uint y = (x | (x >> 2u)) & 0x0F0F0F0Fu;
  y = (y | (y >> 4u)) & 0x00FF00FFu;
  return (y | (y >> 8u)) & 0x0000FFFFu;
}
uint compact3(uint x) {
  uint y = (x & 0x07070707u) | ((x >> 1u) & 0x38383838u);
  y = (y & 0x003F003Fu) | ((y >> 2u) & 0x0FC00FC0u);
  return (y & 0x00000FFFu) | ((y >> 4u) & 0x00FFF000u);
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
      float v = gPixels[k].x;
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
      float sg = clamp(floor(gPixels[k].x * k1 + k0), 0.0, 15.0);
      glo |= uint(sg) << uint(k * 4);
    }
    for (int k = 8; k < 16; k++) {
      float sg = clamp(floor(gPixels[k].x * k1 + k0), 0.0, 15.0);
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
  ivec2 mx = uSrcSize - ivec2(1);

  // Load pass: texels (8-bit integer domain), bbox and the gray test.
  vec4 lo = vec4(255.0);
  vec4 hi = vec4(0.0);
  float gd = 0.0;
  ivec4 xs = min(ivec4(base.x) + ivec4(0, 1, 2, 3), ivec4(mx.x));
  ivec4 ys = min(ivec4(base.y) + ivec4(0, 1, 2, 3), ivec4(mx.y));
  for (int i = 0; i < 16; i++) {
    int py = ys[i >> 2];
    int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - py) : py;
    vec4 px = clamp(floor(texelFetch(uSrc, ivec2(xs[i & 3], sy), 0) * 255.0 + 0.5), vec4(0.0), vec4(255.0));
    gPixels[i] = px;
    lo = min(lo, px);
    hi = max(hi, px);
    gd = max(gd, max(abs(px.x - px.y), abs(px.x - px.z)));
  }

  if (lo.w == 255.0 && gd == 0.0) {
    outColor = encodeGray(lo.x, hi.x);
    return;
  }

  // Covariance (10 symmetric products) over d = px − p0, stored back into
  // gPixels: every later pass works block-relative.
  vec4 p0v = gPixels[0];
  gPixels[0] = vec4(0.0);
  vec4 sd = vec4(0.0);
  vec4 cx = vec4(0.0);
  vec3 cy = vec3(0.0);
  vec2 cz = vec2(0.0);
  float cw = 0.0;
  for (int i = 1; i < 16; i++) {
    vec4 d = gPixels[i] - p0v;
    gPixels[i] = d;
    sd = sd + d;
    cx = cx + d.x * d;
    cy = cy + d.y * d.yzw;
    cz = cz + d.z * d.zw;
    cw = cw + d.w * d.w;
  }
  vec4 md = sd * (1.0 / 16.0);
  vec4 sd4 = sd * 0.25;
  cx = cx - sd4.x * sd4;
  cy = cy - sd4.y * sd4.yzw;
  cz = cz - sd4.z * sd4.zw;
  cw = cw - sd4.w * sd4.w;
  vec4 diag = vec4(cx.x, cy.x, cz.x, cw);
  float trace = diag.x + diag.y + diag.z + diag.w;

  // Mode decision + fit axis. Flat blocks keep axis 0.
  bool use4 = false;
  bool idx1 = false;
  uint ch = 0u;
  vec4 cmask = vec4(1.0);
  vec4 axisF = vec4(0.0);
  if (trace > 0.254) {
    float s = 1.0 / trace;
    vec4 m0 = cx * s;
    vec4 m1 = vec4(cx.y, cy) * s;
    vec4 m2 = vec4(cx.z, cy.y, cz) * s;
    vec4 m3 = vec4(cx.w, cy.z, cz.y, cw) * s;
    vec4 axis = m0;
    float dm = diag.x;
    if (diag.y > dm) { axis = m1; dm = diag.y; }
    if (diag.z > dm) { axis = m2; dm = diag.z; }
    if (diag.w > dm) { axis = m3; }
    axis = axis * inversesqrt(dot(axis, axis));
    axis = vec4(dot(m0, axis), dot(m1, axis), dot(m2, axis), dot(m3, axis));
    axis = vec4(dot(m0, axis), dot(m1, axis), dot(m2, axis), dot(m3, axis));
    float a4 = max(dot(axis, axis), 1e-12);
    axis = axis * inversesqrt(a4);
    float lamn = sqrt(sqrt(a4));
    float lam = lamn * trace;

    // λ3 for all four scalar-channel candidates at once (lane c).
    vec4 a2 = axis * axis;
    vec4 nn = vec4(0.0);
    { vec4 wr = lamn * axis.x - axis * m0; nn = nn + wr * wr * vec4(0.0, 1.0, 1.0, 1.0); }
    { vec4 wr = lamn * axis.y - axis * m1; nn = nn + wr * wr * vec4(1.0, 0.0, 1.0, 1.0); }
    { vec4 wr = lamn * axis.z - axis * m2; nn = nn + wr * wr * vec4(1.0, 1.0, 0.0, 1.0); }
    { vec4 wr = lamn * axis.w - axis * m3; nn = nn + wr * wr * vec4(1.0, 1.0, 1.0, 0.0); }
    vec4 l3v = sqrt(nn / max(vec4(1.0) - a2, vec4(1e-3))) * trace;
    float d1 = max(max(diag.x, diag.y), max(diag.z, diag.w));
    bvec4 oh1 = equal(diag, vec4(d1));
    vec4 dr = mix(diag, vec4(-1.0), oh1);
    float d2 = max(max(dr.x, dr.y), max(dr.z, dr.w));
    l3v = clamp(l3v, mix(vec4(d1), vec4(d2), oh1), vec4(trace) - diag);

    vec4 sb = max(l3v, diag) * (48.0 / 49.0) + min(l3v, diag) * (8.0 / 9.0);
    float smax = max(max(sb.x, sb.y), max(sb.z, sb.w));
    use4 = smax - N4 > lam * (224.0 / 225.0);
    if (sb.y == smax) { ch = 1u; }
    if (sb.z == smax) { ch = 2u; }
    if (sb.w == smax) { ch = 3u; }
    vec4 ohc = mix(vec4(0.0), vec4(1.0), equal(uvec4(ch), uvec4(0u, 1u, 2u, 3u)));
    idx1 = dot(l3v - diag, ohc) > 0.0;

    vec4 col = ch == 3u ? m3 : (ch == 2u ? m2 : (ch == 1u ? m1 : m0));
    cmask = use4 ? vec4(1.0) - ohc : vec4(1.0);
    vec4 v = use4 ? (lamn * axis - dot(axis, ohc) * col) * cmask + (hi - lo) * cmask * (1e-3 / 255.0) : axis;
    v = v * inversesqrt(max(dot(v, v), 1e-12));
    v = vec4(dot(m0, v), dot(m1, v), dot(m2, v), dot(m3, v)) * cmask;
    float vv = dot(v, v);
    axisF = vv > 1e-6 ? v * inversesqrt(max(vv, 1e-12)) : vec4(0.0);
  }

  // Mode-4 scalar plane: 6-bit codes at the channel's exact extremes, index
  // map v = ⌊d·ks + os⌋. chs = 0 leaves ks = 0 for mode 6.
  vec4 chs = vec4(1.0) - cmask;
  float Ls = idx1 ? 3.0 : 7.0;
  uint A0 = uint(floor(dot(lo, chs) * (63.0 / 255.0) + 0.5));
  uint A1 = uint(floor(dot(hi, chs) * (63.0 / 255.0) + 0.5));
  vec4 ks = vec4(0.0);
  float os = 0.0;
  {
    float d0a = float((A0 << 2u) | (A0 >> 4u));
    float d1a = float((A1 << 2u) | (A1 >> 4u));
    float aspan = d1a - d0a;
    if (aspan > 0.0) {
      float sca = Ls / aspan;
      ks = chs * sca;
      os = (dot(p0v, chs) - d0a) * sca + 0.5;
    }
    // Anchor rule up front: pixel 0 (the d-space origin) indexes at ⌊os⌋.
    if (floor(os) >= (Ls + 1.0) * 0.5) {
      uint t = A0; A0 = A1; A1 = t;
      ks = -ks;
      os = Ls + 1.0 - os;
    }
  }

  // ONE extents pass along the fit axis: projections kept for the colour
  // indices, scalar-plane indices ride along as float nibble fields.
  float tMin = 1e30;
  float tMax = -1e30;
  float ga = 0.0;
  float gb = 0.0;
  float gc = 0.0;
  float w3 = 1.0;
  for (int k = 0; k < 16; k++) {
    float t = dot(gPixels[k], axisF);
    gT[k] = t;
    tMin = min(tMin, t);
    tMax = max(tMax, t);
    float v = clamp(floor(dot(gPixels[k], ks) + os), 0.0, Ls);
    if (k == 0) { v = min(v, floor(Ls * 0.5)); }
    if (k < 6) { ga = ga + v * w3; } else if (k < 12) { gb = gb + v * w3; } else { gc = gc + v * w3; }
    w3 = (k == 5 || k == 11) ? 1.0 : w3 * 16.0;
  }
  float tm = dot(md, axisF);
  vec4 mean = p0v + md;
  vec4 seedLo = clamp(mean + (tMin - tm) * axisF, vec4(0.0), vec4(255.0));
  vec4 seedHi = clamp(mean + (tMax - tm) * axisF, vec4(0.0), vec4(255.0));

  // Endpoint codes, one quantiser for both modes: mode 6 = 7-bit + p-bit,
  // mode 4 colour = 5-bit.
  float sc = use4 ? 31.0 / 255.0 : 0.5;
  float cmax = use4 ? 31.0 : 127.0;
  vec4 y0 = seedLo * sc;
  vec4 y1 = seedHi * sc;
  vec4 r0 = min(floor(y0 + 0.5), vec4(cmax));
  vec4 r1 = min(floor(y1 + 0.5), vec4(cmax));
  vec4 f0 = min(floor(y0), vec4(cmax));
  vec4 f1 = min(floor(y1), vec4(cmax));
  vec4 e0r = r0 - y0;
  vec4 e0f = f0 + 0.5 - y0;
  vec4 e1r = r1 - y1;
  vec4 e1f = f1 + 0.5 - y1;
  bool pp0 = !use4 && dot(e0f, e0f) < dot(e0r, e0r);
  bool pp1 = !use4 && dot(e1f, e1f) < dot(e1r, e1r);
  vec4 g0 = pp0 ? f0 : r0;
  vec4 g1 = pp1 ? f1 : r1;
  uvec4 q0c = uvec4(g0);
  uvec4 q1c = uvec4(g1);
  uint P0 = uint(pp0);
  uint P1 = uint(pp1);
  // Decoded 8-bit endpoints: mode 6 2q + p, mode 4 q << 3 | q >> 2.
  vec4 d0 = use4 ? g0 * 8.0 + floor(g0 * 0.25) : g0 * 2.0 + float(P0);
  vec4 d1 = use4 ? g1 * 8.0 + floor(g1 * 0.25) : g1 * 2.0 + float(P1);
  float Lc = use4 ? (idx1 ? 7.0 : 3.0) : 15.0;

  float tau0 = dot(d0 - p0v, axisF);
  float tau1 = dot(d1 - p0v, axisF);
  float span = tau1 - tau0;
  float kc = 0.0;
  float oc = 0.0;
  if (abs(span) > 0.25) {
    kc = Lc / span;
    oc = 0.5 - tau0 * kc;
  }
  // Anchor rule up front (pixel 0 projects to t = 0, index ⌊oc⌋).
  if (min(floor(oc), Lc) >= (Lc + 1.0) * 0.5) {
    uvec4 tq = q0c; q0c = q1c; q1c = tq;
    uint tp = P0; P0 = P1; P1 = tp;
    oc = 0.5 + tau1 * kc;
    kc = -kc;
  }
  float fa = 0.0;
  float fb = 0.0;
  float fc = 0.0;
  float w = 1.0;
  for (int k = 0; k < 16; k++) {
    float sg = clamp(floor(gT[k] * kc + oc), 0.0, Lc);
    if (k == 0) { sg = min(sg, floor(Lc * 0.5)); }
    if (k < 6) { fa = fa + sg * w; } else if (k < 12) { fb = fb + sg * w; } else { fc = fc + sg * w; }
    w = (k == 5 || k == 11) ? 1.0 : w * 16.0;
  }
  uint ub = uint(fb);
  uint ilo = uint(fa) | (ub << 24u);
  uint ihi = (ub >> 8u) | (uint(fc) << 16u);

  if (!use4) {
    outColor = uvec4(
      0x40u | (q0c.x << 7u) | (q1c.x << 14u) | (q0c.y << 21u) | (q1c.y << 28u),
      (q1c.y >> 4u) | (q0c.z << 3u) | (q1c.z << 10u) | (q0c.w << 17u) | (q1c.w << 24u) | (P0 << 31u),
      P1 | ((ilo & 0x7u) << 1u) | (ilo & 0xFFFFFFF0u),
      ihi
    );
  } else {
    uint vb = uint(gb);
    uint slo = uint(ga) | (vb << 24u);
    uint shi = (vb >> 8u) | (uint(gc) << 16u);
    uint c2 = compact2(idx1 ? slo : ilo) | (compact2(idx1 ? shi : ihi) << 16u);
    uint iA = compact3(idx1 ? ilo : slo);
    uint iB = compact3(idx1 ? ihi : shi);
    uint R0 = ch == 0u ? q0c.w : q0c.x;
    uint G0 = ch == 1u ? q0c.w : q0c.y;
    uint B0 = ch == 2u ? q0c.w : q0c.z;
    uint R1 = ch == 0u ? q1c.w : q1c.x;
    uint G1 = ch == 1u ? q1c.w : q1c.y;
    uint B1 = ch == 2u ? q1c.w : q1c.z;
    uint rot = (ch + 1u) & 3u;
    uint field2 = (c2 & 1u) | ((c2 >> 2u) << 1u);
    uint fLo = (iA & 3u) | ((iA >> 3u) << 2u) | (iB << 23u);
    uint fHi = iB >> 9u;
    outColor = uvec4(
      0x10u | (rot << 5u) | (uint(idx1) << 7u) | (R0 << 8u) | (R1 << 13u) | (G0 << 18u) | (G1 << 23u) | (B0 << 28u),
      (B0 >> 4u) | (B1 << 1u) | (A0 << 6u) | (A1 << 12u) | ((field2 & 0x3FFFu) << 18u),
      (field2 >> 14u) | (fLo << 17u),
      (fLo >> 15u) | (fHi << 17u)
    );
  }
}
