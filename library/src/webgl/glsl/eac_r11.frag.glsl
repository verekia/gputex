#version 300 es
// EAC R11 (unsigned) fragment-shader encoder — WebGL2 port of eac_r11.wgsl.
//
// One fragment per 4×4 block → 8-byte EAC R11 block as 2 × u32 in
// outColor.rg (outColor.ba unused), from the source's R channel. Same
// algorithm and arithmetic as eac_r11.wgsl (see that file for the notes),
// all of it exact integer / half-integer math, so the output is
// byte-identical to the WGSL modules:
//   • targets T = round(v·2047/255), relative to the block minimum;
//   • the block's 8-bit span and shape (mean distance from the mid-range)
//     pick two (table, multiplier rule) candidates from a fitted 164-entry
//     table;
//   • each is scored with the palette centred on the mid-range (the
//     magnitude whose threshold interval holds the texel's distance,
//     outermost one clamped to [0, 2047]), summing squared and signed
//     residuals;
//   • the signed sum gives the base step that zeroes the mean residual and
//     the error it predicts; the lower prediction wins at its stepped base;
//   • indices count the thresholds below each texel's distance on the final
//     palette, and pack as Σ i·8^k floats, big-endian and column-major.

precision highp float;
precision highp int;

uniform sampler2D uSrc;
uniform ivec2 uSrcSize;
uniform int uFlipY;

layout(location = 0) out uvec4 outColor;

// 2p + 1 for the four positive modifiers p of each table.
const vec4 QV[16] = vec4[16](
  vec4(5.0, 11.0, 17.0, 29.0), vec4(5.0, 13.0, 19.0, 25.0),
  vec4(3.0, 9.0, 15.0, 25.0),  vec4(3.0, 7.0, 11.0, 25.0),
  vec4(5.0, 11.0, 15.0, 23.0), vec4(5.0, 13.0, 17.0, 21.0),
  vec4(7.0, 13.0, 15.0, 21.0), vec4(5.0, 9.0, 15.0, 21.0),
  vec4(3.0, 11.0, 15.0, 19.0), vec4(3.0, 9.0, 15.0, 19.0),
  vec4(3.0, 7.0, 15.0, 19.0),  vec4(3.0, 9.0, 13.0, 19.0),
  vec4(5.0, 7.0, 13.0, 19.0),  vec4(1.0, 3.0, 5.0, 19.0),
  vec4(7.0, 11.0, 15.0, 17.0), vec4(5.0, 9.0, 13.0, 17.0)
);
// Per table, with q0..q3 the row above: the two inner thresholds
// (q0 + q1) / 2 and (q1 + q2) / 2, then 4·q3 + ½ and 1 / (8·q3) for the
// covering-multiplier rule.
const vec4 QT[16] = vec4[16](
  vec4(8.0, 14.0, 116.5, 1.0 / 232.0), vec4(9.0, 16.0, 100.5, 1.0 / 200.0),
  vec4(6.0, 12.0, 100.5, 1.0 / 200.0), vec4(5.0, 9.0, 100.5, 1.0 / 200.0),
  vec4(8.0, 13.0, 92.5, 1.0 / 184.0), vec4(9.0, 15.0, 84.5, 1.0 / 168.0),
  vec4(10.0, 14.0, 84.5, 1.0 / 168.0), vec4(7.0, 12.0, 84.5, 1.0 / 168.0),
  vec4(7.0, 13.0, 76.5, 1.0 / 152.0), vec4(6.0, 12.0, 76.5, 1.0 / 152.0),
  vec4(5.0, 11.0, 76.5, 1.0 / 152.0), vec4(6.0, 11.0, 76.5, 1.0 / 152.0),
  vec4(6.0, 10.0, 76.5, 1.0 / 152.0), vec4(2.0, 4.0, 76.5, 1.0 / 152.0),
  vec4(9.0, 13.0, 68.5, 1.0 / 136.0), vec4(7.0, 11.0, 68.5, 1.0 / 136.0)
);

// (Span, shape) → two candidates, 6 bits each (low first): table in bits
// 0..3, rule in bits 4..5 (0 = covering multiplier, 1 = that + 1,
// 2 = multiplier 0). Index = span bucket · 4 + shape bucket; span bucket =
// the 8-bit span below 32, then 8 sub-buckets per octave of span, 40 for
// full-range blocks (span 255).
const uint LUT[164] = uint[164](
  0xb63u, 0xb63u, 0xb63u, 0xb63u, 0xae8u, 0xae8u, 0xae8u, 0xae8u, 0xae4u, 0xae4u,
  0x92bu, 0x969u, 0x360u, 0x360u, 0x84du, 0x360u, 0x80du, 0x360u, 0x360u, 0x360u,
  0x80du, 0x80du, 0x360u, 0x340u, 0x342u, 0x342u, 0x74du, 0x74du, 0x74du, 0x74du,
  0x0cdu, 0x303u, 0x35du, 0x35du, 0x0ddu, 0x74cu, 0x0ddu, 0x0ddu, 0x2ddu, 0x09du,
  0x743u, 0x743u, 0x743u, 0x743u, 0x0ddu, 0x0ddu, 0x0ddu, 0x743u, 0x743u, 0x743u,
  0x743u, 0x0cfu, 0x2c3u, 0x2c3u, 0x0cbu, 0x2cfu, 0x2cau, 0x2cau, 0x2cfu, 0x3ceu,
  0x24au, 0x24au, 0x3c9u, 0x3c6u, 0x34bu, 0x289u, 0x3c9u, 0x3ceu, 0x348u, 0x2cau,
  0x24fu, 0x38fu, 0x34bu, 0x2cau, 0x24fu, 0x388u, 0x344u, 0x28bu, 0x24bu, 0x308u,
  0x34au, 0x2cdu, 0x2cau, 0x202u, 0x340u, 0x28bu, 0x2c7u, 0x1c5u, 0x2c7u, 0x2c7u,
  0x2c7u, 0x141u, 0x1c2u, 0x1c2u, 0x147u, 0x144u, 0x084u, 0x084u, 0x081u, 0x044u,
  0x0c2u, 0x0c2u, 0x081u, 0x7c1u, 0x0c2u, 0x0c2u, 0x042u, 0x081u, 0x0c2u, 0x0c2u,
  0x042u, 0x05bu, 0x080u, 0x080u, 0x081u, 0x04fu, 0x002u, 0x002u, 0x040u, 0x38fu,
  0x080u, 0x080u, 0x040u, 0x00eu, 0x002u, 0x002u, 0x200u, 0x00eu, 0x080u, 0x080u,
  0x3c9u, 0x385u, 0x2c2u, 0x2c2u, 0x089u, 0x20eu, 0x08bu, 0x08bu, 0x1c9u, 0x14eu,
  0x084u, 0x084u, 0x207u, 0x10eu, 0x042u, 0x042u, 0x081u, 0x28eu, 0x042u, 0x042u,
  0x042u, 0x14eu, 0x08bu, 0x08bu, 0x2c9u, 0x389u, 0x009u, 0x009u, 0x009u, 0x14eu,
  0x655u, 0x655u, 0x655u, 0x655u
);

// Index weights 8^k for the big-endian, column-major index field (gather
// order x,y,z,w = (0,1),(1,1),(1,0),(0,0)): WA rows 0..1, WB rows 2..3.
const vec4 WA = vec4(262144.0, 64.0, 512.0, 2097152.0);
const vec4 WB = vec4(4096.0, 1.0, 8.0, 32768.0);

float fetchR(ivec2 p, ivec2 maxXY) {
  ivec2 c = clamp(p, ivec2(0), maxXY);
  int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - c.y) : c.y;
  return texelFetch(uSrc, ivec2(c.x, sy), 0).r;
}

void main() {
  ivec2 org = ivec2(gl_FragCoord.xy) * 4;
  ivec2 maxXY = uSrcSize - ivec2(1);

  vec4 u[4];
  for (int q = 0; q < 4; q++) {
    ivec2 qo = org + ivec2((q & 1) * 2, (q >> 1) * 2);
    u[q] = floor(vec4(
      fetchR(qo + ivec2(0, 1), maxXY),
      fetchR(qo + ivec2(1, 1), maxXY),
      fetchR(qo + ivec2(1, 0), maxXY),
      fetchR(qo, maxXY)
    ) * 2047.0 + 0.5);
  }
  vec4 mn = min(min(u[0], u[1]), min(u[2], u[3]));
  vec4 mxv = max(max(u[0], u[1]), max(u[2], u[3]));
  float tmin = min(min(mn.x, mn.y), min(mn.z, mn.w));
  float tmax = max(max(mxv.x, mxv.y), max(mxv.z, mxv.w));
  for (int q = 0; q < 4; q++) u[q] -= tmin;

  float h2f = tmax - tmin;
  float mid2f = tmin + tmax;
  uint s8 = uint(floor(tmax * (255.0 / 2047.0) + 0.5) - floor(tmin * (255.0 / 2047.0) + 0.5));
  uint sub = (s8 >> (2u + uint(s8 >= 64u) + uint(s8 >= 128u))) & 7u;
  // Shape bucket: floor(4 · Σ|2T − (min + max)| / (16 · span + 1)).
  vec4 f4 = abs(u[0] + u[0] - h2f) + abs(u[1] + u[1] - h2f) + abs(u[2] + u[2] - h2f) + abs(u[3] + u[3] - h2f);
  float fx = 4.0 * (f4.x + f4.y + f4.z + f4.w);
  float fd = 16.0 * h2f + 1.0;
  uint fb = uint(fx >= fd) + uint(fx >= 2.0 * fd) + uint(fx >= 3.0 * fd);
  uint lut = LUT[(s8 == 255u ? 40u : (s8 < 32u ? s8 : 32u + sub)) * 4u + fb];

  float best = 3.0e38;
  uint bt = 0u;
  float bmf = 0.0, bbf = 0.0;
  for (uint k = 0u; k < 2u; k++) {
    uint c = (lut >> (6u * k)) & 63u;
    uint tab = c & 15u;
    uint rule = c >> 4u;
    vec4 qv = QV[tab];
    vec4 qt = QT[tab];
    float mulf = clamp(floor((h2f + qt.z) * qt.w), 1.0, 15.0);
    if (rule == 1u) mulf = min(mulf + 1.0, 15.0);
    bool fine = rule == 2u;
    float halfM = fine ? 0.5 : 4.0 * mulf;
    float basf = min(floor((mid2f + 2.0 * halfM) * 0.0625), 255.0);
    float cr = 8.0 * basf + 4.0 - halfM - tmin;
    vec4 mf = qv * halfM;
    vec2 m3f = min(vec2(mf.w), vec2(2047.0 - tmin - cr, cr + tmin));   // (+ side, − side)
    vec2 t2r = (vec2(mf.z) + m3f) * 0.5;
    vec2 t2f = fine ? t2r : floor(t2r);
    vec2 t01 = qt.xy * halfM;
    vec4 ev = vec4(0.0);
    vec4 sv = vec4(0.0);
    for (int q = 0; q < 4; q++) {
      vec4 d = u[q] - cr;
      vec4 a = abs(d);
      bvec4 pos = greaterThanEqual(d, vec4(0.0));
      vec4 ms = mix(vec4(mf.x), vec4(mf.y), greaterThan(a, vec4(t01.x)));
      ms = mix(ms, vec4(mf.z), greaterThan(a, vec4(t01.y)));
      ms = mix(ms, mix(vec4(m3f.y), vec4(m3f.x), pos), greaterThan(a, mix(vec4(t2f.y), vec4(t2f.x), pos)));
      vec4 rh = a - ms;
      vec4 rf = mix(-rh, rh, pos);
      ev += rf * rf;
      sv += rf;
    }
    float e = ev.x + ev.y + ev.z + ev.w;
    float s = sv.x + sv.y + sv.z + sv.w;
    // Base step that zeroes the mean residual, and the error it predicts.
    float nb = clamp(basf + floor((s + 64.0) * 0.0078125), 0.0, 255.0);
    float db = nb - basf;
    float ep = e - 16.0 * db * s + 1024.0 * db * db;
    if (ep < best) { best = ep; bt = tab; bmf = fine ? 0.0 : mulf; bbf = nb; }
  }
  uint bm = uint(bmf);
  uint bb = uint(bbf);

  // Index pass against the final (table, multiplier, base): the number of
  // thresholds below the texel's distance, magnitudes clamped per side.
  bool fine = bm == 0u;
  float halfM = fine ? 0.5 : 4.0 * bmf;
  float cr = 8.0 * bbf + 4.0 - halfM - tmin;
  vec4 mf = QV[bt] * halfM;
  vec4 np = min(mf, vec4(2047.0 - tmin - cr));
  vec4 nn = min(mf, vec4(cr + tmin));
  vec3 tp = (np.xyz + np.yzw) * 0.5;
  vec3 tn = (nn.xyz + nn.yzw) * 0.5;
  if (!fine) {
    tp = floor(tp);
    tn = floor(tn);
  }
  vec2 pk = vec2(0.0);
  for (int q = 0; q < 4; q++) {
    vec4 d = u[q] - cr;
    vec4 a = abs(d);
    bvec4 pos = greaterThanEqual(d, vec4(0.0));
    vec4 i = mix(vec4(0.0), vec4(4.0), pos)
      + vec4(greaterThan(a, mix(vec4(tn.x), vec4(tp.x), pos)))
      + vec4(greaterThan(a, mix(vec4(tn.y), vec4(tp.y), pos)))
      + vec4(greaterThan(a, mix(vec4(tn.z), vec4(tp.z), pos)));
    float p = dot(i, q >= 2 ? WB : WA);
    bool lo = (q & 1) == 1;
    pk += vec2(lo ? 0.0 : p, lo ? p : 0.0);
  }
  uint hi24 = uint(pk.x);
  uint lo24 = uint(pk.y);

  // Bytes 0..7 as two little-endian u32s.
  outColor = uvec4(
    bb | (((bm << 4u) | bt) << 8u) | ((hi24 >> 16u) << 16u) | (((hi24 >> 8u) & 255u) << 24u),
    (hi24 & 255u) | ((lo24 >> 16u) << 8u) | (((lo24 >> 8u) & 255u) << 16u) | ((lo24 & 255u) << 24u),
    0u,
    0u
  );
}
