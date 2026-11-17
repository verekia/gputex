#version 300 es
// EAC R11 (unsigned) fragment-shader encoder — WebGL2 port of eac_r11.wgsl.
//
// One fragment per 4×4 block → 8-byte EAC R11 block as 2 × u32 in
// outColor.rg (outColor.ba unused), from the source's R channel. Same
// algorithm and arithmetic as eac_r11.wgsl (see that file for the notes),
// all of it exact integer / half-integer math, so the output is
// byte-identical to the WGSL modules:
//   • targets T = round(v·2047/255), relative to the block minimum;
//   • the block's 8-bit span picks three (table, multiplier rule) candidates
//     from a fitted 41-entry table;
//   • each is scored exactly (nearest of four magnitudes per side of the
//     palette centre, outermost one clamped to [0, 2047]);
//   • the winner is re-scored at base − 1 and base + 1;
//   • 3-bit indices pack as Σ i·8^k floats, big-endian and column-major.

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
const uint QW[16] = uint[16](29u, 25u, 25u, 25u, 23u, 21u, 21u, 21u, 19u, 19u, 19u, 19u, 19u, 19u, 17u, 17u);

// Span → three candidates, 6 bits each (low first): table in bits 0..3, rule
// in bits 4..5 (0 = covering multiplier, 1 = that + 1, 2 = multiplier 0).
// Index: the 8-bit span below 32, then 8 sub-buckets per octave of span;
// entry 40 is for full-range blocks (span 255).
const uint LUT[41] = uint[41](
  0x01026u, 0x24a2du, 0x2b368u, 0x21360u, 0x2380du, 0x2380du, 0x0c74du, 0x1d0cdu,
  0x1141du, 0x0f2ddu, 0x0b0ddu, 0x0b0ddu, 0x0a203u, 0x033cbu, 0x0a3cbu, 0x013cbu,
  0x0e3cbu, 0x0f389u, 0x0d389u, 0x0d2c9u, 0x0b289u, 0x06247u, 0x05247u, 0x07101u,
  0x04081u, 0x020c1u, 0x03081u, 0x03081u, 0x02001u, 0x0f040u, 0x083c0u, 0x0f200u,
  0x0424fu, 0x0f089u, 0x00247u, 0x05244u, 0x0a042u, 0x021c1u, 0x00389u, 0x0e245u,
  0x1269bu
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

  uint h2 = uint(tmax - tmin);
  uint mid2 = uint(tmin + tmax);
  uint s8 = uint(floor(tmax * (255.0 / 2047.0) + 0.5) - floor(tmin * (255.0 / 2047.0) + 0.5));
  uint sub = (s8 >> (2u + uint(s8 >= 64u) + uint(s8 >= 128u))) & 7u;
  uint lut = LUT[s8 == 255u ? 40u : (s8 < 32u ? s8 : 32u + sub)];

  float best = 3.0e38;
  uint bt = 0u, bm = 0u, bb = 0u;
  uint wt = 0u, wm = 0u, wb = 0u;
  for (uint k = 0u; k < 5u; k++) {
    uint tab, mul, bas;
    if (k < 3u) {
      uint c = (lut >> (6u * k)) & 63u;
      tab = c & 15u;
      uint rule = c >> 4u;
      uint dv = 8u * QW[tab];
      mul = clamp((h2 + dv / 2u) / dv, 1u, 15u);
      if (rule == 1u) mul = min(mul + 1u, 15u);
      if (rule == 2u) mul = 0u;
      bas = min((mid2 + (mul == 0u ? 1u : 8u * mul)) >> 4u, 255u);
    } else {
      if (k == 3u) { wt = bt; wm = bm; wb = bb; }
      tab = wt;
      mul = wm;
      bas = k == 3u ? max(wb, 1u) - 1u : min(wb + 1u, 255u);
    }
    float halfM = mul == 0u ? 0.5 : 4.0 * float(mul);
    float cr = float(8u * bas + 4u) - halfM - tmin;
    vec4 m = QV[tab] * halfM;
    float m3p = min(m.w, 2047.0 - tmin - cr);
    float m3n = min(m.w, cr + tmin);
    float e = 0.0;
    for (int q = 0; q < 4; q++) {
      vec4 d = u[q] - cr;
      vec4 a = abs(d);
      vec4 m3 = mix(vec4(m3n), vec4(m3p), greaterThanEqual(d, vec4(0.0)));
      vec4 r = min(min(abs(a - m.x), abs(a - m.y)), min(abs(a - m.z), abs(a - m3)));
      e += dot(r, r);
    }
    if (e < best) { best = e; bt = tab; bm = mul; bb = bas; }
  }

  // Index pass against the final (table, multiplier, base).
  float halfM = bm == 0u ? 0.5 : 4.0 * float(bm);
  float cr = float(8u * bb + 4u) - halfM - tmin;
  vec4 m = QV[bt] * halfM;
  float m3p = min(m.w, 2047.0 - tmin - cr);
  float m3n = min(m.w, cr + tmin);
  vec2 pk = vec2(0.0);
  for (int q = 0; q < 4; q++) {
    vec4 d = u[q] - cr;
    vec4 a = abs(d);
    bvec4 pos = greaterThanEqual(d, vec4(0.0));
    vec4 r1 = abs(a - m.y);
    vec4 r2 = abs(a - m.z);
    vec4 r3 = abs(a - mix(vec4(m3n), vec4(m3p), pos));
    vec4 b = abs(a - m.x);
    vec4 i = mix(vec4(0.0), vec4(1.0), lessThan(r1, b));
    b = min(b, r1);
    i = mix(i, vec4(2.0), lessThan(r2, b));
    b = min(b, r2);
    i = mix(i, vec4(3.0), lessThan(r3, b));
    float p = dot(i + mix(vec4(0.0), vec4(4.0), pos), q >= 2 ? WB : WA);
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
