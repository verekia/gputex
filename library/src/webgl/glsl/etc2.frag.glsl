#version 300 es
// ETC2 RGB8 fragment-shader encoder — WebGL2 port of etc2.wgsl.
//
// One fragment per 4×4 block. Output is the 8-byte ETC2 block as 2 × u32 in
// outColor.rg (outColor.ba unused), each word byte-swapped because ETC2 is
// big-endian on the wire; the encoder reads back RGBA32UI and keeps the low
// two words per block. Same algorithm and arithmetic as etc2.wgsl (see its
// header for the design and measurements): scalar-luma selection, O(1) flip
// preselect with both flips scored for exactly-gray blocks, a two-candidate
// table search per subblock (lower neighbour skipped when both covers are
// table 0), and a closed-form planar contest with quantised corners, channel
// sums kept in the sampler's unit domain (gray blocks snap theirs back to
// exact integers). WebGL2
// has no textureGather, so each 2×2 quad is four texelFetch()es arranged in
// gather order (w=(0,0) z=(1,0) x=(0,1) y=(1,1)). Output matches etc2.wgsl
// except on rare exact decision ties, which the unit-domain sums' f32
// rounding resolves per compiler (~1 block in 20k).

precision highp float;
precision highp int;

uniform sampler2D uSrc;
uniform ivec2 uSrcSize; // original (unpadded) width, height
uniform int uFlipY;     // 1 = sample bottom-up (matches Three.js flipY)

layout(location = 0) out uvec4 outColor;

const float THR[8] = float[8](15.0, 33.0, 57.0, 82.5, 117.0, 156.0, 208.5, 345.0);
// Per-table score constants (DK, DM, C0, C1) = (b3² − a3², −2(b3 − a3),
// 4(a3² + b3²), −(a3 + b3)), with (a3, b3) = 3 × the modifier magnitudes.
const vec4 TAB[8] = vec4[8](
  vec4(540.0, -36.0, 2448.0, -30.0),
  vec4(2376.0, -72.0, 11304.0, -66.0),
  vec4(6840.0, -120.0, 33192.0, -114.0),
  vec4(14355.0, -174.0, 69588.0, -165.0),
  vec4(29484.0, -252.0, 141264.0, -234.0),
  vec4(52416.0, -336.0, 251136.0, -312.0),
  vec4(91323.0, -438.0, 443700.0, -417.0),
  vec4(281520.0, -816.0, 1285128.0, -690.0)
);

// Planar's closed-form estimate models the QUANTISED corners exactly; only
// decode's floor-rounding (±½ per sample) is unmodelled. This small bias
// keeps near-ties on the predictable ETC1 side.
const float PLANAR_FUDGE = 8.0;
// Fraction of the luma variance the flip preselect treats as absorbed.
const float KAPPA = 0.9;
const vec3 ONE3 = vec3(1.0);
const vec4 ONE4 = vec4(1.0);
// Float -> uint for exact integers in [0, 2^23): x + 2^23 holds x in its
// mantissa, so floatBitsToUint(x + MAGIC) ^ MAGIC_BITS == x, and a left
// shift by >= 8 drops the exponent bits on its own.
const float MAGIC = 8388608.0;
const uint MAGIC_BITS = 0x4B000000u;

uint bswap(uint x) {
  uint t = ((x & 0x00ff00ffu) << 8u) | ((x >> 8u) & 0x00ff00ffu);
  return (t << 16u) | (t >> 16u);
}

// One column's four index bits at bit offset `at` (unrolled by the caller:
// a runtime column index would make col[] indexable memory).
uint indexBits(bvec4 b, uint at) {
  return ((b.x ? 1u : 0u) | (b.y ? 2u : 0u) | (b.z ? 4u : 0u) | (b.w ? 8u : 0u)) << at;
}

float max4(vec4 v) {
  return max(max(v.x, v.y), max(v.z, v.w));
}

// Base colours from subblock SUMS (8 texels each, unit domain: 31·255/2040
// = 3.875, 15·255/2040 = 1.875): codes (as floats) and their 8-bit
// expansions. Differential mode when the 5-bit codes are within the 3-bit
// delta range, else individual 4-bit.
struct Bases { vec3 c0; vec3 c1; vec3 b0; vec3 b1; bool diff; };
Bases quantiseBases(vec3 sum0, vec3 sum1) {
  vec3 q0 = floor(sum0 * 3.875 + 0.5);
  vec3 q1 = floor(sum1 * 3.875 + 0.5);
  vec3 d = q1 - q0;
  Bases o;
  o.diff = all(greaterThanEqual(d, vec3(-4.0))) && all(lessThanEqual(d, vec3(3.0)));
  vec3 i0 = floor(sum0 * 1.875 + 0.5);
  vec3 i1 = floor(sum1 * 1.875 + 0.5);
  o.c0 = o.diff ? q0 : i0;
  o.c1 = o.diff ? q1 : i1;
  float k = o.diff ? 8.25 : 17.0;
  o.b0 = floor(o.c0 * k);
  o.b1 = floor(o.c1 * k);
  return o;
}

// Subblock error (×3) of table t under the threshold rule, in min form
// summed as 4(a3² + b3²) − (a3 + b3)·sad − ½·Σ|x| (min(0, x) = (x − |x|)/2).
float tableScore(vec4 au, vec4 av, float sad, uint t) {
  vec4 k = TAB[int(t)];
  vec4 xu = au * k.y + k.x;
  vec4 xv = av * k.y + k.x;
  return k.w * sad + k.z - 0.5 * dot(abs(xu) + abs(xv), ONE4);
}

// First table whose large modifier reaches mx — a binary search over the 7
// thresholds.
uint coverTable(float mx) {
  bool s1 = mx > 126.0;
  bool s2 = mx > (s1 ? 240.0 : 51.0);
  bool s3 = mx > (s1 ? (s2 ? 318.0 : 180.0) : (s2 ? 87.0 : 24.0));
  return (s1 ? 4u : 0u) + (s2 ? 2u : 0u) + (s3 ? 1u : 0u);
}

// Both subblocks of one flip (lumas u, v against base luma lb): cover
// tables and their scores, then the lower neighbours behind ONE branch —
// skipped when both covers are table 0 (the lower neighbour IS the cover).
struct PairOut { uint t0; uint t1; float acc; };
PairOut sbPair(vec4 u0, vec4 v0, float lb0, vec4 u1, vec4 v1, float lb1) {
  vec4 au0 = abs(u0 - lb0);
  vec4 av0 = abs(v0 - lb0);
  vec4 au1 = abs(u1 - lb1);
  vec4 av1 = abs(v1 - lb1);
  float sad0 = dot(au0 + av0, ONE4);
  float sad1 = dot(au1 + av1, ONE4);
  uint c0 = coverTable(max(max4(au0), max4(av0)));
  uint c1 = coverTable(max(max4(au1), max4(av1)));
  float hi0 = tableScore(au0, av0, sad0, c0);
  float hi1 = tableScore(au1, av1, sad1, c1);
  PairOut o = PairOut(c0, c1, hi0 + hi1);
  if (c0 != 0u || c1 != 0u) {
    uint l0 = max(c0, 1u) - 1u;
    uint l1 = max(c1, 1u) - 1u;
    float lo0 = tableScore(au0, av0, sad0, l0);
    float lo1 = tableScore(au1, av1, sad1, l1);
    bool w0 = lo0 <= hi0;
    bool w1 = lo1 <= hi1;
    o.t0 = w0 ? l0 : c0;
    o.t1 = w1 ? l1 : c1;
    o.acc = (w0 ? lo0 : hi0) + (w1 ? lo1 : hi1);
  }
  return o;
}

// One flip's fit: base quantisation + table search, and its estimate
// (Σ||p||² omitted; sums in the unit domain, hence 2·255 = 510).
struct FlipFit { float est; Bases bases; float lb0; float lb1; uint t0; uint t1; };
FlipFit fitFlip(vec4 s0u, vec4 s0v, vec4 s1u, vec4 s1v, vec3 sum0, vec3 sum1) {
  FlipFit o;
  o.bases = quantiseBases(sum0, sum1);
  vec3 b0 = o.bases.b0;
  vec3 b1 = o.bases.b1;
  o.lb0 = b0.r + b0.g + b0.b;
  o.lb1 = b1.r + b1.g + b1.b;
  PairOut pp = sbPair(s0u, s0v, o.lb0, s1u, s1v, o.lb1);
  o.t0 = pp.t0;
  o.t1 = pp.t1;
  o.est = dot(b0, 8.0 * b0 - 510.0 * sum0) + dot(b1, 8.0 * b1 - 510.0 * sum1) + pp.acc * (1.0 / 3.0);
  return o;
}

// fitFlip for exactly-gray blocks (r = g = b): the same arithmetic on one
// channel; sum0/sum1 are one channel's subblock sums, as exact integers.
FlipFit fitGray(vec4 s0u, vec4 s0v, vec4 s1u, vec4 s1v, float sum0, float sum1) {
  FlipFit o;
  float q0 = floor(sum0 * (31.0 / 2040.0) + 0.5);
  float q1 = floor(sum1 * (31.0 / 2040.0) + 0.5);
  float d = q1 - q0;
  bool diff = d >= -4.0 && d <= 3.0;
  float i0 = floor(sum0 * (15.0 / 2040.0) + 0.5);
  float i1 = floor(sum1 * (15.0 / 2040.0) + 0.5);
  float c0 = diff ? q0 : i0;
  float c1 = diff ? q1 : i1;
  o.bases.diff = diff;
  o.bases.c0 = vec3(c0);
  o.bases.c1 = vec3(c1);
  float k = diff ? 8.25 : 17.0;
  float b0 = floor(c0 * k);
  float b1 = floor(c1 * k);
  o.bases.b0 = vec3(b0);
  o.bases.b1 = vec3(b1);
  o.lb0 = 3.0 * b0;
  o.lb1 = 3.0 * b1;
  PairOut pp = sbPair(s0u, s0v, o.lb0, s1u, s1v, o.lb1);
  o.t0 = pp.t0;
  o.t1 = pp.t1;
  o.est = 3.0 * (b0 * (8.0 * b0 - 2.0 * sum0) + b1 * (8.0 * b1 - 2.0 * sum1)) + pp.acc * (1.0 / 3.0);
  return o;
}

vec3 fetchRGB(ivec2 p) {
  int sy = (uFlipY != 0) ? (uSrcSize.y - 1 - p.y) : p.y;
  return texelFetch(uSrc, ivec2(p.x, sy), 0).rgb;
}

// Blocks straddling the edge of a non-multiple-of-4 image: texel loads
// clamped to the last real texel.
vec3 fetchRGBClamped(ivec2 p) {
  return fetchRGB(min(p, uSrcSize - ivec2(1)));
}

// One 2×2 quad at top-left p: per-texel luma (gather order, exact 0..765),
// unit-domain channel sums, and the sums of its right column and bottom row
// (the planar moments' local parts), and whether R = G = B in all four texels.
struct Quad { vec4 l; vec3 s; vec3 right; vec3 bottom; bool gray; };
Quad makeQuad(vec3 cw, vec3 cz, vec3 cx, vec3 cy) {
  vec4 r = vec4(cx.r, cy.r, cz.r, cw.r);
  vec4 g = vec4(cx.g, cy.g, cz.g, cw.g);
  vec4 b = vec4(cx.b, cy.b, cz.b, cw.b);
  Quad o;
  // Rounded: GLSL ES 3.00 has no fma(), and fast-math may factor the three
  // exact products into (r + g + b)·255, which is not exact.
  o.l = roundEven((r + g + b) * 255.0);
  o.right = vec3(r.z + r.y, g.z + g.y, b.z + b.y);
  o.s = o.right + vec3(r.w + r.x, g.w + g.x, b.w + b.x);
  o.bottom = vec3(r.x + r.y, g.x + g.y, b.x + b.y);
  o.gray = all(equal(r, g)) && all(equal(g, b));
  return o;
}
Quad fetchQuad(ivec2 p) {
  return makeQuad(fetchRGB(p), fetchRGB(p + ivec2(1, 0)), fetchRGB(p + ivec2(0, 1)), fetchRGB(p + ivec2(1, 1)));
}
Quad fetchQuadClamped(ivec2 p) {
  return makeQuad(
    fetchRGBClamped(p),
    fetchRGBClamped(p + ivec2(1, 0)),
    fetchRGBClamped(p + ivec2(0, 1)),
    fetchRGBClamped(p + ivec2(1, 1))
  );
}

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 4;

  // Luma by column: col[x][y]. Quadrant q = (x >= 2) | (y >= 2) << 1.
  vec4 col[4];
  vec3 qsum[4];
  // Planar right-hand sides: Σ x·p and Σ y·p (all sums in the unit domain).
  vec3 sxp = vec3(0.0);
  vec3 syp = vec3(0.0);
  // Exactly gray: R = G = B in every texel — then R and B planar corners
  // coincide (same 6-bit code).
  bool gray;
  if (base.x + 4 <= uSrcSize.x && base.y + 4 <= uSrcSize.y) {
    Quad q0 = fetchQuad(base);
    Quad q1 = fetchQuad(base + ivec2(2, 0));
    Quad q2 = fetchQuad(base + ivec2(0, 2));
    Quad q3 = fetchQuad(base + ivec2(2, 2));
    qsum[0] = q0.s;
    qsum[1] = q1.s;
    qsum[2] = q2.s;
    qsum[3] = q3.s;
    sxp = (q0.right + q1.right) + (q2.right + q3.right) + 2.0 * (q1.s + q3.s);
    syp = (q0.bottom + q1.bottom) + (q2.bottom + q3.bottom) + 2.0 * (q2.s + q3.s);
    col[0] = vec4(q0.l.w, q0.l.x, q2.l.w, q2.l.x);
    col[1] = vec4(q0.l.z, q0.l.y, q2.l.z, q2.l.y);
    col[2] = vec4(q1.l.w, q1.l.x, q3.l.w, q3.l.x);
    col[3] = vec4(q1.l.z, q1.l.y, q3.l.z, q3.l.y);
    gray = q0.gray && q1.gray && q2.gray && q3.gray;
  } else {
    // Edge blocks: the same quads from clamped loads (no runtime-indexed
    // col/qsum writes).
    Quad q0 = fetchQuadClamped(base);
    Quad q1 = fetchQuadClamped(base + ivec2(2, 0));
    Quad q2 = fetchQuadClamped(base + ivec2(0, 2));
    Quad q3 = fetchQuadClamped(base + ivec2(2, 2));
    qsum[0] = q0.s;
    qsum[1] = q1.s;
    qsum[2] = q2.s;
    qsum[3] = q3.s;
    sxp = (q0.right + q1.right) + (q2.right + q3.right) + 2.0 * (q1.s + q3.s);
    syp = (q0.bottom + q1.bottom) + (q2.bottom + q3.bottom) + 2.0 * (q2.s + q3.s);
    col[0] = vec4(q0.l.w, q0.l.x, q2.l.w, q2.l.x);
    col[1] = vec4(q0.l.z, q0.l.y, q2.l.z, q2.l.y);
    col[2] = vec4(q1.l.w, q1.l.x, q3.l.w, q3.l.x);
    col[3] = vec4(q1.l.z, q1.l.y, q3.l.z, q3.l.y);
    gray = q0.gray && q1.gray && q2.gray && q3.gray;
  }

  vec3 total = (qsum[0] + qsum[1]) + (qsum[2] + qsum[3]);
  // Right and bottom halves (subblock 1 of flip 0 / flip 1).
  vec3 right = qsum[1] + qsum[3];
  vec3 bottom = qsum[2] + qsum[3];

  float planarEst;
  vec3 qo;
  vec3 qh;
  vec3 qv;
  uint bflip = 0u;
  FlipFit sel;
  if (gray) {
    // Gray blocks resolve exact est ties (the two flips often tie), so their
    // five unit-domain scalars are snapped back to the exact integers.
    float tr = roundEven(total.r * 255.0);
    float r1 = roundEven(right.r * 255.0);
    float b1 = roundEven(bottom.r * 255.0);
    // Planar on two channels: R and B share the 6-bit solve.
    float rB = roundEven(sxp.r * 255.0) * 0.25;
    float rC = roundEven(syp.r * 255.0) * 0.25;
    float rA = tr - rB - rC;
    float po = 0.2875 * rA - 0.0125 * rB - 0.0125 * rC;
    float ph = -0.0125 * rA + 0.4875 * rB - 0.3125 * rC;
    float pv = -0.0125 * rA - 0.3125 * rB + 0.4875 * rC;
    vec2 pmax = vec2(63.0, 127.0);
    vec2 qo2 = clamp(floor(po * (pmax / 255.0) + 0.5), vec2(0.0), pmax);
    vec2 qh2 = clamp(floor(ph * (pmax / 255.0) + 0.5), vec2(0.0), pmax);
    vec2 qv2 = clamp(floor(pv * (pmax / 255.0) + 0.5), vec2(0.0), pmax);
    vec2 xk = vec2(4.0625, 2.015625);
    vec2 eo = floor(qo2 * xk);
    vec2 eh = floor(qh2 * xk);
    vec2 ev = floor(qv2 * xk);
    vec2 gram = 3.5 * (eo * eo + eh * eh + ev * ev) + 0.5 * eo * (eh + ev) + 4.5 * eh * ev;
    vec2 pe = gram - 2.0 * (eo * rA + eh * rB + ev * rC);
    planarEst = 2.0 * pe.x + pe.y + PLANAR_FUDGE;
    qo = qo2.xyx;
    qh = qh2.xyx;
    qv = qv2.xyx;

    float sum1a = r1;
    float sum0a = tr - r1;
    float sum1b = b1;
    float sum0b = tr - b1;
    sel = fitGray(col[0], col[1], col[2], col[3], sum0a, sum1a);
    FlipFit alt = fitGray(
      vec4(col[0].xy, col[1].xy),
      vec4(col[2].xy, col[3].xy),
      vec4(col[0].zw, col[1].zw),
      vec4(col[2].zw, col[3].zw),
      sum0b,
      sum1b
    );
    if (alt.est < sel.est) {
      sel = alt;
      bflip = 1u;
    }
  } else {
    // LSQ plane in closed form (constant inverse Gram matrix), estimated with
    // the quantised, clamped corners: −2·θ·rhs + θᵀGθ (unit domain: the
    // corner clamp is a saturate, rhs is scaled back by 255).
    vec3 rB = sxp * 0.25;
    vec3 rC = syp * 0.25;
    vec3 rA = total - rB - rC;
    vec3 po = 0.2875 * rA - 0.0125 * rB - 0.0125 * rC;
    vec3 ph = -0.0125 * rA + 0.4875 * rB - 0.3125 * rC;
    vec3 pv = -0.0125 * rA - 0.3125 * rB + 0.4875 * rC;
    vec3 pmax = vec3(63.0, 127.0, 63.0);
    qo = floor(clamp(po, 0.0, 1.0) * pmax + 0.5);
    qh = floor(clamp(ph, 0.0, 1.0) * pmax + 0.5);
    qv = floor(clamp(pv, 0.0, 1.0) * pmax + 0.5);
    // 6-bit expand (q<<2)|(q>>4) = floor(4.0625·q); 7-bit (q<<1)|(q>>6) = floor(2.015625·q).
    vec3 xk = vec3(4.0625, 2.015625, 4.0625);
    vec3 eo = floor(qo * xk);
    vec3 eh = floor(qh * xk);
    vec3 ev = floor(qv * xk);
    vec3 mA = -510.0 * rA;
    vec3 mB = -510.0 * rB;
    vec3 mC = -510.0 * rC;
    vec3 pe = eo * (3.5 * eo + 0.5 * (eh + ev) + mA) + eh * (3.5 * eh + 4.5 * ev + mB) + ev * (3.5 * ev + mC);
    planarEst = dot(pe, ONE3) + PLANAR_FUDGE;

    // Flip 0 splits columns (subblock 1 = right half), flip 1 splits rows
    // (subblock 1 = bottom half). Per flip, the preselect residual minus
    // flip-independent terms, in half-difference form: (κ·δ² − 3·||Δ||²)/48
    // with Δ = s0 − s1, δ = Σ_c Δ_c.
    vec3 da = total - 2.0 * right;
    vec3 db = total - 2.0 * bottom;
    float la = dot(da, ONE3);
    float lb = dot(db, ONE3);
    float resA = KAPPA * la * la - 3.0 * dot(da, da);
    float resB = KAPPA * lb * lb - 3.0 * dot(db, db);
    bool fb = resB < resA;
    bflip = fb ? 1u : 0u;
    vec3 sum1 = fb ? bottom : right;
    // sbPair is order-blind within a subblock: only two half-column pairs
    // swap between the flips.
    vec4 cz = vec4(col[0].zw, col[1].zw);
    vec4 cx = vec4(col[2].xy, col[3].xy);
    sel = fitFlip(
      vec4(col[0].xy, col[1].xy),
      fb ? cx : cz,
      vec4(col[2].zw, col[3].zw),
      fb ? cz : cx,
      total - sum1,
      sum1
    );
  }

  // ------------------------------------------------------------ packing --
  uint hi;
  uint lo;
  if (sel.est <= planarEst) {
    uint t0 = sel.t0;
    uint t1 = sel.t1;
    // Per channel byte: differential = base5 << 3 | (delta & 7), individual
    // = base4a << 4 | base4b — built in float, converted once (MAGIC).
    bool dfl = sel.bases.diff;
    vec3 dd = sel.bases.c1 - sel.bases.c0;
    vec3 ddw = mix(dd, dd + 8.0, lessThan(dd, vec3(0.0)));
    vec3 low = dfl ? ddw : sel.bases.c1;
    uvec3 bytes = floatBitsToUint(sel.bases.c0 * (dfl ? 8.0 : 16.0) + low + MAGIC);
    hi = (bytes.r << 24u) | (bytes.g << 16u) | (bytes.b << 8u) | (t0 << 5u) | (t1 << 2u) | (dfl ? 2u : 0u) | bflip;
    // Wire indices, column by column (bit x·4 + y): flip 0 gives columns
    // 0,1 subblock 0; flip 1 gives rows 0,1 (lanes x, y) subblock 0.
    // LSB = large modifier, MSB = negative.
    bool fb = bflip == 1u;
    float lb0 = sel.lb0;
    float lb1 = sel.lb1;
    float th0 = THR[int(t0)];
    float th1 = THR[int(t1)];
    vec4 lbRows = vec4(lb0, lb0, lb1, lb1);
    vec4 thRows = vec4(th0, th0, th1, th1);
    vec4 lbL = fb ? lbRows : vec4(lb0);
    vec4 lbR = fb ? lbRows : vec4(lb1);
    vec4 thL = fb ? thRows : vec4(th0);
    vec4 thR = fb ? thRows : vec4(th1);
    vec4 d0 = col[0] - lbL;
    vec4 d1 = col[1] - lbL;
    vec4 d2 = col[2] - lbR;
    vec4 d3 = col[3] - lbR;
    uint lsb = indexBits(greaterThan(abs(d0), thL), 0u) | indexBits(greaterThan(abs(d1), thL), 4u)
             | indexBits(greaterThan(abs(d2), thR), 8u) | indexBits(greaterThan(abs(d3), thR), 12u);
    uint msb = indexBits(lessThan(d0, vec4(0.0)), 0u) | indexBits(lessThan(d1, vec4(0.0)), 4u)
             | indexBits(lessThan(d2, vec4(0.0)), 8u) | indexBits(lessThan(d3, vec4(0.0)), 12u);
    lo = lsb | (msb << 16u);
  } else {
    uvec4 pi = floatBitsToUint(vec4(qo, qh.r) + MAGIC) ^ uvec4(MAGIC_BITS);
    uint ro = pi.x; uint go = pi.y; uint bo = pi.z; uint rh = pi.w;
    // ETC1-view overflow fixes. R and G fields read as base + signed 3-bit
    // delta (x >> 3, x & 7) must stay in [0, 31]: signed3(v) = (v ^ 4) − 4.
    uint xr = (ro << 1u) | (go >> 6u);
    uint xg = ((go & 63u) << 1u) | (bo >> 5u);
    uint rFix = (xr >> 3u) + ((xr & 7u) ^ 4u) < 4u ? 0x80000000u : 0u;
    uint gFix = (xg >> 3u) + ((xg & 7u) ^ 4u) < 4u ? 0x800000u : 0u;
    // B must overflow: bits 47-45 = 111 with bit 42 = 0 when p + q >= 4,
    // else 000 with bit 42 = 1.
    uint bFix = ((bo >> 3u) & 3u) + ((bo >> 1u) & 3u) >= 4u ? 0xE000u : 0x400u;
    // go + (go & 64) moves GO bit 6 up one place (bit 24; bit 23 is gFix);
    // rh + (rh & 62) spreads RH around the diff bit.
    hi = rFix | (ro << 25u) | ((go + (go & 64u)) << 17u) | gFix
       | ((bo & 32u) << 11u) | ((bo & 24u) << 8u) | ((bo & 7u) << 7u) | bFix
       | (rh + (rh & 62u)) | 2u;
    // GH·2^25 | BH·2^19 | RV·2^13 | GV·2^6 | BV: two exact float field sums.
    lo = (floatBitsToUint(qh.g * 64.0 + qh.b + MAGIC) << 19u)
       | (floatBitsToUint((qv.r * 128.0 + qv.g) * 64.0 + qv.b + MAGIC) ^ MAGIC_BITS);
  }

  outColor = uvec4(bswap(hi), bswap(lo), 0u, 0u);
}
