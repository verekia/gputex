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
// Per-table score constants: DK = b3² − a3², DM = −2(b3 − a3), A8 = 8·a3²,
// AM = −2·a3, with (a3, b3) = 3 × the modifier magnitudes.
const float DK[8] = float[8](540.0, 2376.0, 6840.0, 14355.0, 29484.0, 52416.0, 91323.0, 281520.0);
const float DM[8] = float[8](-36.0, -72.0, -120.0, -174.0, -252.0, -336.0, -438.0, -816.0);
const float A8[8] = float[8](288.0, 1800.0, 5832.0, 12168.0, 23328.0, 41472.0, 78408.0, 159048.0);
const float AM[8] = float[8](-12.0, -30.0, -54.0, -78.0, -108.0, -144.0, -198.0, -282.0);

// Planar's closed-form estimate models the QUANTISED corners exactly; only
// decode's floor-rounding (±½ per sample) is unmodelled. This small bias
// keeps near-ties on the predictable ETC1 side.
const float PLANAR_FUDGE = 8.0;
// Fraction of the luma variance the flip preselect treats as absorbed.
const float KAPPA = 0.9;
const vec3 ONE3 = vec3(1.0);
const vec4 ONE4 = vec4(1.0);

int signed3(uint bits) {
  return bits > 3u ? int(bits) - 8 : int(bits);
}

uint bswap(uint x) {
  return ((x & 0xffu) << 24u) | ((x & 0xff00u) << 8u) | ((x >> 8u) & 0xff00u) | (x >> 24u);
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
  o.b0 = o.diff ? floor(q0 * 8.25) : i0 * 17.0;
  o.b1 = o.diff ? floor(q1 * 8.25) : i1 * 17.0;
  return o;
}

// Subblock error (×3) of table t under the threshold rule, in min form.
float tableScore(vec4 au, vec4 av, float sad, uint t) {
  int i = int(t);
  float dk = DK[i];
  float dm = DM[i];
  vec4 eu = min(vec4(0.0), au * dm + dk);
  vec4 ev = min(vec4(0.0), av * dm + dk);
  return A8[i] + AM[i] * sad + dot(eu + ev, ONE4);
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
  o.bases.diff = diff;
  o.bases.c0 = vec3(diff ? q0 : i0);
  o.bases.c1 = vec3(diff ? q1 : i1);
  float b0 = diff ? floor(q0 * 8.25) : i0 * 17.0;
  float b1 = diff ? floor(q1 * 8.25) : i1 * 17.0;
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

// One 2×2 quad at top-left p: per-texel luma (gather order, exact 0..765),
// unit-domain channel sums, and the sums of its right column and bottom row
// (the planar moments' local parts).
struct Quad { vec4 l; vec3 s; vec3 right; vec3 bottom; };
Quad fetchQuad(ivec2 p) {
  vec3 cw = fetchRGB(p);
  vec3 cz = fetchRGB(p + ivec2(1, 0));
  vec3 cx = fetchRGB(p + ivec2(0, 1));
  vec3 cy = fetchRGB(p + ivec2(1, 1));
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
  return o;
}

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 4;

  // Luma by column: col[x][y]. Quadrant q = (x >= 2) | (y >= 2) << 1.
  vec4 col[4];
  vec3 qsum[4];
  // Planar right-hand sides: Σ x·p and Σ y·p (all sums in the unit domain).
  vec3 sxp = vec3(0.0);
  vec3 syp = vec3(0.0);
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
  } else {
    // Blocks straddling the edge of a non-multiple-of-4 image: per-texel
    // loads clamped to the last real texel.
    ivec2 maxXY = uSrcSize - ivec2(1);
    for (int q = 0; q < 4; q++) qsum[q] = vec3(0.0);
    for (int i = 0; i < 16; i++) {
      int lx = i & 3;
      int ly = i >> 2;
      vec3 c = roundEven(fetchRGB(clamp(base + ivec2(lx, ly), ivec2(0), maxXY)) * 255.0);
      col[lx][ly] = c.r + c.g + c.b;
      int q = (lx >= 2 ? 1 : 0) | (ly >= 2 ? 2 : 0);
      qsum[q] += c;
      sxp += float(lx) * c;
      syp += float(ly) * c;
    }
    // Edge blocks accumulate in 0..255 units; rescale to the unit domain.
    for (int q = 0; q < 4; q++) qsum[q] *= 1.0 / 255.0;
    sxp *= 1.0 / 255.0;
    syp *= 1.0 / 255.0;
  }

  vec3 total = (qsum[0] + qsum[1]) + (qsum[2] + qsum[3]);
  // Right and bottom halves (subblock 1 of flip 0 / flip 1).
  vec3 right = qsum[1] + qsum[3];
  vec3 bottom = qsum[2] + qsum[3];
  // Exactly gray: every quadrant sum AND both planar moments equal across
  // R, G, B — then R and B planar corners coincide (same 6-bit code).
  bool gray = all(equal(qsum[0].rg, qsum[0].gb)) && all(equal(qsum[1].rg, qsum[1].gb)) &&
              all(equal(qsum[2].rg, qsum[2].gb)) && all(equal(qsum[3].rg, qsum[3].gb)) &&
              all(equal(sxp.rg, sxp.gb)) && all(equal(syp.rg, syp.gb));

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
    sel = fitFlip(
      fb ? vec4(col[0].xy, col[1].xy) : col[0],
      fb ? vec4(col[2].xy, col[3].xy) : col[1],
      fb ? vec4(col[0].zw, col[1].zw) : col[2],
      fb ? vec4(col[2].zw, col[3].zw) : col[3],
      total - sum1,
      sum1
    );
  }

  // ------------------------------------------------------------ packing --
  uint hi;
  uint lo;
  if (sel.est <= planarEst) {
    uvec3 codes0 = uvec3(sel.bases.c0);
    uvec3 codes1 = uvec3(sel.bases.c1);
    uint t0 = sel.t0;
    uint t1 = sel.t1;
    if (sel.bases.diff) {
      uvec3 d = uvec3(ivec3(codes1) - ivec3(codes0)) & uvec3(7u);
      hi = (codes0.r << 27u) | (d.r << 24u) | (codes0.g << 19u) | (d.g << 16u) | (codes0.b << 11u) | (d.b << 8u)
         | (t0 << 5u) | (t1 << 2u) | 2u | bflip;
    } else {
      hi = (codes0.r << 28u) | (codes1.r << 24u) | (codes0.g << 20u) | (codes1.g << 16u) | (codes0.b << 12u) | (codes1.b << 8u)
         | (t0 << 5u) | (t1 << 2u) | bflip;
    }
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
    uint lsb = 0u;
    uint msb = 0u;
    for (int c = 0; c < 4; c++) {
      vec4 d = col[c] - (c >= 2 ? lbR : lbL);
      bvec4 large = greaterThan(abs(d), c >= 2 ? thR : thL);
      bvec4 neg = lessThan(d, vec4(0.0));
      uint nl = (large.x ? 1u : 0u) | (large.y ? 2u : 0u) | (large.z ? 4u : 0u) | (large.w ? 8u : 0u);
      uint nn = (neg.x ? 1u : 0u) | (neg.y ? 2u : 0u) | (neg.z ? 4u : 0u) | (neg.w ? 8u : 0u);
      lsb |= nl << uint(c * 4);
      msb |= nn << uint(c * 4);
    }
    lo = lsb | (msb << 16u);
  } else {
    uint ro = uint(qo.r); uint go = uint(qo.g); uint bo = uint(qo.b);
    uint rh = uint(qh.r); uint gh = uint(qh.g); uint bh = uint(qh.b);
    uint rv = uint(qv.r); uint gv = uint(qv.g); uint bv = uint(qv.b);
    int rSum = int(ro >> 2u) + signed3(((ro & 3u) << 1u) | (go >> 6u));
    uint rFix = rSum < 0 ? 1u : 0u;
    int gSum = int((go >> 2u) & 15u) + signed3(((go & 3u) << 1u) | (bo >> 5u));
    uint gFix = gSum < 0 ? 1u : 0u;
    uint p = (bo >> 3u) & 3u;
    uint q = (bo >> 1u) & 3u;
    uint bFix3 = p + q >= 4u ? 7u : 0u;
    uint bFix1 = p + q >= 4u ? 0u : 1u;
    hi = (rFix << 31u) | (ro << 25u) | ((go >> 6u) << 24u) | (gFix << 23u) | ((go & 63u) << 17u)
       | ((bo >> 5u) << 16u) | (bFix3 << 13u) | (((bo >> 3u) & 3u) << 11u) | (bFix1 << 10u)
       | ((bo & 7u) << 7u) | ((rh >> 1u) << 2u) | 2u | (rh & 1u);
    lo = (gh << 25u) | (bh << 19u) | (rv << 13u) | (gv << 6u) | bv;
  }

  outColor = uvec4(bswap(hi), bswap(lo), 0u, 0u);
}
