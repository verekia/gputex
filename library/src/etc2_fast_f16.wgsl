// ETC2 RGB8 compute shader encoder.
//
// Each invocation encodes one 4x4 pixel block into an 8-byte ETC2 RGB8 block
// written as 2 x u32 into the destination storage buffer. ETC2 blocks are
// big-endian on the wire (byte 0 = bits 63..56), so both words are byte-
// swapped on the way out. This is the f16 module; etc2.wgsl is the f32
// fallback with the same algorithm.
//
// EXACT-VALUE f16: unlike the other formats' f16 fast paths (which accept
// float rounding in a [0,1] domain), the f16 values here are integers (or
// half-integers) that f16 represents exactly — per-texel lumas and base
// lumas (<= 765), luma deviations |D| (<= 765) and the index thresholds
// (<= 345, halves included) all sit below f16's exactness limits. Sums,
// scores and estimates stay f32 (they reach ~1e6). Where the sampler's
// unorm→float conversion is exact (verified on Apple/metal-3), the f16 and
// f32 modules are BYTE-IDENTICAL; elsewhere they can differ only on exact
// decision ties. f16 buys register space (the 16 lumas are 4 × vec4<f16>)
// and measured 1-3% faster than the f32 module on Apple.
//
// ALGORITHM — scalar-luma selection:
//
//   • The ETC1 modifier is a SCALAR shift along (1,1,1), so per texel
//     err(m) = ||e||² − 2mD + 3m² with D = luma(p) − luma(base), where
//     luma(x) = x.r+x.g+x.b. Selection therefore needs only |D| threshold
//     tests, and Σ||e||² per subblock is O(1) from the quadrant sums. The
//     block-constant Σ||p||² is dropped from EVERY estimate (ETC1 flips and
//     planar alike): only differences between estimates are ever used.
//     The estimate is exact for unclamped decode and an upper bound on the
//     true clamped error.
//   • Loads: 4 textureGather quads × R,G,B for interior blocks (the gather
//     point, normalised by the PHYSICAL texture size, sits exactly between
//     the quad's texel centres; interior quads never touch the zeroed
//     padding strip). Blocks straddling the edge of a non-multiple-of-4
//     image fall back to clamped per-texel loads. Lumas are kept as 4
//     COLUMN vectors — wire pixel order is x·4 + y — so both flips' half-
//     blocks and the index packing use only constant indexing.
//   • Flip preselect, O(1): per subblock the residual after continuous luma
//     modulation is within-variance − κ·(luma variance)/3, κ = 0.9. κ = 1
//     is the exact chroma residual; keeping a tenth of the luma variance
//     prefers the split with less luma spread for the 4-level tables to
//     cover (+0.07-0.10 dB on photo colour vs κ = 1, free). Only the chosen
//     flip is searched.
//   • Exactly-gray blocks (every quadrant's R, G and B sums AND both planar
//     moments equal) have no chroma to steer the preselect, so both flips
//     are scored — worth ~0.3 dB on roughness/AO content over any O(1)
//     proxy tried (luma variance, luma range, squared range all land at
//     −0.30 dB; deciding the flip on the cover table's score alone loses
//     0.3-0.5 dB). They use a one-channel copy of the fit (fit_gray), the
//     second flip is a separate straight-line call (the older
//     single-call-site loop cost 7-9% even on colour content that never ran
//     its second iteration), and their planar solve runs on two channels:
//     R and B share the 6-bit code, G takes the 7-bit one (−4..5% on gray
//     maps).
//     Widening the second evaluation to chroma near-ties (the previous
//     rule) cost 12-25% on colour textures through warp divergence for
//     ≤ 0.015 dB.
//   • Table search is pruned to two candidates — the table whose LARGE
//     magnitude covers max|D| and its lower neighbour (outlier hedge).
//     One candidate loses ~0.7-2.9 dB; all eight gain ≤ 0.05 dB. Scores use
//     the min form: per texel min(a3² − 2·a3·ad, b3² − 2·b3·ad) is the
//     threshold rule exactly, and its a3 part sums in closed form. A flip's
//     two subblocks are searched together (sb_pair): the lower-neighbour
//     scores sit behind ONE branch, skipped when both covers are table 0 —
//     then the lower neighbour IS the cover table. Smooth content skips it
//     wholesale (−9..16% on displacement maps at 2K/4K); a per-subblock
//     branch cost 2-4% on noisy content by splitting the score pair.
//   • NO base refit (worth ~0.2 dB on photo colour for ≥ 13% GPU).
//   • PLANAR runs unconditionally: the LSQ solve is O(1) from the block sum
//     and the first moments Σx·p, Σy·p (the Gram inverse of the fixed
//     sample positions is a constant; folding it into fewer coefficients
//     saved ~1% but resolved rounding ties unlike the CPU mirror on ~9% of
//     the colour card's blocks), and its residual is the closed form
//     −2·θ·rhs + θᵀGθ evaluated with the QUANTISED, clamped corners —
//     clamp-aware, which a continuous-corner estimate is not. Gating the
//     quantised evaluation on the continuous plane's residual (an exact
//     lower bound) is byte-identical but measured 0-1%: ~half the warps
//     still hold a block that needs it.
//   • T and H modes are decoded by hardware but never emitted — their win
//     is limited to two-chroma-cluster blocks and needs a clustering pass.
//
// Numeric notes: every m3 in A3/B3 is divisible by 3 so m = m3/3 is exact;
// est values are integer sums held exactly in f32 (< 2^24) apart from the
// planar solve's decimal weights.

enable f16;

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
  y0:       u32, // first block row of this dispatch (row-band encodes)
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var smp: sampler;

const A3  = array<f32, 8>(6.0, 15.0, 27.0, 39.0, 54.0, 72.0, 99.0, 141.0);
const B3  = array<f32, 8>(24.0, 51.0, 87.0, 126.0, 180.0, 240.0, 318.0, 549.0);
const THR = array<f32, 8>(15.0, 33.0, 57.0, 82.5, 117.0, 156.0, 208.5, 345.0);

// Planar's closed-form estimate models the QUANTISED corners exactly; only
// decode's floor-rounding (±½ per sample) is unmodelled. This small bias
// keeps near-ties on the predictable ETC1 side.
const PLANAR_FUDGE = 8.0;
// Fraction of the luma variance the flip preselect treats as absorbed.
const KAPPA = 0.9;
const ONE3 = vec3<f32>(1.0);
const ONE4 = vec4<f32>(1.0);

fn signed3(bits: u32) -> i32 {
  return select(i32(bits), i32(bits) - 8, bits > 3u);
}

fn bswap(x: u32) -> u32 {
  return ((x & 0xffu) << 24u) | ((x & 0xff00u) << 8u) | ((x >> 8u) & 0xff00u) | (x >> 24u);
}

fn max4(v: vec4<f16>) -> f16 {
  return max(max(v.x, v.y), max(v.z, v.w));
}

// Base colours from subblock SUMS (8 texels each): codes (as floats) and
// their 8-bit expansions. Differential mode when the 5-bit codes are within
// the 3-bit delta range, else individual 4-bit. Expansions in float:
// (q<<3)|(q>>2) = floor(8.25·q) for 5 bits, (q<<4)|q = 17·q for 4 bits.
struct Bases {
  c0: vec3<f32>,
  c1: vec3<f32>,
  b0: vec3<f32>,
  b1: vec3<f32>,
  diff: bool,
};
fn quantise_bases(sum0: vec3<f32>, sum1: vec3<f32>) -> Bases {
  let q0 = floor(sum0 * (31.0 / 2040.0) + 0.5);
  let q1 = floor(sum1 * (31.0 / 2040.0) + 0.5);
  let d = q1 - q0;
  var o: Bases;
  o.diff = all(d >= vec3<f32>(-4.0)) && all(d <= vec3<f32>(3.0));
  let i0 = floor(sum0 * (15.0 / 2040.0) + 0.5);
  let i1 = floor(sum1 * (15.0 / 2040.0) + 0.5);
  o.c0 = select(i0, q0, o.diff);
  o.c1 = select(i1, q1, o.diff);
  o.b0 = select(i0 * 17.0, floor(q0 * 8.25), o.diff);
  o.b1 = select(i1 * 17.0, floor(q1 * 8.25), o.diff);
  return o;
}

// Subblock error (×3) of table t under the threshold rule, in min form:
// per texel min(a3² − 2·a3·ad, b3² − 2·b3·ad) = (a3² − 2·a3·ad) +
// min(0, (b3² − a3²) − 2·(b3 − a3)·ad); the a3 part sums in closed form
// from sad = Σ ad.
// Per-table score constants: DK = b3² − a3², DM = −2(b3 − a3), A8 = 8·a3²,
// AM = −2·a3 (precomputed: −1.5% GPU over deriving them per call).
const DK = array<f32, 8>(540.0, 2376.0, 6840.0, 14355.0, 29484.0, 52416.0, 91323.0, 281520.0);
const DM = array<f32, 8>(-36.0, -72.0, -120.0, -174.0, -252.0, -336.0, -438.0, -816.0);
const A8 = array<f32, 8>(288.0, 1800.0, 5832.0, 12168.0, 23328.0, 41472.0, 78408.0, 159048.0);
const AM = array<f32, 8>(-12.0, -30.0, -54.0, -78.0, -108.0, -144.0, -198.0, -282.0);
fn table_score(au: vec4<f32>, av: vec4<f32>, sad: f32, t: u32) -> f32 {
  let dk = DK[t];
  let dm = DM[t];
  let eu = min(vec4<f32>(0.0), au * dm + dk);
  let ev = min(vec4<f32>(0.0), av * dm + dk);
  return A8[t] + AM[t] * sad + dot(eu + ev, ONE4);
}

// Both subblocks of one flip (lumas u, v against base luma lb): cover
// tables and their scores, then the lower neighbours behind ONE branch —
// skipped when both covers are table 0 (the lower neighbour IS the cover
// table; smooth content), branch-free inside so the two scores interleave.
// |D| is exact in f16; the scores need f32.
struct PairOut {
  t0: u32,
  t1: u32,
  acc: f32,
};
fn sb_pair(u0: vec4<f16>, v0: vec4<f16>, lbf0: f32, u1: vec4<f16>, v1: vec4<f16>, lbf1: f32) -> PairOut {
  let lb0 = f16(lbf0);
  let lb1 = f16(lbf1);
  let ah0 = abs(u0 - lb0);
  let bh0 = abs(v0 - lb0);
  let ah1 = abs(u1 - lb1);
  let bh1 = abs(v1 - lb1);
  let mx = vec2<f16>(max(max4(ah0), max4(bh0)), max(max4(ah1), max4(bh1)));
  let au0 = vec4<f32>(ah0);
  let av0 = vec4<f32>(bh0);
  let au1 = vec4<f32>(ah1);
  let av1 = vec4<f32>(bh1);
  let sad0 = dot(au0 + av0, ONE4);
  let sad1 = dot(au1 + av1, ONE4);
  // cover = #{B3[k] < mx : k < 7}, the first table whose large modifier
  // reaches mx — a binary search over the 7 thresholds.
  let s1 = mx > vec2<f16>(126.0h);
  let s2 = mx > select(vec2<f16>(51.0h), vec2<f16>(240.0h), s1);
  let s3 = mx > select(select(vec2<f16>(24.0h), vec2<f16>(87.0h), s2), select(vec2<f16>(180.0h), vec2<f16>(318.0h), s2), s1);
  let cover = select(vec2<u32>(0u), vec2<u32>(4u), s1) + select(vec2<u32>(0u), vec2<u32>(2u), s2) + select(vec2<u32>(0u), vec2<u32>(1u), s3);
  let hi0 = table_score(au0, av0, sad0, cover.x);
  let hi1 = table_score(au1, av1, sad1, cover.y);
  var out: PairOut;
  out.t0 = cover.x;
  out.t1 = cover.y;
  out.acc = hi0 + hi1;
  if (any(cover != vec2<u32>(0u))) {
    let t_lo = max(cover, vec2<u32>(1u)) - vec2<u32>(1u);
    let lo0 = table_score(au0, av0, sad0, t_lo.x);
    let lo1 = table_score(au1, av1, sad1, t_lo.y);
    let w0 = lo0 <= hi0;
    let w1 = lo1 <= hi1;
    out.t0 = select(cover.x, t_lo.x, w0);
    out.t1 = select(cover.y, t_lo.y, w1);
    out.acc = select(hi0, lo0, w0) + select(hi1, lo1, w1);
  }
  return out;
}

// One flip's fit: base quantisation + table search, and its estimate
// (Σ||p||² omitted).
struct FlipFit {
  est: f32,
  bases: Bases,
  lb0: f32,
  lb1: f32,
  t0: u32,
  t1: u32,
};
fn fit_flip(
  s0u: vec4<f16>,
  s0v: vec4<f16>,
  s1u: vec4<f16>,
  s1v: vec4<f16>,
  sum0: vec3<f32>,
  sum1: vec3<f32>,
) -> FlipFit {
  var out: FlipFit;
  out.bases = quantise_bases(sum0, sum1);
  let b0 = out.bases.b0;
  let b1 = out.bases.b1;
  out.lb0 = b0.r + b0.g + b0.b;
  out.lb1 = b1.r + b1.g + b1.b;
  let pp = sb_pair(s0u, s0v, out.lb0, s1u, s1v, out.lb1);
  out.t0 = pp.t0;
  out.t1 = pp.t1;
  out.est = dot(b0, 8.0 * b0 - 2.0 * sum0) + dot(b1, 8.0 * b1 - 2.0 * sum1) + pp.acc * (1.0 / 3.0);
  return out;
}

// fit_flip for exactly-gray blocks (r = g = b): the same arithmetic on one
// channel; sum0/sum1 are one channel's subblock sums.
fn fit_gray(
  s0u: vec4<f16>,
  s0v: vec4<f16>,
  s1u: vec4<f16>,
  s1v: vec4<f16>,
  sum0: f32,
  sum1: f32,
) -> FlipFit {
  var out: FlipFit;
  let q0 = floor(sum0 * (31.0 / 2040.0) + 0.5);
  let q1 = floor(sum1 * (31.0 / 2040.0) + 0.5);
  let d = q1 - q0;
  let diff = d >= -4.0 && d <= 3.0;
  let i0 = floor(sum0 * (15.0 / 2040.0) + 0.5);
  let i1 = floor(sum1 * (15.0 / 2040.0) + 0.5);
  out.bases.diff = diff;
  out.bases.c0 = vec3<f32>(select(i0, q0, diff));
  out.bases.c1 = vec3<f32>(select(i1, q1, diff));
  let b0 = select(i0 * 17.0, floor(q0 * 8.25), diff);
  let b1 = select(i1 * 17.0, floor(q1 * 8.25), diff);
  out.lb0 = 3.0 * b0;
  out.lb1 = 3.0 * b1;
  let pp = sb_pair(s0u, s0v, out.lb0, s1u, s1v, out.lb1);
  out.t0 = pp.t0;
  out.t1 = pp.t1;
  out.est = 3.0 * (b0 * (8.0 * b0 - 2.0 * sum0) + b1 * (8.0 * b1 - 2.0 * sum1)) + pp.acc * (1.0 / 3.0);
  return out;
}

// One gathered 2×2 quad: per-texel luma (gather order), channel sums, and
// the sums of its right column and bottom row (the planar moments' local
// parts). Gather order: w=(0,0) z=(1,0) x=(0,1) y=(1,1).
struct Quad {
  l: vec4<f16>,
  s: vec3<f32>,
  right: vec3<f32>,
  bottom: vec3<f32>,
};
fn gather_quad(cc: vec2<f32>) -> Quad {
  let r = textureGather(0, src_tex, smp, cc) * 255.0;
  let g = textureGather(1, src_tex, smp, cc) * 255.0;
  let b = textureGather(2, src_tex, smp, cc) * 255.0;
  var o: Quad;
  o.l = vec4<f16>(r + g + b);
  o.right = vec3<f32>(r.z + r.y, g.z + g.y, b.z + b.y);
  o.s = o.right + vec3<f32>(r.w + r.x, g.w + g.x, b.w + b.x);
  o.bottom = vec3<f32>(r.x + r.y, g.x + g.y, b.x + b.y);
  return o;
}

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid_raw: vec3<u32>) {
  // Row-band encodes dispatch a slice of the block grid starting at row y0.
  let gid = vec3<u32>(gid_raw.x, gid_raw.y + params.y0, gid_raw.z);
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) {
    return;
  }

  let block_index = gid.y * params.blocks_x + gid.x;
  let base_xy = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);

  // Luma by column: col[x][y]. Quadrant q = (x >= 2) | (y >= 2) << 1.
  var col: array<vec4<f16>, 4>;
  var qsum: array<vec3<f32>, 4>;
  // Planar right-hand sides: Σ x·p and Σ y·p.
  var sxp = vec3<f32>(0.0);
  var syp = vec3<f32>(0.0);
  if (u32(base_xy.x) + 4u <= params.width && u32(base_xy.y) + 4u <= params.height) {
    let inv = vec2<f32>(1.0) / vec2<f32>(textureDimensions(src_tex));
    let c0 = (vec2<f32>(base_xy) + 1.0) * inv;
    let q0 = gather_quad(c0);
    let q1 = gather_quad(c0 + vec2<f32>(2.0, 0.0) * inv);
    let q2 = gather_quad(c0 + vec2<f32>(0.0, 2.0) * inv);
    let q3 = gather_quad(c0 + vec2<f32>(2.0, 2.0) * inv);
    qsum[0] = q0.s;
    qsum[1] = q1.s;
    qsum[2] = q2.s;
    qsum[3] = q3.s;
    sxp = q0.right + q2.right + 2.0 * (q1.s + q3.s) + q1.right + q3.right;
    syp = q0.bottom + q1.bottom + 2.0 * (q2.s + q3.s) + q2.bottom + q3.bottom;
    col[0] = vec4<f16>(q0.l.w, q0.l.x, q2.l.w, q2.l.x);
    col[1] = vec4<f16>(q0.l.z, q0.l.y, q2.l.z, q2.l.y);
    col[2] = vec4<f16>(q1.l.w, q1.l.x, q3.l.w, q3.l.x);
    col[3] = vec4<f16>(q1.l.z, q1.l.y, q3.l.z, q3.l.y);
  } else {
    let max_xy = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
      let lx = i & 3u;
      let ly = i >> 2u;
      let p = clamp(base_xy + vec2<i32>(i32(lx), i32(ly)), vec2<i32>(0, 0), max_xy);
      let c = round(textureLoad(src_tex, p, 0).rgb * 255.0);
      col[lx][ly] = f16(c.r + c.g + c.b);
      let q = u32(lx >= 2u) | (u32(ly >= 2u) << 1u);
      qsum[q] = qsum[q] + c;
      sxp = sxp + f32(lx) * c;
      syp = syp + f32(ly) * c;
    }
  }

  let total = qsum[0] + qsum[1] + qsum[2] + qsum[3];
  // Exactly gray: every quadrant sum AND both planar moments equal across
  // R, G, B — then R and B planar corners coincide (same 6-bit code) and
  // only R (6-bit) and G (7-bit) need solving.
  let gray = all(qsum[0].rg == qsum[0].gb) && all(qsum[1].rg == qsum[1].gb) &&
             all(qsum[2].rg == qsum[2].gb) && all(qsum[3].rg == qsum[3].gb) &&
             all(sxp.rg == sxp.gb) && all(syp.rg == syp.gb);

  var planar_est: f32;
  var qo: vec3<f32>;
  var qh: vec3<f32>;
  var qv: vec3<f32>;
  var bflip = 0u;
  var sel: FlipFit;
  if (gray) {
    // Planar on two channels: R and B share the 6-bit solve.
    let rB = sxp.r * 0.25;
    let rC = syp.r * 0.25;
    let rA = total.r - rB - rC;
    let po = 0.2875 * rA - 0.0125 * rB - 0.0125 * rC;
    let ph = -0.0125 * rA + 0.4875 * rB - 0.3125 * rC;
    let pv = -0.0125 * rA - 0.3125 * rB + 0.4875 * rC;
    let pmax = vec2<f32>(63.0, 127.0);
    let qo2 = clamp(floor(po * (pmax / 255.0) + 0.5), vec2<f32>(0.0), pmax);
    let qh2 = clamp(floor(ph * (pmax / 255.0) + 0.5), vec2<f32>(0.0), pmax);
    let qv2 = clamp(floor(pv * (pmax / 255.0) + 0.5), vec2<f32>(0.0), pmax);
    let xk = vec2<f32>(4.0625, 2.015625);
    let eo = floor(qo2 * xk);
    let eh = floor(qh2 * xk);
    let ev = floor(qv2 * xk);
    let gram = 3.5 * (eo * eo + eh * eh + ev * ev) + 0.5 * eo * (eh + ev) + 4.5 * eh * ev;
    let pe = gram - 2.0 * (eo * rA + eh * rB + ev * rC);
    planar_est = 2.0 * pe.x + pe.y + PLANAR_FUDGE;
    qo = qo2.xyx;
    qh = qh2.xyx;
    qv = qv2.xyx;

    let sum0a = qsum[0].r + qsum[2].r;
    let sum1a = qsum[1].r + qsum[3].r;
    let sum0b = qsum[0].r + qsum[1].r;
    let sum1b = qsum[2].r + qsum[3].r;
    sel = fit_gray(col[0], col[1], col[2], col[3], sum0a, sum1a);
    let alt = fit_gray(
      vec4<f16>(col[0].xy, col[1].xy),
      vec4<f16>(col[2].xy, col[3].xy),
      vec4<f16>(col[0].zw, col[1].zw),
      vec4<f16>(col[2].zw, col[3].zw),
      sum0b,
      sum1b,
    );
    if (alt.est < sel.est) {
      sel = alt;
      bflip = 1u;
    }
  } else {
    // LSQ plane in closed form: rhs rA = Σ(1 − x/4 − y/4)·p, rB = Σ(x/4)·p,
    // rC = Σ(y/4)·p times the constant inverse Gram matrix (the same
    // coefficient form as the CPU mirror, so rounding ties resolve alike);
    // estimate with the quantised, clamped corners: −2·θ·rhs + θᵀGθ.
    let rB = sxp * 0.25;
    let rC = syp * 0.25;
    let rA = total - rB - rC;
    let po = 0.2875 * rA - 0.0125 * rB - 0.0125 * rC;
    let ph = -0.0125 * rA + 0.4875 * rB - 0.3125 * rC;
    let pv = -0.0125 * rA - 0.3125 * rB + 0.4875 * rC;
    let pmax = vec3<f32>(63.0, 127.0, 63.0);
    qo = clamp(floor(po * (pmax / 255.0) + 0.5), vec3<f32>(0.0), pmax);
    qh = clamp(floor(ph * (pmax / 255.0) + 0.5), vec3<f32>(0.0), pmax);
    qv = clamp(floor(pv * (pmax / 255.0) + 0.5), vec3<f32>(0.0), pmax);
    // 6-bit expand (q<<2)|(q>>4) = floor(4.0625·q); 7-bit (q<<1)|(q>>6) = floor(2.015625·q).
    let xk = vec3<f32>(4.0625, 2.015625, 4.0625);
    let eo = floor(qo * xk);
    let eh = floor(qh * xk);
    let ev = floor(qv * xk);
    let gram = 3.5 * (eo * eo + eh * eh + ev * ev) + 0.5 * eo * (eh + ev) + 4.5 * eh * ev;
    planar_est = dot(gram - 2.0 * (eo * rA + eh * rB + ev * rC), ONE3) + PLANAR_FUDGE;

    // Flip 0 splits columns (sum0a = left half), flip 1 splits rows (sum0b =
    // top half). Per flip, the preselect residual minus the flip-independent
    // Σ||p||² and Σℓ² terms: −Σ||s||²/8 + κ·(Σℓ)²/24 over its two subblocks.
    let sum0a = qsum[0] + qsum[2];
    let sum1a = qsum[1] + qsum[3];
    let sum0b = qsum[0] + qsum[1];
    let sum1b = qsum[2] + qsum[3];
    let l0a = dot(sum0a, ONE3);
    let l1a = dot(sum1a, ONE3);
    let l0b = dot(sum0b, ONE3);
    let l1b = dot(sum1b, ONE3);
    let res_a = KAPPA / 24.0 * (l0a * l0a + l1a * l1a) - 0.125 * (dot(sum0a, sum0a) + dot(sum1a, sum1a));
    let res_b = KAPPA / 24.0 * (l0b * l0b + l1b * l1b) - 0.125 * (dot(sum0b, sum0b) + dot(sum1b, sum1b));
    let fb = res_b < res_a;
    bflip = select(0u, 1u, fb);
    sel = fit_flip(
      select(col[0], vec4<f16>(col[0].xy, col[1].xy), fb),
      select(col[1], vec4<f16>(col[2].xy, col[3].xy), fb),
      select(col[2], vec4<f16>(col[0].zw, col[1].zw), fb),
      select(col[3], vec4<f16>(col[2].zw, col[3].zw), fb),
      select(sum0a, sum0b, fb),
      select(sum1a, sum1b, fb),
    );
  }

  // ------------------------------------------------------------ packing --
  var hi: u32;
  var lo: u32;
  if (sel.est <= planar_est) {
    let codes0 = vec3<u32>(sel.bases.c0);
    let codes1 = vec3<u32>(sel.bases.c1);
    let t0 = sel.t0;
    let t1 = sel.t1;
    if (sel.bases.diff) {
      let d = vec3<u32>(vec3<i32>(codes1) - vec3<i32>(codes0)) & vec3<u32>(7u);
      hi = (codes0.r << 27u) | (d.r << 24u) | (codes0.g << 19u) | (d.g << 16u) | (codes0.b << 11u) | (d.b << 8u)
         | (t0 << 5u) | (t1 << 2u) | 2u | bflip;
    } else {
      hi = (codes0.r << 28u) | (codes1.r << 24u) | (codes0.g << 20u) | (codes1.g << 16u) | (codes0.b << 12u) | (codes1.b << 8u)
         | (t0 << 5u) | (t1 << 2u) | bflip;
    }
    // Wire indices, column by column (bit x·4 + y): flip 0 gives columns
    // 0,1 subblock 0; flip 1 gives rows 0,1 (lanes x, y) subblock 0.
    // LSB = large modifier, MSB = negative.
    let fb = bflip == 1u;
    let lb0 = f16(sel.lb0);
    let lb1 = f16(sel.lb1);
    let th0 = f16(THR[t0]);
    let th1 = f16(THR[t1]);
    let lb_rows = vec4<f16>(lb0, lb0, lb1, lb1);
    let th_rows = vec4<f16>(th0, th0, th1, th1);
    let lb_l = select(vec4<f16>(lb0), lb_rows, fb);
    let lb_r = select(vec4<f16>(lb1), lb_rows, fb);
    let th_l = select(vec4<f16>(th0), th_rows, fb);
    let th_r = select(vec4<f16>(th1), th_rows, fb);
    let bitv = vec4<u32>(1u, 2u, 4u, 8u);
    var lsb = 0u;
    var msb = 0u;
    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
      let d = col[c] - select(lb_l, lb_r, c >= 2u);
      let large = select(vec4<u32>(0u), bitv, abs(d) > select(th_l, th_r, c >= 2u));
      let neg = select(vec4<u32>(0u), bitv, d < vec4<f16>(0.0));
      lsb = lsb | ((large.x | large.y | large.z | large.w) << (c * 4u));
      msb = msb | ((neg.x | neg.y | neg.z | neg.w) << (c * 4u));
    }
    lo = lsb | (msb << 16u);
  } else {
    let ro = u32(qo.r); let go = u32(qo.g); let bo = u32(qo.b);
    let rh = u32(qh.r); let gh = u32(qh.g); let bh = u32(qh.b);
    let rv = u32(qv.r); let gv = u32(qv.g); let bv = u32(qv.b);
    let r_sum = i32(ro >> 2u) + signed3(((ro & 3u) << 1u) | (go >> 6u));
    let r_fix = select(0u, 1u, r_sum < 0);
    let g_sum = i32((go >> 2u) & 15u) + signed3(((go & 3u) << 1u) | (bo >> 5u));
    let g_fix = select(0u, 1u, g_sum < 0);
    let p = (bo >> 3u) & 3u;
    let q = (bo >> 1u) & 3u;
    let b_fix3 = select(0u, 7u, p + q >= 4u);
    let b_fix1 = select(1u, 0u, p + q >= 4u);
    hi = (r_fix << 31u) | (ro << 25u) | ((go >> 6u) << 24u) | (g_fix << 23u) | ((go & 63u) << 17u)
       | ((bo >> 5u) << 16u) | (b_fix3 << 13u) | (((bo >> 3u) & 3u) << 11u) | (b_fix1 << 10u)
       | ((bo & 7u) << 7u) | ((rh >> 1u) << 2u) | 2u | (rh & 1u);
    lo = (gh << 25u) | (bh << 19u) | (rv << 13u) | (gv << 6u) | bv;
  }

  let out = block_index * 2u;
  dst[out]      = bswap(hi);
  dst[out + 1u] = bswap(lo);
}
