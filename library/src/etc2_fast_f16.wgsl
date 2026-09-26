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
//     the quad's texel centres, the other three quads by constant texel
//     offsets; interior quads never touch the zeroed padding strip). Blocks
//     straddling the edge of a non-multiple-of-4 image build the same quads
//     from clamped per-texel loads — never a runtime-indexed loop: one that
//     wrote col[x][y]/qsum[q] cost ~2% on EVERY block. Lumas are kept as 4
//     COLUMN vectors — wire pixel order is x·4 + y — so both flips' half-
//     blocks and the index packing use only constant indexing.
//   • Channel sums stay in the sampler's UNIT domain: the ×255 lives in
//     the constants that consume them (quantisers, estimates), not in 48
//     per-texel multiplies (−3..4% GPU). Lumas are exact integers (one FMA
//     chain per texel). The unit sums carry f32 rounding noise, which only
//     moves exact decision ties (quality-neutral: ~0.1-0.5% of colour
//     blocks change bytes, ΔPSNR ±0.001 dB); the gray path, where the two
//     flips often tie exactly, rounds its five scalars back to integers
//     and stays byte-identical to the integer-domain form.
//   • Flip preselect, O(1): per subblock the residual after continuous luma
//     modulation is within-variance − κ·(luma variance)/3, κ = 0.9. κ = 1
//     is the exact chroma residual; keeping a tenth of the luma variance
//     prefers the split with less luma spread for the 4-level tables to
//     cover (+0.07-0.10 dB on photo colour vs κ = 1, free). Only the chosen
//     flip is searched. Evaluated in half-difference form (only the two
//     halves' difference varies between flips; −1..2% GPU).
//   • Exactly-gray blocks (R = G = B in every texel, tested on the raw
//     gathers before any luma or sum is formed; their lumas and sums then
//     come from R alone — a third of the load-side ALU, −4..5% on gray
//     maps) have no chroma to steer the preselect, so both flips
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
//     threshold rule exactly, summed as an |x| form (see table_score). A flip's
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
// the table-search terms are integer sums held exactly in f32 (< 2^24).

enable f16;

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
  y0:       u32, // first block row of this dispatch (row-band encodes)
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<u32>>;
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
// Float -> u32 for exact integers in [0, 2^23): x + 2^23 holds x in its
// mantissa, so bitcast(x + MAGIC) ^ MAGIC_BITS == x, and a left shift by
// >= 8 drops the exponent bits on its own. WGSL's u32(f32) is a SATURATING
// conversion (compares + selects around the convert): packing's 15 of them
// cost ~2.5% GPU.
const MAGIC = 8388608.0;
const MAGIC_BITS = 0x4B000000u;

fn bswap(x: u32) -> u32 {
  let t = ((x & 0x00ff00ffu) << 8u) | ((x >> 8u) & 0x00ff00ffu);
  return (t << 16u) | (t >> 16u);
}

fn max4(v: vec4<f16>) -> f16 {
  return max(max(v.x, v.y), max(v.z, v.w));
}

// Base colours from subblock SUMS (8 texels each, unit domain: 31·255/2040
// = 3.875, 15·255/2040 = 1.875): codes (as floats) and their 8-bit
// expansions. Differential mode when the 5-bit codes are within
// the 3-bit delta range, else individual 4-bit. Expansions in float:
// (q<<3)|(q>>2) = floor(8.25·q) for 5 bits, (q<<4)|q = 17·q for 4 bits —
// the mode's codes are selected first, then expanded once. (The codes
// themselves must stay a select of q/i: recomputing floor(sum·k + 0.5)
// with a selected k could round differently from the q the diff test saw.)
struct Bases {
  c0: vec3<f32>,
  c1: vec3<f32>,
  b0: vec3<f32>,
  b1: vec3<f32>,
  diff: bool,
};
fn quantise_bases(sum0: vec3<f32>, sum1: vec3<f32>) -> Bases {
  let q0 = floor(sum0 * 3.875 + 0.5);
  let q1 = floor(sum1 * 3.875 + 0.5);
  let d = q1 - q0;
  var o: Bases;
  o.diff = all(d >= vec3<f32>(-4.0)) && all(d <= vec3<f32>(3.0));
  let i0 = floor(sum0 * 1.875 + 0.5);
  let i1 = floor(sum1 * 1.875 + 0.5);
  o.c0 = select(i0, q0, o.diff);
  o.c1 = select(i1, q1, o.diff);
  let k = select(17.0, 8.25, o.diff);
  o.b0 = floor(o.c0 * k);
  o.b1 = floor(o.c1 * k);
  return o;
}

// Subblock error (×3) of table t under the threshold rule, in min form:
// per texel min(a3² − 2·a3·ad, b3² − 2·b3·ad) = (a3² − 2·a3·ad) + min(0, x)
// with x = (b3² − a3²) − 2·(b3 − a3)·ad. min(0, x) = (x − |x|)/2, and Σx
// and the a3 part both sum in closed form from sad = Σ ad, so per texel
// only one FMA and one |·| add remain (|·| is a free source modifier):
//   score = 4(a3² + b3²) − (a3 + b3)·sad − ½·Σ|x|
// All terms are integers (or halves) below 2^24 — exact in f32, identical
// to the per-texel min form (−2% GPU; −3% on gray maps, which search twice).
// One vec4 per table (DK = b3² − a3², DM = −2(b3 − a3), C0 = 4(a3² + b3²),
// C1 = −(a3 + b3)): every runtime-indexed lookup also pays a bounds clamp,
// so four scalar arrays cost ~1% more.
const TAB = array<vec4<f32>, 8>(
  vec4<f32>(540.0, -36.0, 2448.0, -30.0),
  vec4<f32>(2376.0, -72.0, 11304.0, -66.0),
  vec4<f32>(6840.0, -120.0, 33192.0, -114.0),
  vec4<f32>(14355.0, -174.0, 69588.0, -165.0),
  vec4<f32>(29484.0, -252.0, 141264.0, -234.0),
  vec4<f32>(52416.0, -336.0, 251136.0, -312.0),
  vec4<f32>(91323.0, -438.0, 443700.0, -417.0),
  vec4<f32>(281520.0, -816.0, 1285128.0, -690.0),
);
fn table_score(au: vec4<f32>, av: vec4<f32>, sad: f32, t: u32) -> f32 {
  let k = TAB[t];
  let xu = au * k.y + k.x;
  let xv = av * k.y + k.x;
  return fma(k.w, sad, k.z) - 0.5 * dot(abs(xu) + abs(xv), ONE4);
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
// (Σ||p||² omitted; sums in the unit domain, hence 2·255 = 510).
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
  out.est = dot(b0, 8.0 * b0 - 510.0 * sum0) + dot(b1, 8.0 * b1 - 510.0 * sum1) + pp.acc * (1.0 / 3.0);
  return out;
}

// fit_flip for exactly-gray blocks (r = g = b): the same arithmetic on one
// channel; sum0/sum1 are one channel's subblock sums, as exact integers.
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
  let c0 = select(i0, q0, diff);
  let c1 = select(i1, q1, diff);
  out.bases.diff = diff;
  out.bases.c0 = vec3<f32>(c0);
  out.bases.c1 = vec3<f32>(c1);
  let k = select(17.0, 8.25, diff);
  let b0 = floor(c0 * k);
  let b1 = floor(c1 * k);
  out.lb0 = 3.0 * b0;
  out.lb1 = 3.0 * b1;
  let pp = sb_pair(s0u, s0v, out.lb0, s1u, s1v, out.lb1);
  out.t0 = pp.t0;
  out.t1 = pp.t1;
  out.est = 3.0 * (b0 * (8.0 * b0 - 2.0 * sum0) + b1 * (8.0 * b1 - 2.0 * sum1)) + pp.acc * (1.0 / 3.0);
  return out;
}

// One gathered 2×2 quad: per-texel luma (gather order, exact 0..765),
// unit-domain channel sums, and the sums of its right column and bottom row
// (the planar moments' local parts). Gather order: w=(0,0) z=(1,0) x=(0,1)
// y=(1,1). `gray` (R = G = B in all four texels) is set by load_quad only.
struct Quad {
  l: vec4<f16>,
  s: vec3<f32>,
  right: vec3<f32>,
  bottom: vec3<f32>,
  gray: bool,
};
fn gather_quad(r: vec4<f32>, g: vec4<f32>, b: vec4<f32>) -> Quad {
  var o: Quad;
  o.l = vec4<f16>(fma(r, vec4<f32>(255.0), fma(g, vec4<f32>(255.0), b * 255.0)));
  o.right = vec3<f32>(r.z + r.y, g.z + g.y, b.z + b.y);
  o.s = o.right + vec3<f32>(r.w + r.x, g.w + g.x, b.w + b.x);
  o.bottom = vec3<f32>(r.x + r.y, g.x + g.y, b.x + b.y);
  return o;
}

// Edge blocks: four clamped texel loads per quad, in gather order (the
// sums then carry the same unit-domain rounding as interior blocks).
fn load_quad(p: vec2<i32>, max_xy: vec2<i32>) -> Quad {
  let a = textureLoad(src_tex, min(p, max_xy), 0);
  let b = textureLoad(src_tex, min(p + vec2<i32>(1, 0), max_xy), 0);
  let c = textureLoad(src_tex, min(p + vec2<i32>(0, 1), max_xy), 0);
  let d = textureLoad(src_tex, min(p + vec2<i32>(1, 1), max_xy), 0);
  let r = vec4<f32>(c.r, d.r, b.r, a.r);
  let g = vec4<f32>(c.g, d.g, b.g, a.g);
  let bl = vec4<f32>(c.b, d.b, b.b, a.b);
  var o = gather_quad(r, g, bl);
  o.gray = all(r == g) && all(g == bl);
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
  // Planar right-hand sides: Σ x·p and Σ y·p (all sums in the unit domain).
  var sxp = vec3<f32>(0.0);
  var syp = vec3<f32>(0.0);
  // Exactly gray (R = G = B in every texel): no chroma, so the gray path
  // below searches both flips on one channel, and R and B share their
  // planar solve (same 6-bit code) — only R (6-bit) and G (7-bit) remain.
  var gray: bool;
  if (u32(base_xy.x) + 4u <= params.width && u32(base_xy.y) + 4u <= params.height) {
    let inv = vec2<f32>(1.0) / vec2<f32>(textureDimensions(src_tex));
    let c0 = (vec2<f32>(base_xy) + 1.0) * inv;
    let r0 = textureGather(0, src_tex, smp, c0);
    let g0 = textureGather(1, src_tex, smp, c0);
    let b0 = textureGather(2, src_tex, smp, c0);
    let r1 = textureGather(0, src_tex, smp, c0, vec2<i32>(2, 0));
    let g1 = textureGather(1, src_tex, smp, c0, vec2<i32>(2, 0));
    let b1 = textureGather(2, src_tex, smp, c0, vec2<i32>(2, 0));
    let r2 = textureGather(0, src_tex, smp, c0, vec2<i32>(0, 2));
    let g2 = textureGather(1, src_tex, smp, c0, vec2<i32>(0, 2));
    let b2 = textureGather(2, src_tex, smp, c0, vec2<i32>(0, 2));
    let r3 = textureGather(0, src_tex, smp, c0, vec2<i32>(2, 2));
    let g3 = textureGather(1, src_tex, smp, c0, vec2<i32>(2, 2));
    let b3 = textureGather(2, src_tex, smp, c0, vec2<i32>(2, 2));
    gray = all(r0 == g0) && all(g0 == b0) && all(r1 == g1) && all(g1 == b1) &&
           all(r2 == g2) && all(g2 == b2) && all(r3 == g3) && all(g3 == b3);
    if (gray) {
      // Lumas and sums from R alone (the gray path reads only the R lanes
      // of qsum/sxp/syp): a third of the colour path's load-side ALU.
      let l0 = vec4<f16>(r0 * 765.0);
      let l1 = vec4<f16>(r1 * 765.0);
      let l2 = vec4<f16>(r2 * 765.0);
      let l3 = vec4<f16>(r3 * 765.0);
      let rt0 = r0.z + r0.y;
      let rt1 = r1.z + r1.y;
      let rt2 = r2.z + r2.y;
      let rt3 = r3.z + r3.y;
      let s0 = rt0 + (r0.w + r0.x);
      let s1 = rt1 + (r1.w + r1.x);
      let s2 = rt2 + (r2.w + r2.x);
      let s3 = rt3 + (r3.w + r3.x);
      qsum[0] = vec3<f32>(s0);
      qsum[1] = vec3<f32>(s1);
      qsum[2] = vec3<f32>(s2);
      qsum[3] = vec3<f32>(s3);
      sxp = vec3<f32>((rt0 + rt1) + (rt2 + rt3) + 2.0 * (s1 + s3));
      syp = vec3<f32>(((r0.x + r0.y) + (r1.x + r1.y)) + ((r2.x + r2.y) + (r3.x + r3.y)) + 2.0 * (s2 + s3));
      col[0] = vec4<f16>(l0.w, l0.x, l2.w, l2.x);
      col[1] = vec4<f16>(l0.z, l0.y, l2.z, l2.y);
      col[2] = vec4<f16>(l1.w, l1.x, l3.w, l3.x);
      col[3] = vec4<f16>(l1.z, l1.y, l3.z, l3.y);
    } else {
      let q0 = gather_quad(r0, g0, b0);
      let q1 = gather_quad(r1, g1, b1);
      let q2 = gather_quad(r2, g2, b2);
      let q3 = gather_quad(r3, g3, b3);
      qsum[0] = q0.s;
      qsum[1] = q1.s;
      qsum[2] = q2.s;
      qsum[3] = q3.s;
      sxp = (q0.right + q1.right) + (q2.right + q3.right) + 2.0 * (q1.s + q3.s);
      syp = (q0.bottom + q1.bottom) + (q2.bottom + q3.bottom) + 2.0 * (q2.s + q3.s);
      col[0] = vec4<f16>(q0.l.w, q0.l.x, q2.l.w, q2.l.x);
      col[1] = vec4<f16>(q0.l.z, q0.l.y, q2.l.z, q2.l.y);
      col[2] = vec4<f16>(q1.l.w, q1.l.x, q3.l.w, q3.l.x);
      col[3] = vec4<f16>(q1.l.z, q1.l.y, q3.l.z, q3.l.y);
    }
  } else {
    // Edge of a non-multiple-of-4 image: clamp to the last texel.
    let max_xy = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    let q0 = load_quad(base_xy, max_xy);
    let q1 = load_quad(base_xy + vec2<i32>(2, 0), max_xy);
    let q2 = load_quad(base_xy + vec2<i32>(0, 2), max_xy);
    let q3 = load_quad(base_xy + vec2<i32>(2, 2), max_xy);
    qsum[0] = q0.s;
    qsum[1] = q1.s;
    qsum[2] = q2.s;
    qsum[3] = q3.s;
    sxp = (q0.right + q1.right) + (q2.right + q3.right) + 2.0 * (q1.s + q3.s);
    syp = (q0.bottom + q1.bottom) + (q2.bottom + q3.bottom) + 2.0 * (q2.s + q3.s);
    col[0] = vec4<f16>(q0.l.w, q0.l.x, q2.l.w, q2.l.x);
    col[1] = vec4<f16>(q0.l.z, q0.l.y, q2.l.z, q2.l.y);
    col[2] = vec4<f16>(q1.l.w, q1.l.x, q3.l.w, q3.l.x);
    col[3] = vec4<f16>(q1.l.z, q1.l.y, q3.l.z, q3.l.y);
    gray = q0.gray && q1.gray && q2.gray && q3.gray;
  }

  let total = (qsum[0] + qsum[1]) + (qsum[2] + qsum[3]);
  // Right and bottom halves (subblock 1 of flip 0 / flip 1).
  let right = qsum[1] + qsum[3];
  let bottom = qsum[2] + qsum[3];

  var planar_est: f32;
  var qo: vec3<f32>;
  var qh: vec3<f32>;
  var qv: vec3<f32>;
  var bflip = 0u;
  var sel: FlipFit;
  if (gray) {
    // The unit-domain sums carry f32 rounding noise; gray blocks resolve
    // exact est ties (the two flips often tie), so their five scalars are
    // snapped back to the exact integers first.
    let tr = round(total.r * 255.0);
    let r1 = round(right.r * 255.0);
    let b1 = round(bottom.r * 255.0);
    // Planar on two channels: R and B share the 6-bit solve.
    let rB = round(sxp.r * 255.0) * 0.25;
    let rC = round(syp.r * 255.0) * 0.25;
    let rA = tr - rB - rC;
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

    let sum1a = r1;
    let sum0a = tr - r1;
    let sum1b = b1;
    let sum0b = tr - b1;
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
    // estimate with the quantised, clamped corners: −2·θ·rhs + θᵀGθ. In
    // the unit domain the corner clamp is a free saturate().
    let rB = sxp * 0.25;
    let rC = syp * 0.25;
    let rA = total - rB - rC;
    let po = 0.2875 * rA - 0.0125 * rB - 0.0125 * rC;
    let ph = -0.0125 * rA + 0.4875 * rB - 0.3125 * rC;
    let pv = -0.0125 * rA - 0.3125 * rB + 0.4875 * rC;
    let pmax = vec3<f32>(63.0, 127.0, 63.0);
    qo = floor(saturate(po) * pmax + 0.5);
    qh = floor(saturate(ph) * pmax + 0.5);
    qv = floor(saturate(pv) * pmax + 0.5);
    // 6-bit expand (q<<2)|(q>>4) = floor(4.0625·q); 7-bit (q<<1)|(q>>6) = floor(2.015625·q).
    let xk = vec3<f32>(4.0625, 2.015625, 4.0625);
    let eo = floor(qo * xk);
    let eh = floor(qh * xk);
    let ev = floor(qv * xk);
    // θᵀGθ − 2θ·rhs in Horner form (rhs scaled back to 0..255 units).
    let mA = -510.0 * rA;
    let mB = -510.0 * rB;
    let mC = -510.0 * rC;
    let pe = eo * (3.5 * eo + 0.5 * (eh + ev) + mA) + eh * (3.5 * eh + 4.5 * ev + mB) + ev * (3.5 * ev + mC);
    planar_est = dot(pe, ONE3) + PLANAR_FUDGE;

    // Flip 0 splits columns (subblock 1 = right half), flip 1 splits rows
    // (subblock 1 = bottom half). Per flip, the preselect residual minus
    // flip-independent terms: with the halves' difference Δ = s0 − s1 and
    // δ = Σ_c Δ_c, −Σ||s||²/8 + κ·(Σℓ)²/24 over the two subblocks is
    // (κ·δ² − 3·||Δ||²)/48 plus a flip-independent constant.
    let da = total - 2.0 * right;
    let db = total - 2.0 * bottom;
    let la = dot(da, ONE3);
    let lb = dot(db, ONE3);
    let res_a = KAPPA * la * la - 3.0 * dot(da, da);
    let res_b = KAPPA * lb * lb - 3.0 * dot(db, db);
    let fb = res_b < res_a;
    bflip = select(0u, 1u, fb);
    let sum1 = select(right, bottom, fb);
    // sb_pair is order-blind within a subblock: rows 0-1 of columns 0-1 are
    // in subblock 0 and rows 2-3 of columns 2-3 in subblock 1 for either
    // flip, so only the other two half-columns swap (8 selects, not 16).
    let cz = vec4<f16>(col[0].zw, col[1].zw);
    let cx = vec4<f16>(col[2].xy, col[3].xy);
    sel = fit_flip(
      vec4<f16>(col[0].xy, col[1].xy),
      select(cz, cx, fb),
      vec4<f16>(col[2].zw, col[3].zw),
      select(cx, cz, fb),
      total - sum1,
      sum1,
    );
  }

  // ------------------------------------------------------------ packing --
  var hi: u32;
  var lo: u32;
  if (sel.est <= planar_est) {
    let t0 = sel.t0;
    let t1 = sel.t1;
    // Per channel byte: differential = base5 << 3 | (delta & 7), individual
    // = base4a << 4 | base4b — built in float, converted once (MAGIC).
    let dfl = sel.bases.diff;
    let dd = sel.bases.c1 - sel.bases.c0;
    let low = select(sel.bases.c1, select(dd, dd + 8.0, dd < vec3<f32>(0.0)), dfl);
    let bytes = bitcast<vec3<u32>>(fma(sel.bases.c0, vec3<f32>(select(16.0, 8.0, dfl)), low) + MAGIC);
    hi = (bytes.r << 24u) | (bytes.g << 16u) | (bytes.b << 8u) | (t0 << 5u) | (t1 << 2u) | select(0u, 2u, dfl) | bflip;
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
    // Unrolled by hand (~1% over the equivalent 4-iteration loop).
    var lsb = 0u;
    var msb = 0u;
    {
      let d = col[0] - lb_l;
      let bits = vec4<u32>(1u, 2u, 4u, 8u);
      let large = select(vec4<u32>(0u), bits, abs(d) > th_l);
      let neg = select(vec4<u32>(0u), bits, d < vec4<f16>(0.0));
      lsb = lsb | (large.x | large.y | large.z | large.w);
      msb = msb | (neg.x | neg.y | neg.z | neg.w);
    }
    {
      let d = col[1] - lb_l;
      let bits = vec4<u32>(16u, 32u, 64u, 128u);
      let large = select(vec4<u32>(0u), bits, abs(d) > th_l);
      let neg = select(vec4<u32>(0u), bits, d < vec4<f16>(0.0));
      lsb = lsb | (large.x | large.y | large.z | large.w);
      msb = msb | (neg.x | neg.y | neg.z | neg.w);
    }
    {
      let d = col[2] - lb_r;
      let bits = vec4<u32>(256u, 512u, 1024u, 2048u);
      let large = select(vec4<u32>(0u), bits, abs(d) > th_r);
      let neg = select(vec4<u32>(0u), bits, d < vec4<f16>(0.0));
      lsb = lsb | (large.x | large.y | large.z | large.w);
      msb = msb | (neg.x | neg.y | neg.z | neg.w);
    }
    {
      let d = col[3] - lb_r;
      let bits = vec4<u32>(4096u, 8192u, 16384u, 32768u);
      let large = select(vec4<u32>(0u), bits, abs(d) > th_r);
      let neg = select(vec4<u32>(0u), bits, d < vec4<f16>(0.0));
      lsb = lsb | (large.x | large.y | large.z | large.w);
      msb = msb | (neg.x | neg.y | neg.z | neg.w);
    }
    lo = lsb | (msb << 16u);
  } else {
    let pi = bitcast<vec4<u32>>(vec4<f32>(qo, qh.r) + MAGIC) ^ vec4<u32>(MAGIC_BITS);
    let ro = pi.x; let go = pi.y; let bo = pi.z; let rh = pi.w;
    // ETC1-view overflow fixes. R and G fields read as base + signed 3-bit
    // delta (x >> 3, x & 7) must stay in [0, 31]: signed3(v) = (v ^ 4) − 4.
    let xr = (ro << 1u) | (go >> 6u);
    let xg = ((go & 63u) << 1u) | (bo >> 5u);
    let r_fix = select(0u, 0x80000000u, (xr >> 3u) + ((xr & 7u) ^ 4u) < 4u);
    let g_fix = select(0u, 0x800000u, (xg >> 3u) + ((xg & 7u) ^ 4u) < 4u);
    // B must overflow: bits 47-45 = 111 with bit 42 = 0 when p + q >= 4,
    // else 000 with bit 42 = 1.
    let b_fix = select(0x400u, 0xE000u, ((bo >> 3u) & 3u) + ((bo >> 1u) & 3u) >= 4u);
    // go + (go & 64) moves GO bit 6 up one place (bit 24; bit 23 is g_fix);
    // rh + (rh & 62) spreads RH around the diff bit.
    hi = r_fix | (ro << 25u) | ((go + (go & 64u)) << 17u) | g_fix
       | ((bo & 32u) << 11u) | ((bo & 24u) << 8u) | ((bo & 7u) << 7u) | b_fix
       | (rh + (rh & 62u)) | 2u;
    // GH·2^25 | BH·2^19 | RV·2^13 | GV·2^6 | BV: two exact float field sums.
    lo = (bitcast<u32>(fma(qh.g, 64.0, qh.b) + MAGIC) << 19u) | (bitcast<u32>(fma(fma(qv.r, 128.0, qv.g), 64.0, qv.b) + MAGIC) ^ MAGIC_BITS);
  }

  dst[block_index] = vec2<u32>(bswap(hi), bswap(lo));
}
