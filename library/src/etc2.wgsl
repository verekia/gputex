// ETC2 RGB8 compute shader encoder — reads the PREPARED split source
// (see etc2_prep.wgsl), not raw RGBA8.
//
// Each invocation encodes one 4x4 pixel block into an 8-byte ETC2 RGB8 block
// written as 2 x u32 into the destination storage buffer. ETC2 blocks are
// big-endian on the wire (byte 0 = bits 63..56), so both words are byte-
// swapped on the way out. f32 only — there is no f16 module (the estimates
// are exact in f32 integer arithmetic and the pass is read-bound).
//
// ARCHITECTURE — bandwidth-first (2026-07 second rewrite). The encoder is
// DRAM-bound: on a 100 GB/s part the old full-RGBA8 read floor was ~0.15 ms
// @2048² with only ~0.03 ms of ALU on top. This shader reads 2 bytes/pixel
// instead of 4 — a packed-luma plane (4 lumas per r32uint fetch = one block
// row) plus a half-resolution quadrant-average plane — because the
// scalar-luma algorithm never needed more: modifier tables and indices are
// pure LUMA decisions, and bases/estimates need only 2×2 quadrant sums.
// Measured 0.11 ms @2048² (vs 0.197 for the RGBA8-source predecessor) at
// −0.5 dB average on the suite textures.
//
// ALGORITHM — scalar-luma selection, leanest form:
//   • The ETC1 modifier is a SCALAR shift along (1,1,1): per texel
//     err(m) = ‖e‖² − 2mD + 3m² with D = luma(p) − luma(base), so table and
//     index selection run on |D| threshold tests alone (A3/B3/THR are 3× the
//     modifier magnitudes; THR = 1.5(a+b)).
//   • Flip preselect, O(1) from quadrant sums: the flip whose quadrant
//     pairing has the smaller CHROMA split (rgb difference minus its luma
//     component — the part luma modulation cannot fix) wins. Exact-grayscale
//     blocks tie at zero, and near-ties evaluate both flips.
//   • Table pick, O(1) + one 8-texel max pass: the table whose large
//     magnitude covers max|D|, downgraded to its neighbour when the D mass
//     sits well below the extreme (mean-square D < max²/4) — the unscored
//     stand-in for the old two-candidate search.
//   • The flip contest estimate is RELATIVE (the flip-invariant Σ‖p‖² is
//     dropped): −2b·Σp + 8‖b‖² per subblock, minus κ·v/3 (κ = 0.85) where
//     v is the luma D-variance the tables largely absorb. The SIGN of that
//     term matters: charging v as a penalty instead of a credit inverts the
//     flip choice (and, historically, handed every block to planar).
//   • NO planar mode and NO base refit — both dropped in the bandwidth
//     rewrite (planar's contest needs estimates the split source can't
//     price reliably, and its win rate no longer justified the divergent
//     packing; the refit's ~0.2 dB cost 13-30% GPU). The 'high' CPU
//     reference still emits planar, bounding what it would buy.
//   • T and H modes are decoded by hardware but never emitted.
//
// Padding is baked into the prepared planes by etc2_prep.wgsl (source reads
// clamp there), so all reads here are unclamped.

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
};

// Prepared source planes (etc2_prep.wgsl): packed luma at binding 0,
// half-res quadrant averages at binding 3 — 2 bytes/pixel total.
// (ab-src-yq32: tells the /ab harness to feed prepared planes, not RGBA8.)
@group(0) @binding(0) var src_tex: texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var q_tex: texture_2d<f32>;

const A3  = array<f32, 8>(6.0, 15.0, 27.0, 39.0, 54.0, 72.0, 99.0, 141.0);
const B3  = array<f32, 8>(24.0, 51.0, 87.0, 126.0, 180.0, 240.0, 318.0, 549.0);
const THR = array<f32, 8>(15.0, 33.0, 57.0, 82.5, 117.0, 156.0, 208.5, 345.0);

fn texel_of(flip: u32, sb: u32, i: u32) -> u32 {
  if (flip == 0u) {
    return (i >> 1u) * 4u + sb * 2u + (i & 1u);
  }
  return (sb * 2u + (i >> 2u)) * 4u + (i & 3u);
}

fn quant_codes(v: vec3<f32>, max_code: vec3<f32>) -> vec3<u32> {
  return vec3<u32>(clamp(floor(v * max_code * (1.0 / 255.0) + 0.5), vec3<f32>(0.0), max_code));
}

fn extend4(c: vec3<u32>) -> vec3<f32> {
  return vec3<f32>((c << vec3<u32>(4u)) | c);
}
fn extend5(c: vec3<u32>) -> vec3<f32> {
  return vec3<f32>((c << vec3<u32>(3u)) | (c >> vec3<u32>(2u)));
}

fn signed3(bits: u32) -> i32 {
  return select(i32(bits), i32(bits) - 8, bits > 3u);
}

fn bswap(x: u32) -> u32 {
  return ((x & 0xffu) << 24u) | ((x & 0xff00u) << 8u) | ((x >> 8u) & 0xff00u) | (x >> 24u);
}

struct BasePair {
  codes0: vec3<u32>,
  codes1: vec3<u32>,
  ok: bool,
};
fn quantise_bases(avg0: vec3<f32>, avg1: vec3<f32>, diff: bool, clamp_delta: bool) -> BasePair {
  var out: BasePair;
  out.ok = true;
  if (!diff) {
    out.codes0 = quant_codes(avg0, vec3<f32>(15.0));
    out.codes1 = quant_codes(avg1, vec3<f32>(15.0));
    return out;
  }
  let q0 = vec3<i32>(quant_codes(avg0, vec3<f32>(31.0)));
  let q1 = vec3<i32>(quant_codes(avg1, vec3<f32>(31.0)));
  let d = q1 - q0;
  if (any(d < vec3<i32>(-4)) || any(d > vec3<i32>(3))) {
    if (!clamp_delta) {
      out.ok = false;
      return out;
    }
  }
  out.codes0 = vec3<u32>(q0);
  out.codes1 = vec3<u32>(q0 + clamp(d, vec3<i32>(-4), vec3<i32>(3)));
  return out;
}

// Table covering the subblock's exact max|D| (from quadrant luma extremes)
// — the shipped shader's "cover" candidate, O(1).
fn table_from_max(mx: f32) -> u32 {
  return min(
    u32(mx > 24.0) + u32(mx > 51.0) + u32(mx > 87.0) + u32(mx > 126.0) +
    u32(mx > 180.0) + u32(mx > 240.0) + u32(mx > 318.0),
    7u,
  );
}

// One flip's base quantisation + table search: everything the flip contest
// and the index derivation need.
struct FlipFit {
  est: f32,
  diff: bool,
  bases: BasePair,
  lb0: f32,
  lb1: f32,
  t0: u32,
  t1: u32,
};
fn sb_maxad(luma: ptr<function, array<f32, 16>>, flip: u32, sb: u32, lb: f32) -> f32 {
  var mx = 0.0;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    mx = max(mx, abs((*luma)[texel_of(flip, sb, i)] - lb));
  }
  return mx;
}
fn eval_flip(
  luma: ptr<function, array<f32, 16>>,
  flip: u32,
  sum0: vec3<f32>,
  lsq0: f32,
  sum1: vec3<f32>,
  lsq1: f32,
) -> FlipFit {
  let avg0 = sum0 * 0.125;
  let avg1 = sum1 * 0.125;
  let try_diff = quantise_bases(avg0, avg1, true, false);
  var out: FlipFit;
  out.diff = try_diff.ok;
  if (out.diff) {
    out.bases = try_diff;
  } else {
    out.bases = quantise_bases(avg0, avg1, false, false);
  }
  var b0: vec3<f32>;
  var b1: vec3<f32>;
  if (out.diff) {
    b0 = extend5(out.bases.codes0);
    b1 = extend5(out.bases.codes1);
  } else {
    b0 = extend4(out.bases.codes0);
    b1 = extend4(out.bases.codes1);
  }
  out.lb0 = b0.r + b0.g + b0.b;
  out.lb1 = b1.r + b1.g + b1.b;
  // D-variance per subblock drives the table pick AND the flip/planar
  // contest. est is RELATIVE: the flip-invariant Σ‖p‖² is dropped (the
  // planar estimate drops the same constant, so the contest is unbiased).
  let v0 = max(lsq0 - 2.0 * out.lb0 * dot(sum0, vec3<f32>(1.0)) + 8.0 * out.lb0 * out.lb0, 0.0);
  let v1 = max(lsq1 - 2.0 * out.lb1 * dot(sum1, vec3<f32>(1.0)) + 8.0 * out.lb1 * out.lb1, 0.0);
  // Hedge: when the D mass sits well below the extreme (mean-square D
  // under half the peak squared), the neighbour table serves the mass
  // better than full coverage serves the outlier.
  let mx0 = sb_maxad(luma, flip, 0u, out.lb0);
  let mx1 = sb_maxad(luma, flip, 1u, out.lb1);
  let c0 = table_from_max(mx0);
  let c1 = table_from_max(mx1);
  out.t0 = select(c0, c0 - 1u, c0 > 0u && v0 * 0.25 < mx0 * mx0);
  out.t1 = select(c1, c1 - 1u, c1 > 0u && v1 * 0.25 < mx1 * mx1);
  // The modifier tables ABSORB most of the luma D energy: the estimate
  // subtracts κ·v/3 (κ<1 for imperfect tables). Sign matters twice — it
  // steers both the flip contest and the ETC1-vs-planar contest.
  out.est = (-2.0 * dot(b0, sum0) + 8.0 * dot(b0, b0)) +
            (-2.0 * dot(b1, sum1) + 8.0 * dot(b1, b1)) -
            (v0 + v1) * (0.85 / 3.0);
  return out;
}

// Wire indices for a chosen table — computed ONCE, from the final base.
fn sb_indices(luma: ptr<function, array<f32, 16>>, flip: u32, sb: u32, lb: f32, t: u32) -> u32 {
  let thr = THR[t];
  var indices = 0u;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    let d = (*luma)[texel_of(flip, sb, i)] - lb;
    let large = abs(d) > thr;
    let neg = d < 0.0;
    indices = indices | ((select(0u, 1u, large) | select(0u, 2u, neg)) << (i * 2u));
  }
  return indices;
}

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) {
    return;
  }

  let block_index = gid.y * params.blocks_x + gid.x;
  let base_xy = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);

  var luma: array<f32, 16>;
  var qsum: array<vec3<f32>, 4>;
  var qlsq: array<f32, 4>;

  // One packed fetch per block row: 4 lumas (round((r+g+b)/3), bytes along
  // x, little-endian). Blocks are 4-aligned so the packed x = block x, and
  // the prep pass wrote every padded row, so no clamping is needed.
  for (var ry: u32 = 0u; ry < 4u; ry = ry + 1u) {
    let w = textureLoad(src_tex, vec2<i32>(i32(gid.x), base_xy.y + i32(ry)), 0).r;
    let l0 = f32(w & 255u) * 3.0;
    let l1 = f32((w >> 8u) & 255u) * 3.0;
    let l2 = f32((w >> 16u) & 255u) * 3.0;
    let l3 = f32(w >> 24u) * 3.0;
    luma[ry * 4u] = l0;
    luma[ry * 4u + 1u] = l1;
    luma[ry * 4u + 2u] = l2;
    luma[ry * 4u + 3u] = l3;
    qlsq[select(2u, 0u, ry < 2u)] += l0 * l0 + l1 * l1;
    qlsq[select(3u, 1u, ry < 2u)] += l2 * l2 + l3 * l3;
  }
  // Quadrant RGB sums from the half-res average plane (avg·4 = sum). The
  // plane covers the full padded grid, so reads are unclamped.
  let qb = vec2<i32>(i32(gid.x) * 2, i32(gid.y) * 2);
  for (var q: u32 = 0u; q < 4u; q = q + 1u) {
    qsum[q] = textureLoad(q_tex, qb + vec2<i32>(i32(q & 1u), i32(q >> 1u)), 0).rgb * 1020.0;
  }

  // ----------------------------------------------- flip + base selection --
  // Flip preselect, O(1) from quadrant sums: per subblock the residual after
  // PERFECT continuous luma modulation is (Σ||p||² − ||Σp||²/8) −
  // (Σℓ² − (Σℓ)²/8)/3 — the within-variance minus the (1,1,1)-direction
  // component the modifier tables can absorb. The flip minimising the summed
  // residual wins and only it gets the table search — EXCEPT when the two
  // residuals are indistinguishable: for exact-grayscale blocks (r=g=b) both
  // are identically zero, so the contest falls back to scoring both flips
  // (this recovered −1.25 dB on roughness/AO-style content).
  let sum0a = qsum[0] + qsum[2];
  let sum1a = qsum[1] + qsum[3];
  let sum0b = qsum[0] + qsum[1];
  let sum1b = qsum[2] + qsum[3];
  let lsq0a = qlsq[0] + qlsq[2];
  let lsq1a = qlsq[1] + qlsq[3];
  let lsq0b = qlsq[0] + qlsq[1];
  let lsq1b = qlsq[2] + qlsq[3];
  // Quadrant-level chroma splits (rgb difference minus its luma component):
  // the unfixable part of pairing quadrants into a subblock.
  let da = qsum[0] - qsum[2];
  let da2 = qsum[1] - qsum[3];
  let db = qsum[0] - qsum[1];
  let db2 = qsum[2] - qsum[3];
  let res_a = (dot(da, da) - dot(da, vec3<f32>(1.0)) * dot(da, vec3<f32>(1.0)) * (1.0 / 3.0)) +
              (dot(da2, da2) - dot(da2, vec3<f32>(1.0)) * dot(da2, vec3<f32>(1.0)) * (1.0 / 3.0));
  let res_b = (dot(db, db) - dot(db, vec3<f32>(1.0)) * dot(db, vec3<f32>(1.0)) * (1.0 / 3.0)) +
              (dot(db2, db2) - dot(db2, vec3<f32>(1.0)) * dot(db2, vec3<f32>(1.0)) * (1.0 / 3.0));

  // Single eval_flip call site (a second inlined copy measured +50% GPU):
  // attempt 0 scores the primary flip, attempt 1 runs only in the dual
  // (indistinguishable-residuals) case and scores the other flip.
  let dual = abs(res_a - res_b) < 1.0;
  let primary = select(select(0u, 1u, res_b < res_a), 0u, dual);
  var bflip = primary;
  var sel: FlipFit;
  for (var attempt = 0u; attempt < 2u; attempt = attempt + 1u) {
    if (attempt == 1u && !dual) {
      break;
    }
    let f = select(primary, 1u, attempt == 1u);
    let cand = eval_flip(
      &luma,
      f,
      select(sum0a, sum0b, f == 1u),
      select(lsq0a, lsq0b, f == 1u),
      select(sum1a, sum1b, f == 1u),
      select(lsq1a, lsq1b, f == 1u),
    );
    if (attempt == 0u || cand.est < sel.est) {
      sel = cand;
      bflip = f;
    }
  }
  let bdiff = sel.diff;

  let codes0 = sel.bases.codes0;
  let codes1 = sel.bases.codes1;
  let t0 = sel.t0;
  let t1 = sel.t1;
  let fit0 = sb_indices(&luma, bflip, 0u, sel.lb0, t0);
  let fit1 = sb_indices(&luma, bflip, 1u, sel.lb1, t1);

  var hi: u32;
  var lo: u32;
  {
    if (bdiff) {
      let d = vec3<u32>(vec3<i32>(codes1) - vec3<i32>(codes0)) & vec3<u32>(7u);
      hi = (codes0.r << 27u) | (d.r << 24u) | (codes0.g << 19u) | (d.g << 16u) | (codes0.b << 11u) | (d.b << 8u)
         | (t0 << 5u) | (t1 << 2u) | 2u | bflip;
    } else {
      hi = (codes0.r << 28u) | (codes1.r << 24u) | (codes0.g << 20u) | (codes1.g << 16u) | (codes0.b << 12u) | (codes1.b << 8u)
         | (t0 << 5u) | (t1 << 2u) | bflip;
    }
    lo = 0u;
    for (var sb: u32 = 0u; sb < 2u; sb = sb + 1u) {
      let indices = select(fit0, fit1, sb == 1u);
      for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let k = texel_of(bflip, sb, i);
        let wire = (k & 3u) * 4u + (k >> 2u);
        let idx = (indices >> (i * 2u)) & 3u;
        lo = lo | ((idx & 1u) << wire) | ((idx >> 1u) << (16u + wire));
      }
    }
  }

  let out = block_index * 2u;
  dst[out]      = bswap(hi);
  dst[out + 1u] = bswap(lo);
}
