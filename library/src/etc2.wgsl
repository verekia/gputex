// ETC2 RGB8 compute shader encoder.
//
// Each invocation encodes one 4x4 pixel block into an 8-byte ETC2 RGB8 block
// written as 2 x u32 into the destination storage buffer. ETC2 blocks are
// big-endian on the wire (byte 0 = bits 63..56), so both words are byte-
// swapped on the way out. f32 only — there is no f16 module (the estimates
// are integer-exact sums that overflow f16, and the pass is read-bound).
//
// ALGORITHM — scalar-luma selection (2026-07; brute force measured 6.0 ms
// @2048² on Apple/metal-3, this shader ~0.18 ms with the DRAM read floor
// at ~0.15):
//
//   • The ETC1 modifier is a SCALAR shift along (1,1,1), so per texel
//     err(m) = ‖e‖² − 2mD + 3m² with D = luma(p) − luma(base), where
//     luma(x) = x.r+x.g+x.b. Selection therefore needs only |D| threshold
//     tests (A3/B3/THR are 3× the modifier magnitudes; THR = 1.5(a+b)),
//     and Σ‖e‖² per subblock is O(1) from the load loop's quadrant sums.
//   • Flip preselect, O(1): per subblock the residual after PERFECT
//     continuous luma modulation is within-variance − (luma variance)/3;
//     the flip with the smaller summed residual wins and only it is
//     evaluated. Exact-grayscale blocks tie at zero, so near-ties evaluate
//     both flips (worth ~1.25 dB on roughness/AO-style content).
//   • Table pick: one 8-texel max pass per subblock gives the table whose
//     large magnitude covers max|D|, downgraded to its neighbour when the D
//     mass sits well below the extreme (mean-square D < max²/4, from the
//     O(1) luma D-variance). The chosen table is then scored EXACTLY in a
//     single pass — replacing the old two-candidate scored search: one
//     score pass saved, and the hedge decides between the same two
//     candidates it used to score. The exact score matters: a variance
//     proxy for it mis-prices perfectly representable blocks and hands
//     them to planar.
//   • NO base refit (its ~0.2 dB cost 13-30% GPU — dropped earlier).
//   • PLANAR runs unconditionally: with the right-hand sides folded into
//     the load loop the LSQ solve is O(1) (constant Gram inverse, det 25)
//     and its residual is the closed-form Σ‖p‖² − 2·θ·rhs + θᵀGθ with the
//     QUANTISED, clamped corners — clamp-aware, which the contest needs.
//   • T and H modes are decoded by hardware but never emitted.
//
// A two-pass variant (prepared 2 B/px split source, etc2_prep.wgsl in git
// history) made the encode pass ~0.115 ms but the preparation pass is also
// DRAM-bound and cannot overlap it, so the per-texture total regressed to
// ~0.30 ms — reverted. Reading the full RGBA8 source once (~0.15 ms of
// bandwidth at 2048² on a 100 GB/s part) is this machine's hard floor for
// any single-pass encoder.
//
// Numeric notes: texel loads use round(load·255) (integer-exact unorm trip);
// every m3 in A3/B3 is divisible by 3 so m = m3/3 is exact; est values are
// integer sums held exactly in f32 (< 2^24) apart from the κ·v/3 credit.

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const A3  = array<f32, 8>(6.0, 15.0, 27.0, 39.0, 54.0, 72.0, 99.0, 141.0);
const B3  = array<f32, 8>(24.0, 51.0, 87.0, 126.0, 180.0, 240.0, 318.0, 549.0);
const THR = array<f32, 8>(15.0, 33.0, 57.0, 82.5, 117.0, 156.0, 208.5, 345.0);

// Planar's closed-form estimate models the QUANTISED corners exactly; only
// decode's floor-rounding (±½ per sample) is unmodelled. This small bias
// keeps near-ties on the predictable ETC1 side.
const PLANAR_FUDGE = 8.0;

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

// Table pick: ONE max pass, then the table whose large magnitude covers
// max|D|, downgraded to its neighbour when the D mass sits well below the
// extreme (mean-square D under max²/4, v = luma D-variance about the
// base). Replaces the two-candidate scored search: one of its two score
// passes is saved, and the remaining one scores only the chosen table —
// keeping the estimate EXACT (a v-variance proxy mis-prices perfectly
// representable blocks and hands them to planar).
fn sb_maxad(luma: ptr<function, array<f32, 16>>, flip: u32, sb: u32, lb: f32) -> f32 {
  var mx = 0.0;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    mx = max(mx, abs((*luma)[texel_of(flip, sb, i)] - lb));
  }
  return mx;
}
fn table_hedged(mx: f32, v: f32) -> u32 {
  let cover = min(
    u32(mx > 24.0) + u32(mx > 51.0) + u32(mx > 87.0) + u32(mx > 126.0) +
    u32(mx > 180.0) + u32(mx > 240.0) + u32(mx > 318.0),
    7u,
  );
  return select(cover, cover - 1u, cover > 0u && v * 0.25 < mx * mx);
}
fn sb_table_score(luma: ptr<function, array<f32, 16>>, flip: u32, sb: u32, lb: f32, t: u32) -> f32 {
  let a3 = A3[t];
  let b3 = B3[t];
  let thr = THR[t];
  var acc = 0.0;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    let ad = abs((*luma)[texel_of(flip, sb, i)] - lb);
    let m3 = select(a3, b3, ad > thr);
    acc = acc + m3 * (m3 - 2.0 * ad);
  }
  return acc;
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
fn eval_flip(
  luma: ptr<function, array<f32, 16>>,
  flip: u32,
  sum0: vec3<f32>,
  sq0: f32,
  lsq0: f32,
  sum1: vec3<f32>,
  sq1: f32,
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
  // Luma D-variance about each base drives the table hedge; the chosen
  // table is then scored exactly (Σ m3(m3 − 2|D|), negative when the
  // modifiers absorb energy), keeping the estimate exact modulo clamping.
  let v0 = max(lsq0 - 2.0 * out.lb0 * dot(sum0, vec3<f32>(1.0)) + 8.0 * out.lb0 * out.lb0, 0.0);
  let v1 = max(lsq1 - 2.0 * out.lb1 * dot(sum1, vec3<f32>(1.0)) + 8.0 * out.lb1 * out.lb1, 0.0);
  out.t0 = table_hedged(sb_maxad(luma, flip, 0u, out.lb0), v0);
  out.t1 = table_hedged(sb_maxad(luma, flip, 1u, out.lb1), v1);
  out.est = (sq0 - 2.0 * dot(b0, sum0) + 8.0 * dot(b0, b0)) +
            (sq1 - 2.0 * dot(b1, sum1) + 8.0 * dot(b1, b1)) +
            (sb_table_score(luma, flip, 0u, out.lb0, out.t0) +
             sb_table_score(luma, flip, 1u, out.lb1, out.t1)) * (1.0 / 3.0);
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
  let max_xy = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  var luma: array<f32, 16>;
  var qsum: array<vec3<f32>, 4>;
  var qsq: array<f32, 4>;
  var qlsq: array<f32, 4>;
  // Planar right-hand sides, folded into the load: rB = Σ (x/4)·p and
  // rC = Σ (y/4)·p accumulate unscaled; rA = Σp − rB − rC afterwards.
  var sxp = vec3<f32>(0.0);
  var syp = vec3<f32>(0.0);

  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let lx = i32(i & 3u);
    let ly = i32(i >> 2u);
    let p = clamp(base_xy + vec2<i32>(lx, ly), vec2<i32>(0, 0), max_xy);
    let c = round(textureLoad(src_tex, p, 0).rgb * 255.0);
    let l = c.r + c.g + c.b;
    luma[i] = l;
    let q = u32(lx >= 2) | (u32(ly >= 2) << 1u);
    qsum[q] = qsum[q] + c;
    qsq[q] = qsq[q] + dot(c, c);
    qlsq[q] = qlsq[q] + l * l;
    sxp = sxp + f32(lx) * c;
    syp = syp + f32(ly) * c;
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
  let sq0a = qsq[0] + qsq[2];
  let sq1a = qsq[1] + qsq[3];
  let sum0b = qsum[0] + qsum[1];
  let sum1b = qsum[2] + qsum[3];
  let sq0b = qsq[0] + qsq[1];
  let sq1b = qsq[2] + qsq[3];
  let lsq0a = qlsq[0] + qlsq[2];
  let lsq1a = qlsq[1] + qlsq[3];
  let lsq0b = qlsq[0] + qlsq[1];
  let lsq1b = qlsq[2] + qlsq[3];
  let res_a = (sq0a - dot(sum0a, sum0a) * 0.125) - (lsq0a - dot(sum0a, vec3<f32>(1.0)) * dot(sum0a, vec3<f32>(1.0)) * 0.125) * (1.0 / 3.0)
            + (sq1a - dot(sum1a, sum1a) * 0.125) - (lsq1a - dot(sum1a, vec3<f32>(1.0)) * dot(sum1a, vec3<f32>(1.0)) * 0.125) * (1.0 / 3.0);
  let res_b = (sq0b - dot(sum0b, sum0b) * 0.125) - (lsq0b - dot(sum0b, vec3<f32>(1.0)) * dot(sum0b, vec3<f32>(1.0)) * 0.125) * (1.0 / 3.0)
            + (sq1b - dot(sum1b, sum1b) * 0.125) - (lsq1b - dot(sum1b, vec3<f32>(1.0)) * dot(sum1b, vec3<f32>(1.0)) * 0.125) * (1.0 / 3.0);

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
      select(sq0a, sq0b, f == 1u),
      select(lsq0a, lsq0b, f == 1u),
      select(sum1a, sum1b, f == 1u),
      select(sq1a, sq1b, f == 1u),
      select(lsq1a, lsq1b, f == 1u),
    );
    if (attempt == 0u || cand.est < sel.est) {
      sel = cand;
      bflip = f;
    }
  }
  let bdiff = sel.diff;

  let best_est = sel.est;
  let codes0 = sel.bases.codes0;
  let codes1 = sel.bases.codes1;
  let t0 = sel.t0;
  let t1 = sel.t1;
  let fit0 = sb_indices(&luma, bflip, 0u, sel.lb0, t0);
  let fit1 = sb_indices(&luma, bflip, 1u, sel.lb1, t1);

  // ------------------------------------------------------------ planar --
  // Always evaluated: with the rhs folded into the load loop this is O(1),
  // and gating it on the ETC1 estimate measured −0.31 dB on smooth content
  // for zero speed.
  let total = qsum[0] + qsum[1] + qsum[2] + qsum[3];
  let sqtotal = qsq[0] + qsq[1] + qsq[2] + qsq[3];
  let rB = sxp * 0.25;
  let rC = syp * 0.25;
  let rA = total - rB - rC;
  let po = 0.2875 * rA - 0.0125 * rB - 0.0125 * rC;
  let ph = -0.0125 * rA + 0.4875 * rB - 0.3125 * rC;
  let pv = -0.0125 * rA - 0.3125 * rB + 0.4875 * rC;
  let pmax = vec3<f32>(63.0, 127.0, 63.0);
  let qo = quant_codes(po, pmax);
  let qh = quant_codes(ph, pmax);
  let qv = quant_codes(pv, pmax);
  // Residual of the plane the hardware will ACTUALLY decode — the
  // quantised, clamped corners — via the normal-equation identity
  // Σ||p − f||² = Σ||p||² − 2·θ·rhs + θᵀGθ (G is the constant Gram matrix
  // of the fixed sample positions). Estimating with the CONTINUOUS corners
  // instead is blind to corner clamping and mis-picks planar on steep
  // gradients (a 1.4-normalised-SSE easy-block artifact on the colour
  // card). Only decode's floor-rounding stays unmodelled (≤ ~12 SSE).
  let shl = vec3<u32>(2u, 1u, 2u);
  let shr = vec3<u32>(4u, 6u, 4u);
  let eo = vec3<f32>((qo << shl) | (qo >> shr));
  let eh = vec3<f32>((qh << shl) | (qh >> shr));
  let ev = vec3<f32>((qv << shl) | (qv >> shr));
  let gram = 3.5 * (eo * eo + eh * eh + ev * ev) + 0.5 * eo * eh + 0.5 * eo * ev + 4.5 * eh * ev;
  let planar_est = sqtotal - 2.0 * (dot(eo, rA) + dot(eh, rB) + dot(ev, rC)) +
                   dot(gram, vec3<f32>(1.0)) + PLANAR_FUDGE;

  // ------------------------------------------------------------ packing --
  var hi: u32;
  var lo: u32;
  if (best_est <= planar_est) {
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
  } else {
    let ro = qo.r; let go = qo.g; let bo = qo.b;
    let rh = qh.r; let gh = qh.g; let bh = qh.b;
    let rv = qv.r; let gv = qv.g; let bv = qv.b;
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
