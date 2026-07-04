// ETC2 RGB8 reference encoder + decoder. CPU implementation.
//
// ETC2 RGB8 compresses a 4×4 RGB block into 8 bytes (big-endian bit order:
// byte 0 holds bits 63..56). Five block modes share the 64 bits:
//
//   • individual (ETC1, diffbit=0): two subblocks (2×4 or 4×2, per the flip
//     bit), each with a 4-bit-per-channel base colour and a 3-bit modifier
//     table; every texel adds one of the table's 4 scalar modifiers to its
//     subblock base (a luma shift), selected by a 2-bit index.
//   • differential (ETC1, diffbit=1): same, but base 1 is 5 bits/channel and
//     base 2 is base 1 + a 3-bit two's-complement delta per channel.
//   • T / H (ETC2): 4-colour paint palettes for chroma-split blocks,
//     signalled by driving the differential R (T) or G (H) sum out of
//     [0, 31]. Decoded here; the encoder does not emit them (see below).
//   • planar (ETC2): three RGB corner colours (O, H, V) define a colour
//     gradient over the block, signalled by driving the differential B sum
//     out of range. The mode that makes ETC2 usable on smooth content.
//
// Texel order on the wire is COLUMN-major: texel (x, y) — x = column — is
// index x·4+y, its 2-bit modifier index split across the low word's halves
// (LSB at bit x·4+y, MSB at bit 16+x·4+y). This module's public API uses the
// same channel-interleaved ROW-major [0, 1] layout as the other `_ref`
// modules; the wire mapping is internal.
//
// QUALITY LEVELS
//   'fast' (default): the CPU mirror of etc2.wgsl's scalar-luma encoder —
//     the ETC1 modifier is a scalar shift along (1,1,1), so table and index
//     selection run on D = luma(p) − luma(base) alone (exact modulo decode
//     clamping): O(1) flip preselect from quadrant variance, subblock-average
//     bases (differential when the delta fits, else individual), a
//     hedged O(1) table pick around max|D| (the scored two-candidate
//     search and the base refit were dropped as speed/quality trades),
//     and a closed-form planar contest.
//   'high': exhaustive — both flips, BOTH differential (delta clamped into
//     range) and individual bases, full 8-table × 4-modifier search with
//     decode-exact clamped errors, up to 4 exact-accepted refit rounds, a
//     planar fit polished by ±1 coordinate descent — and finally the 'fast'
//     result, keeping whichever decodes with lower error. That last step
//     makes 'high' ≤ 'fast' per block BY CONSTRUCTION, which is what lets
//     the GPU suite gate the shader against it as the reference.
//
// The encoder never emits T or H: their win over the modes above is narrow
// (two-chroma-cluster blocks) and needs a clustering pass that would double
// the encoder for a corner case. The decoder handles all five modes so any
// spec-valid ETC2 RGB8 stream round-trips.
//
// Arithmetic happens in integer 0..255 space (errors are exact integer sums)
// so the reference tracks the WGSL port closely — f32 holds every value here
// exactly.

/** 48-value input: 16 RGB triplets, channel-interleaved, row-major, each in [0, 1]. */
export type ETC2Pixels = Readonly<ArrayLike<number>>

/** Exactly 8 bytes of ETC2 RGB8 block data. */
export type ETC2Block = Uint8Array

export type ETC2Quality = 'fast' | 'high'

export type ETC2Mode = 'individual' | 'differential' | 'T' | 'H' | 'planar'

// The 8 ETC1 modifier tables in WIRE index order: index 0 = small positive,
// 1 = large positive, 2 = small negative, 3 = large negative.
const MODIFIERS: readonly (readonly [number, number, number, number])[] = [
  [2, 8, -2, -8],
  [5, 17, -5, -17],
  [9, 29, -9, -29],
  [13, 42, -13, -42],
  [18, 60, -18, -60],
  [24, 80, -24, -80],
  [33, 106, -33, -106],
  [47, 183, -47, -183],
]

// T/H paint-colour distance table (3-bit index).
const TH_DISTANCES: readonly number[] = [3, 6, 11, 16, 23, 32, 41, 64]

const clamp255 = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : v)

// Bit-replication extensions to 8 bits — what the hardware decoder applies.
const extend4 = (x: number): number => (x << 4) | x
const extend5 = (x: number): number => (x << 3) | (x >> 2)
const extend6 = (x: number): number => (x << 2) | (x >> 4)
const extend7 = (x: number): number => (x << 1) | (x >> 6)

// Round-to-nearest quantisation of a 0..255 value into an n-bit code.
const quant = (v: number, maxCode: number): number => {
  const q = Math.floor((v * maxCode) / 255 + 0.5)
  return q < 0 ? 0 : q > maxCode ? maxCode : q
}

/** 3-bit two's complement → signed. */
const signed3 = (bits: number): number => (bits > 3 ? bits - 8 : bits)

// -------------------------------------------------------------------------
// Subblock geometry
// -------------------------------------------------------------------------
//
// Row-major texel indices (k = y·4+x) of subblock `sb` for a flip value.
// flip=0 splits into left/right 2×4 halves (by column), flip=1 into
// top/bottom 4×2 halves (by row). Table/base 1 always covers subblock 0.
function subblockTexels(flip: number, sb: number): number[] {
  const out: number[] = []
  for (let k = 0; k < 16; k++) {
    const x = k & 3
    const y = k >> 2
    const inSb1 = flip === 0 ? x >= 2 : y >= 2
    if ((inSb1 ? 1 : 0) === sb) out.push(k)
  }
  return out
}

const SUBBLOCKS: readonly (readonly number[])[][] = [
  [subblockTexels(0, 0), subblockTexels(0, 1)],
  [subblockTexels(1, 0), subblockTexels(1, 1)],
]

// -------------------------------------------------------------------------
// ETC1-family search (individual + differential)
// -------------------------------------------------------------------------

interface SubblockFit {
  table: number
  /** Wire 2-bit index per texel of the subblock, in `texels` order. */
  indices: Uint8Array
  err: number
  /** Sum of the chosen scalar modifiers — feeds the base-colour refit. */
  modSum: number
}

/**
 * For one subblock and a decoded (8-bit) base colour, pick the modifier
 * table and per-texel indices minimising squared error. Exhaustive over the
 * 8 tables × 4 modifiers; error accumulated with decode-exact clamping.
 */
function fitSubblock(px: Int32Array, texels: readonly number[], base: readonly number[]): SubblockFit {
  const best: SubblockFit = { table: 0, indices: new Uint8Array(8), err: Infinity, modSum: 0 }
  const idx = new Uint8Array(8)
  for (let t = 0; t < 8; t++) {
    const mods = MODIFIERS[t]!
    let err = 0
    let modSum = 0
    for (let i = 0; i < texels.length; i++) {
      const k = texels[i]!
      const r = px[k * 3]!
      const g = px[k * 3 + 1]!
      const b = px[k * 3 + 2]!
      let bestJ = 0
      let bestD = Infinity
      let bestM = 0
      for (let j = 0; j < 4; j++) {
        const m = mods[j]!
        const dr = clamp255(base[0]! + m) - r
        const dg = clamp255(base[1]! + m) - g
        const db = clamp255(base[2]! + m) - b
        const d = dr * dr + dg * dg + db * db
        if (d < bestD) {
          bestD = d
          bestJ = j
          bestM = m
        }
      }
      idx[i] = bestJ
      err += bestD
      modSum += bestM
    }
    if (err < best.err) {
      best.table = t
      best.indices.set(idx)
      best.err = err
      best.modSum = modSum
    }
  }
  return best
}

interface Etc1Candidate {
  diff: boolean
  flip: number
  /** Quantised base codes (4- or 5-bit per channel). */
  codes0: number[]
  codes1: number[]
  fit0: SubblockFit
  fit1: SubblockFit
  err: number
}

/** Quantise two subblock-average colours as a differential pair (5-bit base +
 *  clamped delta) or as independent 4-bit individual bases. Returns the codes,
 *  or null when `diff` is requested and the delta doesn't fit (no clamping). */
function quantiseBases(
  avg0: readonly number[],
  avg1: readonly number[],
  diff: boolean,
  clampDelta: boolean,
): { codes0: number[]; codes1: number[] } | null {
  if (!diff) {
    return { codes0: avg0.map(v => quant(v, 15)), codes1: avg1.map(v => quant(v, 15)) }
  }
  const codes0 = avg0.map(v => quant(v, 31))
  const codes1: number[] = []
  for (let c = 0; c < 3; c++) {
    let d = quant(avg1[c]!, 31) - codes0[c]!
    if (d < -4 || d > 3) {
      if (!clampDelta) return null
      d = d < -4 ? -4 : 3
    }
    codes1.push(codes0[c]! + d)
  }
  return { codes0, codes1 }
}

const decodeBases = (codes: readonly number[], diff: boolean): number[] => codes.map(diff ? extend5 : extend4)

/**
 * Evaluate one ETC1-family candidate: fit both subblocks from the given base
 * codes, then run `refits` rounds of base-colour refinement (base ← subblock
 * mean − mean chosen modifier, requantised), each accepted only on strictly
 * lower total error.
 */
function fitEtc1(
  px: Int32Array,
  flip: number,
  diff: boolean,
  clampDelta: boolean,
  avg0: number[],
  avg1: number[],
  refits: number,
): Etc1Candidate | null {
  const texels0 = SUBBLOCKS[flip]![0]!
  const texels1 = SUBBLOCKS[flip]![1]!
  const q = quantiseBases(avg0, avg1, diff, clampDelta)
  if (!q) return null

  let { codes0, codes1 } = q
  let fit0 = fitSubblock(px, texels0, decodeBases(codes0, diff))
  let fit1 = fitSubblock(px, texels1, decodeBases(codes1, diff))
  let err = fit0.err + fit1.err

  for (let round = 0; round < refits; round++) {
    // Optimal (unquantised) base for the chosen modifiers: the modifier is a
    // scalar luma shift, so per channel base' = mean(texel) − mean(modifier).
    const shift0 = fit0.modSum / texels0.length
    const shift1 = fit1.modSum / texels1.length
    const nAvg0 = avg0.map(v => v - shift0)
    const nAvg1 = avg1.map(v => v - shift1)
    const nq = quantiseBases(nAvg0, nAvg1, diff, true)
    if (!nq) break
    if (nq.codes0.every((v, c) => v === codes0[c]) && nq.codes1.every((v, c) => v === codes1[c])) break
    const nFit0 = fitSubblock(px, texels0, decodeBases(nq.codes0, diff))
    const nFit1 = fitSubblock(px, texels1, decodeBases(nq.codes1, diff))
    if (nFit0.err + nFit1.err >= err) break
    codes0 = nq.codes0
    codes1 = nq.codes1
    fit0 = nFit0
    fit1 = nFit1
    err = fit0.err + fit1.err
  }

  return { diff, flip, codes0, codes1, fit0, fit1, err }
}

// -------------------------------------------------------------------------
// Planar fit
// -------------------------------------------------------------------------

// Least-squares plane fit. The decode is linear in the three corner colours:
// value(x, y) = O·(1 − x/4 − y/4) + H·(x/4) + V·(y/4). The 16 sample
// positions are fixed, so the 3×3 Gram matrix of those basis functions is a
// constant; this is its exact inverse (det = 25).
const PLANAR_GINV: readonly (readonly number[])[] = [
  [0.2875, -0.0125, -0.0125],
  [-0.0125, 0.4875, -0.3125],
  [-0.0125, -0.3125, 0.4875],
]

interface PlanarCandidate {
  /** Quantised codes per corner: [R6, G7, B6]. */
  o: number[]
  h: number[]
  v: number[]
  err: number
}

/** Decode-exact planar error for a set of quantised corner codes. */
function planarError(px: Int32Array, o: readonly number[], h: readonly number[], v: readonly number[]): number {
  const ext = (codes: readonly number[]): number[] => [extend6(codes[0]!), extend7(codes[1]!), extend6(codes[2]!)]
  const eo = ext(o)
  const eh = ext(h)
  const ev = ext(v)
  let err = 0
  for (let k = 0; k < 16; k++) {
    const x = k & 3
    const y = k >> 2
    for (let c = 0; c < 3; c++) {
      const val = clamp255((x * (eh[c]! - eo[c]!) + y * (ev[c]! - eo[c]!) + 4 * eo[c]! + 2) >> 2)
      const d = val - px[k * 3 + c]!
      err += d * d
    }
  }
  return err
}

function fitPlanar(px: Int32Array, polish: boolean): PlanarCandidate {
  // Per channel: accumulate the three basis-weighted sums, then solve with
  // the hardcoded inverse Gram matrix.
  const o: number[] = []
  const h: number[] = []
  const v: number[] = []
  const maxCode = [63, 127, 63]
  for (let c = 0; c < 3; c++) {
    let sA = 0
    let sB = 0
    let sC = 0
    for (let k = 0; k < 16; k++) {
      const x = (k & 3) / 4
      const y = (k >> 2) / 4
      const p = px[k * 3 + c]!
      sA += (1 - x - y) * p
      sB += x * p
      sC += y * p
    }
    const oc = PLANAR_GINV[0]![0]! * sA + PLANAR_GINV[0]![1]! * sB + PLANAR_GINV[0]![2]! * sC
    const hc = PLANAR_GINV[1]![0]! * sA + PLANAR_GINV[1]![1]! * sB + PLANAR_GINV[1]![2]! * sC
    const vc = PLANAR_GINV[2]![0]! * sA + PLANAR_GINV[2]![1]! * sB + PLANAR_GINV[2]![2]! * sC
    o.push(quant(clamp255(oc), maxCode[c]!))
    h.push(quant(clamp255(hc), maxCode[c]!))
    v.push(quant(clamp255(vc), maxCode[c]!))
  }
  let err = planarError(px, o, h, v)

  if (polish) {
    // One round of ±1 coordinate descent over the 9 quantised parameters.
    const params: number[][] = [o, h, v]
    for (let p = 0; p < 3; p++) {
      for (let c = 0; c < 3; c++) {
        const cur = params[p]![c]!
        for (const cand of [cur - 1, cur + 1]) {
          if (cand < 0 || cand > maxCode[c]!) continue
          params[p]![c] = cand
          const e = planarError(px, o, h, v)
          if (e < err) {
            err = e
          } else {
            params[p]![c] = cur
          }
        }
      }
    }
  }

  return { o, h, v, err }
}

// -------------------------------------------------------------------------
// Packing
// -------------------------------------------------------------------------

/** Write logical big-endian words (hi = bits 63..32, lo = bits 31..0) to bytes. */
function wordsToBlock(hi: number, lo: number): ETC2Block {
  const out = new Uint8Array(8)
  out[0] = (hi >>> 24) & 0xff
  out[1] = (hi >>> 16) & 0xff
  out[2] = (hi >>> 8) & 0xff
  out[3] = hi & 0xff
  out[4] = (lo >>> 24) & 0xff
  out[5] = (lo >>> 16) & 0xff
  out[6] = (lo >>> 8) & 0xff
  out[7] = lo & 0xff
  return out
}

/** Pack per-subblock wire indices into the low word (LSB half + MSB half). */
function packIndices(cand: Etc1Candidate): number {
  let lo = 0
  for (let sb = 0; sb < 2; sb++) {
    const texels = SUBBLOCKS[cand.flip]![sb]!
    const fit = sb === 0 ? cand.fit0 : cand.fit1
    for (let i = 0; i < texels.length; i++) {
      const k = texels[i]!
      const wire = (k & 3) * 4 + (k >> 2) // column-major texel number
      const idx = fit.indices[i]!
      lo |= (idx & 1) << wire
      lo |= ((idx >> 1) & 1) << (16 + wire)
    }
  }
  return lo >>> 0
}

function packEtc1(cand: Etc1Candidate): ETC2Block {
  const { codes0: c0, codes1: c1 } = cand
  let hi: number
  if (cand.diff) {
    const d = [c1[0]! - c0[0]!, c1[1]! - c0[1]!, c1[2]! - c0[2]!]
    hi =
      (c0[0]! << 27) |
      ((d[0]! & 7) << 24) |
      (c0[1]! << 19) |
      ((d[1]! & 7) << 16) |
      (c0[2]! << 11) |
      ((d[2]! & 7) << 8) |
      (cand.fit0.table << 5) |
      (cand.fit1.table << 2) |
      2 |
      cand.flip
  } else {
    hi =
      (c0[0]! << 28) |
      (c1[0]! << 24) |
      (c0[1]! << 20) |
      (c1[1]! << 16) |
      (c0[2]! << 12) |
      (c1[2]! << 8) |
      (cand.fit0.table << 5) |
      (cand.fit1.table << 2) |
      0 |
      cand.flip
  }
  return wordsToBlock(hi >>> 0, packIndices(cand))
}

/**
 * Pack a planar block. The planar fields are scattered across the ETC1
 * differential fields; the leftover "opacity" bits must be chosen so the
 * ETC1-decoder view keeps R and G sums IN [0, 31] (else the block would
 * read as T or H) while the B sum overflows (which is what signals planar).
 */
function packPlanar(cand: PlanarCandidate): ETC2Block {
  const ro = cand.o[0]!
  const go = cand.o[1]!
  const bo = cand.o[2]!
  const rh = cand.h[0]!
  const gh = cand.h[1]!
  const bh = cand.h[2]!
  const rv = cand.v[0]!
  const gv = cand.v[1]!
  const bv = cand.v[2]!

  // Free bit 63: the ETC1 view reads R1 = bits 63..59, dR = bits 58..56.
  // With bit 63 clear, R1 = RO >> 2 and dR = signed((RO & 3) << 1 | GO >> 6);
  // if the sum would go negative, setting bit 63 adds 16 and lands it back
  // in [12, 15] — always representable.
  const rSum = (ro >> 2) + signed3(((ro & 3) << 1) | (go >> 6))
  const rFix = rSum < 0 ? 1 : 0
  // Free bit 55: same construction for G1 = bits 55..51, dG = bits 50..48.
  // Only GO's low 6 bits (GO2) sit in the G1/dG fields — bit 6 lives at 56.
  const gSum = ((go >> 2) & 15) + signed3(((go & 3) << 1) | (bo >> 5))
  const gFix = gSum < 0 ? 1 : 0
  // Free bits 47..45 and 42: force the B sum OUT of [0, 31]. With p = the
  // BO bits at 44..43 and q = the BO bits at 41..40, either 111/+q pushes
  // the sum to 28+p+q > 31 (needs p+q ≥ 4) or 000/−(4−q) pulls it to
  // p+q−4 < 0 (needs p+q ≤ 3). Exactly one applies for every p, q.
  const p = (bo >> 3) & 3
  const q = (bo >> 1) & 3
  const bFix3 = p + q >= 4 ? 7 : 0
  const bFix1 = p + q >= 4 ? 0 : 1

  const hi =
    (rFix << 31) |
    (ro << 25) |
    ((go >> 6) << 24) |
    (gFix << 23) |
    ((go & 63) << 17) |
    ((bo >> 5) << 16) |
    (bFix3 << 13) |
    (((bo >> 3) & 3) << 11) |
    (bFix1 << 10) |
    ((bo & 7) << 7) |
    ((rh >> 1) << 2) |
    2 | // diffbit
    (rh & 1)
  const lo = (gh << 25) | (bh << 19) | (rv << 13) | (gv << 6) | bv
  return wordsToBlock(hi >>> 0, lo >>> 0)
}

// -------------------------------------------------------------------------
// Fast path — the CPU mirror of etc2.wgsl (see the shader header for the
// derivations and the measured speed/quality trade-offs)
// -------------------------------------------------------------------------

// 3× the modifier magnitudes (D-domain) and the small/large threshold
// 1.5·(a+b), plus the estimate gates shared with the shader.
const A3 = MODIFIERS.map(m => 3 * m[0])
const B3 = MODIFIERS.map(m => 3 * m[1])
const THR3 = MODIFIERS.map(m => 1.5 * (m[0] + m[1]))
const PLANAR_FUDGE = 8

/**
 * Table pick — mirrors etc2.wgsl's table_hedged: the table whose large
 * magnitude covers max|D|, downgraded to its neighbour when the D mass
 * sits well below the extreme (mean-square D under max²/4, with v the
 * luma D-variance about the base).
 */
function fastTable(D: readonly number[], v: number): number {
  let mx = 0
  for (let i = 0; i < D.length; i++) mx = Math.max(mx, Math.abs(D[i]!))
  let cover = 0
  while (cover < 7 && B3[cover]! < mx) cover++
  return cover > 0 && v * 0.25 < mx * mx ? cover - 1 : cover
}

/** Σ m3·(m3 − 2|D|) for one table over a subblock's D values (×3 scale). */
function fastTableScore(D: readonly number[], t: number): number {
  let acc = 0
  for (let i = 0; i < D.length; i++) {
    const ad = Math.abs(D[i]!)
    const m3 = ad > THR3[t]! ? B3[t]! : A3[t]!
    acc += m3 * (m3 - 2 * ad)
  }
  return acc
}

/** Wire indices + modifier sum for a chosen table (selection on D only). */
function fastIndices(D: readonly number[], t: number): SubblockFit {
  const indices = new Uint8Array(D.length)
  let modSum = 0
  for (let i = 0; i < D.length; i++) {
    const d = D[i]!
    const large = Math.abs(d) > THR3[t]!
    const neg = d < 0
    indices[i] = (large ? 1 : 0) | (neg ? 2 : 0)
    modSum += ((neg ? -1 : 1) * (large ? B3[t]! : A3[t]!)) / 3
  }
  return { table: t, indices, err: 0, modSum }
}

/** The scalar-luma fast encode — mirrors etc2.wgsl decision for decision. */
function encodeFastBlock(px: Int32Array): ETC2Block {
  // Per-texel luma sums and quadrant statistics (Σp, Σ||p||², Σℓ²).
  const luma = new Float64Array(16)
  const qsum = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ]
  const qsq = [0, 0, 0, 0]
  const qlsq = [0, 0, 0, 0]
  for (let k = 0; k < 16; k++) {
    const r = px[k * 3]!
    const g = px[k * 3 + 1]!
    const b = px[k * 3 + 2]!
    const l = r + g + b
    luma[k] = l
    const q = ((k & 3) >= 2 ? 1 : 0) | (k >> 2 >= 2 ? 2 : 0)
    qsum[q]![0]! += r
    qsum[q]![1]! += g
    qsum[q]![2]! += b
    qsq[q]! += r * r + g * g + b * b
    qlsq[q]! += l * l
  }
  const pair = (a: number, b: number): { sum: number[]; sq: number; lsq: number } => ({
    sum: [qsum[a]![0]! + qsum[b]![0]!, qsum[a]![1]! + qsum[b]![1]!, qsum[a]![2]! + qsum[b]![2]!],
    sq: qsq[a]! + qsq[b]!,
    lsq: qlsq[a]! + qlsq[b]!,
  })

  // Estimate of one base pair: Σ||p − b||² (O(1) from the sums) plus the
  // scalar table term. Exact modulo decode clamping; an upper bound on the
  // true error.
  const constErr = (s: { sum: number[]; sq: number }, b: readonly number[]): number =>
    s.sq -
    2 * (b[0]! * s.sum[0]! + b[1]! * s.sum[1]! + b[2]! * s.sum[2]!) +
    8 * (b[0]! * b[0]! + b[1]! * b[1]! + b[2]! * b[2]!)
  const Dof = (texels: readonly number[], lb: number): number[] => texels.map(k => luma[k]! - lb)

  interface FlipEval {
    est: number
    diff: boolean
    codes: { codes0: number[]; codes1: number[] }
    t0: number
    t1: number
    lb0: number
    lb1: number
  }
  const evalCodes = (
    flip: number,
    s0: { sum: number[]; sq: number; lsq: number },
    s1: { sum: number[]; sq: number; lsq: number },
    diff: boolean,
    c: { codes0: number[]; codes1: number[] },
  ): FlipEval => {
    const b0 = decodeBases(c.codes0, diff)
    const b1 = decodeBases(c.codes1, diff)
    const lb0 = b0[0]! + b0[1]! + b0[2]!
    const lb1 = b1[0]! + b1[1]! + b1[2]!
    // Luma D-variance about each base drives the table hedge; the chosen
    // table is then scored exactly, keeping the estimate exact modulo
    // decode clamping.
    const lsum0 = s0.sum[0]! + s0.sum[1]! + s0.sum[2]!
    const lsum1 = s1.sum[0]! + s1.sum[1]! + s1.sum[2]!
    const v0 = Math.max(s0.lsq - 2 * lb0 * lsum0 + 8 * lb0 * lb0, 0)
    const v1 = Math.max(s1.lsq - 2 * lb1 * lsum1 + 8 * lb1 * lb1, 0)
    const D0 = Dof(SUBBLOCKS[flip]![0]!, lb0)
    const D1 = Dof(SUBBLOCKS[flip]![1]!, lb1)
    const t0 = fastTable(D0, v0)
    const t1 = fastTable(D1, v1)
    return {
      est: constErr(s0, b0) + constErr(s1, b1) + (fastTableScore(D0, t0) + fastTableScore(D1, t1)) / 3,
      diff,
      codes: c,
      t0,
      t1,
      lb0,
      lb1,
    }
  }
  const evalFlip = (
    flip: number,
    s0: { sum: number[]; sq: number; lsq: number },
    s1: { sum: number[]; sq: number; lsq: number },
  ): FlipEval => {
    const avg0 = s0.sum.map(v => v / 8)
    const avg1 = s1.sum.map(v => v / 8)
    const tryDiff = quantiseBases(avg0, avg1, true, false)
    const diff = tryDiff !== null
    const c = tryDiff ?? quantiseBases(avg0, avg1, false, false)!
    return evalCodes(flip, s0, s1, diff, c)
  }

  // O(1) flip preselect: within-variance minus the luma-direction component
  // the modifier tables can absorb, summed over both subblocks. For
  // exact-grayscale blocks BOTH residuals are identically zero (all
  // variance is along luma), so near-ties fall back to scoring both flips.
  const residual = (s: { sum: number[]; sq: number; lsq: number }): number => {
    const dotSum = s.sum[0]! * s.sum[0]! + s.sum[1]! * s.sum[1]! + s.sum[2]! * s.sum[2]!
    const lsum = s.sum[0]! + s.sum[1]! + s.sum[2]!
    return s.sq - dotSum / 8 - (s.lsq - (lsum * lsum) / 8) / 3
  }
  const resA = residual(pair(0, 2)) + residual(pair(1, 3))
  const resB = residual(pair(0, 1)) + residual(pair(2, 3))
  let flip: number
  let cur: FlipEval
  if (Math.abs(resA - resB) < 1) {
    const fa = evalFlip(0, pair(0, 2), pair(1, 3))
    const fb = evalFlip(1, pair(0, 1), pair(2, 3))
    flip = fb.est < fa.est ? 1 : 0
    cur = fb.est < fa.est ? fb : fa
  } else {
    flip = resB < resA ? 1 : 0
    cur = evalFlip(flip, flip === 0 ? pair(0, 2) : pair(0, 1), flip === 0 ? pair(1, 3) : pair(2, 3))
  }
  const texels0 = SUBBLOCKS[flip]![0]!
  const texels1 = SUBBLOCKS[flip]![1]!
  const diff = cur.diff
  const codes = cur.codes
  // NO base refit — dropped with the shader (2026-07): ~0.2 dB on
  // photographic colour for 13-30% GPU. The 'high' path keeps its exact
  // refit rounds, so the reference still bounds what a refit could buy.
  const fit0 = fastIndices(Dof(texels0, cur.lb0), cur.t0)
  const fit1 = fastIndices(Dof(texels1, cur.lb1), cur.t1)

  // Planar contest, closed-form: the residual of the plane the hardware
  // will ACTUALLY decode — quantised, clamped corners — via the
  // normal-equation identity Σ||p − f||² = Σ||p||² − 2·θ·rhs + θᵀGθ
  // (G = the constant Gram matrix of the sample positions). Only decode's
  // floor-rounding is unmodelled (PLANAR_FUDGE). Always evaluated — it is
  // O(1) and gating it loses smooth content.
  const o: number[] = []
  const h: number[] = []
  const v: number[] = []
  const maxCode = [63, 127, 63]
  const extendOHV = [extend6, extend7, extend6]
  let planarEst = PLANAR_FUDGE
  for (let c = 0; c < 3; c++) {
    let sA = 0
    let sB = 0
    let sC = 0
    let sq = 0
    for (let k = 0; k < 16; k++) {
      const x = (k & 3) / 4
      const y = (k >> 2) / 4
      const p = px[k * 3 + c]!
      sA += (1 - x - y) * p
      sB += x * p
      sC += y * p
      sq += p * p
    }
    const oc = PLANAR_GINV[0]![0]! * sA + PLANAR_GINV[0]![1]! * sB + PLANAR_GINV[0]![2]! * sC
    const hc = PLANAR_GINV[1]![0]! * sA + PLANAR_GINV[1]![1]! * sB + PLANAR_GINV[1]![2]! * sC
    const vc = PLANAR_GINV[2]![0]! * sA + PLANAR_GINV[2]![1]! * sB + PLANAR_GINV[2]![2]! * sC
    const qO = quant(oc, maxCode[c]!)
    const qH = quant(hc, maxCode[c]!)
    const qV = quant(vc, maxCode[c]!)
    o.push(qO)
    h.push(qH)
    v.push(qV)
    const eO = extendOHV[c]!(qO)
    const eH = extendOHV[c]!(qH)
    const eV = extendOHV[c]!(qV)
    planarEst +=
      sq -
      2 * (eO * sA + eH * sB + eV * sC) +
      3.5 * (eO * eO + eH * eH + eV * eV) +
      0.5 * eO * eH +
      0.5 * eO * eV +
      4.5 * eH * eV
  }

  if (cur.est <= planarEst) {
    return packEtc1({ diff, flip, codes0: codes.codes0, codes1: codes.codes1, fit0, fit1, err: cur.est })
  }
  return packPlanar({ o, h, v, err: planarEst })
}

// -------------------------------------------------------------------------
// Encoder
// -------------------------------------------------------------------------

/** Decoded squared error of a packed block against the source, 0..255 units. */
function blockSse(px: Int32Array, block: ETC2Block): number {
  const decoded = decodeETC2Block(block)
  let sse = 0
  for (let i = 0; i < 48; i++) {
    const d = decoded[i]! * 255 - px[i]!
    sse += d * d
  }
  return sse
}

/**
 * Encode a 4×4 RGB block (48 floats in [0, 1], row-major) into an 8-byte
 * ETC2 RGB8 block. Emits individual, differential or planar mode.
 */
export function encodeETC2Block(pixels: ETC2Pixels, { quality = 'fast' }: { quality?: ETC2Quality } = {}): ETC2Block {
  if (pixels.length !== 48) {
    throw new Error(`encodeETC2Block: expected 48 values (16 RGB), got ${pixels.length}`)
  }
  const px = new Int32Array(48)
  for (let i = 0; i < 48; i++) {
    const v = Math.round(pixels[i]! * 255)
    px[i] = v < 0 ? 0 : v > 255 ? 255 : v
  }

  const fastBlock = encodeFastBlock(px)
  if (quality === 'fast') return fastBlock

  // 'high': exhaustive search with decode-exact errors.
  let bestEtc1: Etc1Candidate | null = null
  for (let flip = 0; flip < 2; flip++) {
    // Subblock averages (float, 0..255 space).
    const avgs: number[][] = []
    for (let sb = 0; sb < 2; sb++) {
      const texels = SUBBLOCKS[flip]![sb]!
      const a = [0, 0, 0]
      for (const k of texels) {
        a[0]! += px[k * 3]!
        a[1]! += px[k * 3 + 1]!
        a[2]! += px[k * 3 + 2]!
      }
      avgs.push(a.map(s => s / texels.length))
    }
    const [avg0, avg1] = avgs as [number[], number[]]

    // Try BOTH differential (delta clamped into range) and individual bases.
    for (const t of [
      { diff: true, clampDelta: true },
      { diff: false, clampDelta: false },
    ]) {
      const cand = fitEtc1(px, flip, t.diff, t.clampDelta, avg0, avg1, 4)
      if (cand && (!bestEtc1 || cand.err < bestEtc1.err)) bestEtc1 = cand
    }
  }

  const planar = fitPlanar(px, true)
  const highBlock = bestEtc1 && bestEtc1.err <= planar.err ? packEtc1(bestEtc1) : packPlanar(planar)

  // The fast path can occasionally beat the exhaustive trajectory (its
  // estimate-accepted refit reaches configs the exact-accepted rounds
  // reject); folding it in makes 'high' ≤ 'fast' by construction.
  return blockSse(px, highBlock) <= blockSse(px, fastBlock) ? highBlock : fastBlock
}

// -------------------------------------------------------------------------
// Decoder
// -------------------------------------------------------------------------

/** Which of the five ETC2 RGB8 modes a packed block uses. */
export function readETC2Mode(block: ETC2Block): ETC2Mode {
  const hi = ((block[0]! << 24) | (block[1]! << 16) | (block[2]! << 8) | block[3]!) >>> 0
  if (((hi >> 1) & 1) === 0) return 'individual'
  const r = ((hi >> 27) & 31) + signed3((hi >> 24) & 7)
  if (r < 0 || r > 31) return 'T'
  const g = ((hi >> 19) & 31) + signed3((hi >> 16) & 7)
  if (g < 0 || g > 31) return 'H'
  const b = ((hi >> 11) & 31) + signed3((hi >> 8) & 7)
  if (b < 0 || b > 31) return 'planar'
  return 'differential'
}

/**
 * Decode an 8-byte ETC2 RGB8 block to 48 normalised [0, 1] values (16 RGB
 * triplets, row-major). Handles all five modes.
 */
export function decodeETC2Block(block: ETC2Block): Float32Array {
  if (block.length !== 8) {
    throw new Error(`decodeETC2Block: expected 8 bytes, got ${block.length}`)
  }
  const hi = ((block[0]! << 24) | (block[1]! << 16) | (block[2]! << 8) | block[3]!) >>> 0
  const lo = ((block[4]! << 24) | (block[5]! << 16) | (block[6]! << 8) | block[7]!) >>> 0
  const mode = readETC2Mode(block)
  const out = new Float32Array(48)

  const wireIndex = (k: number): number => {
    const wire = (k & 3) * 4 + (k >> 2)
    return (((lo >>> (16 + wire)) & 1) << 1) | ((lo >>> wire) & 1)
  }
  const writeTexel = (k: number, r: number, g: number, b: number): void => {
    out[k * 3] = clamp255(r) / 255
    out[k * 3 + 1] = clamp255(g) / 255
    out[k * 3 + 2] = clamp255(b) / 255
  }

  if (mode === 'planar') {
    const ro = extend6((hi >> 25) & 63)
    const go = extend7((((hi >> 24) & 1) << 6) | ((hi >> 17) & 63))
    const bo = extend6((((hi >> 16) & 1) << 5) | (((hi >> 11) & 3) << 3) | ((hi >> 7) & 7))
    const rh = extend6((((hi >> 2) & 31) << 1) | (hi & 1))
    const gh = extend7((lo >>> 25) & 127)
    const bh = extend6((lo >>> 19) & 63)
    const rv = extend6((lo >>> 13) & 63)
    const gv = extend7((lo >>> 6) & 127)
    const bv = extend6(lo & 63)
    for (let k = 0; k < 16; k++) {
      const x = k & 3
      const y = k >> 2
      writeTexel(
        k,
        (x * (rh - ro) + y * (rv - ro) + 4 * ro + 2) >> 2,
        (x * (gh - go) + y * (gv - go) + 4 * go + 2) >> 2,
        (x * (bh - bo) + y * (bv - bo) + 4 * bo + 2) >> 2,
      )
    }
    return out
  }

  if (mode === 'T' || mode === 'H') {
    let base1: number[]
    let base2: number[]
    let dIdx: number
    if (mode === 'T') {
      base1 = [(((hi >> 27) & 3) << 2) | ((hi >> 24) & 3), (hi >> 20) & 15, (hi >> 16) & 15]
      base2 = [(hi >> 12) & 15, (hi >> 8) & 15, (hi >> 4) & 15]
      dIdx = (((hi >> 2) & 3) << 1) | (hi & 1)
    } else {
      base1 = [(hi >> 27) & 15, (((hi >> 24) & 7) << 1) | ((hi >> 20) & 1), (((hi >> 19) & 1) << 3) | ((hi >> 15) & 7)]
      base2 = [(hi >> 11) & 15, (hi >> 7) & 15, (hi >> 3) & 15]
      const v1 = (base1[0]! << 8) | (base1[1]! << 4) | base1[2]!
      const v2 = (base2[0]! << 8) | (base2[1]! << 4) | base2[2]!
      dIdx = (((hi >> 2) & 1) << 2) | ((hi & 1) << 1) | (v1 >= v2 ? 1 : 0)
    }
    const c1 = base1.map(extend4)
    const c2 = base2.map(extend4)
    const d = TH_DISTANCES[dIdx]!
    // T: palette = { c1, c2+d, c2, c2−d }. H: { c1+d, c1−d, c2+d, c2−d }.
    const palette: number[][] =
      mode === 'T'
        ? [c1, c2.map(v => v + d), c2, c2.map(v => v - d)]
        : [c1.map(v => v + d), c1.map(v => v - d), c2.map(v => v + d), c2.map(v => v - d)]
    for (let k = 0; k < 16; k++) {
      const pc = palette[wireIndex(k)]!
      writeTexel(k, pc[0]!, pc[1]!, pc[2]!)
    }
    return out
  }

  // individual / differential
  const diff = mode === 'differential'
  let base1: number[]
  let base2: number[]
  if (diff) {
    const codes1 = [(hi >> 27) & 31, (hi >> 19) & 31, (hi >> 11) & 31]
    const codes2 = codes1.map((v, c) => v + signed3((hi >> (24 - c * 8)) & 7))
    base1 = codes1.map(extend5)
    base2 = codes2.map(extend5)
  } else {
    base1 = [(hi >> 28) & 15, (hi >> 20) & 15, (hi >> 12) & 15].map(extend4)
    base2 = [(hi >> 24) & 15, (hi >> 16) & 15, (hi >> 8) & 15].map(extend4)
  }
  const table1 = (hi >> 5) & 7
  const table2 = (hi >> 2) & 7
  const flip = hi & 1
  for (let k = 0; k < 16; k++) {
    const x = k & 3
    const y = k >> 2
    const sb1 = flip === 0 ? x >= 2 : y >= 2
    const base = sb1 ? base2 : base1
    const m = MODIFIERS[sb1 ? table2 : table1]![wireIndex(k)]!
    writeTexel(k, base[0]! + m, base[1]! + m, base[2]! + m)
  }
  return out
}

// Exposed for tests so they can assert quantisation / packing without
// re-deriving the bit layout.
export const _internal = {
  MODIFIERS,
  TH_DISTANCES,
  extend4,
  extend5,
  extend6,
  extend7,
  quant,
  signed3,
  wordsToBlock,
  fitPlanar,
  packPlanar,
  planarError,
}
