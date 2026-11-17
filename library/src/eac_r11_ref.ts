// EAC R11 (unsigned) reference encoder + decoder. CPU implementation.
//
// EAC R11 compresses 16 single-channel values into 8 bytes (big-endian bit
// order):
//   • byte 0     : base codeword
//   • byte 1     : multiplier (high nibble) | modifier table (low nibble)
//   • bytes 2..7 : 16 × 3-bit modifier indices, pixel 0 in the top 3 bits
//
// Decode, in 11-bit units:
//   value = clamp(base·8 + 4 + modifier·M, 0, 2047)
// with M = multiplier·8, or 1 when the multiplier is 0 (an eighth of the
// multiplier-1 step: fine precision for nearly flat blocks). The sampler
// returns value / 2047.
//
// Texel order on the wire is COLUMN-major: texel (x, y) is pixel x·4 + y.
// This module's public API uses the same ROW-major [0, 1] layout as the
// other `_ref` modules; the wire mapping is internal.
//
// QUALITY LEVELS
//   'fast' (default): the CPU mirror of eac_r11.wgsl, in the same exact
//     integer arithmetic — the GPU output matches it byte for byte.
//     Targets T = round(v·2047); the block's 8-bit span picks three (table,
//     multiplier rule) candidates from a fitted table; each is scored
//     exactly with the palette centred on the mid-range; the winner is
//     re-scored at base ± 1.
//   'high': exhaustive — every table × every multiplier × a window of base
//     codewords around the mid-range, decode-exact errors — and finally the
//     'fast' result, keeping whichever decodes with lower error. That makes
//     'high' ≤ 'fast' per block by construction, so it bounds what a wider
//     search buys.

/** 16-value input, row-major, each in [0, 1]. */
export type EACR11Values = Readonly<ArrayLike<number>>

/** Exactly 8 bytes of EAC R11 block data. */
export type EACR11Block = Uint8Array

export type EACR11Quality = 'fast' | 'high'

// Modifier tables (Khronos Data Format Specification, ETC2/EAC section).
// Every table is symmetric about −½: entries 0..3 are −(p + 1) for the
// positive entries p at 4..7.
const MODIFIERS: readonly (readonly number[])[] = [
  [-3, -6, -9, -15, 2, 5, 8, 14],
  [-3, -7, -10, -13, 2, 6, 9, 12],
  [-2, -5, -8, -13, 1, 4, 7, 12],
  [-2, -4, -6, -13, 1, 3, 5, 12],
  [-3, -6, -8, -12, 2, 5, 7, 11],
  [-3, -7, -9, -11, 2, 6, 8, 10],
  [-4, -7, -8, -11, 3, 6, 7, 10],
  [-3, -5, -8, -11, 2, 4, 7, 10],
  [-2, -6, -8, -10, 1, 5, 7, 9],
  [-2, -5, -8, -10, 1, 4, 7, 9],
  [-2, -4, -8, -10, 1, 3, 7, 9],
  [-2, -5, -7, -10, 1, 4, 6, 9],
  [-3, -4, -7, -10, 2, 3, 6, 9],
  [-1, -2, -3, -10, 0, 1, 2, 9],
  [-4, -6, -8, -9, 3, 5, 7, 8],
  [-3, -5, -7, -9, 2, 4, 6, 8],
]

// Span → three candidates, 6 bits each (low first): table in bits 0..3, rule
// in bits 4..5 (0 = covering multiplier, 1 = that + 1, 2 = multiplier 0).
// Index: the 8-bit span below 32, then 8 sub-buckets per octave of span;
// entry 40 is for full-range blocks (span 255: both black and white).
// Fitted offline per span on roughness / AO / displacement / mask /
// normal-channel textures (greedy, on the encoder's own policy). Must stay
// identical to the copy in eac_r11.wgsl, eac_r11_fast_f16.wgsl and
// webgl/glsl/eac_r11.frag.glsl.
const CANDIDATES: readonly number[] = [
  0x01026, 0x24a2d, 0x2b368, 0x21360, 0x2380d, 0x2380d, 0x0c74d, 0x1d0cd, 0x1141d, 0x0f2dd, 0x0b0dd, 0x0b0dd, 0x0a203,
  0x033cb, 0x0a3cb, 0x013cb, 0x0e3cb, 0x0f389, 0x0d389, 0x0d2c9, 0x0b289, 0x06247, 0x05247, 0x07101, 0x04081, 0x020c1,
  0x03081, 0x03081, 0x02001, 0x0f040, 0x083c0, 0x0f200, 0x0424f, 0x0f089, 0x00247, 0x05244, 0x0a042, 0x021c1, 0x00389,
  0x0e245, 0x1269b,
]

function clampInt(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v
}

/** Decoded 11-bit value of one modifier index. */
function decode11(base: number, multiplier: number, table: number, index: number): number {
  const m = MODIFIERS[table]![index]!
  return clampInt(base * 8 + 4 + (multiplier ? m * multiplier * 8 : m), 0, 2047)
}

function spanBucket(s8: number): number {
  if (s8 === 255) return 40
  if (s8 < 32) return s8
  return 32 + ((s8 >> (2 + (s8 >= 64 ? 1 : 0) + (s8 >= 128 ? 1 : 0))) & 7)
}

/**
 * Exact squared error (×4, so multiplier-0 half-integers stay integral) of
 * one (table, multiplier, base) against 11-bit targets, with the nearest
 * modifier per texel. Mirrors the shader: the texel picks among the four
 * magnitudes on its side of the palette centre, the outermost one clamped
 * to the [0, 2047] range. `indices`, when given, receives the wire indices.
 */
function scoreCandidate(
  targets: ArrayLike<number>,
  table: number,
  multiplier: number,
  base: number,
  indices?: Uint8Array,
): number {
  const halfM2 = multiplier ? 8 * multiplier : 1 // 2 · (M / 2)
  const centre2 = 16 * base + 8 - halfM2 // 2 · centre
  const limitPos = 2 * 2047 - centre2
  const limitNeg = centre2
  const mods = MODIFIERS[table]!
  let err = 0
  for (let p = 0; p < 16; p++) {
    const d = 2 * targets[p]! - centre2
    const a = Math.abs(d)
    let best = Infinity
    let bestI = 0
    for (let i = 0; i < 4; i++) {
      let m = (2 * mods[4 + i]! + 1) * halfM2
      if (i === 3) m = Math.min(m, d >= 0 ? limitPos : limitNeg)
      const r = Math.abs(a - m)
      if (r < best) {
        best = r
        bestI = i
      }
    }
    err += best * best
    if (indices) indices[p] = (d >= 0 ? 4 : 0) + bestI
  }
  return err
}

/** Decode-exact squared error (11-bit units) with free nearest-index choice. */
function exactError(
  targets: ArrayLike<number>,
  table: number,
  multiplier: number,
  base: number,
  limit: number,
): number {
  let err = 0
  for (let p = 0; p < 16 && err < limit; p++) {
    let best = Infinity
    for (let i = 0; i < 8; i++) {
      const d = decode11(base, multiplier, table, i) - targets[p]!
      if (d * d < best) best = d * d
    }
    err += best
  }
  return err
}

interface Choice {
  table: number
  multiplier: number
  base: number
}

/** The shader's search: three span-picked candidates, then base ± 1. */
function searchFast(targets: ArrayLike<number>, tmin: number, tmax: number): Choice {
  const h2 = tmax - tmin
  const mid2 = tmin + tmax
  const s8 = Math.floor((tmax * 255) / 2047 + 0.5) - Math.floor((tmin * 255) / 2047 + 0.5)
  const lut = CANDIDATES[spanBucket(s8)]!
  let best = Infinity
  let choice: Choice = { table: 0, multiplier: 0, base: 0 }
  let winner = choice
  for (let k = 0; k < 5; k++) {
    let table: number
    let multiplier: number
    let base: number
    if (k < 3) {
      const c = (lut >> (6 * k)) & 63
      table = c & 15
      const rule = c >> 4
      const divisor = 8 * (2 * MODIFIERS[table]![7]! + 1)
      multiplier = clampInt(Math.floor((h2 + divisor / 2) / divisor), 1, 15)
      if (rule === 1) multiplier = Math.min(multiplier + 1, 15)
      if (rule === 2) multiplier = 0
      base = Math.min((mid2 + (multiplier ? 8 * multiplier : 1)) >> 4, 255)
    } else {
      if (k === 3) winner = choice
      table = winner.table
      multiplier = winner.multiplier
      base = k === 3 ? Math.max(winner.base, 1) - 1 : Math.min(winner.base + 1, 255)
    }
    const e = scoreCandidate(targets, table, multiplier, base)
    if (e < best) {
      best = e
      choice = { table, multiplier, base }
    }
  }
  return choice
}

/** Exhaustive search, seeded with the fast result so it can only improve. */
function searchHigh(targets: ArrayLike<number>, tmin: number, tmax: number): Choice {
  let choice = searchFast(targets, tmin, tmax)
  let best = exactError(targets, choice.table, choice.multiplier, choice.base, Infinity)
  const mid = (tmin + tmax) / 2
  for (let table = 0; table < 16; table++) {
    for (let multiplier = 0; multiplier < 16; multiplier++) {
      const step = multiplier ? multiplier * 8 : 1
      const b0 = Math.floor((mid - 4 + step / 2) / 8 + 0.5)
      const window = 3 + multiplier
      for (let base = Math.max(0, b0 - window); base <= Math.min(255, b0 + window); base++) {
        const e = exactError(targets, table, multiplier, base, best)
        if (e < best) {
          best = e
          choice = { table, multiplier, base }
        }
      }
    }
  }
  return choice
}

/**
 * Encode 16 single-channel values in [0, 1] (row-major) into an 8-byte EAC
 * R11 block.
 */
export function encodeEACR11Block(
  values: EACR11Values,
  { quality = 'fast' }: { quality?: EACR11Quality } = {},
): EACR11Block {
  if (values.length !== 16) {
    throw new Error(`encodeEACR11Block: expected 16 values, got ${values.length}`)
  }
  // 11-bit targets in WIRE (column-major) order.
  const targets = new Int32Array(16)
  let tmin = 2047
  let tmax = 0
  for (let x = 0; x < 4; x++) {
    for (let y = 0; y < 4; y++) {
      const t = clampInt(Math.floor(values[y * 4 + x]! * 2047 + 0.5), 0, 2047)
      targets[x * 4 + y] = t
      if (t < tmin) tmin = t
      if (t > tmax) tmax = t
    }
  }

  const { table, multiplier, base } =
    quality === 'high' ? searchHigh(targets, tmin, tmax) : searchFast(targets, tmin, tmax)

  const indices = new Uint8Array(16)
  if (quality === 'high') {
    // Free nearest-index choice against the decoded palette.
    for (let p = 0; p < 16; p++) {
      let best = Infinity
      for (let i = 0; i < 8; i++) {
        const d = Math.abs(decode11(base, multiplier, table, i) - targets[p]!)
        if (d < best) {
          best = d
          indices[p] = i
        }
      }
    }
  } else {
    scoreCandidate(targets, table, multiplier, base, indices)
  }

  const out = new Uint8Array(8)
  out[0] = base
  out[1] = (multiplier << 4) | table
  let bits = 0n
  for (let p = 0; p < 16; p++) bits |= BigInt(indices[p]!) << BigInt(45 - 3 * p)
  for (let i = 0; i < 6; i++) out[2 + i] = Number((bits >> BigInt(40 - 8 * i)) & 0xffn)
  return out
}

/** Decode an 8-byte EAC R11 block to 16 normalized [0, 1] values, row-major. */
export function decodeEACR11Block(block: EACR11Block): Float32Array {
  if (block.length !== 8) {
    throw new Error(`decodeEACR11Block: expected 8 bytes, got ${block.length}`)
  }
  const base = block[0]!
  const multiplier = block[1]! >> 4
  const table = block[1]! & 15
  let bits = 0n
  for (let i = 0; i < 6; i++) bits |= BigInt(block[2 + i]!) << BigInt(40 - 8 * i)
  const out = new Float32Array(16)
  for (let p = 0; p < 16; p++) {
    const index = Number((bits >> BigInt(45 - 3 * p)) & 7n)
    // Wire pixel p = x·4 + y → row-major y·4 + x.
    out[(p & 3) * 4 + (p >> 2)] = decode11(base, multiplier, table, index) / 2047
  }
  return out
}

// Exposed for tests.
export const _internal = {
  MODIFIERS,
  CANDIDATES,
  decode11,
  spanBucket,
}
