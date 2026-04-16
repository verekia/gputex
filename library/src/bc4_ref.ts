// BC4 (RGTC1) reference encoder + decoder. CPU implementation.
//
// BC4 compresses 16 single-channel values into 8 bytes:
//   • 2 × 8-bit endpoints (red0, red1)                       // bytes 0, 1
//   • 16 × 3-bit indices, LSB-first, pixel 0 at bit 0        // bytes 2..7
//
// There are two decode modes, selected by the ordering of the endpoints:
//
//   6-interpolation mode (red0 > red1) — used for smooth data:
//     palette[0] = red0
//     palette[1] = red1
//     palette[n+2] = ((6-n)*red0 + (n+1)*red1) / 7   for n = 0..5
//
//   4-interpolation mode (red0 <= red1) — adds hard 0.0 / 1.0 rails:
//     palette[0] = red0
//     palette[1] = red1
//     palette[n+2] = ((4-n)*red0 + (n+1)*red1) / 5   for n = 0..3
//     palette[6] = 0.0
//     palette[7] = 1.0  (255 in 8-bit)
//
// The encoder here only produces the 6-interpolation mode: we use BC4 as
// a BC5 half-block for normal maps, which are smooth and don't benefit
// from the hard 0/1 rails. The decoder handles both modes so it can
// round-trip externally-encoded blocks.
//
// Numeric precision: all arithmetic happens in normalized [0, 1] f32 so
// the reference matches the WGSL port bit-for-bit. Real hardware does
// integer-domain interpolation which can differ from f32 by ~1 LSB per
// interpolated palette entry; we accept that delta in tests.
//
// Algorithm (per Stage 2 of the plan):
//   1. Initial endpoints: red0 = max(values), red1 = min(values).
//   2. Build the 8-entry palette, assign each texel its nearest entry.
//   3. Refinement pass: with the chosen indices, solve the 2×2 normal
//      equations for the (r0, r1) pair that minimizes Σ (palette[i_k] − v_k)².
//      Quantize the refined endpoints and re-assign indices.
//   4. Accept the refinement if it reduces total squared error and keeps
//      red0 > red1 after quantization; otherwise keep the bbox endpoints.
//   5. Pack 2 endpoint bytes + 6 index bytes = 8-byte block.

/** 16-value input: accepts both real arrays and typed arrays. */
export type BC4Values = Readonly<ArrayLike<number>>

/** Exactly 8 bytes, indexed numerically. */
export type BC4Block = Uint8Array

const PALETTE_SIZE = 8

// 6-interp mode weights. `w0[j]` and `w1[j]` are the normalized
// coefficients applied to red0 and red1 when reconstructing palette[j]:
//   palette[j] = w0[j] * red0 + w1[j] * red1
// palette[0] = red0, palette[1] = red1; the six interpolants between them
// fill palette[2..7].
const W0_6: readonly number[] = [1, 0, 6 / 7, 5 / 7, 4 / 7, 3 / 7, 2 / 7, 1 / 7]
const W1_6: readonly number[] = [0, 1, 1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7]

// 4-interp mode weights. Indices 6 and 7 are the hard 0.0 / 1.0 rails
// with zero coefficient on the endpoints.
const W0_4: readonly number[] = [1, 0, 4 / 5, 3 / 5, 2 / 5, 1 / 5, 0, 0]
const W1_4: readonly number[] = [0, 1, 1 / 5, 2 / 5, 3 / 5, 4 / 5, 0, 0]
const PAL4_CONST: readonly number[] = [0, 0, 0, 0, 0, 0, 0, 1] // additive constant for indices 6, 7

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v
}
function quantize8(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v * 255)))
}

function buildPalette6(r0: number, r1: number): Float32Array {
  // r0, r1 in [0, 1]. Used when red0_byte > red1_byte.
  const p = new Float32Array(PALETTE_SIZE)
  for (let j = 0; j < PALETTE_SIZE; j++) p[j] = W0_6[j]! * r0 + W1_6[j]! * r1
  return p
}

function buildPalette4(r0: number, r1: number): Float32Array {
  // Used when red0_byte <= red1_byte. Hard rails at indices 6, 7.
  const p = new Float32Array(PALETTE_SIZE)
  for (let j = 0; j < PALETTE_SIZE; j++) {
    p[j] = W0_4[j]! * r0 + W1_4[j]! * r1 + PAL4_CONST[j]!
  }
  return p
}

/**
 * Nearest-palette-index selection for one texel. Full 8-entry L2 search.
 * Returns the index in 0..7 and the squared error at that index.
 */
function assignIndex(value: number, palette: Float32Array): { idx: number; err: number } {
  let bestIdx = 0
  let bestD = Infinity
  for (let j = 0; j < PALETTE_SIZE; j++) {
    const d = palette[j]! - value
    const d2 = d * d
    if (d2 < bestD) {
      bestD = d2
      bestIdx = j
    }
  }
  return { idx: bestIdx, err: bestD }
}

function sumSquaredError(values: BC4Values, palette: Float32Array, indices: ArrayLike<number>): number {
  let e = 0
  for (let k = 0; k < 16; k++) {
    const d = palette[indices[k]!]! - values[k]!
    e += d * d
  }
  return e
}

/**
 * One-pass least-squares refit. Given the current per-texel indices,
 * solve the 2×2 normal equations for the (r0, r1) pair that minimizes
 * the total squared error. See file-level comment for the derivation.
 *
 * Returns null if the system is degenerate (e.g. all texels landed on
 * the same palette entry) — caller should keep the bbox endpoints.
 */
function refitEndpoints6Interp(values: BC4Values, indices: ArrayLike<number>): { r0: number; r1: number } | null {
  let sumAA = 0,
    sumBB = 0,
    sumAB = 0,
    sumAV = 0,
    sumBV = 0
  for (let k = 0; k < 16; k++) {
    const a = W0_6[indices[k]!]!
    const b = W1_6[indices[k]!]!
    const v = values[k]!
    sumAA += a * a
    sumBB += b * b
    sumAB += a * b
    sumAV += a * v
    sumBV += b * v
  }
  const det = sumAA * sumBB - sumAB * sumAB
  // A very small det means the A and B weight vectors are almost
  // colinear — the least-squares problem is underdetermined.
  if (Math.abs(det) < 1e-9) return null

  const r0 = (sumBB * sumAV - sumAB * sumBV) / det
  const r1 = (sumAA * sumBV - sumAB * sumAV) / det
  return { r0: clamp01(r0), r1: clamp01(r1) }
}

/**
 * Pack (red0_byte, red1_byte, 16 × 3-bit indices) into the 8-byte block
 * layout. Indices occupy a 48-bit field in bytes 2..7, little-endian,
 * with pixel 0 at bit 0. We build the whole field as a BigInt so the
 * straddling indices (those that span a byte boundary) come out right
 * without manual bit-fiddling per byte.
 */
function packBlock(red0: number, red1: number, indices: ArrayLike<number>): BC4Block {
  const out = new Uint8Array(8)
  out[0] = red0
  out[1] = red1

  let bits = 0n
  for (let k = 0; k < 16; k++) {
    bits |= BigInt(indices[k]! & 7) << BigInt(3 * k)
  }
  for (let i = 0; i < 6; i++) {
    out[2 + i] = Number((bits >> BigInt(i * 8)) & 0xffn)
  }
  return out
}

/**
 * Encode 16 single-channel values in [0, 1] into an 8-byte BC4 block.
 *
 * Always produces a 6-interpolation-mode block (red0 > red1). This is
 * the right choice for the BC5 normal-map use case; see file comment.
 */
export function encodeBC4Block(values: BC4Values): BC4Block {
  if (values.length !== 16) {
    throw new Error(`encodeBC4Block: expected 16 values, got ${values.length}`)
  }

  // 1. Initial endpoints = bbox of input.
  let vmin = Infinity,
    vmax = -Infinity
  for (let k = 0; k < 16; k++) {
    const v = values[k]!
    if (v < vmin) vmin = v
    if (v > vmax) vmax = v
  }
  vmin = clamp01(vmin)
  vmax = clamp01(vmax)

  // Quantize to 8-bit. Force 6-interp mode by ensuring red0_byte > red1_byte.
  let r0b = quantize8(vmax)
  let r1b = quantize8(vmin)
  if (r0b === r1b) {
    // Flat (or near-flat) block. Nudge to keep 6-interp mode. The choice
    // between nudging up vs. down only matters at the 0 / 255 extremes.
    if (r1b > 0) r1b -= 1
    else r0b += 1
  }

  // 2. Build palette from quantized endpoints, assign indices.
  const pal = buildPalette6(r0b / 255, r1b / 255)
  const indices = new Uint8Array(16)
  for (let k = 0; k < 16; k++) {
    indices[k] = assignIndex(values[k]!, pal).idx
  }
  let bestErr = sumSquaredError(values, pal, indices)
  let bestR0 = r0b,
    bestR1 = r1b
  let bestIndices = indices.slice()

  // 3. Refinement: one pass of least-squares fit on the current indices.
  const refit = refitEndpoints6Interp(values, indices)
  if (refit) {
    const qR0 = quantize8(refit.r0)
    const qR1 = quantize8(refit.r1)
    // The refit can flip or equalize endpoints; only accept outputs that
    // stay in 6-interp mode, otherwise the decoder would interpret the
    // block in the wrong mode.
    if (qR0 > qR1) {
      const pal2 = buildPalette6(qR0 / 255, qR1 / 255)
      const idx2 = new Uint8Array(16)
      for (let k = 0; k < 16; k++) idx2[k] = assignIndex(values[k]!, pal2).idx
      const err2 = sumSquaredError(values, pal2, idx2)
      if (err2 < bestErr) {
        bestErr = err2
        bestR0 = qR0
        bestR1 = qR1
        bestIndices = idx2
      }
    }
  }

  // 4. Pack.
  return packBlock(bestR0, bestR1, bestIndices)
}

/**
 * Decode an 8-byte BC4 block to 16 normalized [0, 1] values.
 *
 * Handles both 6-interpolation and 4-interpolation modes so we can
 * round-trip blocks produced externally (hardware, other encoders).
 */
export function decodeBC4Block(block: BC4Block): Float32Array {
  if (block.length !== 8) {
    throw new Error(`decodeBC4Block: expected 8 bytes, got ${block.length}`)
  }
  const r0b = block[0]!
  const r1b = block[1]!
  const palette = r0b > r1b ? buildPalette6(r0b / 255, r1b / 255) : buildPalette4(r0b / 255, r1b / 255)

  // Read the 48-bit index field as a BigInt.
  let bits = 0n
  for (let i = 0; i < 6; i++) bits |= BigInt(block[2 + i]!) << BigInt(i * 8)

  const out = new Float32Array(16)
  for (let k = 0; k < 16; k++) {
    const idx = Number((bits >> BigInt(3 * k)) & 7n)
    out[k] = palette[idx]!
  }
  return out
}

// Exposed for tests so they can assert bit layouts without re-deriving
// the packing rules.
export const _internal = {
  buildPalette6,
  buildPalette4,
  packBlock,
  W0_6,
  W1_6,
}
