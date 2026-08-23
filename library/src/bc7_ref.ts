// BC7 reference encoder + decoder. CPU implementation.
//
// BC7 is a multi-mode 16-byte RGBA block format. The mode is selected by
// a variable-width unary-coded prefix: mode N is N zero bits followed by
// a single 1 bit. The reference ENCODER implements mode 6 only — the
// strongest single-subset mode, well-suited to smooth content — and is
// the quality yardstick the GPU encoder is gated against. The DECODER
// additionally supports mode 1 (2-subset partitioning), which the GPU
// encoder emits for multi-modal blocks; see `BC7Encoder.ts`.
//
// -----------------------------------------------------------------------
// MODE 6 LAYOUT (LSB-first, bit 0 = byte 0's bit 0)
//   bits 0..6    mode field           (0b0000001 — only bit 6 is 1)
//   bits 7..13   R0 (7-bit)
//   bits 14..20  R1
//   bits 21..27  G0
//   bits 28..34  G1
//   bits 35..41  B0
//   bits 42..48  B1
//   bits 49..55  A0
//   bits 56..62  A1
//   bit  63      P0  (shared p-bit for endpoint 0, i.e. same p-bit for R0, G0, B0, A0)
//   bit  64      P1
//   bits 65..67  pixel 0 index (3 bits; MSB is implicit 0 — the anchor rule)
//   bits 68..71  pixel 1 index (4 bits)
//   ...
//   bits 124..127 pixel 15 index
//
// Effective 8-bit endpoint channel = (7_bit_value << 1) | p_bit.
//
// Palette uses 16-entry interpolation between e0 and e1:
//   palette[i] = ((64 − w[i]) × e0_8 + w[i] × e1_8 + 32) >> 6
// with w = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64].
// This is integer arithmetic, matching the hardware decoder.
//
// Anchor rule: pixel 0 of mode 6 has an implicit MSB = 0, so its stored
// index occupies only 3 bits. If the chosen index for pixel 0 ever lands
// in [8, 15], swap the endpoints and invert every index (new = 15 − old)
// — the decode palette is reflected, so the image is unchanged but the
// anchor now has MSB = 0.

/** 16 pixels × 4 channels = 64 normalized floats, interleaved RGBA. */
export type BC7Pixels = Readonly<ArrayLike<number>>

/** 16 bytes of BC7 block data. */
export type BC7Block = Uint8Array

// --- Constants --------------------------------------------------------------

/** Mode 6 interpolation weights (× 1/64). Fixed by the spec. */
const W4: readonly number[] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64]

/** Mode 6's pixel-0 anchor MSB must be 0. 4-bit indices in [0..7] satisfy this. */
const MODE6_ANCHOR_MSB_CUTOFF = 8

// --- Helpers ----------------------------------------------------------------

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v
}

/** Round a normalized [0,1] value to its 8-bit representation. */
function to8(v: number): number {
  return clamp(Math.round(v * 255), 0, 255)
}

/** Hardware interpolation rule. Integer domain, matches the decoder. */
function interp8(e0: number, e1: number, w: number): number {
  return ((64 - w) * e0 + w * e1 + 32) >> 6
}

/**
 * Low `n` bits set, as an unsigned 32-bit value. `1 << 32` is `1` in JS
 * (shift counts are taken mod 32), so the n = 32 case needs its own arm.
 */
function mask32(n: number): number {
  return n >= 32 ? 0xffffffff : ((1 << n) - 1) >>> 0
}

/**
 * 128-bit little-endian bit store as four u32 words, word 0 holding bits
 * 0..31. Fields are read/written LSB-first and may straddle a word
 * boundary, so both directions loop over the (at most two) words a field
 * touches.
 *
 * Words rather than a BigInt: every field in a BC7 block is ≤ 8 bits, and
 * BigInt allocates a fresh heap value per operation — packing and
 * unpacking blocks this way was a large share of the reference encoder's
 * runtime. Field values are limited to 32 bits, which every BC7 field
 * satisfies.
 */
function writeBits(w: Uint32Array, pos: number, nBits: number, value: number): void {
  let v = (value & mask32(nBits)) >>> 0
  let p = pos
  let left = nBits < 32 ? nBits : 32
  while (left > 0) {
    const idx = p >>> 5
    const off = p & 31
    const take = 32 - off < left ? 32 - off : left
    const m = mask32(take)
    // Clear then OR, so a rewrite at the same position replaces rather
    // than merges.
    w[idx] = ((w[idx]! & ~((m << off) >>> 0)) | ((v & m) << off)) >>> 0
    v = take >= 32 ? 0 : v >>> take
    p += take
    left -= take
  }
}

function readBits(w: Uint32Array, pos: number, nBits: number): number {
  let out = 0
  let scale = 1
  let p = pos
  let left = nBits
  while (left > 0) {
    const idx = p >>> 5
    const off = p & 31
    const take = 32 - off < left ? 32 - off : left
    // `* scale` rather than `<< done`: keeps the accumulator a positive
    // JS number instead of wrapping into int32's sign bit.
    out += ((w[idx]! >>> off) & mask32(take)) * scale
    scale *= 2 ** take
    p += take
    left -= take
  }
  return out
}

function wordsToBytes(w: Uint32Array): Uint8Array {
  const out = new Uint8Array(16)
  for (let i = 0; i < 4; i++) {
    const v = w[i]!
    const b = i * 4
    out[b] = v & 0xff
    out[b + 1] = (v >>> 8) & 0xff
    out[b + 2] = (v >>> 16) & 0xff
    out[b + 3] = (v >>> 24) & 0xff
  }
  return out
}

function bytesToWords(block: BC7Block): Uint32Array {
  const w = new Uint32Array(4)
  for (let i = 0; i < 4; i++) {
    const b = i * 4
    w[i] = (block[b]! | (block[b + 1]! << 8) | (block[b + 2]! << 16) | (block[b + 3]! << 24)) >>> 0
  }
  return w
}

/**
 * Append-only 128-bit writer. Matches the BC7 "LSB-first" bitstream
 * convention — the first value written occupies the low-order bits.
 */
class BitWriter128 {
  private readonly w = new Uint32Array(4)
  private pos = 0

  write(value: number, nBits: number): void {
    if (this.pos + nBits > 128) throw new Error('BitWriter128 overflow')
    writeBits(this.w, this.pos, nBits, value)
    this.pos += nBits
  }

  toBytes(): Uint8Array {
    return wordsToBytes(this.w)
  }
}

class BitReader128 {
  private readonly w: Uint32Array
  private pos = 0

  constructor(block: BC7Block) {
    this.w = bytesToWords(block)
  }

  read(nBits: number): number {
    const v = readBits(this.w, this.pos, nBits)
    this.pos += nBits
    return v
  }
}

/** Read the variable-width unary-coded mode field. Returns the mode number. */
export function readBC7Mode(block: BC7Block): number {
  const br = new BitReader128(block)
  // Mode N = N zero bits then a 1. Scan until we see the 1 (max 8 modes).
  for (let m = 0; m < 8; m++) {
    if (br.read(1) === 1) return m
  }
  throw new Error('BC7: no mode bit found in first 8 bits')
}

// --- Mode 6 encode ----------------------------------------------------------

/**
 * Build the 16-entry 8-bit-per-channel palette from two 8-bit RGBA endpoints.
 */
function buildPalette6(
  e0: readonly [number, number, number, number],
  e1: readonly [number, number, number, number],
): Uint8Array {
  const pal = new Uint8Array(16 * 4)
  fillPalette6(pal, e0[0]!, e0[1]!, e0[2]!, e0[3]!, e1[0]!, e1[1]!, e1[2]!, e1[3]!)
  return pal
}

/**
 * `buildPalette6` into a caller-owned buffer, endpoints passed as loose
 * channels. Identical arithmetic to `interp8`, inlined: the p-bit search
 * rebuilds this palette eight times per block, so neither the array
 * allocation nor the per-channel call survives the hot path.
 */
function fillPalette6(
  pal: Uint8Array,
  e0r: number,
  e0g: number,
  e0b: number,
  e0a: number,
  e1r: number,
  e1g: number,
  e1b: number,
  e1a: number,
): void {
  for (let i = 0; i < 16; i++) {
    const w = W4[i]!
    const iw = 64 - w
    const base = i * 4
    pal[base] = (iw * e0r + w * e1r + 32) >> 6
    pal[base + 1] = (iw * e0g + w * e1g + 32) >> 6
    pal[base + 2] = (iw * e0b + w * e1b + 32) >> 6
    pal[base + 3] = (iw * e0a + w * e1a + 32) >> 6
  }
}

/**
 * Nearest-palette-entry assignment for all 16 pixels at once, writing the
 * indices into `indices` and returning the summed squared error.
 *
 * Linear scan per pixel with strict `<`, so the lowest index wins a tie.
 * Pixels and results are carried in loose locals and the caller's buffer
 * rather than a 4-tuple and a result object per pixel — that is 32
 * short-lived objects per call, and this runs eight times per block.
 */
function assignAllIndices6(pixels8: Uint8Array, palette: Uint8Array, indices: Uint8Array): number {
  let total = 0
  for (let k = 0; k < 16; k++) {
    const b = k * 4
    const pr = pixels8[b]!
    const pg = pixels8[b + 1]!
    const pb = pixels8[b + 2]!
    const pa = pixels8[b + 3]!
    let bestIdx = 0
    let bestD = Infinity
    for (let i = 0; i < 16; i++) {
      const c = i * 4
      const dr = palette[c]! - pr
      const dg = palette[c + 1]! - pg
      const db = palette[c + 2]! - pb
      const da = palette[c + 3]! - pa
      const d = dr * dr + dg * dg + db * db + da * da
      if (d < bestD) {
        bestD = d
        bestIdx = i
      }
    }
    indices[k] = bestIdx
    total += bestD
  }
  return total
}

/**
 * One-pass least-squares refit of endpoints, channel-independent. Given
 * current indices, find the (e0, e1) pair per channel that minimizes total
 * squared error. Returns 8-bit ideal endpoints (pre-p-bit-quantization).
 *
 * For index i with weight w_i / 64, palette is a_i * e0 + b_i * e1 where
 * a_i = (64 - w_i)/64, b_i = w_i/64. Normal equations:
 *   ΣAA * e0 + ΣAB * e1 = ΣAV
 *   ΣAB * e0 + ΣBB * e1 = ΣBV
 * Returns null if the system is degenerate.
 */
function refitEndpointsMode6(
  pixels8: Uint8Array,
  indices: Uint8Array,
): { e0: [number, number, number, number]; e1: [number, number, number, number] } | null {
  let sAA = 0,
    sBB = 0,
    sAB = 0
  // Per-channel sums of (a * v) and (b * v).
  const sAV: [number, number, number, number] = [0, 0, 0, 0]
  const sBV: [number, number, number, number] = [0, 0, 0, 0]
  for (let k = 0; k < 16; k++) {
    const i = indices[k]!
    const a = (64 - W4[i]!) / 64
    const b = W4[i]! / 64
    sAA += a * a
    sBB += b * b
    sAB += a * b
    const base = k * 4
    for (let c = 0; c < 4; c++) {
      sAV[c] += a * pixels8[base + c]!
      sBV[c] += b * pixels8[base + c]!
    }
  }
  const det = sAA * sBB - sAB * sAB
  if (Math.abs(det) < 1e-9) return null

  const e0: [number, number, number, number] = [0, 0, 0, 0]
  const e1: [number, number, number, number] = [0, 0, 0, 0]
  for (let c = 0; c < 4; c++) {
    const r0 = (sBB * sAV[c] - sAB * sBV[c]) / det
    const r1 = (sAA * sBV[c] - sAB * sAV[c]) / det
    e0[c] = clamp(Math.round(r0), 0, 255)
    e1[c] = clamp(Math.round(r1), 0, 255)
  }
  return { e0, e1 }
}

/**
 * Pack a mode 6 block into 16 bytes.
 *   e0_7, e1_7 are the 7-bit-per-channel RGBA endpoint vectors (any
 *   4-element indexable — the search hands over its Int32Array scratch).
 *   p0, p1 are the shared p-bits (one per endpoint).
 *   indices are 4-bit values, with indices[0]'s MSB already guaranteed 0.
 */
function packMode6Block(
  e0_7: ArrayLike<number>,
  e1_7: ArrayLike<number>,
  p0: 0 | 1,
  p1: 0 | 1,
  indices: Uint8Array,
): BC7Block {
  if ((indices[0]! & 0x8) !== 0) {
    throw new Error('packMode6Block: pixel 0 index MSB must be 0 (anchor rule)')
  }
  const bw = new BitWriter128()
  // Mode 6: 6 zero bits then a 1 (LSB first).
  bw.write(0, 6)
  bw.write(1, 1)
  // Endpoints: R0, R1, G0, G1, B0, B1, A0, A1 — 7 bits each.
  for (let c = 0; c < 4; c++) {
    bw.write(e0_7[c]!, 7)
    bw.write(e1_7[c]!, 7)
  }
  // P-bits.
  bw.write(p0, 1)
  bw.write(p1, 1)
  // Pixel 0 index: 3 bits (MSB implicit 0).
  bw.write(indices[0]! & 0x7, 3)
  // Pixels 1..15: 4 bits each.
  for (let k = 1; k < 16; k++) bw.write(indices[k]! & 0xf, 4)
  return bw.toBytes()
}

/**
 * Pick the two pixels (out of 16) that maximize 4-channel L2 distance.
 * O(N²) = 256 comparisons — trivial cost, and immune to the "per-channel
 * min/max gives bbox corners, not data-line endpoints" failure mode of a
 * naive bbox. For linear data the chosen pair IS the data-line endpoints;
 * for noisy data it's a reasonable seed that refinement can improve on.
 */
function farthestPair(pixels8: Uint8Array): { i0: number; i1: number } {
  let best = -1
  let bi0 = 0,
    bi1 = 1
  for (let i = 0; i < 16; i++) {
    const bi = i * 4
    for (let j = i + 1; j < 16; j++) {
      const bj = j * 4
      const dr = pixels8[bi]! - pixels8[bj]!
      const dg = pixels8[bi + 1]! - pixels8[bj + 1]!
      const db = pixels8[bi + 2]! - pixels8[bj + 2]!
      const da = pixels8[bi + 3]! - pixels8[bj + 3]!
      const d = dr * dr + dg * dg + db * db + da * da
      if (d > best) {
        best = d
        bi0 = i
        bi1 = j
      }
    }
  }
  return { i0: bi0, i1: bi1 }
}

/**
 * Seed endpoints from the block's principal axis: power-iterate the 4×4
 * channel covariance, then take the extremes of the pixels' projection
 * onto the dominant eigenvector.
 *
 * This is the seed the GPU encoders use, and it is a genuinely different
 * candidate from `farthestPair` rather than a refinement of it. The
 * farthest pair is by construction two actual pixels, which pins the line
 * to the data but lets a single outlier drag it off the distribution's
 * axis; the principal axis is fitted to all sixteen, so it survives
 * outliers but need not pass through any pixel. Neither dominates, so
 * both are tried and the lower-error result wins.
 */
function principalAxisSeed(pixels8: Uint8Array, e0: Int32Array, e1: Int32Array): void {
  const mean = [0, 0, 0, 0]
  for (let k = 0; k < 16; k++) {
    for (let c = 0; c < 4; c++) mean[c]! += pixels8[k * 4 + c]! / 16
  }
  // Upper triangle is enough — the covariance is symmetric — but the full
  // 4×4 keeps the multiply below branch-free.
  const cov = new Float64Array(16)
  const d = [0, 0, 0, 0]
  for (let k = 0; k < 16; k++) {
    for (let c = 0; c < 4; c++) d[c] = pixels8[k * 4 + c]! - mean[c]!
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) cov[i * 4 + j]! += d[i]! * d[j]!
    }
  }
  // Power iteration, stopped as soon as the vector settles. A 4×4 with a
  // dominant eigenvalue converges in a handful of rounds; the zero-length
  // guard covers the degenerate (flat block) case with no axis at all.
  let vx = 1,
    vy = 1,
    vz = 1,
    vw = 1
  for (let it = 0; it < 12; it++) {
    const nx = cov[0]! * vx + cov[1]! * vy + cov[2]! * vz + cov[3]! * vw
    const ny = cov[4]! * vx + cov[5]! * vy + cov[6]! * vz + cov[7]! * vw
    const nz = cov[8]! * vx + cov[9]! * vy + cov[10]! * vz + cov[11]! * vw
    const nw = cov[12]! * vx + cov[13]! * vy + cov[14]! * vz + cov[15]! * vw
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
    if (len < 1e-12) break
    const px = nx / len
    const py = ny / len
    const pz = nz / len
    const pw = nw / len
    const settled = Math.abs(px - vx) + Math.abs(py - vy) + Math.abs(pz - vz) + Math.abs(pw - vw) < 1e-6
    vx = px
    vy = py
    vz = pz
    vw = pw
    if (settled) break
  }
  let lo = Infinity
  let hi = -Infinity
  for (let k = 0; k < 16; k++) {
    const b = k * 4
    const t =
      (pixels8[b]! - mean[0]!) * vx +
      (pixels8[b + 1]! - mean[1]!) * vy +
      (pixels8[b + 2]! - mean[2]!) * vz +
      (pixels8[b + 3]! - mean[3]!) * vw
    if (t < lo) lo = t
    if (t > hi) hi = t
  }
  const v = [vx, vy, vz, vw]
  for (let c = 0; c < 4; c++) {
    e0[c] = clamp(Math.round(mean[c]! + lo * v[c]!), 0, 255)
    e1[c] = clamp(Math.round(mean[c]! + hi * v[c]!), 0, 255)
  }
}

/** Copy one fit record over another (scratch records are reused). */
function copyFit(dst: Mode6Fit, src: Mode6Fit): void {
  dst.e0_7.set(src.e0_7)
  dst.e1_7.set(src.e1_7)
  dst.p0 = src.p0
  dst.p1 = src.p1
  dst.indices.set(src.indices)
  dst.err = src.err
}

/**
 * Refit rounds per seed. Measured on the repo-style test card, a second
 * round is worth ~0.01 dB against ~40% more encode time — the seed choice
 * is what matters, not how long the refit is walked — so one round per
 * seed it is, matching what the single-seed encoder always did.
 */
const MAX_REFIT_ROUNDS = 1

/**
 * Run one seed to convergence: p-bit search, then alternate least-squares
 * endpoint refit / p-bit search while the error strictly falls. Anything
 * that beats `best` is copied into it.
 */
function runSeedToConvergence(pixels8: Uint8Array, e0: Int32Array, e1: Int32Array, best: Mode6Fit): void {
  searchPbitCombos(pixels8, e0, e1, fitRun)
  if (fitRun.err < best.err) copyFit(best, fitRun)
  if (fitRun.err === 0) return
  let prevErr = fitRun.err
  for (let round = 0; round < MAX_REFIT_ROUNDS; round++) {
    const refit = refitEndpointsMode6(pixels8, fitRun.indices)
    if (!refit) break
    searchPbitCombos(pixels8, refit.e0, refit.e1, fitRun)
    // Strict: a round that ties has reached a fixed point, and the next
    // one would refit from the same indices and tie again.
    if (fitRun.err >= prevErr) break
    prevErr = fitRun.err
    if (fitRun.err < best.err) copyFit(best, fitRun)
    if (fitRun.err === 0) return
  }
}

/**
 * Encode 16 RGBA pixels (64 floats in [0,1]) as a BC7 mode 6 block.
 *
 * Algorithm:
 *   1. Convert to 8-bit per channel.
 *   2. Two candidate endpoint seeds — the farthest pixel pair in 4D, and
 *      the block's principal axis. Neither dominates the other (see
 *      farthestPair and principalAxisSeed), so both are tried.
 *   3. For each seed, and each p-bit combo {(0,0),(0,1),(1,0),(1,1)}:
 *      quantize each endpoint to 7-bit under its chosen p-bit,
 *      rebuild palette, reassign indices, measure total error.
 *   4. Alternate least-squares endpoint refit / p-bit search while the
 *      error strictly falls, capped at MAX_REFIT_ROUNDS.
 *   5. Keep the lowest-error fit found across both seeds and every round.
 *   6. Anchor fix: if pixel 0's index has MSB=1, swap endpoints and
 *      invert all indices.
 *   7. Pack.
 *
 * Steps 2 and 4 are pure keep-the-best searches over a superset of what a
 * single seed with a single refit round explores, so no block encodes
 * worse than it otherwise would.
 */
export function encodeBC7Mode6Block(pixels: BC7Pixels): BC7Block {
  if (pixels.length !== 64) {
    throw new Error(`encodeBC7Mode6Block: expected 64 values (16 RGBA), got ${pixels.length}`)
  }

  // Step 1: 8-bit pixel buffer.
  const pixels8 = new Uint8Array(64)
  for (let k = 0; k < 64; k++) pixels8[k] = to8(pixels[k]!)

  // Steps 2-5: two seeds, each run to convergence, lowest error wins.
  //
  // Seed A, farthest pair. Per-channel min/max gives the corners of the
  // RGBA bounding box, which only coincides with the data-line endpoints
  // when every channel varies in the same direction. If R rises while G
  // falls (common for colourful gradients), the diagonal of the bbox
  // doesn't pass through the data at all — the resulting palette is
  // sideways, and refinement can't escape because the initial indices are
  // already misassigned. Picking the two pixels that are farthest apart in
  // 4D pins the endpoints to actual points on the data line, which works
  // for any channel-orientation combination.
  fitBest.err = Infinity
  const farthest = farthestPair(pixels8)
  for (let c = 0; c < 4; c++) {
    seedA[c] = pixels8[farthest.i0 * 4 + c]!
    seedB[c] = pixels8[farthest.i1 * 4 + c]!
  }
  runSeedToConvergence(pixels8, seedA, seedB, fitBest)

  // Seed B, principal axis — fitted to all sixteen pixels rather than
  // pinned to two of them, so it wins where an outlier drags the farthest
  // pair off the distribution's axis (see principalAxisSeed).
  //
  // Skipped when seed A already encodes the block exactly (flat and
  // two-tone blocks, which are common), and when the principal axis lands
  // on the same endpoints seed A used — re-running an identical search
  // cannot change the answer.
  if (fitBest.err !== 0) {
    const farA0 = seedA[0]!
    const farA1 = seedA[1]!
    const farA2 = seedA[2]!
    const farA3 = seedA[3]!
    const farB0 = seedB[0]!
    const farB1 = seedB[1]!
    const farB2 = seedB[2]!
    const farB3 = seedB[3]!
    principalAxisSeed(pixels8, seedA, seedB)
    const same =
      seedA[0] === farA0 &&
      seedA[1] === farA1 &&
      seedA[2] === farA2 &&
      seedA[3] === farA3 &&
      seedB[0] === farB0 &&
      seedB[1] === farB1 &&
      seedB[2] === farB2 &&
      seedB[3] === farB3
    if (!same) runSeedToConvergence(pixels8, seedA, seedB, fitBest)
  }

  // Step 6: anchor rule — pixel 0's index MSB must be 0.
  const best = fitBest
  let e0_7 = best.e0_7
  let e1_7 = best.e1_7
  let p0 = best.p0
  let p1 = best.p1
  const indices = best.indices
  if ((indices[0]! & 0x8) !== 0) {
    // Swap endpoints and invert every index. The decoded palette is the
    // mirror of the original palette, so reflecting indices preserves the
    // decoded image.
    const tmp7 = e0_7
    e0_7 = e1_7
    e1_7 = tmp7
    const tmpP = p0
    p0 = p1
    p1 = tmpP
    for (let k = 0; k < 16; k++) indices[k] = 15 - indices[k]!
  }

  // Step 7: pack.
  return packMode6Block(e0_7, e1_7, p0, p1, indices)
}

/**
 * One p-bit search result. Reused across calls (see `fitRun` / `fitBest`) so the
 * search allocates nothing per block.
 */
interface Mode6Fit {
  /** 7-bit endpoint channels, RGBA. */
  e0_7: Int32Array
  e1_7: Int32Array
  p0: 0 | 1
  p1: 0 | 1
  indices: Uint8Array
  err: number
}

function makeFit(): Mode6Fit {
  return { e0_7: new Int32Array(4), e1_7: new Int32Array(4), p0: 0, p1: 0, indices: new Uint8Array(16), err: 0 }
}

// Scratch for the mode-6 search. `encodeBC7Mode6Block` is synchronous and
// non-reentrant, so one module-level set is enough — and it takes the
// block's allocation count from a few hundred to zero.
const fitRun = makeFit()
const fitBest = makeFit()
const seedA = new Int32Array(4)
const seedB = new Int32Array(4)
const scratchPalette = new Uint8Array(16 * 4)
const scratchIndices = new Uint8Array(16)
// Quantised endpoint candidates, [p=0 RGBA, p=1 RGBA]: 7-bit stored values
// and the 8-bit values they reconstruct to.
const q0Seven = new Int32Array(8)
const q0Eight = new Int32Array(8)
const q1Seven = new Int32Array(8)
const q1Eight = new Int32Array(8)

/**
 * Quantise one ideal 8-bit endpoint under both p-bit choices, filling
 * `seven` / `eight` with the p=0 vector at offset 0 and p=1 at offset 4.
 *
 * The quantisation error is deliberately not computed: the p-bit search
 * scores combos by the palette's total decode error, which already
 * accounts for where quantisation moved the endpoints.
 */
function quantizeBothPbits(ideal: ArrayLike<number>, seven: Int32Array, eight: Int32Array): void {
  for (let p = 0; p < 2; p++) {
    const base = p * 4
    for (let c = 0; c < 4; c++) {
      const q = clamp(Math.round((ideal[c]! - p) / 2), 0, 127)
      seven[base + c] = q
      eight[base + c] = (q << 1) | p
    }
  }
}

/**
 * Exhaustively try the 4 p-bit combinations (p0, p1) ∈ {0,1}² against the
 * given ideal-8-bit endpoints. For each combo, quantize endpoints, rebuild
 * palette, reassign indices, and sum the squared decode error. The combo
 * with the smallest total error is written into `out`.
 *
 * Combos are visited in (0,0), (0,1), (1,0), (1,1) order and ties keep the
 * earlier combo, matching the nested-loop search this replaced.
 */
function searchPbitCombos(
  pixels8: Uint8Array,
  ideal0: ArrayLike<number>,
  ideal1: ArrayLike<number>,
  out: Mode6Fit,
): void {
  quantizeBothPbits(ideal0, q0Seven, q0Eight)
  quantizeBothPbits(ideal1, q1Seven, q1Eight)

  let bestErr = Infinity
  for (let p0 = 0; p0 < 2; p0++) {
    const a = p0 * 4
    for (let p1 = 0; p1 < 2; p1++) {
      const b = p1 * 4
      fillPalette6(
        scratchPalette,
        q0Eight[a]!,
        q0Eight[a + 1]!,
        q0Eight[a + 2]!,
        q0Eight[a + 3]!,
        q1Eight[b]!,
        q1Eight[b + 1]!,
        q1Eight[b + 2]!,
        q1Eight[b + 3]!,
      )
      const err = assignAllIndices6(pixels8, scratchPalette, scratchIndices)
      if (err < bestErr) {
        bestErr = err
        out.p0 = p0 as 0 | 1
        out.p1 = p1 as 0 | 1
        for (let c = 0; c < 4; c++) {
          out.e0_7[c] = q0Seven[a + c]!
          out.e1_7[c] = q1Seven[b + c]!
        }
        out.indices.set(scratchIndices)
      }
    }
  }
  out.err = bestErr
}

// --- Mode 6 decode ----------------------------------------------------------

/**
 * Decode a BC7 mode 6 block to 16 normalized [0, 1] RGBA pixels (64 floats).
 * Throws if the block doesn't start with the mode 6 field.
 */
export function decodeBC7Mode6Block(block: BC7Block): Float32Array {
  if (block.length !== 16) {
    throw new Error(`decodeBC7Mode6Block: expected 16 bytes, got ${block.length}`)
  }
  const mode = readBC7Mode(block)
  if (mode !== 6) {
    throw new Error(`decodeBC7Mode6Block: expected mode 6, got mode ${mode}`)
  }

  const br = new BitReader128(block)
  br.read(7) // skip mode field

  const r0 = br.read(7),
    r1 = br.read(7)
  const g0 = br.read(7),
    g1 = br.read(7)
  const b0 = br.read(7),
    b1 = br.read(7)
  const a0 = br.read(7),
    a1 = br.read(7)
  const p0 = br.read(1),
    p1 = br.read(1)

  const e0: [number, number, number, number] = [(r0 << 1) | p0, (g0 << 1) | p0, (b0 << 1) | p0, (a0 << 1) | p0]
  const e1: [number, number, number, number] = [(r1 << 1) | p1, (g1 << 1) | p1, (b1 << 1) | p1, (a1 << 1) | p1]
  const pal = buildPalette6(e0, e1)

  const out = new Float32Array(64)
  // Pixel 0: 3-bit stored (MSB implicit 0).
  const idx0 = br.read(3)
  for (let c = 0; c < 4; c++) out[c] = pal[idx0 * 4 + c]! / 255
  // Pixels 1..15: 4-bit.
  for (let k = 1; k < 16; k++) {
    const idx = br.read(4)
    const base = k * 4
    for (let c = 0; c < 4; c++) out[base + c] = pal[idx * 4 + c]! / 255
  }
  return out
}

// --- Mode 1 decode ----------------------------------------------------------
//
// MODE 1 LAYOUT (LSB-first):
//   bits 0..1    mode field (0b01 — one zero bit then a 1)
//   bits 2..7    partition index (6 bits, 64 two-subset patterns)
//   bits 8..31   R endpoints  (4 × 6 bits: s0.e0, s0.e1, s1.e0, s1.e1)
//   bits 32..55  G endpoints  (4 × 6 bits)
//   bits 56..79  B endpoints  (4 × 6 bits)
//   bit  80      P0 (p-bit shared by BOTH endpoints of subset 0)
//   bit  81      P1 (shared by subset 1)
//   bits 82..127 weights: 3-bit per pixel, except the two anchors (pixel 0
//                and BC7_ANCHOR2[partition]) which store 2 bits (MSB
//                implicit 0) — 46 bits total
//
// Endpoint dequant: v7 = (v6 << 1) | p, then e8 = (v7 << 1) | (v7 >> 6).
// Alpha decodes to 255. Weights are W3 (× 1/64). Tables cross-checked
// against bc7enc's bc7decomp.cpp (public domain).

/** Mode 1/3/7 two-subset partition patterns as 16-bit masks (bit k = subset of pixel k). */
const BC7_PARTITION2: readonly number[] = [
  0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80, 0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00,
  0xfff0, 0xf000, 0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce, 0x088c, 0x3110, 0x6666, 0x366c,
  0x17e8, 0x0ff0, 0x718e, 0x399c, 0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a, 0x73ce, 0x13c8,
  0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660, 0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c,
  0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
]

/** Anchor pixel of the second subset, per partition. */
const BC7_ANCHOR2: readonly number[] = [
  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, 15,
  15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
]

/** Mode 1 interpolation weights (× 1/64). Fixed by the spec. */
const W3: readonly number[] = [0, 9, 18, 27, 37, 46, 55, 64]

/**
 * Decode a BC7 mode 1 block to 16 normalized [0, 1] RGBA pixels (64 floats).
 * Throws if the block doesn't start with the mode 1 field.
 */
export function decodeBC7Mode1Block(block: BC7Block): Float32Array {
  if (block.length !== 16) {
    throw new Error(`decodeBC7Mode1Block: expected 16 bytes, got ${block.length}`)
  }
  const mode = readBC7Mode(block)
  if (mode !== 1) {
    throw new Error(`decodeBC7Mode1Block: expected mode 1, got mode ${mode}`)
  }

  const br = new BitReader128(block)
  br.read(2) // skip mode field
  const part = br.read(6)

  // 4 endpoints × RGB, channel-major: R×4, G×4, B×4.
  const ep: number[][] = [[], [], [], []]
  for (let c = 0; c < 3; c++) {
    for (let e = 0; e < 4; e++) ep[e]!.push(br.read(6))
  }
  const p0 = br.read(1)
  const p1 = br.read(1)
  const pbits = [p0, p0, p1, p1]

  // Dequantize 6-bit + shared p-bit to 8 bits.
  const e8: number[][] = ep.map((rgb, e) =>
    rgb.map(v6 => {
      const v7 = (v6 << 1) | pbits[e]!
      return (v7 << 1) | (v7 >> 6)
    }),
  )

  // Two 8-entry RGB palettes.
  const pal = [0, 1].map(s => {
    const lo = e8[s * 2]!
    const hi = e8[s * 2 + 1]!
    return W3.map(w => [0, 1, 2].map(c => (((64 - w) * lo[c]! + w * hi[c]! + 32) >> 6) / 255))
  })

  const anchor2 = BC7_ANCHOR2[part]!
  const mask = BC7_PARTITION2[part]!
  const out = new Float32Array(64)
  for (let k = 0; k < 16; k++) {
    const w = br.read(k === 0 || k === anchor2 ? 2 : 3)
    const subset = (mask >> k) & 1
    const rgb = pal[subset]![w]!
    out[k * 4] = rgb[0]!
    out[k * 4 + 1] = rgb[1]!
    out[k * 4 + 2] = rgb[2]!
    out[k * 4 + 3] = 1
  }
  return out
}

// --- Mode 4 decode ----------------------------------------------------------
//
// MODE 4 LAYOUT (LSB-first):
//   bits 0..4    mode field (0b00001 — four zero bits then a 1)
//   bits 5..6    rotation: 0 = none, 1 = A↔R, 2 = A↔G, 3 = A↔B (the swap is
//                applied to each decoded pixel at the END, so the "colour"
//                triple and the "alpha" scalar below are in rotated space)
//   bit  7       idxMode: 0 → colour uses the 2-bit index field and alpha
//                the 3-bit field; 1 → swapped
//   bits 8..37   colour endpoints, 5 bits each: R0 R1 G0 G1 B0 B1
//   bits 38..49  alpha endpoints, 6 bits each: A0 A1
//   bits 50..80  31-bit index field: 16 × 2-bit, pixel 0 anchored (1 bit)
//   bits 81..127 47-bit index field: 16 × 3-bit, pixel 0 anchored (2 bits)
//
// Endpoint dequant: colour e8 = (v5 << 3) | (v5 >> 2), alpha
// e8 = (v6 << 2) | (v6 >> 4). No p-bits. Weights: W2 for the 2-bit set,
// W3 for the 3-bit set. Layout cross-checked against bc7enc's
// bc7decomp.cpp (public domain) and validated against hardware
// bc7-rgba-unorm sampling.

/** 2-bit interpolation weights (× 1/64). Fixed by the spec. */
const W2: readonly number[] = [0, 21, 43, 64]

/**
 * Decode a BC7 mode 4 block to 16 normalized [0, 1] RGBA pixels (64 floats).
 * Throws if the block doesn't start with the mode 4 field.
 */
export function decodeBC7Mode4Block(block: BC7Block): Float32Array {
  if (block.length !== 16) {
    throw new Error(`decodeBC7Mode4Block: expected 16 bytes, got ${block.length}`)
  }
  const mode = readBC7Mode(block)
  if (mode !== 4) {
    throw new Error(`decodeBC7Mode4Block: expected mode 4, got mode ${mode}`)
  }

  const br = new BitReader128(block)
  br.read(5) // skip mode field
  const rotation = br.read(2)
  const idxMode = br.read(1)

  const c5: number[] = []
  for (let i = 0; i < 6; i++) c5.push(br.read(5))
  const a6 = [br.read(6), br.read(6)]

  // Dequantize (bit replication).
  const c8 = c5.map(v => (v << 3) | (v >> 2))
  const a8 = a6.map(v => (v << 2) | (v >> 4))
  const e0 = [c8[0]!, c8[2]!, c8[4]!]
  const e1 = [c8[1]!, c8[3]!, c8[5]!]

  // Index fields: 2-bit set first (31 bits), then 3-bit set (47 bits).
  const w2idx = new Uint8Array(16)
  for (let k = 0; k < 16; k++) w2idx[k] = br.read(k === 0 ? 1 : 2)
  const w3idx = new Uint8Array(16)
  for (let k = 0; k < 16; k++) w3idx[k] = br.read(k === 0 ? 2 : 3)

  const out = new Float32Array(64)
  for (let k = 0; k < 16; k++) {
    const cw = idxMode === 0 ? W2[w2idx[k]!]! : W3[w3idx[k]!]!
    const aw = idxMode === 0 ? W3[w3idx[k]!]! : W2[w2idx[k]!]!
    const px = [
      interp8(e0[0]!, e1[0]!, cw),
      interp8(e0[1]!, e1[1]!, cw),
      interp8(e0[2]!, e1[2]!, cw),
      interp8(a8[0]!, a8[1]!, aw),
    ]
    if (rotation > 0) {
      const c = rotation - 1
      const t = px[3]!
      px[3] = px[c]!
      px[c] = t
    }
    out[k * 4] = px[0]! / 255
    out[k * 4 + 1] = px[1]! / 255
    out[k * 4 + 2] = px[2]! / 255
    out[k * 4 + 3] = px[3]! / 255
  }
  return out
}

// --- Top-level entry points -------------------------------------------------

/**
 * Encode a BC7 block. Currently this always produces a mode 6 block —
 * see the file header for why other modes are deliberately out of scope.
 */
export function encodeBC7Block(pixels: BC7Pixels): BC7Block {
  return encodeBC7Mode6Block(pixels)
}

/**
 * Decode a BC7 block, dispatching on the mode field. This reference
 * supports modes 1, 4 and 6 — the modes the GPU encoder emits (or has
 * emitted); other modes throw. Encoders outside this project (AMD
 * Compressonator, bc7enc, ...) routinely pick other modes, so this
 * decoder is mainly for round-tripping our own output.
 */
export function decodeBC7Block(block: BC7Block): Float32Array {
  const mode = readBC7Mode(block)
  switch (mode) {
    case 1:
      return decodeBC7Mode1Block(block)
    case 4:
      return decodeBC7Mode4Block(block)
    case 6:
      return decodeBC7Mode6Block(block)
    default:
      throw new Error(`decodeBC7Block: mode ${mode} not supported by this reference decoder`)
  }
}

// Exposed for tests — tight coupling to the bitstream layout tests and
// the anchor-rule check. Not part of the stable public surface.
export const _internal = {
  W4,
  MODE6_ANCHOR_MSB_CUTOFF,
  BitWriter128,
  BitReader128,
  buildPalette6,
  interp8,
  to8,
}
