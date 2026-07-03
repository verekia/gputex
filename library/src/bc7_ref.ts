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
 * 128-bit little-endian bit writer backed by a single BigInt. Matches
 * the BC7 "LSB-first" bitstream convention — the first value written
 * occupies the low-order bits.
 */
class BitWriter128 {
  private bits = 0n
  private pos = 0

  write(value: number, nBits: number): void {
    if (this.pos + nBits > 128) throw new Error('BitWriter128 overflow')
    const v = BigInt(value) & ((1n << BigInt(nBits)) - 1n)
    this.bits |= v << BigInt(this.pos)
    this.pos += nBits
  }

  toBytes(): Uint8Array {
    const out = new Uint8Array(16)
    for (let i = 0; i < 16; i++) {
      out[i] = Number((this.bits >> BigInt(i * 8)) & 0xffn)
    }
    return out
  }
}

class BitReader128 {
  private bits: bigint
  private pos = 0

  constructor(block: BC7Block) {
    let b = 0n
    for (let i = 0; i < 16; i++) b |= BigInt(block[i]!) << BigInt(i * 8)
    this.bits = b
  }

  read(nBits: number): number {
    const v = Number((this.bits >> BigInt(this.pos)) & ((1n << BigInt(nBits)) - 1n))
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
 * Quantize an 8-bit channel value to (7-bit, p-bit) where p is already
 * chosen. The effective reconstructed 8-bit is (q << 1) | p.
 * Returns {q, error²} where error² is squared distance from the ideal.
 */
function quantizeChannelWithPbit(ideal8: number, p: 0 | 1): { q: number; err: number } {
  // Closest 7-bit `q` such that (q<<1)|p is near ideal8.
  const q = clamp(Math.round((ideal8 - p) / 2), 0, 127)
  const effective = (q << 1) | p
  const d = effective - ideal8
  return { q, err: d * d }
}

/**
 * Quantize one endpoint (4 channels) under a fixed shared p-bit choice.
 * Returns the 7-bit RGBA vector and the total squared quantization error.
 */
function quantizeEndpoint(
  ideal8: readonly number[],
  p: 0 | 1,
): { rgba: [number, number, number, number]; err: number } {
  const out: [number, number, number, number] = [0, 0, 0, 0]
  let err = 0
  for (let c = 0; c < 4; c++) {
    const r = quantizeChannelWithPbit(ideal8[c]!, p)
    out[c] = r.q
    err += r.err
  }
  return { rgba: out, err }
}

/**
 * Build the 16-entry 8-bit-per-channel palette from two 8-bit RGBA endpoints.
 */
function buildPalette6(
  e0: readonly [number, number, number, number],
  e1: readonly [number, number, number, number],
): Uint8Array {
  // palette[i * 4 + c] = interp8(e0[c], e1[c], W4[i])
  const pal = new Uint8Array(16 * 4)
  for (let i = 0; i < 16; i++) {
    const w = W4[i]!
    const base = i * 4
    for (let c = 0; c < 4; c++) {
      pal[base + c] = interp8(e0[c]!, e1[c]!, w)
    }
  }
  return pal
}

/**
 * Nearest palette index for a single RGBA pixel. L2 across all 4 channels.
 * Operates on 8-bit values so the distance metric matches hardware behavior.
 */
function assignIndex6(
  pixel: readonly [number, number, number, number],
  palette: Uint8Array,
): { idx: number; err: number } {
  let bestIdx = 0
  let bestD = Infinity
  for (let i = 0; i < 16; i++) {
    const base = i * 4
    const dr = palette[base]! - pixel[0]!
    const dg = palette[base + 1]! - pixel[1]!
    const db = palette[base + 2]! - pixel[2]!
    const da = palette[base + 3]! - pixel[3]!
    const d = dr * dr + dg * dg + db * db + da * da
    if (d < bestD) {
      bestD = d
      bestIdx = i
    }
  }
  return { idx: bestIdx, err: bestD }
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

function totalSqErrorForEndpoints(
  pixels8: Uint8Array,
  e0_8: readonly [number, number, number, number],
  e1_8: readonly [number, number, number, number],
): { indices: Uint8Array; err: number } {
  const pal = buildPalette6(e0_8, e1_8)
  const indices = new Uint8Array(16)
  let err = 0
  for (let k = 0; k < 16; k++) {
    const base = k * 4
    const sel = assignIndex6([pixels8[base]!, pixels8[base + 1]!, pixels8[base + 2]!, pixels8[base + 3]!], pal)
    indices[k] = sel.idx
    err += sel.err
  }
  return { indices, err }
}

/**
 * Pack a mode 6 block into 16 bytes.
 *   e0_7, e1_7 are the 7-bit-per-channel RGBA endpoint vectors.
 *   p0, p1 are the shared p-bits (one per endpoint).
 *   indices are 4-bit values, with indices[0]'s MSB already guaranteed 0.
 */
function packMode6Block(
  e0_7: readonly [number, number, number, number],
  e1_7: readonly [number, number, number, number],
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
 * Encode 16 RGBA pixels (64 floats in [0,1]) as a BC7 mode 6 block.
 *
 * Algorithm:
 *   1. Convert to 8-bit per channel.
 *   2. Farthest-pair in 4D → initial 8-bit endpoints. (See farthestPair
 *      for why bbox corners aren't safe when channels vary in different
 *      directions along the data line.)
 *   3. For each p-bit combo {(0,0),(0,1),(1,0),(1,1)}:
 *      quantize each endpoint to 7-bit under its chosen p-bit,
 *      rebuild palette, reassign indices, measure total error.
 *   4. Keep the best p-bit combo.
 *   5. One-pass least-squares refinement on the surviving indices,
 *      re-quantize with p-bit search, accept if error decreases.
 *   6. Anchor fix: if pixel 0's index has MSB=1, swap endpoints and
 *      invert all indices.
 *   7. Pack.
 */
export function encodeBC7Mode6Block(pixels: BC7Pixels): BC7Block {
  if (pixels.length !== 64) {
    throw new Error(`encodeBC7Mode6Block: expected 64 values (16 RGBA), got ${pixels.length}`)
  }

  // Step 1: 8-bit pixel buffer.
  const pixels8 = new Uint8Array(64)
  for (let k = 0; k < 64; k++) pixels8[k] = to8(pixels[k]!)

  // Step 2: initial endpoints via farthest-pair. Per-channel min/max gives
  // the corners of the RGBA bounding box, which only coincides with the
  // data-line endpoints when every channel varies in the same direction.
  // If R rises while G falls (common for colorful gradients), the diagonal
  // of the bbox doesn't pass through the data at all — the resulting
  // palette is sideways, and refinement can't escape because the initial
  // indices are already misassigned. Picking the two pixels that are
  // farthest apart in 4D pins the endpoints to actual points on the data
  // line, which works for any channel-orientation combination.
  const farthest = farthestPair(pixels8)
  const ideal0: [number, number, number, number] = [
    pixels8[farthest.i0 * 4]!,
    pixels8[farthest.i0 * 4 + 1]!,
    pixels8[farthest.i0 * 4 + 2]!,
    pixels8[farthest.i0 * 4 + 3]!,
  ]
  const ideal1: [number, number, number, number] = [
    pixels8[farthest.i1 * 4]!,
    pixels8[farthest.i1 * 4 + 1]!,
    pixels8[farthest.i1 * 4 + 2]!,
    pixels8[farthest.i1 * 4 + 3]!,
  ]

  // Helper: given ideal 8-bit endpoints and a p-bit choice, quantize to
  // 7-bit and return (7-bit vec, reconstructed 8-bit vec, quantization err).
  function quantPair(
    ideal: readonly [number, number, number, number],
    p: 0 | 1,
  ): {
    seven: [number, number, number, number]
    eight: [number, number, number, number]
    err: number
  } {
    const r = quantizeEndpoint(ideal, p)
    const eight: [number, number, number, number] = [
      (r.rgba[0] << 1) | p,
      (r.rgba[1] << 1) | p,
      (r.rgba[2] << 1) | p,
      (r.rgba[3] << 1) | p,
    ]
    return { seven: r.rgba, eight, err: r.err }
  }

  // Step 3: try all 4 p-bit combos for the initial bbox endpoints.
  let best = tryPbitCombos(pixels8, ideal0, ideal1, quantPair)

  // Step 5: one-pass least-squares refinement + p-bit search.
  const refit = refitEndpointsMode6(pixels8, best.indices)
  if (refit) {
    const candidate = tryPbitCombos(pixels8, refit.e0, refit.e1, quantPair)
    if (candidate.err < best.err) best = candidate
  }

  // Step 6: anchor rule — pixel 0's index MSB must be 0.
  let e0_7 = best.e0_7
  let e1_7 = best.e1_7
  let p0 = best.p0
  let p1 = best.p1
  let indices = best.indices
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
    const inv = new Uint8Array(16)
    for (let k = 0; k < 16; k++) inv[k] = 15 - indices[k]!
    indices = inv
  }

  // Step 7: pack.
  return packMode6Block(e0_7, e1_7, p0, p1, indices)
}

/**
 * Exhaustively try the 4 p-bit combinations (p0, p1) ∈ {0,1}² against the
 * given ideal-8-bit endpoints. For each combo, quantize endpoints, rebuild
 * palette, reassign indices, and sum the squared decode error. Returns the
 * combo with the smallest total error.
 */
function tryPbitCombos(
  pixels8: Uint8Array,
  ideal0: readonly [number, number, number, number],
  ideal1: readonly [number, number, number, number],
  quantPair: (
    ideal: readonly [number, number, number, number],
    p: 0 | 1,
  ) => {
    seven: [number, number, number, number]
    eight: [number, number, number, number]
    err: number
  },
): {
  e0_7: [number, number, number, number]
  e1_7: [number, number, number, number]
  p0: 0 | 1
  p1: 0 | 1
  indices: Uint8Array
  err: number
} {
  let best: ReturnType<typeof tryPbitCombos> | null = null
  for (const p0 of [0, 1] as const) {
    const q0 = quantPair(ideal0, p0)
    for (const p1 of [0, 1] as const) {
      const q1 = quantPair(ideal1, p1)
      // Decode error = decode-distance of each pixel to nearest palette entry.
      const { indices, err } = totalSqErrorForEndpoints(pixels8, q0.eight, q1.eight)
      if (best == null || err < best.err) {
        best = { e0_7: q0.seven, e1_7: q1.seven, p0, p1, indices, err }
      }
    }
  }
  // `best` is always set after the loop — the guard on Array.prototype.forEach
  // wouldn't fire here either; narrow explicitly.
  if (best == null) throw new Error('unreachable: tryPbitCombos no-op')
  return best
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
