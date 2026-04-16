// ASTC 4×4 LDR reference encoder + decoder. CPU implementation.
//
// The full ASTC format is sprawling — variable block sizes, trits/quints
// endpoint quantisation, partition tables, dual-plane, HDR modes, void
// extents. This encoder deliberately occupies a tiny, cleanly-specified
// corner of that space so the CPU and GPU ports can both be audited
// line-by-line:
//
//   • Single partition (no multi-subset fitting).
//   • No dual-plane.
//   • Color endpoint mode 12 = LDR RGBA, direct. Works for opaque and
//     translucent inputs alike (opaque → A0 = A1 = 255).
//   • Weight grid 4×4 (one weight per footprint texel, no upsampling).
//   • 2-bit weights (QUANT_4), weight-stream 32 bits total.
//   • 8-bit endpoints (QUANT_256) — bit-replication is a no-op, so the
//     stored byte equals the unquantised byte.
//
// This produces *fully valid* ASTC 4×4 blocks: any conforming decoder
// (software or hardware) will accept them and reconstruct RGBA pixels
// close to the source image. Quality is bounded below that of a full
// ASTC encoder (which would search partitions, weight precisions, etc.)
// but mode-6-only BC7 suffers from the same narrowing and still looks
// excellent on smooth content. See the test file for the error budgets.
//
// -----------------------------------------------------------------------
// BLOCK LAYOUT (128 bits total, LSB-first, bit 0 = byte 0's bit 0)
//
//   bits [10:0]   block mode = 0x042
//                 (decoded as: W=4, H=4, R=4 → QUANT_4 weights, H=D=0)
//   bits [12:11]  partition_count − 1 = 0 (one partition)
//   bits [16:13]  CEM = 12 (LDR RGBA direct)
//   bits [80:17]  endpoint data: 8 values × 8 bits each, in this order:
//                 R0, R1, G0, G1, B0, B1, A0, A1
//   bits [95:81]  unused (15 bits; must be zero)
//   bits [127:96] weight data. For weight k in 0..15 (row-major, x fast):
//                 block_bit(127 − 2k)     = weight_k[0]   (LSB)
//                 block_bit(126 − 2k)     = weight_k[1]   (MSB)
//                 i.e. the first weight is at the "top" of the block,
//                 the last weight at bits [97:96].
//
// BLOCK MODE (0x042) DERIVATION
//   Using the decode formulas in Khronos DF spec §22.11 / ARM astc-encoder
//   `decode_block_mode_2d`:
//     base_quant_mode = ((block_mode & 3) << 1) | ((block_mode >> 4) & 1)
//     With block_mode & 3 = 2 and bit 4 = 0 → base_quant_mode = 4 = R.
//     The `(block_mode >> 2) & 3 = 0` selects case 0: W = B+4, H = A+2
//     where B = bits[8:7], A = bits[6:5].
//     For W=H=4: B = 0 (bits 7,8 = 00), A = 2 (bits 5,6 = 0,1).
//     H-flag (bit 9) = 0, D-flag (bit 10) = 0.
//   Laying out: bit 0=0, 1=1, 2-5=0, 6=1, 7-10=0 → 0b00001000010 = 0x042.
//
// ENDPOINT ORDERING (avoiding blue contraction)
//   The decoder for CEM 12 branches on the RGB sum comparison:
//     if (v0 + v2 + v4) > (v1 + v3 + v5):
//       swap + blue_contract   ← we avoid this path
//     else:
//       e0 = (v0, v2, v4, v6)
//       e1 = (v1, v3, v5, v7)
//   Blue-contraction is a lossy remap that squeezes the dynamic range of
//   R and G toward B. Our encoder ensures sum(e0.rgb) ≤ sum(e1.rgb) by
//   swapping endpoints (and inverting all weight indices w → 3−w) before
//   packing, which lands us unambiguously in the else-branch. The
//   decoder below still handles the swap path correctly so externally-
//   encoded blocks round-trip, but our own blocks never exercise it.
//
// WEIGHT UNQUANTISATION (QUANT_4)
//   Bit-replicate the 2-bit index to 6 bits, then bump >32 by one:
//     q=0 → 000000 = 0            (≤32, no bump)
//     q=1 → 010101 = 21           (≤32, no bump)
//     q=2 → 101010 = 42 → 43      (>32, +1)
//     q=3 → 111111 = 63 → 64      (>32, +1)
//   The encoder and decoder here both use the resulting [0, 21, 43, 64]
//   table, and the hardware-exact interpolation formula
//     color_8 = ((64 − w) · e0_8 + w · e1_8 + 32) >> 6
//   which matches BC7's.

/** 16 pixels × 4 channels = 64 normalised floats, interleaved RGBA. */
export type ASTC4x4Pixels = Readonly<ArrayLike<number>>

/** 16 bytes of ASTC 4×4 block data. */
export type ASTC4x4Block = Uint8Array

// --- Constants --------------------------------------------------------------

/** Block mode: 4×4 grid, 2-bit weights, single plane, LDR. See header. */
const BLOCK_MODE_4x4_2BIT = 0x042

/** CEM 12 = "LDR RGBA, direct". 8 endpoint values per block. */
const CEM_RGBA_DIRECT = 12

/** Weight unquantisation table for QUANT_4. Indexed by 2-bit weight. */
const WEIGHT_UNQ_4: readonly number[] = [0, 21, 43, 64]

// --- Scalar helpers ---------------------------------------------------------

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v
}

/** Normalised [0, 1] → clamped 8-bit. */
function to8(v: number): number {
  return clamp(Math.round(v * 255), 0, 255)
}

/**
 * Hardware-exact integer interpolation, shared with BC7. Given 8-bit
 * endpoints and a 0..64 weight, returns the reconstructed 8-bit channel.
 */
function interp8(e0: number, e1: number, w: number): number {
  return ((64 - w) * e0 + w * e1 + 32) >> 6
}

// --- 128-bit positioned bit writer / reader ---------------------------------
//
// BC7's BitWriter128 is append-only (it tracks a monotonically advancing
// position). ASTC's layout has two growth directions — config + endpoints
// from the low end, weights from bit 127 downward — so a positioned API
// is easier to audit: the caller states exactly where each field lives.

class BitWriter128 {
  private bits = 0n

  write(pos: number, nBits: number, value: number): void {
    if (pos < 0 || nBits < 0 || pos + nBits > 128) {
      throw new Error(`BitWriter128: out-of-range write pos=${pos}, n=${nBits}`)
    }
    const mask = (1n << BigInt(nBits)) - 1n
    // Clear first, then OR in; makes the writer safe against re-writes at
    // the same position (not used today, but removes a sharp edge).
    this.bits &= ~(mask << BigInt(pos))
    this.bits |= (BigInt(value) & mask) << BigInt(pos)
  }

  toBytes(): Uint8Array {
    const out = new Uint8Array(16)
    let b = this.bits
    for (let i = 0; i < 16; i++) {
      out[i] = Number(b & 0xffn)
      b >>= 8n
    }
    return out
  }
}

class BitReader128 {
  private bits: bigint

  constructor(block: ASTC4x4Block) {
    let b = 0n
    for (let i = 0; i < 16; i++) b |= BigInt(block[i]!) << BigInt(i * 8)
    this.bits = b
  }

  read(pos: number, nBits: number): number {
    const mask = (1n << BigInt(nBits)) - 1n
    return Number((this.bits >> BigInt(pos)) & mask)
  }
}

// --- Endpoint line fit ------------------------------------------------------

/**
 * Pick the pair of texels (out of 16) that maximise 4D RGBA L2 distance.
 * Same rationale as BC7's `farthestPair`: this is immune to the "channels
 * vary in different directions along the data line → bbox diagonal misses
 * the data" failure mode of a per-channel min/max seed. O(N²) = 120
 * comparisons, trivial cost.
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

/** Build the 4-entry RGBA palette from 8-bit endpoints and WEIGHT_UNQ_4. */
function buildPalette(
  e0: readonly [number, number, number, number],
  e1: readonly [number, number, number, number],
): Uint8Array {
  const pal = new Uint8Array(4 * 4)
  for (let i = 0; i < 4; i++) {
    const w = WEIGHT_UNQ_4[i]!
    const base = i * 4
    for (let c = 0; c < 4; c++) {
      pal[base + c] = interp8(e0[c]!, e1[c]!, w)
    }
  }
  return pal
}

/** Nearest palette index for one 8-bit RGBA pixel. Full 4-entry L2 search. */
function assignIndex(
  pixel: readonly [number, number, number, number],
  palette: Uint8Array,
): { idx: number; err: number } {
  let bestIdx = 0
  let bestD = Infinity
  for (let i = 0; i < 4; i++) {
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
 * Assign all 16 pixels to nearest palette entries; return indices and
 * summed squared error in 8-bit decode space. Same shape as BC7's
 * `totalSqErrorForEndpoints`.
 */
function totalSqError(
  pixels8: Uint8Array,
  e0: readonly [number, number, number, number],
  e1: readonly [number, number, number, number],
): { indices: Uint8Array; err: number } {
  const pal = buildPalette(e0, e1)
  const indices = new Uint8Array(16)
  let err = 0
  for (let k = 0; k < 16; k++) {
    const base = k * 4
    const sel = assignIndex([pixels8[base]!, pixels8[base + 1]!, pixels8[base + 2]!, pixels8[base + 3]!], pal)
    indices[k] = sel.idx
    err += sel.err
  }
  return { indices, err }
}

/**
 * One-pass least-squares refit. Given current per-texel 2-bit indices,
 * find the (e0, e1) pair that minimises Σ (palette[idx_k] − v_k)².
 *
 * Per-channel normal equations are identical to BC7's:
 *   sAA · e0 + sAB · e1 = sAV
 *   sAB · e0 + sBB · e1 = sBV
 * with a_k = (64 − unq_{i_k}) / 64, b_k = unq_{i_k} / 64.
 *
 * Returns null when the system is degenerate (all texels landed on a
 * single palette entry — the weight vectors are colinear).
 */
function refitEndpoints(
  pixels8: Uint8Array,
  indices: Uint8Array,
): { e0: [number, number, number, number]; e1: [number, number, number, number] } | null {
  let sAA = 0,
    sBB = 0,
    sAB = 0
  const sAV: [number, number, number, number] = [0, 0, 0, 0]
  const sBV: [number, number, number, number] = [0, 0, 0, 0]
  for (let k = 0; k < 16; k++) {
    const unq = WEIGHT_UNQ_4[indices[k]!]!
    const a = (64 - unq) / 64
    const b = unq / 64
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
    e0[c] = clamp(Math.round((sBB * sAV[c] - sAB * sBV[c]) / det), 0, 255)
    e1[c] = clamp(Math.round((sAA * sBV[c] - sAB * sAV[c]) / det), 0, 255)
  }
  return { e0, e1 }
}

// --- Encode -----------------------------------------------------------------

/**
 * Encode 16 RGBA pixels (64 floats in [0, 1]) as a single ASTC 4×4 LDR
 * block using the narrow subset described at the top of this file.
 *
 * Algorithm:
 *   1. Quantise input to 8-bit.
 *   2. Farthest-pair in 4D → initial (e0, e1).
 *   3. Assign per-texel 2-bit indices by nearest palette entry.
 *   4. One LSQ refit pass; accept only if total error strictly decreases.
 *   5. Flip endpoints (and invert indices) if sum(e0.rgb) > sum(e1.rgb)
 *      so the decoder doesn't apply blue contraction.
 *   6. Pack block mode, CEM, endpoints, weights into 128 bits.
 */
export function encodeASTC4x4Block(pixels: ASTC4x4Pixels): ASTC4x4Block {
  if (pixels.length !== 64) {
    throw new Error(`encodeASTC4x4Block: expected 64 values (16 RGBA), got ${pixels.length}`)
  }

  // Step 1.
  const pixels8 = new Uint8Array(64)
  for (let k = 0; k < 64; k++) pixels8[k] = to8(pixels[k]!)

  // Step 2.
  const fp = farthestPair(pixels8)
  let e0: [number, number, number, number] = [
    pixels8[fp.i0 * 4]!,
    pixels8[fp.i0 * 4 + 1]!,
    pixels8[fp.i0 * 4 + 2]!,
    pixels8[fp.i0 * 4 + 3]!,
  ]
  let e1: [number, number, number, number] = [
    pixels8[fp.i1 * 4]!,
    pixels8[fp.i1 * 4 + 1]!,
    pixels8[fp.i1 * 4 + 2]!,
    pixels8[fp.i1 * 4 + 3]!,
  ]

  // Step 3.
  let { indices, err } = totalSqError(pixels8, e0, e1)

  // Step 4.
  const refit = refitEndpoints(pixels8, indices)
  if (refit) {
    const { indices: idx2, err: err2 } = totalSqError(pixels8, refit.e0, refit.e1)
    if (err2 < err) {
      e0 = refit.e0
      e1 = refit.e1
      indices = idx2
      err = err2
    }
  }

  // Step 5: endpoint ordering. Strict '>' so a tie (s0 == s1) doesn't
  // cause a gratuitous swap.
  const s0 = e0[0] + e0[1] + e0[2]
  const s1 = e1[0] + e1[1] + e1[2]
  if (s0 > s1) {
    const tmp = e0
    e0 = e1
    e1 = tmp
    const inv = new Uint8Array(16)
    // Reflect weights: w' = 3 − w. The decoded palette is mirrored, so
    // the reconstructed colour is unchanged.
    for (let k = 0; k < 16; k++) inv[k] = 3 - indices[k]!
    indices = inv
  }

  // Step 6.
  return packBlock(e0, e1, indices)
}

function packBlock(
  e0: readonly [number, number, number, number],
  e1: readonly [number, number, number, number],
  indices: Uint8Array,
): ASTC4x4Block {
  const bw = new BitWriter128()

  // Config header.
  bw.write(0, 11, BLOCK_MODE_4x4_2BIT)
  bw.write(11, 2, 0) // partition_count − 1
  bw.write(13, 4, CEM_RGBA_DIRECT)

  // Endpoints, 8 × 8-bit values at bits [80:17], LSB of each byte at the
  // byte boundary (normal bit order).
  // Order matches the CEM 12 decoder's (v0..v7) = (R0, R1, G0, G1, B0, B1, A0, A1).
  const ep: readonly number[] = [e0[0], e1[0], e0[1], e1[1], e0[2], e1[2], e0[3], e1[3]]
  for (let i = 0; i < 8; i++) bw.write(17 + i * 8, 8, ep[i]!)

  // Weights: 16 × 2-bit values, LSB of weight k at block bit (127 − 2k),
  // MSB at (126 − 2k). Two 1-bit writes per weight keeps the mapping
  // obvious at the cost of 32 calls — unmeasurable versus the encode
  // cost, and worth the clarity.
  for (let k = 0; k < 16; k++) {
    const w = indices[k]! & 0x3
    bw.write(127 - 2 * k, 1, w & 1)
    bw.write(126 - 2 * k, 1, (w >> 1) & 1)
  }

  return bw.toBytes()
}

// --- Decode -----------------------------------------------------------------

/**
 * Decode an ASTC 4×4 block produced by this encoder (or by any other
 * encoder that respects our narrow subset: block mode 0x042, single
 * partition, CEM 12). Handles the blue-contraction branch even though
 * our encoder doesn't produce it, so externally-supplied blocks round-
 * trip predictably.
 *
 * Output: 16 RGBA pixels as 64 floats in [0, 1].
 */
export function decodeASTC4x4Block(block: ASTC4x4Block): Float32Array {
  if (block.length !== 16) {
    throw new Error(`decodeASTC4x4Block: expected 16 bytes, got ${block.length}`)
  }
  const br = new BitReader128(block)

  const mode = br.read(0, 11)
  if (mode !== BLOCK_MODE_4x4_2BIT) {
    throw new Error(
      `decodeASTC4x4Block: expected block mode 0x${BLOCK_MODE_4x4_2BIT.toString(16)}, ` + `got 0x${mode.toString(16)}`,
    )
  }
  const partCount = br.read(11, 2)
  if (partCount !== 0) {
    throw new Error(`decodeASTC4x4Block: multi-partition blocks not supported (count=${partCount + 1})`)
  }
  const cem = br.read(13, 4)
  if (cem !== CEM_RGBA_DIRECT) {
    throw new Error(`decodeASTC4x4Block: only CEM 12 supported, got ${cem}`)
  }

  // Endpoint values as stored.
  const v: number[] = []
  for (let i = 0; i < 8; i++) v.push(br.read(17 + i * 8, 8))
  const [v0, v1, v2, v3, v4, v5, v6, v7] = v as [number, number, number, number, number, number, number, number]

  // CEM 12 ordering + blue contraction.
  let e0: [number, number, number, number]
  let e1: [number, number, number, number]
  if (v0 + v2 + v4 <= v1 + v3 + v5) {
    e0 = [v0, v2, v4, v6]
    e1 = [v1, v3, v5, v7]
  } else {
    // blue_contract(r, g, b) = ((r + b) >> 1, (g + b) >> 1, b). Alpha
    // tags along unchanged, but the endpoint swap is full (RGB + A).
    e0 = [(v1 + v5) >> 1, (v3 + v5) >> 1, v5, v7]
    e1 = [(v0 + v4) >> 1, (v2 + v4) >> 1, v4, v6]
  }

  // 16 × 2-bit weights at the top of the block.
  const weights = new Uint8Array(16)
  for (let k = 0; k < 16; k++) {
    const lsb = br.read(127 - 2 * k, 1)
    const msb = br.read(126 - 2 * k, 1)
    weights[k] = lsb | (msb << 1)
  }

  const out = new Float32Array(64)
  for (let k = 0; k < 16; k++) {
    const w = WEIGHT_UNQ_4[weights[k]!]!
    const base = k * 4
    for (let c = 0; c < 4; c++) {
      out[base + c] = interp8(e0[c]!, e1[c]!, w) / 255
    }
  }
  return out
}

// --- Test-only exposure ----------------------------------------------------

// Exposed so tests can assert bit layouts and round-trip constants without
// re-deriving them. Not part of the stable public surface; the encoder /
// decoder functions above are.
export const _internal = {
  BLOCK_MODE_4x4_2BIT,
  CEM_RGBA_DIRECT,
  WEIGHT_UNQ_4,
  BitWriter128,
  BitReader128,
  buildPalette,
  interp8,
  to8,
}
