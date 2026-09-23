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
//   • Endpoints either 8-bit (QUANT_256 — bit-replication is a no-op, so
//     the stored byte equals the unquantised byte) or QUANT_192 (one trit
//     + 6 bits per value, integer-sequence encoded — see below).
//   • Plain-bit weights; weight grid 4×4 (one weight per footprint texel,
//     no upsampling).
//   • Block classes chosen per block by content — the ASTC bit budget
//     trades endpoint bits against weight bits, so a block should only pay
//     for the endpoint channels it actually uses:
//       gray + opaque   → CEM 0  (LDR luminance direct), 5-bit weights
//                         (QUANT_32): 2×8 endpoint bits + 80 weight bits
//       opaque          → CEM 8  (LDR RGB direct), one of two budgets:
//                         3-bit weights (QUANT_8) + 6×8 endpoint bits, or
//                         4-bit weights (QUANT_16) + 6 QUANT_192 values
//                         (46 bits). The encoder below tries both and keeps
//                         the lower error; the GPU encoders pick by the
//                         block's colour span (exact endpoints win on
//                         small spans, finer weights on wide ones).
//       translucent     → CEM 12 (LDR RGBA direct), 2-bit weights
//                         (QUANT_4): 8×8 endpoint bits + 32 weight bits
//     "gray" means every texel has R == G == B exactly in 8-bit;
//     "opaque" means every texel has A == 255. A grayscale roughness/AO
//     map gets a 32-level palette instead of 4 — the difference between
//     visible banding and near-BC4 quality — while translucent content
//     keeps the full RGBA endpoint pair.
//
// This produces *fully valid* ASTC 4×4 blocks: any conforming decoder
// (software or hardware) will accept them and reconstruct RGBA pixels
// close to the source image. Quality is bounded below that of a full
// ASTC encoder (which would search partitions, weight precisions, etc.).
//
// -----------------------------------------------------------------------
// BLOCK LAYOUT (128 bits total, LSB-first, bit 0 = byte 0's bit 0)
//
//   bits [10:0]   block mode: 0x042 (QUANT_4), 0x053 (QUANT_8), 0x242
//                 (QUANT_16) or 0x253 (QUANT_32) — all decode to a 4×4
//                 single-plane weight grid, see derivation below
//   bits [12:11]  partition_count − 1 = 0 (one partition)
//   bits [16:13]  CEM = 0, 8 or 12
//   bits [17+]    endpoint data, v0 first:
//                 CEM 0:  L0 L1, 8 bits each                   (ends at 33)
//                 CEM 8:  R0 R1 G0 G1 B0 B1, 8 bits each       (ends at 65)
//                         or QUANT_192 ISE (mode 0x242)        (ends at 63)
//                 CEM 12: R0 R1 G0 G1 B0 B1 A0 A1, 8 bits each (ends at 81)
//   bits [..]     unused, zero
//   top bits      weight data, growing DOWN from bit 127: bit j (LSB=0)
//                 of weight k lives at block bit (127 − nBits·k − j).
//                 QUANT_4: 32 bits (down to 96), QUANT_8: 48 (down to
//                 80), QUANT_16: 64 (down to 64), QUANT_32: 80 (down to
//                 48).
//
// The endpoint range is not stored: the decoder derives it as the largest
// range whose ISE encoding of the CEM's values fits the bits left after
// the weights. 128 − 17 − 48 (QUANT_8) = 63 → QUANT_256 for 6 values;
// 128 − 17 − 64 (QUANT_16) = 47 → QUANT_192 (46 bits — QUANT_256 would
// need 48).
//
// QUANT_192 ENDPOINTS (ISE, one trit + 6 bits per value)
//   Values are coded in groups of five: v0 bits, T[1:0], v1 bits, T[3:2],
//   v2 bits, T[4], v3 bits, T[6:5], v4 bits, T[7], where T packs the five
//   trits (spec trit-block table; `encodeTrits` is its inverse). A partial
//   last group stops after its last value's T bits — CEM 8's sixth value
//   is followed by T[1:0] only. Unquantisation (spec, trit ranges): with a
//   = bit 0 of the 6 bits and b..f the rest, A = a ? 0x1FF : 0,
//   B = fedcb000f (9 bits), C = 5, D = the trit:
//     T = D·C + B;  T ^= A;  unq = (A & 0x80) | (T >> 2)
//   which yields every value ≤ 127 not ≡ 3 (mod 4) plus their mirror
//   images 255 − u — 192 levels, hardware-verified on Apple M3.
//
// BLOCK MODE DERIVATION
//   Using the decode formulas in Khronos DF spec §22.11 / ARM astc-encoder
//   `decode_block_mode_2d`:
//     base_quant_mode R = ((block_mode & 3) << 1) | ((block_mode >> 4) & 1)
//     (block_mode >> 2) & 3 = 0 selects case 0: W = B+4, H = A+2
//     where B = bits[8:7], A = bits[6:5]; H-flag = bit 9, D-flag = bit 10.
//   For W = H = 4: B = 0, A = 2 (bit 6 set). The weight range comes from
//   (R, H-flag): (4, 0) → QUANT_4, (7, 0) → QUANT_8, (4, 1) → QUANT_16,
//   (7, 1) → QUANT_32.
//     0x042 = 0b000_0100_0010: R = 100₂ = 4, H = 0 → QUANT_4
//     0x053 = 0b000_0101_0011: R = 111₂ = 7, H = 0 → QUANT_8
//     0x242 = 0b010_0100_0010: R = 100₂ = 4, H = 1 → QUANT_16
//     0x253 = 0b010_0101_0011: R = 111₂ = 7, H = 1 → QUANT_32
//
// ENDPOINT ORDERING (avoiding blue contraction)
//   The decoders for CEM 8 and CEM 12 branch on the RGB sum comparison:
//     if (v0 + v2 + v4) > (v1 + v3 + v5): swap + blue_contract
//   Blue-contraction is a lossy remap that squeezes the dynamic range of
//   R and G toward B. Our encoder ensures sum(e0.rgb) ≤ sum(e1.rgb) by
//   swapping endpoints (and reflecting all weights w → wmax − w) before
//   packing. The decoder below still handles the swap path correctly so
//   externally-encoded blocks round-trip, but our own blocks never
//   exercise it. CEM 0 has no such rule.
//
// WEIGHT UNQUANTISATION (bit-only ranges)
//   Spec rule: bit-replicate to 6 bits (shift left until the MSB is in
//   bit 5, OR the original top bits into the low bits), then add 1 to
//   every value greater than 32 — mapping 0..63 onto 0..64 with 32 fixed.
//     QUANT_4  (2-bit): [0, 21, 43, 64]
//     QUANT_8  (3-bit): [0, 9, 18, 27, 37, 46, 55, 64]
//     QUANT_16 (4-bit): [0, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52,
//                        56, 60, 64]
//     QUANT_32 (5-bit): w ≤ 15 → 2w; w ≥ 16 → 2w + 2   (0..30, 34..64)
//   Decode interpolation follows the spec's LDR rule exactly: endpoints
//   are first expanded to 16 bits by byte replication (e16 = e8 · 257),
//   then interpolated in 16-bit space
//     C = ((64 − w) · e0_16 + w · e1_16 + 32) >> 6
//   and the result is a unorm16 (C / 65535). This differs from BC7's
//   8-bit formula by ±1/255 on some (endpoint, weight) combinations.
//   Hardware probe (M3, 2026-07, exhaustive (v0,v1,w) sweeps rendered to
//   rgba16float): Apple's effective weight is exactly unq/64 for
//   unq ≤ 32 and 64, but unq/64 − 2⁻¹¹ for unq 34..62 — a tiny off-spec
//   bias worth at most ±1/255 on ~3% of decoded 8-bit texels (≈0.001 dB).
//   The reference implements the spec rule; every block class round-trips
//   the hardware within that ±1/255 envelope, which is what validates the
//   three block-mode/CEM layouts above.

/** 16 pixels × 4 channels = 64 normalised floats, interleaved RGBA. */
export type ASTC4x4Pixels = Readonly<ArrayLike<number>>

/** 16 bytes of ASTC 4×4 block data. */
export type ASTC4x4Block = Uint8Array

// --- Constants --------------------------------------------------------------

/** Block modes: 4×4 grid, single plane, LDR. See header derivation. */
const BLOCK_MODE_4x4_2BIT = 0x042
const BLOCK_MODE_4x4_3BIT = 0x053
const BLOCK_MODE_4x4_4BIT = 0x242
const BLOCK_MODE_4x4_5BIT = 0x253

/** Color endpoint modes we emit (all "LDR, direct"). */
const CEM_LUM_DIRECT = 0
const CEM_RGB_DIRECT = 8
const CEM_RGBA_DIRECT = 12

/** Weight unquantisation tables (spec bit-replicate-then-bump rule). */
const WEIGHT_UNQ_4: readonly number[] = [0, 21, 43, 64]
const WEIGHT_UNQ_8: readonly number[] = [0, 9, 18, 27, 37, 46, 55, 64]
const WEIGHT_UNQ_16: readonly number[] = Array.from({ length: 16 }, (_, w) => {
  const u = (w << 2) | (w >> 2)
  return u > 32 ? u + 1 : u
})
const WEIGHT_UNQ_32: readonly number[] = Array.from({ length: 32 }, (_, w) => (w <= 15 ? 2 * w : 2 * w + 2))

/** Per-class packing description, keyed by CEM. */
interface BlockClass {
  cem: number
  blockMode: number
  /** Channels the endpoint stores (subset of RGBA indices). */
  channels: readonly number[]
  weightBits: number
  unq: readonly number[]
  /** Endpoint range: plain bytes (256) or trit-ISE QUANT_192. */
  endpointRange: 256 | 192
}

const CLASS_LUM: BlockClass = {
  cem: CEM_LUM_DIRECT,
  blockMode: BLOCK_MODE_4x4_5BIT,
  channels: [0],
  weightBits: 5,
  unq: WEIGHT_UNQ_32,
  endpointRange: 256,
}
const CLASS_RGB: BlockClass = {
  cem: CEM_RGB_DIRECT,
  blockMode: BLOCK_MODE_4x4_3BIT,
  channels: [0, 1, 2],
  weightBits: 3,
  unq: WEIGHT_UNQ_8,
  endpointRange: 256,
}
const CLASS_RGB_Q192: BlockClass = {
  cem: CEM_RGB_DIRECT,
  blockMode: BLOCK_MODE_4x4_4BIT,
  channels: [0, 1, 2],
  weightBits: 4,
  unq: WEIGHT_UNQ_16,
  endpointRange: 192,
}
const CLASS_RGBA: BlockClass = {
  cem: CEM_RGBA_DIRECT,
  blockMode: BLOCK_MODE_4x4_2BIT,
  channels: [0, 1, 2, 3],
  weightBits: 2,
  unq: WEIGHT_UNQ_4,
  endpointRange: 256,
}

// --- Scalar helpers ---------------------------------------------------------

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v
}

/** Normalised [0, 1] → clamped 8-bit. */
function to8(v: number): number {
  return clamp(Math.round(v * 255), 0, 255)
}

/**
 * Hardware-exact ASTC LDR interpolation (spec §decode). Given 8-bit
 * endpoints and a 0..64 weight, expands the endpoints to 16 bits by byte
 * replication (e·257) and interpolates there. Returns the raw 16-bit
 * result in [0, 65535]; divide by 65535 for the normalised value or by
 * 257 for the "8-bit scale" used in the encoder's error metric.
 */
function interp16(e0: number, e1: number, w: number): number {
  return ((64 - w) * e0 * 257 + w * e1 * 257 + 32) >> 6
}

// --- QUANT_192 endpoint range (trit ISE) ------------------------------------

/** QUANT_192 ISE value (trit·64 + 6 bits) → unquantised byte (spec rule). */
function unq192(value: number): number {
  const trit = value >> 6
  const bits = value & 63
  const A = bits & 1 ? 0x1ff : 0
  const f = (bits >> 5) & 1
  // B = f e d c b 0 0 0 f
  const B = (((bits >> 1) & 31) << 4) | f
  const T = ((trit * 5 + B) ^ A) & 0x1ff
  return ((A & 0x80) | (T >> 2)) & 0xff
}

/** unquantised byte → ISE value, −1 for the 64 unrepresentable bytes. */
const Q192_VALUE_OF: Int16Array = (() => {
  const t = new Int16Array(256).fill(-1)
  for (let v = 0; v < 192; v++) t[unq192(v)] = v
  return t
})()

/** Nearest representable QUANT_192 byte to x (ties go down). */
function nearest192(x: number): number {
  const v = clamp(Math.round(x), 0, 255)
  if (Q192_VALUE_OF[v]! >= 0) return v
  const lo = v - 1
  const hi = v + 1
  const loOk = lo >= 0 && Q192_VALUE_OF[lo]! >= 0
  const hiOk = hi <= 255 && Q192_VALUE_OF[hi]! >= 0
  if (loOk && hiOk) return x - lo <= hi - x ? lo : hi
  if (loOk) return x - lo <= 2 ? lo : v + 2
  return hiOk ? hi : v - 2
}

/** Spec trit-block decode: 8-bit T → five trits. */
function decodeTrits(T: number): [number, number, number, number, number] {
  const bit = (x: number, i: number): number => (x >> i) & 1
  let C: number
  let t3: number
  let t4: number
  if (((T >> 2) & 7) === 7) {
    C = (((T >> 5) & 7) << 2) | (T & 3)
    t4 = 2
    t3 = 2
  } else {
    C = T & 31
    if (((T >> 5) & 3) === 3) {
      t4 = 2
      t3 = bit(T, 7)
    } else {
      t4 = bit(T, 7)
      t3 = (T >> 5) & 3
    }
  }
  let t0: number
  let t1: number
  let t2: number
  if ((C & 3) === 3) {
    t2 = 2
    t1 = bit(C, 4)
    t0 = (bit(C, 3) << 1) | (bit(C, 2) & ~bit(C, 3) & 1)
  } else if (((C >> 2) & 3) === 3) {
    t2 = 2
    t1 = 2
    t0 = C & 3
  } else {
    t2 = bit(C, 4)
    t1 = (C >> 2) & 3
    t0 = (bit(C, 1) << 1) | (bit(C, 0) & ~bit(C, 1) & 1)
  }
  return [t0, t1, t2, t3, t4]
}

/** Five trits → 8-bit T: the inverse of `decodeTrits` (the GPU shaders
 *  carry the same branch structure as selects). */
function encodeTrits(t0: number, t1: number, t2: number, t3: number, t4: number): number {
  let C: number
  if (t2 === 2 && t1 === 2) C = 0b01100 | t0
  else if (t2 === 2) C = (t1 << 4) | (t0 << 2) | 3
  else C = (t2 << 4) | (t1 << 2) | t0
  if (t3 === 2 && t4 === 2) return ((C >> 2) << 5) | (7 << 2) | (C & 3)
  if (t4 === 2) return (t3 << 7) | (3 << 5) | C
  return (t4 << 7) | (t3 << 5) | C
}

// Bit positions of each trit-field slice inside a 5-value group: after
// value i come T bits [start, start + count).
const TRIT_SLICES: readonly (readonly [number, number])[] = [
  [0, 2],
  [2, 2],
  [4, 1],
  [5, 2],
  [7, 1],
]

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

// --- Endpoint line fit (C channels, L levels) -------------------------------

/**
 * Pick the pair of texels (out of 16) that maximise L2 distance over the
 * class's channels. Same rationale as BC7's `farthestPair`: this is immune
 * to the "channels vary in different directions along the data line → bbox
 * diagonal misses the data" failure mode of a per-channel min/max seed.
 * O(N²) = 120 comparisons, trivial cost.
 */
function farthestPair(vals: Float64Array, C: number): { i0: number; i1: number } {
  let best = -1
  let bi0 = 0,
    bi1 = 1
  for (let i = 0; i < 16; i++) {
    for (let j = i + 1; j < 16; j++) {
      let d = 0
      for (let c = 0; c < C; c++) {
        const t = vals[i * C + c]! - vals[j * C + c]!
        d += t * t
      }
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
 * Build the palette (levels × C) from 8-bit endpoints and an unq table,
 * in "8-bit scale" (decoded 16-bit value / 257) so the encoder's error
 * metric measures exactly what the hardware will reconstruct.
 */
function buildPalette(e0: readonly number[], e1: readonly number[], unq: readonly number[]): Float64Array {
  const C = e0.length
  const pal = new Float64Array(unq.length * C)
  for (let i = 0; i < unq.length; i++) {
    for (let c = 0; c < C; c++) {
      pal[i * C + c] = interp16(e0[c]!, e1[c]!, unq[i]!) / 257
    }
  }
  return pal
}

/**
 * Assign all 16 texels to nearest palette entries; return indices and
 * summed squared error in 8-bit decode space.
 */
function totalSqError(
  vals: Float64Array,
  C: number,
  e0: readonly number[],
  e1: readonly number[],
  unq: readonly number[],
): { indices: Uint8Array; err: number } {
  const pal = buildPalette(e0, e1, unq)
  const levels = unq.length
  const indices = new Uint8Array(16)
  let err = 0
  for (let k = 0; k < 16; k++) {
    let bestIdx = 0
    let bestD = Infinity
    for (let i = 0; i < levels; i++) {
      let d = 0
      for (let c = 0; c < C; c++) {
        const t = pal[i * C + c]! - vals[k * C + c]!
        d += t * t
      }
      if (d < bestD) {
        bestD = d
        bestIdx = i
      }
    }
    indices[k] = bestIdx
    err += bestD
  }
  return { indices, err }
}

/**
 * One-pass least-squares refit. Given current per-texel indices, find the
 * (e0, e1) pair that minimises Σ (palette[idx_k] − v_k)².
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
  vals: Float64Array,
  C: number,
  indices: Uint8Array,
  unq: readonly number[],
): { e0: number[]; e1: number[] } | null {
  let sAA = 0,
    sBB = 0,
    sAB = 0
  const sAV = new Float64Array(C)
  const sBV = new Float64Array(C)
  for (let k = 0; k < 16; k++) {
    const u = unq[indices[k]!]!
    const a = (64 - u) / 64
    const b = u / 64
    sAA += a * a
    sBB += b * b
    sAB += a * b
    for (let c = 0; c < C; c++) {
      sAV[c] += a * vals[k * C + c]!
      sBV[c] += b * vals[k * C + c]!
    }
  }
  const det = sAA * sBB - sAB * sAB
  if (Math.abs(det) < 1e-9) return null

  const e0: number[] = Array.from({ length: C }, () => 0)
  const e1: number[] = Array.from({ length: C }, () => 0)
  for (let c = 0; c < C; c++) {
    e0[c] = clamp(Math.round((sBB * sAV[c]! - sAB * sBV[c]!) / det), 0, 255)
    e1[c] = clamp(Math.round((sAA * sBV[c]! - sAB * sAV[c]!) / det), 0, 255)
  }
  return { e0, e1 }
}

// --- Encode -----------------------------------------------------------------

/** Quantise an endpoint to the class's range (identity for QUANT_256). */
function quantEndpoint(cls: BlockClass, e: readonly number[]): number[] {
  return cls.endpointRange === 192 ? e.map(nearest192) : e.map(v => clamp(Math.round(v), 0, 255))
}

/**
 * Steps 2–4 for one class: farthest-pair seed, nearest-palette indices,
 * one least-squares refit accepted only if the error drops. Endpoints are
 * quantised to the class's range before every error evaluation.
 */
function fitClass(
  cls: BlockClass,
  pixels8: Uint8Array,
): { e0: number[]; e1: number[]; indices: Uint8Array; err: number } {
  const C = cls.channels.length
  const vals = new Float64Array(16 * C)
  for (let k = 0; k < 16; k++) {
    for (let c = 0; c < C; c++) vals[k * C + c] = pixels8[k * 4 + cls.channels[c]!]!
  }
  const fp = farthestPair(vals, C)
  let e0 = quantEndpoint(
    cls,
    Array.from({ length: C }, (_, c) => vals[fp.i0 * C + c]!),
  )
  let e1 = quantEndpoint(
    cls,
    Array.from({ length: C }, (_, c) => vals[fp.i1 * C + c]!),
  )
  let { indices, err } = totalSqError(vals, C, e0, e1, cls.unq)
  const refit = refitEndpoints(vals, C, indices, cls.unq)
  if (refit) {
    const r0 = quantEndpoint(cls, refit.e0)
    const r1 = quantEndpoint(cls, refit.e1)
    const second = totalSqError(vals, C, r0, r1, cls.unq)
    if (second.err < err) {
      e0 = r0
      e1 = r1
      indices = second.indices
      err = second.err
    }
  }
  return { e0, e1, indices, err }
}

/**
 * Encode 16 RGBA pixels (64 floats in [0, 1]) as a single ASTC 4×4 LDR
 * block using the narrow subset described at the top of this file. The
 * block class (CEM 0 / 8 / 12) is picked from the 8-bit content: exact
 * grayscale + opaque → luminance, opaque → RGB, otherwise RGBA.
 *
 * Algorithm per class:
 *   1. Quantise input to 8-bit; classify (opaque colour tries both CEM 8
 *      bit budgets and keeps the lower-error one).
 *   2. Farthest-pair over the class's channels → initial (e0, e1),
 *      quantised to the class's endpoint range.
 *   3. Assign per-texel indices by nearest palette entry.
 *   4. One LSQ refit pass; accept only if total error strictly decreases.
 *   5. CEM 8/12: flip endpoints (and reflect weights) if
 *      sum(e0.rgb) > sum(e1.rgb) so the decoder skips blue contraction.
 *   6. Pack block mode, CEM, endpoints, weights into 128 bits.
 */
export function encodeASTC4x4Block(pixels: ASTC4x4Pixels): ASTC4x4Block {
  if (pixels.length !== 64) {
    throw new Error(`encodeASTC4x4Block: expected 64 values (16 RGBA), got ${pixels.length}`)
  }

  // Step 1: quantise + classify in the 8-bit domain (the GPU encoders use
  // the same rule, so CPU and GPU agree on every block's class).
  const pixels8 = new Uint8Array(64)
  for (let k = 0; k < 64; k++) pixels8[k] = to8(pixels[k]!)
  let gray = true
  let opaque = true
  for (let k = 0; k < 16; k++) {
    const r = pixels8[k * 4]!
    if (pixels8[k * 4 + 1] !== r || pixels8[k * 4 + 2] !== r) gray = false
    if (pixels8[k * 4 + 3] !== 255) opaque = false
  }
  // Opaque colour has two bit budgets for CEM 8 — encode both, keep the
  // lower error (ties keep exact 8-bit endpoints).
  const classes = opaque ? (gray ? [CLASS_LUM] : [CLASS_RGB, CLASS_RGB_Q192]) : [CLASS_RGBA]
  let best: { cls: BlockClass; e0: number[]; e1: number[]; indices: Uint8Array; err: number } | null = null
  for (const cls of classes) {
    const fit = fitClass(cls, pixels8)
    if (!best || fit.err < best.err) best = { cls, ...fit }
  }
  const { cls } = best!
  let { e0, e1, indices } = best!
  const C = cls.channels.length

  // Step 5: endpoint ordering. For CEM 8/12 this dodges the decoder's
  // blue-contraction branch (RGB sums); for CEM 0 it is purely a
  // normalisation (L0 ≤ L1, matching the GPU encoder's min/max seed).
  // Strict '>' so a tie (s0 == s1) doesn't cause a gratuitous swap.
  {
    const nSum = Math.min(C, 3)
    let s0 = 0
    let s1 = 0
    for (let c = 0; c < nSum; c++) {
      s0 += e0[c]!
      s1 += e1[c]!
    }
    if (s0 > s1) {
      const tmp = e0
      e0 = e1
      e1 = tmp
      const wmax = cls.unq.length - 1
      const inv = new Uint8Array(16)
      // Reflect weights: w' = wmax − w. The decoded palette is mirrored,
      // so the reconstructed colour is unchanged.
      for (let k = 0; k < 16; k++) inv[k] = wmax - indices[k]!
      indices = inv
    }
  }

  // Step 6.
  return packBlock(cls, e0, e1, indices)
}

function packBlock(cls: BlockClass, e0: readonly number[], e1: readonly number[], indices: Uint8Array): ASTC4x4Block {
  const bw = new BitWriter128()

  // Config header.
  bw.write(0, 11, cls.blockMode)
  bw.write(11, 2, 0) // partition_count − 1
  bw.write(13, 4, cls.cem)

  // Endpoints from bit 17, interleaved (v0, v1) per channel — matching the
  // decoder's (v0, v1) = channel 0 lo/hi, (v2, v3) = channel 1…
  const values: number[] = []
  for (let c = 0; c < e0.length; c++) values.push(e0[c]!, e1[c]!)
  if (cls.endpointRange === 192) {
    // Trit ISE: groups of five, each value's 6 bits followed by its slice of
    // the group's trit field.
    let pos = 17
    for (let g = 0; g < values.length; g += 5) {
      const grp = values.slice(g, g + 5).map(v => {
        const q = Q192_VALUE_OF[v]!
        if (q < 0) throw new Error(`packBlock: ${v} is not a QUANT_192 level`)
        return q
      })
      const t = [0, 0, 0, 0, 0]
      grp.forEach((q, i) => (t[i] = q >> 6))
      const T = encodeTrits(t[0]!, t[1]!, t[2]!, t[3]!, t[4]!)
      grp.forEach((q, i) => {
        bw.write(pos, 6, q & 63)
        pos += 6
        const [start, count] = TRIT_SLICES[i]!
        bw.write(pos, count, (T >> start) & ((1 << count) - 1))
        pos += count
      })
    }
  } else {
    values.forEach((v, i) => bw.write(17 + i * 8, 8, v))
  }

  // Weights: bit j (LSB = 0) of weight k at block bit (127 − nBits·k − j).
  // One 1-bit write per weight bit keeps the mapping obvious at the cost
  // of a few dozen calls — unmeasurable versus the encode cost.
  for (let k = 0; k < 16; k++) {
    const w = indices[k]!
    for (let j = 0; j < cls.weightBits; j++) {
      bw.write(127 - cls.weightBits * k - j, 1, (w >> j) & 1)
    }
  }

  return bw.toBytes()
}

// --- Decode -----------------------------------------------------------------

/** The (block mode, CEM) pairs we accept, keyed `mode:cem`. */
const CLASS_BY_MODE_CEM: Record<string, BlockClass> = {
  [`${BLOCK_MODE_4x4_2BIT}:${CEM_RGBA_DIRECT}`]: CLASS_RGBA,
  [`${BLOCK_MODE_4x4_3BIT}:${CEM_RGB_DIRECT}`]: CLASS_RGB,
  [`${BLOCK_MODE_4x4_4BIT}:${CEM_RGB_DIRECT}`]: CLASS_RGB_Q192,
  [`${BLOCK_MODE_4x4_5BIT}:${CEM_LUM_DIRECT}`]: CLASS_LUM,
}
const SUPPORTED_MODES = new Set([BLOCK_MODE_4x4_2BIT, BLOCK_MODE_4x4_3BIT, BLOCK_MODE_4x4_4BIT, BLOCK_MODE_4x4_5BIT])

/**
 * Decode an ASTC 4×4 block produced by this encoder (or by any other
 * encoder that respects our narrow subset: block modes 0x042/0x053/0x242/
 * 0x253, single partition, CEM 0/8/12 — QUANT_192 endpoints with 0x242,
 * 8-bit endpoints otherwise). Handles the
 * blue-contraction branch even though our encoder doesn't produce it, so
 * externally-supplied blocks round-trip predictably.
 *
 * The CEM is validated against the block mode's expected pairing (we only
 * ever emit the four fixed combinations above).
 *
 * Output: 16 RGBA pixels as 64 floats in [0, 1].
 */
export function decodeASTC4x4Block(block: ASTC4x4Block): Float32Array {
  if (block.length !== 16) {
    throw new Error(`decodeASTC4x4Block: expected 16 bytes, got ${block.length}`)
  }
  const br = new BitReader128(block)

  const mode = br.read(0, 11)
  if (!SUPPORTED_MODES.has(mode)) {
    throw new Error(`decodeASTC4x4Block: unsupported block mode 0x${mode.toString(16)}`)
  }
  const partCount = br.read(11, 2)
  if (partCount !== 0) {
    throw new Error(`decodeASTC4x4Block: multi-partition blocks not supported (count=${partCount + 1})`)
  }
  const cem = br.read(13, 4)
  const cls = CLASS_BY_MODE_CEM[`${mode}:${cem}`]
  if (!cls) {
    throw new Error(`decodeASTC4x4Block: unsupported CEM ${cem} with block mode 0x${mode.toString(16)}`)
  }

  // Endpoint values, unquantised.
  const nVals = cls.channels.length * 2
  const v: number[] = []
  if (cls.endpointRange === 192) {
    let pos = 17
    for (let g = 0; g < nVals; g += 5) {
      const n = Math.min(5, nVals - g)
      const bits: number[] = []
      let T = 0
      for (let i = 0; i < n; i++) {
        bits.push(br.read(pos, 6))
        pos += 6
        const [start, count] = TRIT_SLICES[i]!
        T |= br.read(pos, count) << start
        pos += count
      }
      const t = decodeTrits(T)
      for (let i = 0; i < n; i++) v.push(unq192((t[i]! << 6) | bits[i]!))
    }
  } else {
    for (let i = 0; i < nVals; i++) v.push(br.read(17 + i * 8, 8))
  }

  // Reconstruct RGBA endpoints per CEM.
  let e0: [number, number, number, number]
  let e1: [number, number, number, number]
  if (cls.cem === CEM_LUM_DIRECT) {
    e0 = [v[0]!, v[0]!, v[0]!, 255]
    e1 = [v[1]!, v[1]!, v[1]!, 255]
  } else {
    const a0 = cls.cem === CEM_RGBA_DIRECT ? v[6]! : 255
    const a1 = cls.cem === CEM_RGBA_DIRECT ? v[7]! : 255
    if (v[0]! + v[2]! + v[4]! <= v[1]! + v[3]! + v[5]!) {
      e0 = [v[0]!, v[2]!, v[4]!, a0]
      e1 = [v[1]!, v[3]!, v[5]!, a1]
    } else {
      // blue_contract(r, g, b) = ((r + b) >> 1, (g + b) >> 1, b). Alpha
      // tags along unchanged, but the endpoint swap is full (RGB + A).
      e0 = [(v[1]! + v[5]!) >> 1, (v[3]! + v[5]!) >> 1, v[5]!, a1]
      e1 = [(v[0]! + v[4]!) >> 1, (v[2]! + v[4]!) >> 1, v[4]!, a0]
    }
  }

  // Weights from the top of the block.
  const weights = new Uint8Array(16)
  for (let k = 0; k < 16; k++) {
    let w = 0
    for (let j = 0; j < cls.weightBits; j++) {
      w |= br.read(127 - cls.weightBits * k - j, 1) << j
    }
    weights[k] = w
  }

  const out = new Float32Array(64)
  for (let k = 0; k < 16; k++) {
    const u = cls.unq[weights[k]!]!
    const base = k * 4
    for (let c = 0; c < 4; c++) {
      out[base + c] = interp16(e0[c]!, e1[c]!, u) / 65535
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
  BLOCK_MODE_4x4_3BIT,
  BLOCK_MODE_4x4_4BIT,
  BLOCK_MODE_4x4_5BIT,
  CEM_LUM_DIRECT,
  CEM_RGB_DIRECT,
  CEM_RGBA_DIRECT,
  WEIGHT_UNQ_4,
  WEIGHT_UNQ_8,
  WEIGHT_UNQ_16,
  WEIGHT_UNQ_32,
  unq192,
  nearest192,
  encodeTrits,
  decodeTrits,
  BitWriter128,
  BitReader128,
  buildPalette,
  interp16,
  to8,
}
