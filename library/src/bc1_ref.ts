// BC1 (DXT1) reference encoder + decoder. CPU implementation.
//
// BC1 compresses a 4×4 RGB block into 8 bytes:
//   • 2 × RGB565 endpoints, color0 (bytes 0..1) and color1 (bytes 2..3)
//   • 16 × 2-bit indices, LSB-first, pixel 0 at bit 0           // bytes 4..7
//
// There are two decode modes, selected by the *numeric* ordering of the two
// 16-bit endpoints:
//
//   4-colour mode (color0 > color1) — opaque, used here for every block:
//     palette[0] = color0
//     palette[1] = color1
//     palette[2] = (2·color0 +   color1) / 3
//     palette[3] = (  color0 + 2·color1) / 3
//
//   3-colour mode (color0 <= color1) — 1-bit punch-through alpha:
//     palette[0] = color0
//     palette[1] = color1
//     palette[2] = (color0 + color1) / 2
//     palette[3] = transparent black
//
// The encoder only ever emits the 4-colour mode (we force color0 > color1):
// this library treats BC1 as an opaque RGB format with alpha forced to 1, so
// the 3-colour mode's punch-through rail buys nothing. The decoder handles
// both modes so it can round-trip blocks produced elsewhere.
//
// QUALITY LEVELS
//   'fast' (default): bounding-box endpoints, inset by ~half a 565 cell, then a
//     single least-squares endpoint refit. The WebGL2 fragment fallback runs
//     this algorithm; the WGSL compute shaders run a faster projection-based
//     variant of it (see bc1.wgsl / bc1_fast_f16.wgsl) with equivalent quality.
//   'high': the endpoints are seeded from the block's principal colour axis
//     (covariance power-iteration) instead of the bbox diagonal, then refined
//     by several least-squares passes. Mirrors the `QUALITY_HIGH` branch of
//     bc1.wgsl. Strictly ≥ 'fast' in quality (the seed that yields the lower
//     error wins), at the cost of the eigen-solve.
//
// Numeric precision: arithmetic happens in normalised [0, 1] f32-ish space so
// the reference tracks the WGSL port closely. As with BC4, real hardware does
// integer-domain interpolation that can differ by ~1 LSB per palette entry; we
// accept that delta in tests.

/** 48-value input: 16 RGB triplets, channel-interleaved, each in [0, 1]. */
export type BC1Pixels = Readonly<ArrayLike<number>>

/** Exactly 8 bytes of BC1 block data. */
export type BC1Block = Uint8Array

export type BC1Quality = 'fast' | 'high'

// 4-colour-mode interpolation weights. palette[j] = A[j]·c0 + B[j]·c1, where
// c0/c1 are the decoded endpoint colours (palette[0] = c0, palette[1] = c1).
const WA: readonly number[] = [1, 0, 2 / 3, 1 / 3]
const WB: readonly number[] = [0, 1, 1 / 3, 2 / 3]

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v
}

/** Round-to-nearest quantization of a normalised colour into a 16-bit RGB565 word. */
function to565(r: number, g: number, b: number): number {
  const ri = Math.max(0, Math.min(31, Math.floor(r * 31 + 0.5)))
  const gi = Math.max(0, Math.min(63, Math.floor(g * 63 + 0.5)))
  const bi = Math.max(0, Math.min(31, Math.floor(b * 31 + 0.5)))
  return (ri << 11) | (gi << 5) | bi
}

/**
 * Expand an RGB565 word to a normalised [0, 1] colour. The 5→8 / 6→8 bit
 * replication matches what a typical hardware BC1 decoder produces, so index
 * selection here agrees with the on-GPU result.
 */
function from565(c: number): [number, number, number] {
  const r = (c >> 11) & 31
  const g = (c >> 5) & 63
  const b = c & 31
  // 5/6-bit → 8-bit. (x·527 + 23) >> 6 == (x << 3) | (x >> 2), i.e. the exact
  // bit-replication a BC1 hardware decoder performs (white → 255, not 63). The
  // 6-bit form uses 259/33. floor keeps the result in [0, 255] and bit-exact;
  // the inputs are small integers and ÷64 is exact in float, so it's portable.
  const r8 = Math.floor((r * 527 + 23) / 64)
  const g8 = Math.floor((g * 259 + 33) / 64)
  const b8 = Math.floor((b * 527 + 23) / 64)
  return [r8 / 255, g8 / 255, b8 / 255]
}

/** Build the 4-entry 4-colour-mode palette (12 floats: rgb × 4) from two 565 endpoints. */
function buildPalette4(c0: number, c1: number): Float32Array {
  const p0 = from565(c0)
  const p1 = from565(c1)
  const pal = new Float32Array(12)
  for (let j = 0; j < 4; j++) {
    pal[j * 3] = WA[j]! * p0[0] + WB[j]! * p1[0]
    pal[j * 3 + 1] = WA[j]! * p0[1] + WB[j]! * p1[1]
    pal[j * 3 + 2] = WA[j]! * p0[2] + WB[j]! * p1[2]
  }
  return pal
}

/**
 * Assign every texel its nearest palette entry (full 4-entry L2 search over
 * RGB), writing indices into `outIdx` and returning the total squared error.
 */
function assignIndices(pixels: BC1Pixels, pal: Float32Array, outIdx: Uint8Array): number {
  let err = 0
  for (let k = 0; k < 16; k++) {
    const r = pixels[k * 3]!
    const g = pixels[k * 3 + 1]!
    const b = pixels[k * 3 + 2]!
    let bestJ = 0
    let bestD = Infinity
    for (let j = 0; j < 4; j++) {
      const dr = pal[j * 3]! - r
      const dg = pal[j * 3 + 1]! - g
      const db = pal[j * 3 + 2]! - b
      const d = dr * dr + dg * dg + db * db
      if (d < bestD) {
        bestD = d
        bestJ = j
      }
    }
    outIdx[k] = bestJ
    err += bestD
  }
  return err
}

/**
 * One least-squares refit pass. Given the current per-texel indices, solve the
 * 2×2 normal equations for the endpoint colours (e0, e1) that minimise
 * Σ‖A[i_k]·e0 + B[i_k]·e1 − c_k‖². The three RGB channels share the same scalar
 * sums (sAA/sBB/sAB), so it's a single 2×2 solve with vec3 right-hand sides.
 *
 * Returns the refined endpoints clamped to [0, 1], or null if the system is
 * degenerate (e.g. every texel landed on one endpoint).
 */
function refitEndpoints(pixels: BC1Pixels, indices: Uint8Array): { e0: number[]; e1: number[] } | null {
  let sAA = 0,
    sBB = 0,
    sAB = 0
  const sAV = [0, 0, 0]
  const sBV = [0, 0, 0]
  for (let k = 0; k < 16; k++) {
    const a = WA[indices[k]!]!
    const b = WB[indices[k]!]!
    sAA += a * a
    sBB += b * b
    sAB += a * b
    for (let c = 0; c < 3; c++) {
      const v = pixels[k * 3 + c]!
      sAV[c]! += a * v
      sBV[c]! += b * v
    }
  }
  const det = sAA * sBB - sAB * sAB
  if (Math.abs(det) < 1e-9) return null
  const e0: number[] = [0, 0, 0]
  const e1: number[] = [0, 0, 0]
  for (let c = 0; c < 3; c++) {
    e0[c] = clamp01((sBB * sAV[c]! - sAB * sBV[c]!) / det)
    e1[c] = clamp01((sAA * sBV[c]! - sAB * sAV[c]!) / det)
  }
  return { e0, e1 }
}

/**
 * Principal colour axis of the block via covariance power-iteration, seeded
 * with the bbox diagonal (already a good guess). Returns the unit axis, or null
 * for a degenerate (constant) block.
 */
function principalAxis(pixels: BC1Pixels, mean: number[], seed: number[]): number[] | null {
  // Symmetric 3×3 covariance: [rr rg rb; rg gg gb; rb gb bb].
  let rr = 0,
    rg = 0,
    rb = 0,
    gg = 0,
    gb = 0,
    bb = 0
  for (let k = 0; k < 16; k++) {
    const dr = pixels[k * 3]! - mean[0]!
    const dg = pixels[k * 3 + 1]! - mean[1]!
    const db = pixels[k * 3 + 2]! - mean[2]!
    rr += dr * dr
    rg += dr * dg
    rb += dr * db
    gg += dg * dg
    gb += dg * db
    bb += db * db
  }
  let vx = seed[0]!,
    vy = seed[1]!,
    vz = seed[2]!
  let len = Math.sqrt(vx * vx + vy * vy + vz * vz)
  if (len < 1e-9) return null
  vx /= len
  vy /= len
  vz /= len
  // A handful of iterations is plenty for a 3×3 system to converge.
  for (let iter = 0; iter < 8; iter++) {
    const nx = rr * vx + rg * vy + rb * vz
    const ny = rg * vx + gg * vy + gb * vz
    const nz = rb * vx + gb * vy + bb * vz
    len = Math.sqrt(nx * nx + ny * ny + nz * nz)
    if (len < 1e-12) return null
    vx = nx / len
    vy = ny / len
    vz = nz / len
  }
  return [vx, vy, vz]
}

/**
 * Pack two 565 endpoints + 16 × 2-bit indices into the 8-byte block.
 * Layout (little-endian): bytes 0..1 = color0, 2..3 = color1, 4..7 = indices
 * with pixel 0 at bit 0.
 */
function packBlock(c0: number, c1: number, indices: Uint8Array): BC1Block {
  const out = new Uint8Array(8)
  out[0] = c0 & 0xff
  out[1] = (c0 >> 8) & 0xff
  out[2] = c1 & 0xff
  out[3] = (c1 >> 8) & 0xff
  let bits = 0
  for (let k = 0; k < 16; k++) bits |= (indices[k]! & 3) << (k * 2)
  // `bits` is a full 32-bit field; write it out byte-wise (>>> keeps it unsigned).
  out[4] = bits & 0xff
  out[5] = (bits >>> 8) & 0xff
  out[6] = (bits >>> 16) & 0xff
  out[7] = (bits >>> 24) & 0xff
  return out
}

/**
 * Given two ideal endpoint colours (normalised), quantize to 565, force
 * 4-colour mode (color0 > color1), assign indices and refine with up to
 * `maxRefits` least-squares passes. Commits to `best` only on strict error
 * improvement. Returns the achieved squared error.
 */
function fitFromEndpoints(
  pixels: BC1Pixels,
  hi: number[],
  lo: number[],
  maxRefits: number,
  best: { c0: number; c1: number; indices: Uint8Array; err: number },
): void {
  let c0 = to565(hi[0]!, hi[1]!, hi[2]!)
  let c1 = to565(lo[0]!, lo[1]!, lo[2]!)
  // 4-colour mode requires color0 > color1.
  if (c0 === c1) {
    if (c1 > 0) c1 -= 1
    else c0 += 1
  } else if (c0 < c1) {
    const t = c0
    c0 = c1
    c1 = t
  }

  const idx = new Uint8Array(16)
  let pal = buildPalette4(c0, c1)
  let err = assignIndices(pixels, pal, idx)
  if (err < best.err) {
    best.c0 = c0
    best.c1 = c1
    best.indices.set(idx)
    best.err = err
  }

  for (let pass = 0; pass < maxRefits; pass++) {
    const refit = refitEndpoints(pixels, idx)
    if (!refit) break
    let nc0 = to565(refit.e0[0]!, refit.e0[1]!, refit.e0[2]!)
    let nc1 = to565(refit.e1[0]!, refit.e1[1]!, refit.e1[2]!)
    // A refit that flips or equalises the endpoints would change decode mode;
    // only accept candidates that stay in 4-colour mode.
    if (nc0 < nc1) {
      const t = nc0
      nc0 = nc1
      nc1 = t
    }
    if (nc0 === nc1) break
    if (nc0 === c0 && nc1 === c1) break // converged
    pal = buildPalette4(nc0, nc1)
    const nerr = assignIndices(pixels, pal, idx)
    c0 = nc0
    c1 = nc1
    err = nerr
    if (nerr < best.err) {
      best.c0 = nc0
      best.c1 = nc1
      best.indices.set(idx)
      best.err = nerr
    }
  }
}

/**
 * Encode a 4×4 RGB block (48 floats in [0, 1]) into an 8-byte BC1 block.
 * Always emits 4-colour (opaque) mode.
 */
export function encodeBC1Block(pixels: BC1Pixels, { quality = 'fast' }: { quality?: BC1Quality } = {}): BC1Block {
  if (pixels.length !== 48) {
    throw new Error(`encodeBC1Block: expected 48 values (16 RGB), got ${pixels.length}`)
  }

  // Bounding box + channel means in one pass.
  const bbMin = [1, 1, 1]
  const bbMax = [0, 0, 0]
  const mean = [0, 0, 0]
  for (let k = 0; k < 16; k++) {
    for (let c = 0; c < 3; c++) {
      const v = pixels[k * 3 + c]!
      if (v < bbMin[c]!) bbMin[c] = v
      if (v > bbMax[c]!) bbMax[c] = v
      mean[c]! += v
    }
  }
  for (let c = 0; c < 3; c++) mean[c]! /= 16

  const best = { c0: 0, c1: 0, indices: new Uint8Array(16), err: Infinity }

  // Inset the bbox by ~half a 565 cell (1/16) so the quantised palette covers
  // the real data range more tightly — the classic stb_dxt heuristic.
  const inset = [(bbMax[0]! - bbMin[0]!) / 16, (bbMax[1]! - bbMin[1]!) / 16, (bbMax[2]! - bbMin[2]!) / 16]
  const bboxHi = [clamp01(bbMax[0]! - inset[0]!), clamp01(bbMax[1]! - inset[1]!), clamp01(bbMax[2]! - inset[2]!)]
  const bboxLo = [clamp01(bbMin[0]! + inset[0]!), clamp01(bbMin[1]! + inset[1]!), clamp01(bbMin[2]! + inset[2]!)]

  if (quality === 'high') {
    // Seed from the principal colour axis: project all texels onto it and take
    // the extreme projections as the endpoint colours, then inset along the
    // axis. Falls back to the bbox seed for degenerate (constant) blocks.
    const diag = [bbMax[0]! - bbMin[0]!, bbMax[1]! - bbMin[1]!, bbMax[2]! - bbMin[2]!]
    const axis = principalAxis(pixels, mean, diag)
    if (axis) {
      let tMin = Infinity
      let tMax = -Infinity
      for (let k = 0; k < 16; k++) {
        const t =
          (pixels[k * 3]! - mean[0]!) * axis[0]! +
          (pixels[k * 3 + 1]! - mean[1]!) * axis[1]! +
          (pixels[k * 3 + 2]! - mean[2]!) * axis[2]!
        if (t < tMin) tMin = t
        if (t > tMax) tMax = t
      }
      const pad = (tMax - tMin) / 16
      const tHi = tMax - pad
      const tLo = tMin + pad
      const pcaHi = [
        clamp01(mean[0]! + tHi * axis[0]!),
        clamp01(mean[1]! + tHi * axis[1]!),
        clamp01(mean[2]! + tHi * axis[2]!),
      ]
      const pcaLo = [
        clamp01(mean[0]! + tLo * axis[0]!),
        clamp01(mean[1]! + tLo * axis[1]!),
        clamp01(mean[2]! + tLo * axis[2]!),
      ]
      fitFromEndpoints(pixels, pcaHi, pcaLo, 3, best)
    }
    // Also try the bbox seed; keep whichever endpoint family wins.
    fitFromEndpoints(pixels, bboxHi, bboxLo, 3, best)
  } else {
    fitFromEndpoints(pixels, bboxHi, bboxLo, 1, best)
  }

  return packBlock(best.c0, best.c1, best.indices)
}

/**
 * Decode an 8-byte BC1 block to 48 normalised [0, 1] values (16 RGB triplets).
 * Handles both 4-colour and 3-colour modes; the 3-colour transparent index
 * decodes to black (this library samples BC1 as opaque RGB).
 */
export function decodeBC1Block(block: BC1Block): Float32Array {
  if (block.length !== 8) {
    throw new Error(`decodeBC1Block: expected 8 bytes, got ${block.length}`)
  }
  const c0 = block[0]! | (block[1]! << 8)
  const c1 = block[2]! | (block[3]! << 8)
  const p0 = from565(c0)
  const p1 = from565(c1)

  // palette[2] / palette[3] depend on the mode (color0 > color1 ? 4 : 3).
  const pal: number[][] = [p0, p1, [0, 0, 0], [0, 0, 0]]
  if (c0 > c1) {
    for (let c = 0; c < 3; c++) {
      pal[2]![c] = (2 * p0[c]! + p1[c]!) / 3
      pal[3]![c] = (p0[c]! + 2 * p1[c]!) / 3
    }
  } else {
    for (let c = 0; c < 3; c++) {
      pal[2]![c] = (p0[c]! + p1[c]!) / 2
      pal[3]![c] = 0 // transparent → black
    }
  }

  const bits = (block[4]! | (block[5]! << 8) | (block[6]! << 16) | (block[7]! << 24)) >>> 0
  const out = new Float32Array(48)
  for (let k = 0; k < 16; k++) {
    const idx = (bits >>> (k * 2)) & 3
    out[k * 3] = pal[idx]![0]!
    out[k * 3 + 1] = pal[idx]![1]!
    out[k * 3 + 2] = pal[idx]![2]!
  }
  return out
}

// Exposed for tests so they can assert quantization / packing without
// re-deriving the bit layout.
export const _internal = {
  to565,
  from565,
  buildPalette4,
  packBlock,
  WA,
  WB,
}
