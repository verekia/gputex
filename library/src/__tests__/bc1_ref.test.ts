import { encodeBC1Block, decodeBC1Block, _internal, type BC1Pixels } from '../bc1_ref.js'

const { to565, buildPalette4, packBlock } = _internal

/** Nearest of the 4 palette entries for one RGB texel (full L2). */
function nearestIndex(pal: Float32Array, r: number, g: number, b: number): number {
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
  return bestJ
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v)

/** bbox + 1/16 inset → forced-4-colour 565 endpoints, shared by both baselines. */
function bboxEndpoints(pixels: BC1Pixels): [number, number] {
  const bbMin = [1, 1, 1]
  const bbMax = [0, 0, 0]
  for (let k = 0; k < 16; k++) {
    for (let c = 0; c < 3; c++) {
      const v = pixels[k * 3 + c]!
      if (v < bbMin[c]!) bbMin[c] = v
      if (v > bbMax[c]!) bbMax[c] = v
    }
  }
  const inset = [(bbMax[0]! - bbMin[0]!) / 16, (bbMax[1]! - bbMin[1]!) / 16, (bbMax[2]! - bbMin[2]!) / 16]
  const hi = [clamp01(bbMax[0]! - inset[0]!), clamp01(bbMax[1]! - inset[1]!), clamp01(bbMax[2]! - inset[2]!)]
  const lo = [clamp01(bbMin[0]! + inset[0]!), clamp01(bbMin[1]! + inset[1]!), clamp01(bbMin[2]! + inset[2]!)]
  let c0 = to565(hi[0]!, hi[1]!, hi[2]!)
  let c1 = to565(lo[0]!, lo[1]!, lo[2]!)
  if (c0 === c1) {
    if (c1 > 0) c1 -= 1
    else c0 += 1
  } else if (c0 < c1) {
    const t = c0
    c0 = c1
    c1 = t
  }
  return [c0, c1]
}

/** Build a 4×4 RGB block (48 floats) from a per-pixel builder. */
function makeBlock(fn: (k: number) => [number, number, number]): Float32Array {
  const out = new Float32Array(48)
  for (let k = 0; k < 16; k++) {
    const [r, g, b] = fn(k)
    out[k * 3] = r
    out[k * 3 + 1] = g
    out[k * 3 + 2] = b
  }
  return out
}

/** Sum of squared error between an original block and its decode. */
function blockSE(orig: BC1Pixels, decoded: Float32Array): number {
  let s = 0
  for (let i = 0; i < 48; i++) {
    const d = orig[i]! - decoded[i]!
    s += d * d
  }
  return s
}

/** PSNR in dB over a set of (original, decoded) pairs, MAX = 1.0. */
function psnr(totalSE: number, sampleCount: number): number {
  const mse = totalSE / sampleCount
  if (mse === 0) return Infinity
  return 10 * Math.log10(1 / mse)
}

// Deterministic LCG so the random blocks are reproducible.
function seededRand(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 0x1_0000_0000
  }
}

/**
 * Old buggy 5/6→8-bit expansion (÷256 instead of ÷64) used by the naïve
 * baseline so it reproduces the *exact* index choices the shipped shader made.
 * Decoding that block with the correct (hardware) decoder measures what users
 * actually saw — so the comparison captures the full end-to-end quality gain.
 */
function from565Buggy(c: number): [number, number, number] {
  const r = (c >> 11) & 31
  const g = (c >> 5) & 63
  const b = c & 31
  return [(r * 527 + 23) / 256 / 255, (g * 259 + 33) / 256 / 255, (b * 527 + 23) / 256 / 255]
}

function buildPalette4Buggy(c0: number, c1: number): Float32Array {
  const p0 = from565Buggy(c0)
  const p1 = from565Buggy(c1)
  const wa = [1, 0, 2 / 3, 1 / 3]
  const wb = [0, 1, 1 / 3, 2 / 3]
  const pal = new Float32Array(12)
  for (let j = 0; j < 4; j++) {
    pal[j * 3] = wa[j]! * p0[0] + wb[j]! * p1[0]
    pal[j * 3 + 1] = wa[j]! * p0[1] + wb[j]! * p1[1]
    pal[j * 3 + 2] = wa[j]! * p0[2] + wb[j]! * p1[2]
  }
  return pal
}

/**
 * The exact algorithm BC1 shipped before this change: bbox endpoints, no
 * least-squares refit, AND the old ÷256 palette bug for index selection.
 * Decoding its output with the correct decoder measures what users saw.
 */
function encodeNaive(pixels: BC1Pixels): Uint8Array {
  const [c0, c1] = bboxEndpoints(pixels)
  const pal = buildPalette4Buggy(c0, c1)
  const idx = new Uint8Array(16)
  for (let k = 0; k < 16; k++) {
    idx[k] = nearestIndex(pal, pixels[k * 3]!, pixels[k * 3 + 1]!, pixels[k * 3 + 2]!)
  }
  return packBlock(c0, c1, idx)
}

/**
 * Same bbox seed and *correct* palette as the new fast path but WITHOUT the
 * refit. The fast path is this plus a refit accepted only when it lowers error,
 * so fast ≤ this on every block — a clean per-block monotonicity check.
 */
function encodeNoRefit(pixels: BC1Pixels): Uint8Array {
  const [c0, c1] = bboxEndpoints(pixels)
  const pal = buildPalette4(c0, c1)
  const idx = new Uint8Array(16)
  for (let k = 0; k < 16; k++) {
    idx[k] = nearestIndex(pal, pixels[k * 3]!, pixels[k * 3 + 1]!, pixels[k * 3 + 2]!)
  }
  return packBlock(c0, c1, idx)
}

/** A representative spread of block types for quality measurement. */
function sampleBlocks(): Float32Array[] {
  const rand = seededRand(0xc0ffee)
  const blocks: Float32Array[] = []

  // Smooth horizontal + diagonal gradients (different orientations than the
  // bbox diagonal — where the principal axis helps most).
  for (let g = 0; g < 6; g++) {
    const a = [rand(), rand(), rand()] as const
    const b = [rand(), rand(), rand()] as const
    blocks.push(
      makeBlock(k => {
        const x = k & 3
        const y = k >> 2
        const t = (x + y) / 6
        return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
      }),
    )
  }

  // Two-tone blocks with a sprinkle of noise (typical of detailed textures).
  for (let n = 0; n < 8; n++) {
    const a = [rand(), rand(), rand()] as const
    const b = [rand(), rand(), rand()] as const
    blocks.push(
      makeBlock(() => {
        const pick = rand() < 0.5 ? a : b
        const j = 0.05
        return [
          Math.min(1, Math.max(0, pick[0] + (rand() - 0.5) * j)),
          Math.min(1, Math.max(0, pick[1] + (rand() - 0.5) * j)),
          Math.min(1, Math.max(0, pick[2] + (rand() - 0.5) * j)),
        ]
      }),
    )
  }

  // Fully random (worst case — little structure for the line to fit).
  for (let n = 0; n < 6; n++) {
    blocks.push(makeBlock(() => [rand(), rand(), rand()]))
  }

  return blocks
}

describe('BC1 block bitstream', () => {
  it('encodes to 8 bytes in forced 4-colour mode (color0 > color1)', () => {
    const block = encodeBC1Block(makeBlock(() => [0.2, 0.6, 0.9]))
    expect(block.byteLength).toBe(8)
    const c0 = block[0]! | (block[1]! << 8)
    const c1 = block[2]! | (block[3]! << 8)
    // Flat block: endpoints land on the same cell, then color0 is nudged up.
    expect(c0).toBeGreaterThan(c1)
  })

  it('round-trips a flat block to within one 565 cell', () => {
    const flat = makeBlock(() => [0.25, 0.5, 0.75])
    const decoded = decodeBC1Block(encodeBC1Block(flat))
    // 5-bit R/B → worst case ~1/32; allow a hair over for rounding.
    expect(blockSE(flat, decoded) / 48).toBeLessThan(0.001)
  })

  it('preserves a two-endpoint ramp better than 1/32 RMS', () => {
    const a: [number, number, number] = [0.1, 0.15, 0.2]
    const b: [number, number, number] = [0.85, 0.8, 0.9]
    const ramp = makeBlock(k => {
      const t = (k & 3) / 3 // 4 distinct levels along X — exactly the 4 palette slots
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
    })
    const decoded = decodeBC1Block(encodeBC1Block(ramp, { quality: 'high' }))
    expect(Math.sqrt(blockSE(ramp, decoded) / 48)).toBeLessThan(1 / 32)
  })

  it('throws on malformed input', () => {
    expect(() => encodeBC1Block(new Float32Array(45))).toThrow(/48/)
    expect(() => decodeBC1Block(new Uint8Array(7))).toThrow(/8/)
  })
})

describe('BC1 decoder modes', () => {
  it('decodes 3-colour mode (color0 <= color1): midpoint + transparent black', () => {
    // Hand-build a block with color0 < color1 so the decoder picks 3-colour mode.
    const cLo = to565(0.0, 0.0, 0.0) // 0
    const cHi = to565(1.0, 1.0, 1.0) // 0xFFFF
    const idx = new Uint8Array(16)
    idx[0] = 0 // color0 (≈ black)
    idx[1] = 1 // color1 (≈ white)
    idx[2] = 2 // midpoint (≈ 0.5)
    idx[3] = 3 // transparent → black
    const block = packBlock(cLo, cHi, idx) // color0(=0) <= color1 → 3-colour mode
    const out = decodeBC1Block(block)
    expect(out[2]! + out[1]! + out[0]!).toBeLessThan(0.05) // pixel 0 ≈ black
    expect(out[5]!).toBeGreaterThan(0.95) // pixel 1 ≈ white
    expect(Math.abs(out[6]! - 0.5)).toBeLessThan(0.05) // pixel 2 ≈ midpoint
    expect(out[9]! + out[10]! + out[11]!).toBe(0) // pixel 3 = black rail
  })
})

describe('BC1 encode quality', () => {
  const blocks = sampleBlocks()
  const N = blocks.length * 48

  it('the refit never loses to the same-seed no-refit encoder', () => {
    for (const block of blocks) {
      const noRefit = blockSE(block, decodeBC1Block(encodeNoRefit(block)))
      const fast = blockSE(block, decodeBC1Block(encodeBC1Block(block, { quality: 'fast' })))
      // Refit is gated on strict error improvement, so it can only help.
      expect(fast).toBeLessThanOrEqual(noRefit + 1e-9)
    }
  })

  it('the new fast path beats the old shipped (buggy) encoder by a wide PSNR margin', () => {
    let oldSE = 0
    let fastSE = 0
    for (const block of blocks) {
      oldSE += blockSE(block, decodeBC1Block(encodeNaive(block)))
      fastSE += blockSE(block, decodeBC1Block(encodeBC1Block(block, { quality: 'fast' })))
    }
    const oldPsnr = psnr(oldSE, N)
    const fastPsnr = psnr(fastSE, N)
    // The ÷256→÷64 palette fix alone is worth several dB; the refit adds more.
    expect(fastPsnr).toBeGreaterThan(oldPsnr + 1.0)
  })

  it('high (PCA + iterative refit) is at least as good as fast on every block', () => {
    for (const block of blocks) {
      const fast = blockSE(block, decodeBC1Block(encodeBC1Block(block, { quality: 'fast' })))
      const high = blockSE(block, decodeBC1Block(encodeBC1Block(block, { quality: 'high' })))
      expect(high).toBeLessThanOrEqual(fast + 1e-9)
    }
  })

  it('high beats fast in aggregate PSNR', () => {
    let fastSE = 0
    let highSE = 0
    for (const block of blocks) {
      fastSE += blockSE(block, decodeBC1Block(encodeBC1Block(block, { quality: 'fast' })))
      highSE += blockSE(block, decodeBC1Block(encodeBC1Block(block, { quality: 'high' })))
    }
    expect(psnr(highSE, N)).toBeGreaterThan(psnr(fastSE, N))
  })
})
