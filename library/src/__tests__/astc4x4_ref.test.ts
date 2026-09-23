import { encodeASTC4x4Block, decodeASTC4x4Block, _internal } from '../astc4x4_ref.js'

/** Build 16 RGBA pixels (64 floats) from a per-pixel builder. */
function makePixels(fn: (i: number) => [number, number, number, number]): Float32Array {
  const out = new Float32Array(64)
  for (let k = 0; k < 16; k++) {
    const [r, g, b, a] = fn(k)
    out[k * 4] = r
    out[k * 4 + 1] = g
    out[k * 4 + 2] = b
    out[k * 4 + 3] = a
  }
  return out
}

function maxAbs(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!))
  return m
}

function rmse(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!
    s += d * d
  }
  return Math.sqrt(s / a.length)
}

// Deterministic LCG — same generator shape as the BC7 tests.
function seededRand(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 0x1_0000_0000
  }
}

/** Pull the 128 bits out of a 16-byte block as a single BigInt for sub-field asserts. */
function bitsOf(block: Uint8Array): bigint {
  let b = 0n
  for (let i = 0; i < 16; i++) b |= BigInt(block[i]!) << BigInt(i * 8)
  return b
}

const blockMode = (block: Uint8Array): number => Number(bitsOf(block) & 0x7ffn)
const blockCem = (block: Uint8Array): number => Number((bitsOf(block) >> 13n) & 0xfn)

// Class-typical inputs: exact-gray opaque → CEM 0, coloured opaque → CEM 8,
// translucent → CEM 12.
const grayPixels = makePixels(k => [k / 15, k / 15, k / 15, 1])
const colorPixels = makePixels(k => [k / 15, 0.2, 1 - k / 15, 1])
const alphaPixels = makePixels(k => [0.3, 0.7, 0.4, k / 15])

describe('ASTC 4×4 block class selection', () => {
  it('encodes exact-grayscale opaque blocks as CEM 0 with 5-bit weights (mode 0x253)', () => {
    const block = encodeASTC4x4Block(grayPixels)
    expect(blockMode(block)).toBe(_internal.BLOCK_MODE_4x4_5BIT)
    expect(blockMode(block)).toBe(0x253)
    expect(blockCem(block)).toBe(_internal.CEM_LUM_DIRECT)
  })

  it('encodes wide-span coloured opaque blocks as CEM 8 with 4-bit weights + QUANT_192 (mode 0x242)', () => {
    const block = encodeASTC4x4Block(colorPixels)
    expect(blockMode(block)).toBe(_internal.BLOCK_MODE_4x4_4BIT)
    expect(blockMode(block)).toBe(0x242)
    expect(blockCem(block)).toBe(_internal.CEM_RGB_DIRECT)
  })

  it('encodes small-span coloured opaque blocks as CEM 8 with 3-bit weights + 8-bit endpoints (mode 0x053)', () => {
    // A 4-level-wide ramp: exact endpoints beat finer weights here.
    const block = encodeASTC4x4Block(makePixels(k => [(100 + (k & 3)) / 255, 150 / 255, (50 + (k >> 2)) / 255, 1]))
    expect(blockMode(block)).toBe(_internal.BLOCK_MODE_4x4_3BIT)
    expect(blockMode(block)).toBe(0x053)
    expect(blockCem(block)).toBe(_internal.CEM_RGB_DIRECT)
  })

  it('encodes translucent blocks as CEM 12 with 2-bit weights (mode 0x042)', () => {
    const block = encodeASTC4x4Block(alphaPixels)
    expect(blockMode(block)).toBe(_internal.BLOCK_MODE_4x4_2BIT)
    expect(blockMode(block)).toBe(0x042)
    expect(blockCem(block)).toBe(_internal.CEM_RGBA_DIRECT)
  })

  it('a single ±1/255 chroma deviation demotes gray → RGB', () => {
    const nearGray = makePixels(k =>
      k === 5 ? [128 / 255, 130 / 255, 128 / 255, 1] : [128 / 255, 128 / 255, 128 / 255, 1],
    )
    expect(blockCem(encodeASTC4x4Block(nearGray))).toBe(_internal.CEM_RGB_DIRECT)
  })

  it('a single non-255 alpha texel demotes to RGBA', () => {
    const nearOpaque = makePixels(k => [k / 15, k / 15, k / 15, k === 5 ? 254 / 255 : 1])
    expect(blockCem(encodeASTC4x4Block(nearOpaque))).toBe(_internal.CEM_RGBA_DIRECT)
  })

  it('writes partition_count − 1 = 0 at bits [12:11] for every class', () => {
    for (const pixels of [grayPixels, colorPixels, alphaPixels]) {
      const partCount = Number((bitsOf(encodeASTC4x4Block(pixels)) >> 11n) & 0x3n)
      expect(partCount).toBe(0)
    }
  })
})

describe('ASTC 4×4 endpoint ordering', () => {
  it('ensures sum(e0.rgb) ≤ sum(e1.rgb) to avoid blue contraction (CEM 8)', () => {
    // Input whose natural farthest-pair seed would place the brighter
    // pixel first. If the encoder doesn't swap, the decoder would apply
    // blue contraction and round-trip error would spike.
    const pixels = makePixels(k => (k === 0 ? [1, 0.9, 1, 1] : [0.05, 0.1, 0.05, 1]))
    const block = encodeASTC4x4Block(pixels)
    expect(blockCem(block)).toBe(_internal.CEM_RGB_DIRECT)
    const bits = bitsOf(block)
    const v = (idx: number): number => Number((bits >> BigInt(17 + idx * 8)) & 0xffn)
    expect(v(0) + v(2) + v(4)).toBeLessThanOrEqual(v(1) + v(3) + v(5))
  })

  it('ensures sum(e0.rgb) ≤ sum(e1.rgb) to avoid blue contraction (CEM 12)', () => {
    const pixels = makePixels(k => (k === 0 ? [1, 0.9, 1, 0.5] : [0.05, 0.1, 0.05, 0.5]))
    const block = encodeASTC4x4Block(pixels)
    expect(blockCem(block)).toBe(_internal.CEM_RGBA_DIRECT)
    const bits = bitsOf(block)
    const v = (idx: number): number => Number((bits >> BigInt(17 + idx * 8)) & 0xffn)
    expect(v(0) + v(2) + v(4)).toBeLessThanOrEqual(v(1) + v(3) + v(5))
  })
})

describe('ASTC 4×4 encode/decode round-trip', () => {
  it('round-trips a flat opaque gray block with tiny error (CEM 0)', () => {
    const pixels = makePixels(() => [0.5, 0.5, 0.5, 1.0])
    const block = encodeASTC4x4Block(pixels)
    const decoded = decodeASTC4x4Block(block)
    expect(maxAbs(decoded, pixels)).toBeLessThan(2 / 255)
  })

  it('round-trips a flat opaque colour block with tiny error (CEM 8)', () => {
    const pixels = makePixels(() => [0.3, 0.55, 0.8, 1.0])
    const block = encodeASTC4x4Block(pixels)
    const decoded = decodeASTC4x4Block(block)
    expect(maxAbs(decoded, pixels)).toBeLessThan(2 / 255)
  })

  it('round-trips the all-zero block', () => {
    const pixels = new Float32Array(64)
    const block = encodeASTC4x4Block(pixels)
    const decoded = decodeASTC4x4Block(block)
    expect(maxAbs(decoded, pixels)).toBeLessThan(1 / 255)
  })

  it('round-trips the all-one opaque white block', () => {
    const pixels = new Float32Array(64).fill(1)
    const block = encodeASTC4x4Block(pixels)
    const decoded = decodeASTC4x4Block(block)
    expect(maxAbs(decoded, pixels)).toBeLessThan(1 / 255)
  })

  it('round-trips a bimodal (half black, half white) block exactly', () => {
    const pixels = makePixels(k => (k < 8 ? [0, 0, 0, 1] : [1, 1, 1, 1]))
    const block = encodeASTC4x4Block(pixels)
    const decoded = decodeASTC4x4Block(block)
    // Endpoints can be 0 and 255 exactly; the extreme weights unquantise
    // to 0 and 64 → zero reconstruction error is achievable.
    expect(maxAbs(decoded, pixels)).toBeLessThan(1 / 255)
  })

  it('round-trips a 16-step opaque grayscale ramp near-losslessly (CEM 0, 32 levels)', () => {
    const block = encodeASTC4x4Block(grayPixels)
    const decoded = decodeASTC4x4Block(block)
    // 32 weight levels across 16 distinct targets: every texel should land
    // within half a 5-bit weight step (a step is 2/64 of the full range,
    // i.e. ~8/255) of its exact value.
    expect(maxAbs(decoded, grayPixels)).toBeLessThan(5 / 255)
    // Decoded gray must stay exactly gray with full alpha.
    for (let k = 0; k < 16; k++) {
      expect(decoded[k * 4]).toBe(decoded[k * 4 + 1]!)
      expect(decoded[k * 4]).toBe(decoded[k * 4 + 2]!)
      expect(decoded[k * 4 + 3]).toBe(1)
    }
  })

  it('round-trips a 16-step RGB gradient with small error (CEM 8, 16 levels)', () => {
    const block = encodeASTC4x4Block(colorPixels)
    const decoded = decodeASTC4x4Block(block)
    // 16 palette entries for 16 targets; the residual is the QUANT_192
    // endpoint rounding plus the non-uniform QUANT_16 weight grid.
    expect(rmse(decoded, colorPixels)).toBeLessThan(0.01)
    expect(maxAbs(decoded, colorPixels)).toBeLessThan(0.025)
  })

  it('round-trips random wide-span opaque colour blocks through both CEM 8 budgets', () => {
    const rand = seededRand(0xc0ffee)
    const modes = new Set<number>()
    for (let trial = 0; trial < 200; trial++) {
      const span = trial % 2 ? 0.02 : 0.6
      const base = [rand() * (1 - span), rand() * (1 - span), rand() * (1 - span)]
      const pixels = makePixels(() => [base[0]! + rand() * span, base[1]! + rand() * span, base[2]! + rand() * span, 1])
      const block = encodeASTC4x4Block(pixels)
      modes.add(blockMode(block))
      const decoded = decodeASTC4x4Block(block)
      for (let k = 0; k < 16; k++) expect(decoded[k * 4 + 3]).toBe(1)
      expect(rmse(decoded, pixels)).toBeLessThan(span * 0.5)
    }
    expect(modes).toEqual(new Set([0x053, 0x242]))
  })

  it('round-trips a block with non-trivial alpha variation (CEM 12)', () => {
    const block = encodeASTC4x4Block(alphaPixels)
    const decoded = decodeASTC4x4Block(block)
    // Alpha moves along the endpoint line; RGB roughly constant. 4 palette
    // entries across 16 alpha targets → ≈ 1/8 worst-case stride.
    expect(maxAbs(decoded, alphaPixels)).toBeLessThan(0.17)
  })

  it('keeps smooth-random error in range over a battery of blocks', () => {
    const rand = seededRand(0xa57c)
    let worst = 0
    for (let trial = 0; trial < 30; trial++) {
      const e0 = [rand(), rand(), rand(), rand()] as const
      const e1 = [rand(), rand(), rand(), rand()] as const
      const pixels = makePixels(k => {
        const t = k / 15 + (rand() - 0.5) * 0.05
        return [
          e0[0] + t * (e1[0] - e0[0]),
          e0[1] + t * (e1[1] - e0[1]),
          e0[2] + t * (e1[2] - e0[2]),
          e0[3] + t * (e1[3] - e0[3]),
        ]
      })
      const block = encodeASTC4x4Block(pixels)
      const decoded = decodeASTC4x4Block(block)
      worst = Math.max(worst, rmse(decoded, pixels))
    }
    // Random alpha ⇒ these stay in the 2-bit RGBA class: 4 palette entries
    // over a 16-point smooth gradient → expected RMSE of ~1/16 = 0.0625 for
    // a perfect fit. Budget is loose but catches regressions.
    expect(worst).toBeLessThan(0.09)
  })
})

describe('ASTC 4×4 determinism', () => {
  it('same input yields identical 16 bytes', () => {
    const pixels = makePixels(k => [0.1 + k * 0.05, 0.2, 0.9 - k * 0.03, 1])
    const a = encodeASTC4x4Block(pixels)
    const b = encodeASTC4x4Block(pixels)
    expect([...a]).toEqual([...b])
  })
})

describe('ASTC 4×4 weight layout', () => {
  // In every class, pixel 0 is bright and the rest dark → after endpoint
  // ordering pixel 0 maps to the top palette entry, whose weight is
  // all-ones. Weight 0's bits sit at the very top of the block, LSB at
  // bit 127, so an all-ones weight pins bits [127 : 128−nBits].
  it('packs weight 0 of a 2-bit block at bits [127:126] (LSB at 127)', () => {
    const pixels = makePixels(k => (k === 0 ? [1, 0.8, 0.9, 0.5] : [0, 0.1, 0, 0.5]))
    const block = encodeASTC4x4Block(pixels)
    expect(blockCem(block)).toBe(_internal.CEM_RGBA_DIRECT)
    expect(Number((bitsOf(block) >> 126n) & 0x3n)).toBe(0x3)
  })

  it('packs weight 0 of a 3-bit block at bits [127:125] (LSB at 127)', () => {
    const pixels = makePixels(k => (k === 0 ? [1, 0.8, 0.9, 1] : [0, 0.1, 0, 1]))
    const block = encodeASTC4x4Block(pixels)
    expect(blockCem(block)).toBe(_internal.CEM_RGB_DIRECT)
    expect(Number((bitsOf(block) >> 125n) & 0x7n)).toBe(0x7)
  })

  it('packs weight 0 of a 5-bit block at bits [127:123] (LSB at 127)', () => {
    const pixels = makePixels(k => (k === 0 ? [1, 1, 1, 1] : [0, 0, 0, 1]))
    const block = encodeASTC4x4Block(pixels)
    expect(blockCem(block)).toBe(_internal.CEM_LUM_DIRECT)
    expect(Number((bitsOf(block) >> 123n) & 0x1fn)).toBe(0x1f)
  })
})

describe('ASTC 4×4 input validation', () => {
  it('throws on non-64-length pixel arrays', () => {
    expect(() => encodeASTC4x4Block(new Float32Array(63))).toThrow(/64/)
    expect(() => encodeASTC4x4Block(new Float32Array(65))).toThrow(/64/)
  })

  it('throws on non-16-length block arrays', () => {
    expect(() => decodeASTC4x4Block(new Uint8Array(15))).toThrow(/16/)
    expect(() => decodeASTC4x4Block(new Uint8Array(17))).toThrow(/16/)
  })

  it('throws when asked to decode a block with an unsupported block mode', () => {
    const fake = new Uint8Array(16)
    // Zero block has mode 0x000, which we never emit.
    expect(() => decodeASTC4x4Block(fake)).toThrow(/block mode/i)
  })

  it('throws when the CEM does not match the block mode pairing', () => {
    // A valid 0x253 (luminance) block with its CEM field rewritten to 12.
    const block = encodeASTC4x4Block(grayPixels)
    block[1] = (block[1]! & ~(0xf << 5)) | ((12 & 0x7) << 5)
    block[2] = (block[2]! & ~1) | (12 >> 3)
    expect(() => decodeASTC4x4Block(block)).toThrow(/CEM/)
  })
})

describe('ASTC 4×4 QUANT_192 endpoint range', () => {
  it('trit blocks round-trip for all 243 trit tuples', () => {
    for (let i = 0; i < 243; i++) {
      const t = [
        i % 3,
        Math.floor(i / 3) % 3,
        Math.floor(i / 9) % 3,
        Math.floor(i / 27) % 3,
        Math.floor(i / 81) % 3,
      ] as const
      const T = _internal.encodeTrits(t[0], t[1], t[2], t[3], t[4])
      expect(T).toBeGreaterThanOrEqual(0)
      expect(T).toBeLessThan(256)
      expect(_internal.decodeTrits(T)).toEqual([...t])
    }
  })

  it('unquantises to 192 distinct levels: u ≤ 127 with u mod 4 ≠ 3, and their mirrors 255 − u', () => {
    const levels = new Set<number>()
    for (let v = 0; v < 192; v++) levels.add(_internal.unq192(v))
    expect(levels.size).toBe(192)
    for (let x = 0; x < 256; x++) {
      const u = x <= 127 ? x : 255 - x
      expect(levels.has(x)).toBe(u % 4 !== 3)
    }
  })

  it('nearest192 returns the nearest representable level', () => {
    const levels = Array.from({ length: 192 }, (_, v) => _internal.unq192(v))
    for (let i = 0; i <= 2550; i++) {
      const x = i / 10
      const got = _internal.nearest192(x)
      const bestDist = Math.min(...levels.map(l => Math.abs(l - x)))
      expect(levels).toContain(got)
      expect(Math.abs(got - x)).toBeCloseTo(bestDist, 9)
    }
  })

  it('uses the spec 4-bit weight unquantisation (bit replication + bump above 32)', () => {
    expect(_internal.WEIGHT_UNQ_16).toEqual([0, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64])
  })

  it('keeps sum(e0.rgb) ≤ sum(e1.rgb) on unquantised levels in mode 0x242', () => {
    // Bright first pixel, dark rest; a wide span selects the QUANT_192
    // budget. A wrong ordering would trigger blue contraction on decode.
    // Colinear: pixel 0 bright, the rest a dark ramp along the same line.
    const pixels = makePixels(k => {
      const t = k === 0 ? 1 : (k - 1) / 30
      return [0.05 + 0.9 * t, 0.1 + 0.8 * t, 0.05 + 0.95 * t, 1]
    })
    const block = encodeASTC4x4Block(pixels)
    expect(blockMode(block)).toBe(0x242)
    const decoded = decodeASTC4x4Block(block)
    expect(maxAbs(decoded, pixels)).toBeLessThan(0.04)
  })
})

describe('ASTC 4×4 interpolation primitive', () => {
  it('matches the spec 16-bit rule at endpoint and midpoint weights', () => {
    // Endpoints replicate to 16 bits (×257); result is a raw unorm16.
    expect(_internal.interp16(13, 240, 0)).toBe(13 * 257)
    expect(_internal.interp16(13, 240, 64)).toBe(240 * 257)
    // Midpoint of full range: (32·65535 + 32) >> 6 = 32768 (0.50000763…).
    expect(_internal.interp16(0, 255, 32)).toBe(32768)
  })

  it('uses the spec bit-replicate-then-bump unquantisation tables', () => {
    expect(_internal.WEIGHT_UNQ_4).toEqual([0, 21, 43, 64])
    expect(_internal.WEIGHT_UNQ_8).toEqual([0, 9, 18, 27, 37, 46, 55, 64])
    // 5-bit: (w << 1) | (w >> 4), then +1 above 32 → 2w below the middle,
    // 2w + 2 above. Symmetric: unq(w) + unq(31 − w) = 64.
    expect(_internal.WEIGHT_UNQ_32.length).toBe(32)
    expect(_internal.WEIGHT_UNQ_32[0]).toBe(0)
    expect(_internal.WEIGHT_UNQ_32[15]).toBe(30)
    expect(_internal.WEIGHT_UNQ_32[16]).toBe(34)
    expect(_internal.WEIGHT_UNQ_32[31]).toBe(64)
    for (let w = 0; w < 32; w++) {
      expect(_internal.WEIGHT_UNQ_32[w]! + _internal.WEIGHT_UNQ_32[31 - w]!).toBe(64)
    }
  })
})

describe('ASTC 4×4 bit writer round-trip', () => {
  it('writes and reads values back at arbitrary positions', () => {
    const bw = new _internal.BitWriter128()
    bw.write(0, 11, 0x042) // block mode
    bw.write(11, 2, 0) // partition count
    bw.write(13, 4, 12) // CEM
    bw.write(126, 2, 0b10) // near the top (weight MSB at 127)
    bw.write(17, 8, 0xab) // endpoint byte
    const bytes = bw.toBytes()
    const br = new _internal.BitReader128(bytes)
    expect(br.read(0, 11)).toBe(0x042)
    expect(br.read(11, 2)).toBe(0)
    expect(br.read(13, 4)).toBe(12)
    expect(br.read(17, 8)).toBe(0xab)
    expect(br.read(126, 2)).toBe(0b10)
  })

  it('refuses out-of-range writes', () => {
    const bw = new _internal.BitWriter128()
    expect(() => bw.write(127, 2, 0)).toThrow(/out-of-range/)
    expect(() => bw.write(-1, 1, 0)).toThrow(/out-of-range/)
  })
})
