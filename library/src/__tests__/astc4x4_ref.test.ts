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

describe('ASTC 4×4 block header', () => {
  it('writes the 0x042 block mode in bits [10:0]', () => {
    // Any input — we're only checking the header bits. A flat block is
    // a convenient choice because its contents don't depend on tie-breaks.
    const pixels = makePixels(() => [0.5, 0.5, 0.5, 1.0])
    const block = encodeASTC4x4Block(pixels)
    expect(block.byteLength).toBe(16)
    const bits = bitsOf(block)
    // Low 11 bits must equal 0x042 (derivation in astc4x4_ref.ts header).
    expect(Number(bits & 0x7ffn)).toBe(_internal.BLOCK_MODE_4x4_2BIT)
    expect(_internal.BLOCK_MODE_4x4_2BIT).toBe(0x042)
  })

  it('writes partition_count − 1 = 0 at bits [12:11]', () => {
    const pixels = makePixels(() => [0.2, 0.6, 0.9, 1.0])
    const block = encodeASTC4x4Block(pixels)
    const partCount = Number((bitsOf(block) >> 11n) & 0x3n)
    expect(partCount).toBe(0)
  })

  it('writes CEM 12 at bits [16:13]', () => {
    const pixels = makePixels(() => [0.1, 0.2, 0.3, 0.4])
    const block = encodeASTC4x4Block(pixels)
    const cem = Number((bitsOf(block) >> 13n) & 0xfn)
    expect(cem).toBe(_internal.CEM_RGBA_DIRECT)
    expect(cem).toBe(12)
  })
})

describe('ASTC 4×4 endpoint ordering', () => {
  it('ensures sum(e0.rgb) ≤ sum(e1.rgb) to avoid blue contraction', () => {
    // Input whose natural farthest-pair seed would place the brighter
    // pixel first. If the encoder doesn't swap, the decoder would apply
    // blue contraction and round-trip error would spike.
    const pixels = makePixels(k => (k === 0 ? [1, 1, 1, 1] : [0.05, 0.05, 0.05, 1]))
    const block = encodeASTC4x4Block(pixels)
    const bits = bitsOf(block)
    const v = (idx: number): number => Number((bits >> BigInt(17 + idx * 8)) & 0xffn)
    const v0 = v(0),
      v2 = v(2),
      v4 = v(4) // R0, G0, B0
    const v1 = v(1),
      v3 = v(3),
      v5 = v(5) // R1, G1, B1
    expect(v0 + v2 + v4).toBeLessThanOrEqual(v1 + v3 + v5)
  })
})

describe('ASTC 4×4 encode/decode round-trip', () => {
  it('round-trips a flat opaque block with tiny error', () => {
    const pixels = makePixels(() => [0.5, 0.5, 0.5, 1.0])
    const block = encodeASTC4x4Block(pixels)
    const decoded = decodeASTC4x4Block(block)
    // Flat colour: encoder can pick any endpoint pair producing the
    // target as a palette entry. QUANT_256 endpoints ≈ exact.
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
    // Endpoints can be 0 and 255 exactly. Indices 0 and 3 map exactly
    // to them via the unquantisation table → zero reconstruction error
    // is achievable.
    expect(maxAbs(decoded, pixels)).toBeLessThan(1 / 255)
  })

  it('round-trips a 16-step RGB gradient with small error', () => {
    // Smooth variation along a 4D line. 4 palette entries for 16 targets
    // is a looser fit than BC7's 16:16, so the tolerance is wider.
    const pixels = makePixels(k => [k / 15, k / 15, 1 - k / 15, 1])
    const block = encodeASTC4x4Block(pixels)
    const decoded = decodeASTC4x4Block(block)
    expect(rmse(decoded, pixels)).toBeLessThan(0.09)
    expect(maxAbs(decoded, pixels)).toBeLessThan(0.17)
  })

  it('round-trips a block with non-trivial alpha variation', () => {
    const pixels = makePixels(k => [0.3, 0.7, 0.4, k / 15])
    const block = encodeASTC4x4Block(pixels)
    const decoded = decodeASTC4x4Block(block)
    // Alpha moves along the endpoint line; RGB roughly constant. 4 palette
    // entries across 16 alpha targets → ≈ 1/8 worst-case stride.
    expect(maxAbs(decoded, pixels)).toBeLessThan(0.17)
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
    // 4 palette entries over a 16-point smooth gradient → expected RMSE
    // of ~1/16 = 0.0625 for a perfect fit. Real fit + quantisation
    // produces slightly worse; budget is loose but still tight enough
    // to catch regressions.
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
  it('packs weight 0 at block bits [127:126] (LSB at 127)', () => {
    // Construct a block that forces weight[0] = 3 (top of the palette):
    // after endpoint-ordering, the encoder maps the brightest pixel to
    // idx 3. Pixel 0 is bright, all others dark → natural idx for pixel 0
    // lands in the high end of the palette.
    const pixels = makePixels(k => (k === 0 ? [1, 1, 1, 1] : [0, 0, 0, 1]))
    const block = encodeASTC4x4Block(pixels)
    const bits = bitsOf(block)
    const idx0_lsb = Number((bits >> 127n) & 1n)
    const idx0_msb = Number((bits >> 126n) & 1n)
    const idx0 = idx0_lsb | (idx0_msb << 1)
    // Pixel 0 sits at the bright end; after the encoder's ordering
    // ensures sum(e0) ≤ sum(e1), bright ↦ idx 3 (nearest e1).
    expect(idx0).toBe(3)
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

  it('throws when asked to decode a block with the wrong block mode', () => {
    const fake = new Uint8Array(16)
    // Zero block has mode 0x000, not 0x042.
    expect(() => decodeASTC4x4Block(fake)).toThrow(/block mode/i)
  })
})

describe('ASTC 4×4 interpolation primitive', () => {
  it('matches the hardware rule at endpoint and midpoint weights', () => {
    // Same integer formula as BC7 (see the file header comment).
    expect(_internal.interp8(13, 240, 0)).toBe(13)
    expect(_internal.interp8(13, 240, 64)).toBe(240)
    // Near-midpoint. ((64-32)*0 + 32*255 + 32) >> 6 = 8192 >> 6 = 128.
    expect(_internal.interp8(0, 255, 32)).toBe(128)
  })

  it('uses [0, 21, 43, 64] as the QUANT_4 unquantisation table', () => {
    // Spec-mandated bit-replicate-then-bump rule. Lock it in.
    expect(_internal.WEIGHT_UNQ_4).toEqual([0, 21, 43, 64])
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
