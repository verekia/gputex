import { _internal, decodeEACR11Block, encodeEACR11Block } from '../eac_r11_ref.js'

const { MODIFIERS, CANDIDATES, decode11, spanBucket } = _internal

// Deterministic PRNG so failures reproduce.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Squared error in 8-bit units between a block's decode and its source. */
function sse8(values: ArrayLike<number>, block: Uint8Array): number {
  const dec = decodeEACR11Block(block)
  let e = 0
  for (let k = 0; k < 16; k++) e += ((dec[k]! - values[k]!) * 255) ** 2
  return e
}

const grid8 = (f: (k: number) => number): number[] => Array.from({ length: 16 }, (_, k) => f(k) / 255)

describe('EAC R11 tables', () => {
  it('every modifier table is symmetric about −½', () => {
    expect(MODIFIERS.length).toBe(16)
    for (const t of MODIFIERS) {
      expect(t.length).toBe(8)
      for (let i = 0; i < 4; i++) expect(t[i]).toBe(-(t[4 + i]! + 1))
      // Magnitudes grow with the index on each side.
      for (let i = 5; i < 8; i++) expect(t[i]!).toBeGreaterThan(t[i - 1]!)
    }
  })

  it('the candidate table holds three valid picks per span bucket', () => {
    expect(CANDIDATES.length).toBe(41)
    for (const entry of CANDIDATES) {
      expect(entry).toBeLessThan(1 << 18)
      for (let k = 0; k < 3; k++) expect(((entry >> (6 * k)) & 63) >> 4).toBeLessThan(3)
    }
  })

  it('multiplier-0 candidates only appear where spans are a few levels', () => {
    CANDIDATES.forEach((entry, bucket) => {
      for (let k = 0; k < 3; k++) {
        if (((entry >> (6 * k)) & 63) >> 4 === 2) expect(bucket).toBeLessThanOrEqual(5)
      }
    })
  })

  it('span buckets cover 0..255 → 0..40, one per span below 32, 40 = full range', () => {
    const seen = new Set<number>()
    for (let s = 0; s < 256; s++) {
      const b = spanBucket(s)
      expect(b).toBeGreaterThanOrEqual(0)
      expect(b).toBeLessThan(41)
      if (s < 32) expect(b).toBe(s)
      seen.add(b)
    }
    expect(seen.size).toBe(41)
    expect(spanBucket(255)).toBe(40)
  })
})

describe('EAC R11 decoder', () => {
  it('rejects blocks that are not 8 bytes', () => {
    expect(() => decodeEACR11Block(new Uint8Array(7))).toThrow()
  })

  it('decodes base·8 + 4 + modifier·multiplier·8, clamped to 11 bits', () => {
    // base 100, multiplier 2, table 0 → modifiers ×16.
    expect(decode11(100, 2, 0, 4)).toBe(100 * 8 + 4 + 2 * 16)
    expect(decode11(100, 2, 0, 3)).toBe(100 * 8 + 4 - 15 * 16)
    expect(decode11(0, 15, 0, 3)).toBe(0)
    expect(decode11(255, 15, 0, 7)).toBe(2047)
  })

  it('multiplier 0 applies the modifier directly (1/8 steps)', () => {
    expect(decode11(100, 0, 13, 4)).toBe(804)
    expect(decode11(100, 0, 13, 7)).toBe(813)
    expect(decode11(100, 0, 13, 0)).toBe(803)
  })

  it('reads indices big-endian, pixel 0 on top, column-major', () => {
    // base 128, multiplier 1, table 13 (0, 1, 2, 9 / −1, −2, −3, −10).
    const block = new Uint8Array([128, (1 << 4) | 13, 0, 0, 0, 0, 0, 0])
    // Wire pixel 0 = (x 0, y 0) gets index 7; wire pixel 1 = (x 0, y 1) index 5.
    block[2] = (7 << 5) | (5 << 2)
    const dec = decodeEACR11Block(block)
    const at = (x: number, y: number) => Math.round(dec[y * 4 + x]! * 2047)
    expect(at(0, 0)).toBe(128 * 8 + 4 + 9 * 8)
    expect(at(0, 1)).toBe(128 * 8 + 4 + 1 * 8)
    // Everything else is index 0 → modifier −1.
    expect(at(1, 0)).toBe(128 * 8 + 4 - 8)
    expect(at(3, 3)).toBe(128 * 8 + 4 - 8)
  })
})

describe('EAC R11 encoder', () => {
  it('rejects inputs that are not 16 values', () => {
    expect(() => encodeEACR11Block(new Float32Array(15))).toThrow()
  })

  it('emits 8 bytes', () => {
    expect(encodeEACR11Block(new Float32Array(16)).length).toBe(8)
  })

  it('reproduces pure black and pure white exactly', () => {
    for (const v of [0, 1]) {
      const dec = decodeEACR11Block(encodeEACR11Block(new Float32Array(16).fill(v)))
      for (let k = 0; k < 16; k++) expect(dec[k]).toBe(v)
    }
  })

  it('keeps black and white exact in a two-level mask block', () => {
    const values = grid8(k => (k % 4 < 2 ? 0 : 255))
    for (const quality of ['fast', 'high'] as const) {
      const dec = decodeEACR11Block(encodeEACR11Block(values, { quality }))
      for (let k = 0; k < 16; k++) expect(dec[k]).toBe(values[k]!)
    }
  })

  it('encodes every flat 8-bit level to within 1/16 of a level', () => {
    for (let v = 0; v < 256; v++) {
      const values = grid8(() => v)
      const dec = decodeEACR11Block(encodeEACR11Block(values))
      for (let k = 0; k < 16; k++) expect(Math.abs(dec[k]! * 255 - v)).toBeLessThanOrEqual(0.0625 + 1e-6)
    }
  })

  it('keeps small-span blocks within a quarter of a level per texel on average', () => {
    const rand = mulberry32(7)
    for (let trial = 0; trial < 200; trial++) {
      const lo = Math.floor(rand() * 250)
      const span = 1 + Math.floor(rand() * 3)
      const values = grid8(() => lo + Math.floor(rand() * (span + 1)))
      expect(sse8(values, encodeEACR11Block(values)) / 16).toBeLessThan(0.25 * 0.25)
    }
  })

  it('respects the row-major API layout (a horizontal ramp stays horizontal)', () => {
    const values = grid8(k => 40 + (k % 4) * 30)
    const dec = decodeEACR11Block(encodeEACR11Block(values))
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) expect(Math.abs(dec[y * 4 + x]! * 255 - (40 + x * 30))).toBeLessThan(6)
    }
    // And the transposed input decodes transposed.
    const t = grid8(k => 40 + (k >> 2) * 30)
    const decT = decodeEACR11Block(encodeEACR11Block(t))
    for (let k = 0; k < 16; k++) expect(decT[k]).toBe(dec[(k & 3) * 4 + (k >> 2)]!)
  })

  it('is deterministic', () => {
    const rand = mulberry32(3)
    const values = grid8(() => Math.floor(rand() * 256))
    expect(encodeEACR11Block(values)).toEqual(encodeEACR11Block(values))
  })

  it("'high' never decodes worse than 'fast', and the gap stays small", () => {
    const rand = mulberry32(11)
    let fastTotal = 0
    let highTotal = 0
    for (let trial = 0; trial < 300; trial++) {
      // Inputs on the 11-bit grid: both searches minimise error against the
      // rounded 11-bit targets, so off-grid inputs would compare them on
      // rounding noise.
      const lo = Math.floor(rand() * 1600)
      const span = Math.floor(rand() * rand() * (2047 - lo))
      const values = Array.from({ length: 16 }, () => (lo + Math.floor(rand() * (span + 1))) / 2047)
      const fast = sse8(values, encodeEACR11Block(values))
      const high = sse8(values, encodeEACR11Block(values, { quality: 'high' }))
      // (tolerance: the decoder returns Float32 values)
      expect(high).toBeLessThanOrEqual(fast * (1 + 1e-5) + 1e-6)
      fastTotal += fast
      highTotal += high
    }
    // The three-candidate search stays within ~2 dB of the exhaustive one
    // even on uniform noise, its worst case.
    expect(fastTotal).toBeLessThan(highTotal * 1.6)
  })

  it('handles full-range noise without leaving the valid field ranges', () => {
    const rand = mulberry32(5)
    for (let trial = 0; trial < 100; trial++) {
      const values = grid8(() => Math.floor(rand() * 256))
      const block = encodeEACR11Block(values)
      expect(sse8(values, block) / 16).toBeLessThan(40 * 40)
      const dec = decodeEACR11Block(block)
      for (let k = 0; k < 16; k++) {
        expect(dec[k]).toBeGreaterThanOrEqual(0)
        expect(dec[k]).toBeLessThanOrEqual(1)
      }
    }
  })
})
