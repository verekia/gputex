import { describe, expect, it } from 'bun:test'

import { _internal, decodeETC2Block, encodeETC2Block, readETC2Mode } from '../etc2_ref.js'

const { extend4, extend5, extend6, extend7, signed3, packPlanar, wordsToBlock } = _internal

/** Deterministic PRNG so failures reproduce. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function sse(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!
    s += d * d
  }
  return s
}

describe('etc2_ref internals', () => {
  it('bit-replication extensions hit the endpoints', () => {
    expect(extend4(0)).toBe(0)
    expect(extend4(15)).toBe(255)
    expect(extend5(0)).toBe(0)
    expect(extend5(31)).toBe(255)
    expect(extend5(16)).toBe(132)
    expect(extend6(63)).toBe(255)
    expect(extend7(127)).toBe(255)
  })

  it('signed3 decodes two’s complement', () => {
    expect(signed3(0)).toBe(0)
    expect(signed3(3)).toBe(3)
    expect(signed3(4)).toBe(-4)
    expect(signed3(7)).toBe(-1)
  })
})

describe('encodeETC2Block basics', () => {
  it('rejects wrong-size input', () => {
    expect(() => encodeETC2Block(new Float32Array(47))).toThrow()
  })

  it('is deterministic', () => {
    const rand = mulberry32(1)
    const px = Float32Array.from({ length: 48 }, () => rand())
    expect(encodeETC2Block(px)).toEqual(encodeETC2Block(px))
    expect(encodeETC2Block(px, { quality: 'high' })).toEqual(encodeETC2Block(px, { quality: 'high' }))
  })

  it('encodes an exactly-representable flat block losslessly', () => {
    // 134 = extend5(16) + modifier 2 (table 0) — reachable exactly in
    // differential mode, so the round trip must be bit-perfect.
    const px = new Float32Array(48).fill(134 / 255)
    const decoded = decodeETC2Block(encodeETC2Block(px))
    for (let i = 0; i < 48; i++) {
      expect(decoded[i]!).toBeCloseTo(134 / 255, 6)
    }
  })

  it('keeps flat blocks within a small quantisation error', () => {
    const rand = mulberry32(2)
    for (let n = 0; n < 32; n++) {
      const c = [rand(), rand(), rand()]
      const px = new Float32Array(48)
      for (let k = 0; k < 16; k++) px.set(c, k * 3)
      const decoded = decodeETC2Block(encodeETC2Block(px))
      for (let i = 0; i < 48; i++) {
        // 5-bit base (max half-step 4/255) + smallest modifier reach (2/255).
        expect(Math.abs(decoded[i]! - px[i]!)).toBeLessThanOrEqual(6.5 / 255)
      }
    }
  })

  it('never emits T or H, and the emitted mode reads back consistently', () => {
    const rand = mulberry32(3)
    for (let n = 0; n < 200; n++) {
      // Mix of noise, flats and gradients so all emitted modes appear.
      const kind = n % 3
      const px = new Float32Array(48)
      const base = [rand(), rand(), rand()]
      for (let k = 0; k < 16; k++) {
        const x = k & 3
        const y = k >> 2
        for (let c = 0; c < 3; c++) {
          const v = kind === 0 ? rand() : kind === 1 ? base[c]! : base[c]! * 0.2 + (x / 6) * 0.4 + (y / 6) * 0.4
          px[k * 3 + c] = Math.min(1, Math.max(0, v))
        }
      }
      for (const quality of ['fast', 'high'] as const) {
        const mode = readETC2Mode(encodeETC2Block(px, { quality }))
        expect(['individual', 'differential', 'planar']).toContain(mode)
      }
    }
  })

  it("tracks a clean colour gradient closely ('high' via planar, 'fast' via ETC1 only)", () => {
    const px = new Float32Array(48)
    for (let k = 0; k < 16; k++) {
      const x = k & 3
      const y = k >> 2
      px[k * 3] = 0.2 + x * 0.12
      px[k * 3 + 1] = 0.3 + y * 0.1
      px[k * 3 + 2] = 0.5 + x * 0.04 - y * 0.06
    }
    // 'high' still emits planar — the natural fit for a smooth gradient.
    const high = encodeETC2Block(px, { quality: 'high' })
    expect(readETC2Mode(high)).toBe('planar')
    const decodedHigh = decodeETC2Block(high)
    for (let i = 0; i < 48; i++) {
      expect(Math.abs(decodedHigh[i]! - px[i]!)).toBeLessThanOrEqual(4 / 255)
    }
    // 'fast' (the GPU mirror) dropped planar in the bandwidth rewrite; it
    // must still track the gradient via ETC1 modes, much more loosely
    // (scalar luma modulation cannot follow two independent chroma slopes
    // — worst sample ≈ 34/255 here; the bound just rules out catastrophe).
    const fast = encodeETC2Block(px)
    expect(readETC2Mode(fast)).not.toBe('planar')
    const decodedFast = decodeETC2Block(fast)
    for (let i = 0; i < 48; i++) {
      expect(Math.abs(decodedFast[i]! - px[i]!)).toBeLessThanOrEqual(48 / 255)
    }
  })

  it('splits a two-tone block along the correct axis via an ETC1 mode', () => {
    // Left half dark red, right half bright blue — a column split (flip=0)
    // with per-subblock bases; T/H would fit too but aren't emitted.
    const px = new Float32Array(48)
    for (let k = 0; k < 16; k++) {
      const right = (k & 3) >= 2
      px.set(right ? [0.15, 0.2, 0.85] : [0.6, 0.1, 0.1], k * 3)
    }
    const block = encodeETC2Block(px)
    expect(['individual', 'differential']).toContain(readETC2Mode(block))
    const decoded = decodeETC2Block(block)
    // Each half is flat, so it must decode near-exactly.
    expect(sse(decoded, px)).toBeLessThanOrEqual(48 * (6.5 / 255) ** 2)
  })

  it("quality 'high' never loses to 'fast'", () => {
    // Inputs on the 8-bit grid: the encoder quantises internally, so its
    // high-vs-fast tie-break is decided against the quantised values —
    // off-grid inputs can flip near-ties in this test's metric.
    const rand = mulberry32(4)
    for (let n = 0; n < 100; n++) {
      const px = Float32Array.from({ length: 48 }, () => Math.round(rand() * 255) / 255)
      const fast = sse(decodeETC2Block(encodeETC2Block(px)), px)
      const high = sse(decodeETC2Block(encodeETC2Block(px, { quality: 'high' })), px)
      expect(high).toBeLessThanOrEqual(fast + 1e-6)
    }
  })
})

describe('planar packing', () => {
  it('signals planar (not T/H/differential) for every corner-code combination', () => {
    // The scattered planar fields leave "opacity" bits the encoder must set
    // so the ETC1-decoder view keeps R and G in range while B overflows.
    // Sweep randomised codes to exercise every p+q branch.
    const rand = mulberry32(5)
    for (let n = 0; n < 500; n++) {
      const c6 = () => Math.floor(rand() * 64)
      const c7 = () => Math.floor(rand() * 128)
      const cand = { o: [c6(), c7(), c6()], h: [c6(), c7(), c6()], v: [c6(), c7(), c6()], err: 0 }
      const block = packPlanar(cand)
      expect(readETC2Mode(block)).toBe('planar')
      // Round trip: the decoded output must equal the plane equation
      // computed straight from the extended corner colours.
      const decoded = decodeETC2Block(block)
      const ext = (codes: number[]) => [extend6(codes[0]!), extend7(codes[1]!), extend6(codes[2]!)]
      const [o, h, v] = [ext(cand.o), ext(cand.h), ext(cand.v)]
      for (let k = 0; k < 16; k++) {
        const x = k & 3
        const y = k >> 2
        for (let c = 0; c < 3; c++) {
          const raw = (x * (h[c]! - o[c]!) + y * (v[c]! - o[c]!) + 4 * o[c]! + 2) >> 2
          const expected = Math.min(255, Math.max(0, raw)) / 255
          expect(decoded[k * 3 + c]!).toBeCloseTo(expected, 6)
        }
      }
    }
  })
})

describe('decodeETC2Block: T and H modes (decode-only)', () => {
  it('decodes a hand-crafted T-mode block', () => {
    // bits 63..56 = 1111 1011: don't-care bits set so the ETC1 R sum (31 + 3)
    // overflows → T. R1a = 3, R1b = 3 → R1 = 15; G1 = B1 = 0 → base1 = red.
    // R2 = 0, G2 = 15, B2 = 0 → base2 = green. da=0, db=0 → distance 3.
    const hi = 0xfb000f02
    // All indices 0 → palette[0] = base1 exactly.
    let decoded = decodeETC2Block(wordsToBlock(hi, 0))
    expect(readETC2Mode(wordsToBlock(hi, 0))).toBe('T')
    for (let k = 0; k < 16; k++) {
      expect(decoded[k * 3]!).toBeCloseTo(1, 6)
      expect(decoded[k * 3 + 1]!).toBeCloseTo(0, 6)
      expect(decoded[k * 3 + 2]!).toBeCloseTo(0, 6)
    }
    // Texel (0,0) → wire bit 0. MSB set → index 2 → palette[2] = base2.
    decoded = decodeETC2Block(wordsToBlock(hi, 1 << 16))
    expect(decoded[0]!).toBeCloseTo(0, 6)
    expect(decoded[1]!).toBeCloseTo(1, 6)
    // LSB set → index 1 → base2 + d, clamped: (3, 255+3→255, 3).
    decoded = decodeETC2Block(wordsToBlock(hi, 1))
    expect(decoded[0]!).toBeCloseTo(3 / 255, 6)
    expect(decoded[1]!).toBeCloseTo(1, 6)
    expect(decoded[2]!).toBeCloseTo(3 / 255, 6)
  })

  it('decodes a hand-crafted H-mode block', () => {
    // R fields in range (0), G forced out of range via dG = -4 (bit 50 set)
    // → H. base1 = (0,0,0), base2: R2 = 15 → (255,0,0). v1 < v2 → distance
    // index 0 → d = 3. All indices 0 → palette[0] = base1 + d = (3,3,3).
    const hi = 0x00047802
    const block = wordsToBlock(hi, 0)
    expect(readETC2Mode(block)).toBe('H')
    const decoded = decodeETC2Block(block)
    for (let i = 0; i < 48; i++) {
      expect(decoded[i]!).toBeCloseTo(3 / 255, 6)
    }
    // Index 2 (MSB) → palette[2] = base2 + d = (255+3→255, 3, 3).
    const d2 = decodeETC2Block(wordsToBlock(hi, 1 << 16))
    expect(d2[0]!).toBeCloseTo(1, 6)
    expect(d2[1]!).toBeCloseTo(3 / 255, 6)
  })

  it('rejects wrong-size blocks', () => {
    expect(() => decodeETC2Block(new Uint8Array(7))).toThrow()
  })
})
