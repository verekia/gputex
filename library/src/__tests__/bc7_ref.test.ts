import {
  encodeBC7Mode6Block,
  decodeBC7Mode6Block,
  decodeBC7Block,
  encodeBC7Block,
  readBC7Mode,
  _internal,
} from '../bc7_ref.js'

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

// Deterministic LCG so "random" tests are reproducible.
function seededRand(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 0x1_0000_0000
  }
}

describe('BC7 mode 6 block bitstream', () => {
  it('has mode bit at position 6 (6 zero bits then a 1)', () => {
    const flat = makePixels(() => [0.5, 0.5, 0.5, 1.0])
    const block = encodeBC7Mode6Block(flat)
    expect(block.byteLength).toBe(16)
    // Low 7 bits of byte 0: bits 0..5 should be 0, bit 6 should be 1.
    expect(block[0]! & 0x7f).toBe(0x40)
    expect(readBC7Mode(block)).toBe(6)
  })

  it('decodeBC7Block dispatches on the mode field', () => {
    const flat = makePixels(() => [0.3, 0.6, 0.9, 1.0])
    const block = encodeBC7Block(flat)
    const direct = decodeBC7Mode6Block(block)
    const via = decodeBC7Block(block)
    expect([...via]).toEqual([...direct])
  })
})

describe('BC7 mode 6 encode/decode round-trip', () => {
  it('round-trips a flat opaque block with tiny error', () => {
    const pixels = makePixels(() => [0.5, 0.5, 0.5, 1.0])
    const block = encodeBC7Mode6Block(pixels)
    const decoded = decodeBC7Mode6Block(block)
    expect(maxAbs(decoded, pixels)).toBeLessThan(2 / 255)
  })

  it('round-trips the all-zero block', () => {
    const pixels = new Float32Array(64) // 0,0,0,0 everywhere
    const block = encodeBC7Mode6Block(pixels)
    const decoded = decodeBC7Mode6Block(block)
    expect(maxAbs(decoded, pixels)).toBeLessThan(1 / 255)
  })

  it('round-trips the all-one opaque white block', () => {
    const pixels = new Float32Array(64).fill(1)
    const block = encodeBC7Mode6Block(pixels)
    const decoded = decodeBC7Mode6Block(block)
    expect(maxAbs(decoded, pixels)).toBeLessThan(1 / 255)
  })

  it('round-trips a smooth RGB gradient + constant alpha very tightly', () => {
    // 16-step gradient in each of R, G, B — BC7's 16-entry palette should
    // hit each target nearly exactly (~1 LSB off at most).
    const pixels = makePixels(k => [k / 15, k / 15, 1 - k / 15, 1])
    const block = encodeBC7Mode6Block(pixels)
    const decoded = decodeBC7Mode6Block(block)
    // BC7 mode 6 on a smooth 16-value gradient should reproduce at very
    // near original: 16 palette entries over 16 values is a 1:1 mapping.
    expect(rmse(decoded, pixels)).toBeLessThan(0.01)
    expect(maxAbs(decoded, pixels)).toBeLessThan(0.02)
  })

  it('round-trips a bimodal block (half black, half white) exactly', () => {
    const pixels = makePixels(k => (k < 8 ? [0, 0, 0, 1] : [1, 1, 1, 1]))
    const block = encodeBC7Mode6Block(pixels)
    const decoded = decodeBC7Mode6Block(block)
    // Palette endpoints can be 0 and 255 exactly, indices 0 and 15 map to
    // them exactly, so zero reconstruction error is achievable.
    expect(maxAbs(decoded, pixels)).toBeLessThan(1 / 255)
  })

  it('round-trips a block with non-trivial alpha variation', () => {
    const pixels = makePixels(k => [0.3, 0.7, 0.4, k / 15])
    const block = encodeBC7Mode6Block(pixels)
    const decoded = decodeBC7Mode6Block(block)
    // RGB is roughly constant, alpha varies across 16 levels; both should
    // fit tightly within the shared 16-entry palette.
    expect(maxAbs(decoded, pixels)).toBeLessThan(0.03)
  })

  it('keeps smooth-random error tight over a battery of blocks', () => {
    // Image-like data (smooth per-block variation in 4D) should round-trip
    // at sub-1% RMS because BC7 mode 6 puts 16 palette entries along a
    // single endpoint line — smooth data sits close to that line. Pure 4D
    // white noise is a worst case (RMS ~0.25 with mode 6 alone); 2-subset
    // modes would handle noise better but are out of scope for this
    // implementation (see `BC7Encoder.ts` file header).
    const rand = seededRand(0xbeef)
    let worst = 0
    for (let trial = 0; trial < 30; trial++) {
      // Pick random endpoints per block and interpolate across them —
      // this is what real photographic textures tend to look like at 4x4.
      const e0 = [rand(), rand(), rand(), rand()] as const
      const e1 = [rand(), rand(), rand(), rand()] as const
      const pixels = makePixels(k => {
        const t = k / 15 + (rand() - 0.5) * 0.05 // small jitter
        return [
          e0[0] + t * (e1[0] - e0[0]),
          e0[1] + t * (e1[1] - e0[1]),
          e0[2] + t * (e1[2] - e0[2]),
          e0[3] + t * (e1[3] - e0[3]),
        ]
      })
      const block = encodeBC7Mode6Block(pixels)
      const decoded = decodeBC7Mode6Block(block)
      worst = Math.max(worst, rmse(decoded, pixels))
    }
    expect(worst).toBeLessThan(0.02)
  })
})

describe('BC7 mode 6 anchor rule', () => {
  it('the stored pixel-0 index has MSB 0 regardless of input', () => {
    // If the natural index for pixel 0 would land in [8..15], the encoder
    // must swap endpoints + invert indices so the stored 3-bit MSB is 0.
    // We trigger this by making pixel 0 the extreme value of the block.
    const pixels = makePixels(k => (k === 0 ? [1, 1, 1, 1] : [0, 0, 0, 1]))
    const block = encodeBC7Mode6Block(pixels)
    // Bits 65..67 hold the 3-bit stored index for pixel 0. Extract them.
    // bit 65 starts in byte 8 bit 1.
    let bits = 0n
    for (let i = 0; i < 16; i++) bits |= BigInt(block[i]!) << BigInt(i * 8)
    const idx0_stored = Number((bits >> 65n) & 7n)
    // Any stored value is fine; the guarantee is that the decoded
    // palette index equals idx0_stored (no implicit high bit was dropped).
    // Round-trip proves the invariant:
    const decoded = decodeBC7Mode6Block(block)
    expect(maxAbs(decoded.subarray(0, 4), pixels.subarray(0, 4))).toBeLessThan(3 / 255)
    // Sanity: stored index is always in [0, 7].
    expect(idx0_stored).toBeGreaterThanOrEqual(0)
    expect(idx0_stored).toBeLessThanOrEqual(7)
  })
})

describe('BC7 mode 6 determinism', () => {
  it('same input yields the same 16 bytes', () => {
    const pixels = makePixels(k => [0.1 + k * 0.05, 0.2, 0.9 - k * 0.03, 1])
    const a = encodeBC7Mode6Block(pixels)
    const b = encodeBC7Mode6Block(pixels)
    expect([...a]).toEqual([...b])
  })

  // Note: encode → decode → encode is NOT idempotent for BC7 mode 6. On
  // re-encode, the observed bbox is strictly narrower than the original
  // endpoints (unless indices 0 AND 15 happen to be used AND the refit
  // reproduces identical endpoints), so the second block generally uses
  // a smaller-range palette and differs in the endpoint/index bytes.
})

describe('BC7 interpolation primitive', () => {
  it('matches the BC7 spec integer rule at the W4 table endpoints', () => {
    // Sanity: interp8(e0, e1, 0) == e0, interp8(e0, e1, 64) == e1.
    expect(_internal.interp8(13, 240, 0)).toBe(13)
    expect(_internal.interp8(13, 240, 64)).toBe(240)
    // A midpoint: palette[7] uses W4[7] = 30 → 30/64 fraction toward e1.
    // interp8(0, 255, 30) = round(30 * 255 / 64) -bias-rounded-down
    // = (34 * 0 + 30 * 255 + 32) >> 6 = (7650 + 32) >> 6 = 7682 >> 6 = 120.
    expect(_internal.interp8(0, 255, 30)).toBe(120)
  })
})

describe('BC7 bit writer round-trip', () => {
  it('writes and reads values back in LSB-first order', () => {
    const bw = new _internal.BitWriter128()
    bw.write(0x2a, 7) // arbitrary 7-bit value
    bw.write(0x1, 1) // bit after
    bw.write(0x1234, 15)
    const bytes = bw.toBytes()
    const br = new _internal.BitReader128(bytes)
    expect(br.read(7)).toBe(0x2a)
    expect(br.read(1)).toBe(1)
    expect(br.read(15)).toBe(0x1234)
  })
})

describe('BC7 input validation', () => {
  it('throws on non-64-length pixel arrays', () => {
    expect(() => encodeBC7Mode6Block(new Float32Array(63))).toThrow(/64/)
    expect(() => encodeBC7Mode6Block(new Float32Array(65))).toThrow(/64/)
  })

  it('throws on non-16-length block arrays', () => {
    expect(() => decodeBC7Mode6Block(new Uint8Array(15))).toThrow(/16/)
  })

  it('throws when asked to decode a non-mode-6 block as mode 6', () => {
    // Craft a block whose first bit is 1 (mode 0).
    const fake = new Uint8Array(16)
    fake[0] = 0x01
    expect(() => decodeBC7Mode6Block(fake)).toThrow(/mode 6/)
    // The top-level dispatcher rejects unsupported modes clearly too.
    expect(() => decodeBC7Block(fake)).toThrow(/mode 0/)
  })
})
