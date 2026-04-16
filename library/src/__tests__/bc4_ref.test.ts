import { encodeBC4Block, decodeBC4Block, _internal } from '../bc4_ref.js'

type NumArray = ArrayLike<number>

// RMS error over 16 values, in the normalized [0,1] domain.
function rmse(actual: NumArray, expected: NumArray): number {
  let s = 0
  for (let i = 0; i < actual.length; i++) {
    const d = actual[i]! - expected[i]!
    s += d * d
  }
  return Math.sqrt(s / actual.length)
}

// Max absolute error across 16 values.
function maxAbs(actual: NumArray, expected: NumArray): number {
  let m = 0
  for (let i = 0; i < actual.length; i++) {
    m = Math.max(m, Math.abs(actual[i]! - expected[i]!))
  }
  return m
}

describe('encodeBC4Block / decodeBC4Block', () => {
  it('produces 8 bytes with red0 > red1 (6-interp mode)', () => {
    const values = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    const block = encodeBC4Block(values)
    expect(block.byteLength).toBe(8)
    expect(block[0]).toBeGreaterThan(block[1])
  })

  it('round-trips a smooth horizontal gradient within BC4 quality bounds', () => {
    // Smooth gradients are what BC4 is designed for. 8 palette entries
    // over a single endpoint line should track a linear ramp tightly.
    // Theoretical worst case for 16 evenly-spaced values against 8 palette
    // entries is ~1/14 ≈ 0.07 max abs; refinement pulls both RMS and max
    // well below that in practice.
    const values = new Float32Array(16)
    for (let i = 0; i < 16; i++) values[i] = i / 15
    const block = encodeBC4Block(values)
    const decoded = decodeBC4Block(block)
    expect(rmse(decoded, values)).toBeLessThan(0.04)
    expect(maxAbs(decoded, values)).toBeLessThan(0.05)
  })

  it('round-trips a flat block exactly (post-nudge)', () => {
    // All-same value. Encoder has to pick endpoints s.t. red0 > red1,
    // so the palette can't be flat; but one of the 8 entries should
    // land within 1/255 of the target (the entry closest to the value).
    const values = new Float32Array(16).fill(0.5)
    const block = encodeBC4Block(values)
    const decoded = decodeBC4Block(block)
    expect(maxAbs(decoded, values)).toBeLessThan(2 / 255)
  })

  it('handles the all-zero block (0.0 is the min and max, red1 must stay >= 0)', () => {
    const values = new Float32Array(16)
    const block = encodeBC4Block(values)
    const decoded = decodeBC4Block(block)
    // All zeros — the closest palette entry should be extremely close
    // to 0. With the nudge (red1=0 → red0=1), palette[1] = 0 exactly.
    expect(maxAbs(decoded, values)).toBeLessThan(1 / 255)
  })

  it('handles the all-one block', () => {
    const values = new Float32Array(16).fill(1)
    const block = encodeBC4Block(values)
    const decoded = decodeBC4Block(block)
    expect(maxAbs(decoded, values)).toBeLessThan(1 / 255)
  })

  it('round-trips a two-value block (half 0, half 1) with near-zero error', () => {
    // Bimodal block — the two endpoint palette entries should land
    // exactly on red0=255 / red1=0, so max error is 0.
    const values = new Float32Array(16)
    for (let i = 0; i < 16; i++) values[i] = i < 8 ? 0 : 1
    const block = encodeBC4Block(values)
    // Endpoints should hit the extremes.
    expect(block[0]).toBe(255)
    expect(block[1]).toBe(0)
    const decoded = decodeBC4Block(block)
    // Palette entries 0 and 1 are exactly 1.0 and 0.0, so indices can
    // encode each extreme losslessly.
    expect(maxAbs(decoded, values)).toBe(0)
  })

  it('is deterministic — same input yields same 8 bytes', () => {
    const values = new Float32Array([0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.8])
    const a = encodeBC4Block(values)
    const b = encodeBC4Block(values)
    expect([...a]).toEqual([...b])
  })

  it('encode → decode → encode is idempotent', () => {
    // After one round-trip, the decoded palette values are already on
    // the palette grid, so re-encoding should produce the same bytes.
    const values = new Float32Array(16)
    for (let i = 0; i < 16; i++) values[i] = 0.1 + i * 0.05
    const b1 = encodeBC4Block(values)
    const decoded = decodeBC4Block(b1)
    const b2 = encodeBC4Block(decoded)
    expect([...b1]).toEqual([...b2])
  })

  it('packs 16 × 3-bit indices little-endian, pixel 0 at bit 0', () => {
    // Build a block with known endpoints and a known index pattern.
    // Index = (pixel_index % 8) so we exercise all 8 values.
    const indices = new Uint8Array(16)
    for (let i = 0; i < 16; i++) indices[i] = i & 7
    const block = _internal.packBlock(200, 50, indices)

    // Reconstruct the 48-bit index field and compare.
    let bits = 0n
    for (let i = 0; i < 6; i++) bits |= BigInt(block[2 + i]) << BigInt(i * 8)
    for (let k = 0; k < 16; k++) {
      const idx = Number((bits >> BigInt(3 * k)) & 7n)
      expect(idx).toBe(k & 7)
    }
    expect(block[0]).toBe(200)
    expect(block[1]).toBe(50)
  })

  it('decodes a hand-constructed 6-interp block with the expected palette', () => {
    // red0 = 255, red1 = 0 → 6-interp mode.
    // palette should be { 1, 0, 6/7, 5/7, 4/7, 3/7, 2/7, 1/7 }.
    const block = new Uint8Array(8)
    block[0] = 255
    block[1] = 0
    // indices: 0, 1, 2, 3, 4, 5, 6, 7 for first 8 pixels; then repeat.
    const indices = new Uint8Array(16)
    for (let i = 0; i < 16; i++) indices[i] = i & 7
    const packed = _internal.packBlock(255, 0, indices)
    const decoded = decodeBC4Block(packed)
    const expected = [1, 0, 6 / 7, 5 / 7, 4 / 7, 3 / 7, 2 / 7, 1 / 7]
    for (let i = 0; i < 8; i++) {
      expect(decoded[i]).toBeCloseTo(expected[i], 6)
    }
  })

  it('decodes a 4-interp-mode block (red0 <= red1) with hard 0/1 rails', () => {
    // 4-interp mode: red0 <= red1.
    // palette = { red0, red1, (4r0+r1)/5, (3r0+2r1)/5, (2r0+3r1)/5, (r0+4r1)/5, 0, 1 }
    const r0 = 50,
      r1 = 200 // forces 4-interp mode
    // All indices point at the "rail" entries 6 and 7.
    const indices = new Uint8Array(16)
    for (let i = 0; i < 16; i++) indices[i] = i < 8 ? 6 : 7
    const packed = _internal.packBlock(r0, r1, indices)
    const decoded = decodeBC4Block(packed)
    // First 8 texels = palette[6] = 0.0
    for (let i = 0; i < 8; i++) expect(decoded[i]).toBe(0)
    // Last 8 texels = palette[7] = 1.0
    for (let i = 8; i < 16; i++) expect(decoded[i]).toBe(1)
  })

  it('throws on bad input length', () => {
    expect(() => encodeBC4Block(new Float32Array(15))).toThrow(/16/)
    expect(() => encodeBC4Block(new Float32Array(17))).toThrow(/16/)
    expect(() => decodeBC4Block(new Uint8Array(7))).toThrow(/8/)
  })

  it('random-texel RMS error stays below ~3% over a battery of blocks', () => {
    // This is a sanity test: BC4 should handle arbitrary texels with
    // bounded error. Use a fixed seed via a tiny LCG so the test is
    // deterministic across runs.
    let state = 0xc0ffee
    const rand = () => {
      state = (state * 1664525 + 1013904223) >>> 0
      return state / 0x1_0000_0000
    }

    let worstRmse = 0
    for (let trial = 0; trial < 50; trial++) {
      const values = new Float32Array(16)
      for (let i = 0; i < 16; i++) values[i] = rand()
      const block = encodeBC4Block(values)
      const decoded = decodeBC4Block(block)
      worstRmse = Math.max(worstRmse, rmse(decoded, values))
    }
    expect(worstRmse).toBeLessThan(0.05)
  })
})
