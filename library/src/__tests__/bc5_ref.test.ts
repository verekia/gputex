import { encodeBC4Block } from '../bc4_ref.js'
import { encodeBC5Block, decodeBC5Block } from '../bc5_ref.js'

function maxAbs(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!))
  return m
}

describe('BC5 reference encoder', () => {
  it('produces 16 bytes per block', () => {
    const r = new Float32Array(16)
    const g = new Float32Array(16)
    const block = encodeBC5Block(r, g)
    expect(block.byteLength).toBe(16)
  })

  it('is exactly two BC4 blocks concatenated', () => {
    // The BC5 bitstream is literally two BC4 halves: R on [0..8), G on
    // [8..16). Encoding R and G independently via BC4 must match the
    // BC5 output byte-for-byte.
    const r = new Float32Array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.7, 0.3, 0.1, 0.5, 0.9, 0.25, 0.75, 0.35, 0.65, 0.05])
    const g = new Float32Array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.3, 0.7, 0.9, 0.5, 0.1, 0.75, 0.25, 0.65, 0.35, 0.95])
    const bc5 = encodeBC5Block(r, g)
    const bc4_r = encodeBC4Block(r)
    const bc4_g = encodeBC4Block(g)
    expect([...bc5.subarray(0, 8)]).toEqual([...bc4_r])
    expect([...bc5.subarray(8, 16)]).toEqual([...bc4_g])
  })

  it('round-trips a smooth gradient on both channels', () => {
    const r = new Float32Array(16)
    const g = new Float32Array(16)
    for (let i = 0; i < 16; i++) {
      r[i] = i / 15 // x gradient (16 steps)
      g[i] = Math.floor(i / 4) / 3 // y gradient (4 stair-steps of 4)
    }
    const block = encodeBC5Block(r, g)
    const { r: rOut, g: gOut } = decodeBC5Block(block)
    // R: smooth 16-step gradient, hits the sweet spot for 8 palette entries.
    expect(maxAbs(rOut, r)).toBeLessThan(0.05)
    // G: four distinct levels {0, 1/3, 2/3, 1} — BC4's palette can't hit
    // 1/3 and 2/3 exactly with r0 > r1 in [0,1], so residual is bigger.
    expect(maxAbs(gOut, g)).toBeLessThan(0.06)
  })

  it('round-trips identical R and G channels with identical halves', () => {
    const r = new Float32Array(16)
    for (let i = 0; i < 16; i++) r[i] = ((i * 17) % 16) / 15
    const g = r.slice()
    const block = encodeBC5Block(r, g)
    // Both halves must be byte-identical when R and G are the same.
    expect([...block.subarray(0, 8)]).toEqual([...block.subarray(8, 16)])
  })

  it('handles a unit-length-normal-map encoded patch', () => {
    // Synthesize a 4×4 patch of unit-length normals that vary smoothly.
    // Remap from [-1, 1] to [0, 1] for storage, which is how BC5
    // normal maps are conventionally authored.
    const r = new Float32Array(16)
    const g = new Float32Array(16)
    for (let i = 0; i < 16; i++) {
      const x = ((i % 4) / 3) * 2 - 1 // [-1, 1]
      const y = (Math.floor(i / 4) / 3) * 2 - 1
      // Reduce magnitude so x² + y² ≤ 1.
      const k = 0.9
      r[i] = (x * k + 1) * 0.5 // [0, 1]
      g[i] = (y * k + 1) * 0.5
    }
    const block = encodeBC5Block(r, g)
    const { r: rOut, g: gOut } = decodeBC5Block(block)
    // Smooth 2D variation in each channel; error should be comparable
    // to the 1D-gradient case.
    expect(maxAbs(rOut, r)).toBeLessThan(0.04)
    expect(maxAbs(gOut, g)).toBeLessThan(0.04)
  })

  it('throws on bad input length', () => {
    expect(() => encodeBC5Block(new Float32Array(15), new Float32Array(16))).toThrow(/16/)
    expect(() => decodeBC5Block(new Uint8Array(15))).toThrow(/16/)
  })
})
