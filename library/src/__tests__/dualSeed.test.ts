// The two-seed endpoint search in the BC7 and ASTC reference encoders.
//
// Both encoders try a farthest-pixel-pair seed and a principal-axis seed and
// keep the lower-error fit. The two disagree most when one texel drags the
// farthest pair off the axis the other fifteen actually lie on — the pair is
// pinned to real texels, the principal axis is fitted to the distribution —
// so dropping either seed silently costs quality on exactly the blocks that
// are already hardest.
//
// These tests pin that: the blocks below are ones where the second seed wins
// outright, with bounds a single-seed search does not meet. Error is measured
// the way the browser suite's PSNR does (`example/lib/gpuTestSuite.ts`):
// against the decoded value unrounded, which is what the encoders minimise.

import { encodeASTC4x4Block, decodeASTC4x4Block } from '../astc4x4_ref.js'
import { encodeBC7Mode6Block, decodeBC7Block } from '../bc7_ref.js'

/**
 * Summed squared error over RGBA, decoded values left unrounded.
 *
 * Decoders return f32, so an "exact" block lands a few times 1e-10 off zero
 * rather than at it; assertions below allow for that.
 */
function blockError(pixels: Float32Array, decoded: Float32Array): number {
  let e = 0
  for (let k = 0; k < 16; k++) {
    for (let c = 0; c < 4; c++) {
      const d = decoded[k * 4 + c]! * 255 - Math.round(pixels[k * 4 + c]! * 255)
      e += d * d
    }
  }
  return e
}

function block(texels: readonly (readonly [number, number, number, number])[]): Float32Array {
  const p = new Float32Array(64)
  texels.forEach((t, k) => {
    for (let c = 0; c < 4; c++) p[k * 4 + c] = t[c]! / 255
  })
  return p
}

// Sixteen texels scattered across the RGBA cube's corners: the farthest pair
// is a cube diagonal that most of the block sits nowhere near, while the
// principal axis follows the bulk of the distribution.
const SCATTERED_BC7 = block([
  [0, 255, 255, 255],
  [255, 255, 0, 255],
  [0, 255, 0, 255],
  [255, 255, 255, 255],
  [255, 0, 0, 255],
  [255, 255, 0, 255],
  [0, 255, 0, 0],
  [255, 0, 255, 255],
  [0, 255, 255, 0],
  [0, 255, 0, 0],
  [0, 255, 0, 0],
  [255, 255, 255, 255],
  [255, 0, 0, 0],
  [0, 255, 0, 0],
  [255, 0, 255, 255],
  [255, 0, 255, 255],
])

const SCATTERED_ASTC = block([
  [0, 0, 0, 255],
  [0, 255, 255, 255],
  [0, 255, 255, 0],
  [255, 255, 255, 255],
  [0, 255, 255, 0],
  [255, 0, 0, 0],
  [255, 255, 0, 255],
  [255, 255, 0, 255],
  [255, 255, 0, 255],
  [255, 255, 255, 0],
  [0, 255, 255, 255],
  [255, 0, 0, 255],
  [0, 255, 255, 0],
  [0, 0, 255, 0],
  [255, 255, 0, 255],
  [0, 0, 255, 0],
])

describe('BC7 mode 6 two-seed search', () => {
  it('beats what a farthest-pair-only search reaches on a scattered block', () => {
    const err = blockError(SCATTERED_BC7, decodeBC7Block(encodeBC7Mode6Block(SCATTERED_BC7)))
    // Farthest pair alone lands at ~798_500 here; both seeds reach ~452_500.
    expect(err).toBeLessThan(500_000)
  })
})

describe('ASTC 4×4 two-seed search', () => {
  it('beats what a farthest-pair-only search reaches on a scattered block', () => {
    const err = blockError(SCATTERED_ASTC, decodeASTC4x4Block(encodeASTC4x4Block(SCATTERED_ASTC)))
    // Farthest pair alone lands at ~833_400 here; both seeds reach ~477_700.
    expect(err).toBeLessThan(520_000)
  })
})

describe('two-seed search leaves easy blocks alone', () => {
  // A flat block has zero covariance, so it has no principal axis at all.
  // The seed must degenerate gracefully (e0 == e1) rather than produce NaN
  // endpoints from a normalise-by-zero.
  it('handles a flat block, which has no principal axis', () => {
    const flat = block(Array.from({ length: 16 }, () => [37, 199, 84, 255] as const))
    // ASTC stores endpoints as plain bytes, so a flat block is exact.
    expect(blockError(flat, decodeASTC4x4Block(encodeASTC4x4Block(flat)))).toBeLessThan(1e-6)
    // BC7 mode 6 cannot be exact here: each endpoint's four channels share
    // one p-bit, and 37/199/255 want p=1 while 84 wants p=0. One channel
    // lands 1 LSB out on both endpoints — 16, the format's floor for this
    // block, not a search failure.
    expect(blockError(flat, decodeBC7Block(encodeBC7Mode6Block(flat)))).toBeCloseTo(16, 3)
  })

  // A pure two-tone block is exactly representable — the endpoints are the
  // two colours and every index is an extreme. Both seeds agree here, and
  // the early exit taken on a zero-error fit must not change the answer.
  it('encodes a two-tone block at the format floor', () => {
    const twoTone = block(
      Array.from({ length: 16 }, (_, k) => (k < 8 ? ([10, 20, 30, 255] as const) : ([200, 210, 220, 255] as const))),
    )
    expect(blockError(twoTone, decodeASTC4x4Block(encodeASTC4x4Block(twoTone)))).toBeLessThan(1e-6)
    // Same shared-p-bit floor as above.
    expect(blockError(twoTone, decodeBC7Block(encodeBC7Mode6Block(twoTone)))).toBeCloseTo(16, 3)
  })

  // Anti-correlated channels are the case the farthest pair was introduced
  // for. Adding a second seed must not have cost anything there, so both
  // formats should still sit near their interpolation floor: BC7's 16-entry
  // palette covers a 4-step ramp almost exactly, while ASTC's 8-entry one
  // cannot place all four steps on grid.
  it('keeps a clean anti-correlated ramp near the interpolation floor', () => {
    const ramp = block(
      Array.from({ length: 16 }, (_, k) => {
        const t = k & 3
        return [40 + t * 60, 240 - t * 60, 128, 255] as const
      }),
    )
    expect(blockError(ramp, decodeBC7Block(encodeBC7Mode6Block(ramp)))).toBeLessThan(40)
    expect(blockError(ramp, decodeASTC4x4Block(encodeASTC4x4Block(ramp)))).toBeLessThan(1200)
  })

  // Every value finite: a NaN leaking out of the power iteration would show
  // up as a NaN error rather than a large one.
  it('never produces non-finite output', () => {
    for (const b of [SCATTERED_BC7, SCATTERED_ASTC]) {
      expect(Number.isFinite(blockError(b, decodeBC7Block(encodeBC7Mode6Block(b))))).toBe(true)
      expect(Number.isFinite(blockError(b, decodeASTC4x4Block(encodeASTC4x4Block(b))))).toBe(true)
    }
  })
})
