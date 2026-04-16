import { generateMipChain, padToBlockMultiple, type MipLevel } from '../mipgen.js'

function makeLevel(w: number, h: number, fill: (x: number, y: number) => [number, number, number, number]): MipLevel {
  const data = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b, a] = fill(x, y)
      const i = (y * w + x) * 4
      data[i] = r
      data[i + 1] = g
      data[i + 2] = b
      data[i + 3] = a
    }
  }
  return { data, width: w, height: h }
}

function getPixel(level: MipLevel, x: number, y: number): [number, number, number, number] {
  const i = (y * level.width + x) * 4
  return [level.data[i]!, level.data[i + 1]!, level.data[i + 2]!, level.data[i + 3]!]
}

describe('generateMipChain', () => {
  it('produces log2(max) + 1 levels for power-of-two inputs', () => {
    const level0 = makeLevel(16, 16, () => [0, 0, 0, 255])
    const chain = generateMipChain(level0)
    // 16, 8, 4, 2, 1 → 5 levels
    expect(chain.length).toBe(5)
    expect(chain.map(l => `${l.width}×${l.height}`)).toEqual(['16×16', '8×8', '4×4', '2×2', '1×1'])
  })

  it('handles non-square power-of-two', () => {
    const level0 = makeLevel(16, 8, () => [0, 0, 0, 255])
    const chain = generateMipChain(level0)
    // 16×8, 8×4, 4×2, 2×1, 1×1 → 5 levels (bounded by max(w,h) = 16)
    expect(chain.map(l => `${l.width}×${l.height}`)).toEqual(['16×8', '8×4', '4×2', '2×1', '1×1'])
  })

  it('handles odd dimensions via clamp-to-edge on the rightmost column', () => {
    // 5×1 image, pixel 4 = white, rest black. Level 1 should be 2×1:
    //   dst[0] = avg(src[0], src[1]) = (0+0)/4 = 0
    //   dst[1] = avg(src[2], src[3]) = 0
    // Level 2 (1×1) averages level 1's two pixels.
    // The rightmost pixel src[4] is clamped into src[3] position for
    // dst-x=2 calculations, but dst only has 2 columns for a 5→2 halving
    // (floor(5/2) = 2). So src[4] isn't actually sampled at level 1;
    // it gets lost. That's the standard CPU-side mip behaviour and the
    // test locks it in.
    const level0 = makeLevel(5, 1, x => [x === 4 ? 255 : 0, 0, 0, 255])
    const chain = generateMipChain(level0)
    // 5×1 → 2×1 → 1×1 = 3 levels. `5 >> 1 = 2`, `2 >> 1 = 1`; the chain
    // stops once both dimensions are 1.
    expect(chain.length).toBe(3)
    expect(chain[1]!.width).toBe(2)
    expect(chain[1]!.height).toBe(1)
    expect(getPixel(chain[1]!, 0, 0)).toEqual([0, 0, 0, 255])
    expect(getPixel(chain[1]!, 1, 0)).toEqual([0, 0, 0, 255])
  })

  it('clamp-to-edge kicks in for odd HEIGHT dimensions', () => {
    // 2×3 image: rows [black, black, white]. Level 1 is 1×1 (height
    // halves to 1 immediately since ceil(3/2)=2 then 1).
    // Actually: 3>>1 = 1, so level 1 is 1×1. Source y0=0, y1=min(1,2)=1.
    // Average of (0,0,0), (0,0,0), (0,0,0), (0,0,0) = black.
    const level0 = makeLevel(2, 3, (_, y) => (y === 2 ? [255, 255, 255, 255] : [0, 0, 0, 255]))
    const chain = generateMipChain(level0)
    expect(chain[1]!.width).toBe(1)
    expect(chain[1]!.height).toBe(1)
    expect(getPixel(chain[1]!, 0, 0)).toEqual([0, 0, 0, 255])
  })

  it('averages four distinct pixels correctly with rounding', () => {
    // 2×2 → 1×1. Corners [10, 20, 30, 40]. Average = (10+20+30+40)/4 = 25.
    // Our integer rule: (a+b+c+d+2) >> 2 = (100+2) >> 2 = 25.
    const level0 = makeLevel(2, 2, (x, y) => {
      const v = [10, 20, 30, 40][y * 2 + x]!
      return [v, v, v, v]
    })
    const chain = generateMipChain(level0)
    expect(chain.length).toBe(2)
    expect(getPixel(chain[1]!, 0, 0)).toEqual([25, 25, 25, 25])
  })

  it('does not drift under repeated averaging of a flat colour', () => {
    // 64×64 of a single colour must survive all 7 mip levels without
    // rounding drift. Catches missing-bias bugs like (a+b+c+d) >> 2.
    const colour: [number, number, number, number] = [127, 64, 32, 200]
    const level0 = makeLevel(64, 64, () => colour)
    const chain = generateMipChain(level0)
    for (const level of chain) {
      for (let y = 0; y < level.height; y++) {
        for (let x = 0; x < level.width; x++) {
          expect(getPixel(level, x, y)).toEqual(colour)
        }
      }
    }
  })

  it('first level is the literal input (not copied)', () => {
    // Saves a full pass — callers that want independence can copy first.
    const level0 = makeLevel(4, 4, () => [1, 2, 3, 4])
    const chain = generateMipChain(level0)
    expect(chain[0]).toBe(level0)
  })

  it('throws on malformed input', () => {
    expect(() => generateMipChain({ data: new Uint8ClampedArray(0), width: 0, height: 1 })).toThrow(/at least 1×1/)
    // Data length doesn't match stated dimensions.
    expect(() => generateMipChain({ data: new Uint8ClampedArray(12), width: 4, height: 4 })).toThrow(/does not match/)
  })
})

describe('padToBlockMultiple', () => {
  it('returns the input unchanged when already block-aligned', () => {
    const level = makeLevel(8, 4, (x, y) => [x, y, 0, 255])
    const padded = padToBlockMultiple(level)
    expect(padded).toBe(level)
  })

  it('pads a 2×2 level up to 4×4 with clamp-to-edge', () => {
    // 2×2 input: ((10,0,0,255), (20,0,0,255) / (30,0,0,255), (40,0,0,255))
    // After padding the extra columns/rows replicate the edge samples.
    const level = makeLevel(2, 2, (x, y) => {
      const v = [10, 20, 30, 40][y * 2 + x]!
      return [v, 0, 0, 255]
    })
    const padded = padToBlockMultiple(level)
    expect(padded.width).toBe(4)
    expect(padded.height).toBe(4)
    // Corner stays.
    expect(getPixel(padded, 0, 0)).toEqual([10, 0, 0, 255])
    // Padded columns clamp from x=1.
    expect(getPixel(padded, 2, 0)).toEqual([20, 0, 0, 255])
    expect(getPixel(padded, 3, 0)).toEqual([20, 0, 0, 255])
    // Padded rows clamp from y=1.
    expect(getPixel(padded, 0, 2)).toEqual([30, 0, 0, 255])
    expect(getPixel(padded, 0, 3)).toEqual([30, 0, 0, 255])
    // Bottom-right corner is the clamped-to-(1,1) sample.
    expect(getPixel(padded, 3, 3)).toEqual([40, 0, 0, 255])
  })

  it('pads a 1×1 level up to 4×4 as a solid block', () => {
    // Covers the sub-4×4 mip-level case: every texel should be the one
    // input pixel, so the encoder sees a flat block and wastes zero
    // palette entries on phantom black edges.
    const level = makeLevel(1, 1, () => [200, 100, 50, 255])
    const padded = padToBlockMultiple(level)
    expect(padded.width).toBe(4)
    expect(padded.height).toBe(4)
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        expect(getPixel(padded, x, y)).toEqual([200, 100, 50, 255])
      }
    }
  })

  it('pads independently in each dimension', () => {
    // 5×3 → 8×4. Columns 5..7 replicate col 4; row 3 replicates row 2.
    const level = makeLevel(5, 3, (x, y) => [x, y, 0, 255])
    const padded = padToBlockMultiple(level)
    expect(padded.width).toBe(8)
    expect(padded.height).toBe(4)
    expect(getPixel(padded, 5, 0)).toEqual([4, 0, 0, 255])
    expect(getPixel(padded, 7, 0)).toEqual([4, 0, 0, 255])
    expect(getPixel(padded, 0, 3)).toEqual([0, 2, 0, 255])
    expect(getPixel(padded, 7, 3)).toEqual([4, 2, 0, 255])
  })
})
