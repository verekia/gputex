import { imageHeaderIsOpaque } from '../sourceAlpha.js'

const ascii = (s: string): number[] => [...s].map(c => c.charCodeAt(0))
const u32be = (n: number): number[] => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255]
const chunk = (type: string, data: number[]): number[] => [...u32be(data.length), ...ascii(type), ...data, 0, 0, 0, 0]

/** A PNG prefix: signature, IHDR with the given colour type, the given
 *  ancillary chunks, then the start of an IDAT. */
const png = (colourType: number, chunks: number[][] = [], withIdat = true): Uint8Array =>
  Uint8Array.from([
    0x89,
    ...ascii('PNG\r\n'),
    0x1a,
    0x0a,
    ...chunk('IHDR', [...u32be(4), ...u32be(4), 8, colourType, 0, 0, 0]),
    ...chunks.flat(),
    ...(withIdat ? chunk('IDAT', [1, 2, 3, 4]) : []),
  ])

const webp = (fourcc: string, payload: number[]): Uint8Array =>
  Uint8Array.from([...ascii('RIFF'), 0, 0, 0, 0, ...ascii('WEBP'), ...ascii(fourcc), 0, 0, 0, 0, ...payload])

describe('imageHeaderIsOpaque', () => {
  it('JPEG is always opaque', () => {
    expect(imageHeaderIsOpaque(Uint8Array.from([0xff, 0xd8, 0xff, 0xe0, 0, 16]))).toBe(true)
  })

  it('PNG: gray / RGB / palette without tRNS are opaque', () => {
    expect(imageHeaderIsOpaque(png(0))).toBe(true)
    expect(imageHeaderIsOpaque(png(2))).toBe(true)
    expect(imageHeaderIsOpaque(png(3, [chunk('PLTE', [0, 0, 0, 255, 255, 255])]))).toBe(true)
    // Ancillary chunks before the pixel data are walked over.
    expect(
      imageHeaderIsOpaque(png(2, [chunk('gAMA', [0, 0, 177, 143]), chunk('pHYs', [0, 0, 0, 1, 0, 0, 0, 1, 0])])),
    ).toBe(true)
  })

  it('PNG: an alpha channel or a tRNS chunk means "may be transparent"', () => {
    expect(imageHeaderIsOpaque(png(4))).toBe(false)
    expect(imageHeaderIsOpaque(png(6))).toBe(false)
    expect(imageHeaderIsOpaque(png(2, [chunk('tRNS', [0, 0, 0, 0, 0, 0])]))).toBe(false)
    expect(imageHeaderIsOpaque(png(3, [chunk('PLTE', [0, 0, 0]), chunk('tRNS', [0])]))).toBe(false)
  })

  it('PNG: answers "may be transparent" when the pixel data is beyond the prefix', () => {
    expect(imageHeaderIsOpaque(png(2, [], false))).toBe(false)
    // A chunk whose declared length runs past the prefix (a large iCCP).
    const big = [...u32be(1 << 20), ...ascii('iCCP'), 1, 2, 3]
    expect(imageHeaderIsOpaque(png(2, [big], false))).toBe(false)
    expect(imageHeaderIsOpaque(png(2).subarray(0, 20))).toBe(false)
  })

  it('WebP: lossy is opaque; extended / lossless follow their alpha flag', () => {
    const pad = Array.from({ length: 16 }, () => 0)
    expect(imageHeaderIsOpaque(webp('VP8 ', pad))).toBe(true)
    expect(imageHeaderIsOpaque(webp('VP8X', [0x00, ...pad]))).toBe(true)
    expect(imageHeaderIsOpaque(webp('VP8X', [0x10, ...pad]))).toBe(false)
    // VP8L: signature 0x2F, then width-1 (14 bits), height-1 (14 bits),
    // alpha_is_used (bit 28): byte 4 of the stream, mask 0x10.
    expect(imageHeaderIsOpaque(webp('VP8L', [0x2f, 0xff, 0xff, 0xff, 0x0f, ...pad]))).toBe(true)
    expect(imageHeaderIsOpaque(webp('VP8L', [0x2f, 0xff, 0xff, 0xff, 0x1f, ...pad]))).toBe(false)
  })

  it('anything else is "may be transparent"', () => {
    expect(imageHeaderIsOpaque(new Uint8Array(0))).toBe(false)
    expect(imageHeaderIsOpaque(Uint8Array.from(ascii('GIF89a')))).toBe(false)
    expect(
      imageHeaderIsOpaque(
        Uint8Array.from([0, 0, 0, 0x1c, ...ascii('ftypavif'), ...Array.from({ length: 32 }, () => 0)]),
      ),
    ).toBe(false)
    expect(imageHeaderIsOpaque(Uint8Array.from(ascii('<svg xmlns="http://www.w3.org/2000/svg"/>')))).toBe(false)
  })
})
