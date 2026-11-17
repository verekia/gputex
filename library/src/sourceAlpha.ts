// Can an encoded image file carry transparency at all?
//
// Decided from the file's own header — never from pixels — so it costs one
// small read and no pass over the image. `compressTexture` uses a "no" to
// promise the encoder an opaque source (`EncodeOptions.opaque`), which lets
// encoders with an opaque-source kernel skip everything alpha costs. The
// answer is deliberately one-sided: `true` only when the format rules
// transparency out; anything unrecognised, truncated or merely *possibly*
// transparent is `false` ("might have alpha"), which just keeps the general
// encoder path.

const u32be = (b: Uint8Array, o: number): number =>
  ((b[o]! << 24) | (b[o + 1]! << 16) | (b[o + 2]! << 8) | b[o + 3]!) >>> 0

const tag = (b: Uint8Array, o: number): string => String.fromCharCode(b[o]!, b[o + 1]!, b[o + 2]!, b[o + 3]!)

/**
 * True when the leading bytes of an image file prove that every decoded
 * pixel is opaque:
 *   • JPEG — the format has no alpha.
 *   • PNG — colour type 0 / 2 / 3 (gray, RGB, palette) with no `tRNS` chunk
 *     before the image data. Types 4 and 6 have an alpha channel.
 *   • WebP — lossy without the extended header, or with the header's /
 *     lossless stream's alpha flag clear.
 * `bytes` must start at the beginning of the file; a prefix is enough (the
 * PNG check walks chunk headers up to the first `IDAT` and answers `false`
 * if that lies beyond the prefix).
 */
export function imageHeaderIsOpaque(bytes: Uint8Array): boolean {
  const b = bytes
  // JPEG: FF D8 FF.
  if (b.length >= 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return true

  // PNG: 8-byte signature, then IHDR (colour type at byte 25).
  if (b.length >= 33 && b[0] === 0x89 && tag(b, 1) === 'PNG\r' && b[5] === 0x0a && b[6] === 0x1a && b[7] === 0x0a) {
    if (tag(b, 12) !== 'IHDR') return false
    const colourType = b[25]!
    if (colourType !== 0 && colourType !== 2 && colourType !== 3) return false
    // Chunks: length(4) type(4) data(length) crc(4). tRNS, when present,
    // comes before the first IDAT.
    let off = 8
    while (off + 8 <= b.length) {
      const type = tag(b, off + 4)
      if (type === 'tRNS') return false
      if (type === 'IDAT') return true
      off += 12 + u32be(b, off)
    }
    return false
  }

  // WebP: 'RIFF' size 'WEBP' then the first chunk.
  if (b.length >= 30 && tag(b, 0) === 'RIFF' && tag(b, 8) === 'WEBP') {
    const chunk = tag(b, 12)
    // Simple lossy: a bare VP8 key frame has no alpha plane.
    if (chunk === 'VP8 ') return true
    // Extended: flags byte, bit 4 = "contains transparency".
    if (chunk === 'VP8X') return (b[20]! & 0x10) === 0
    // Lossless: signature 0x2F, then 14-bit width, 14-bit height and the
    // alpha_is_used bit (bit 28 of the little-endian header word).
    if (chunk === 'VP8L') return b[20] === 0x2f && (b[24]! & 0x10) === 0
  }
  return false
}

/** Bytes of an image file read for the header check. Covers the chunks an
 *  encoder puts before the pixel data unless an embedded colour profile is
 *  unusually large (then the answer is the safe `false`). */
const HEADER_PREFIX = 64 * 1024

/** `imageHeaderIsOpaque` for a Blob. Never rejects. */
export async function blobIsOpaque(blob: Blob): Promise<boolean> {
  try {
    return imageHeaderIsOpaque(new Uint8Array(await blob.slice(0, HEADER_PREFIX).arrayBuffer()))
  } catch {
    return false
  }
}
