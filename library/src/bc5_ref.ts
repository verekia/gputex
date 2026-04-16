// BC5 (RGTC2) reference encoder + decoder. CPU implementation.
//
// BC5 is just two concatenated BC4 blocks: one for R (bytes 0..7), one
// for G (bytes 8..15), 16 bytes total per 4×4 tile. No shared metadata,
// no mode bits — it's literally byte-wise concatenation.
//
// We use BC5 for tangent-space normal maps: R encodes normal.x, G
// encodes normal.y, and Z is reconstructed in the shader via
//   z = sqrt(1 - x² - y²)
// BC5's two-channel linear storage (no sRGB variant) is exactly what
// this use case wants.

import { encodeBC4Block, decodeBC4Block, type BC4Values } from './bc4_ref.js'

/** Exactly 16 bytes of BC5 block data. */
export type BC5Block = Uint8Array

export interface BC5Decoded {
  r: Float32Array
  g: Float32Array
}

export function encodeBC5Block(r: BC4Values, g: BC4Values): BC5Block {
  if (r.length !== 16 || g.length !== 16) {
    throw new Error(`encodeBC5Block: expected two arrays of 16, got ${r.length} / ${g.length}`)
  }
  const out = new Uint8Array(16)
  out.set(encodeBC4Block(r), 0)
  out.set(encodeBC4Block(g), 8)
  return out
}

export function decodeBC5Block(block: BC5Block): BC5Decoded {
  if (block.length !== 16) {
    throw new Error(`decodeBC5Block: expected 16 bytes, got ${block.length}`)
  }
  return {
    r: decodeBC4Block(block.subarray(0, 8)),
    g: decodeBC4Block(block.subarray(8, 16)),
  }
}
