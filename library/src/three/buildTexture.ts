// Three.js texture assembly for the encoder core.
//
// The core encoders (`../Encoder.ts`, `../webgl/*`) are deliberately Three.js-
// free: they only produce raw block bytes (`encodeToBytes`). This module is the
// Three.js side — it maps a logical `TextureFormat` to the matching Three.js
// `CompressedPixelFormat` constant and wraps encoded levels into a
// `CompressedTexture`. Keeping it out of the core is what lets `gputex` (the
// root entry) be imported with no `three` install; the helpers here live behind
// the `gputex/three` entry instead.

import { RED_GREEN_RGTC2_Format, RGBA_ASTC_4x4_Format, RGBA_BPTC_Format, RGBA_S3TC_DXT1_Format } from 'three'

import { TextureFormat } from '../TextureFormat.js'
import { assembleCompressedTexture, type EncodedLevel } from './textureAssembly.js'

import type { CompressedPixelFormat, CompressedTexture } from 'three'

import type { Encoder, EncoderImageSource, EncodeQuality } from '../Encoder.js'

/**
 * Logical format → Three.js `CompressedPixelFormat`. sRGB and linear variants
 * of a family share one constant; the sRGB flag rides on `texture.colorSpace`
 * (see textureAssembly.ts), so both map to the same value here.
 */
const THREE_FORMAT: Record<TextureFormat, CompressedPixelFormat> = {
  [TextureFormat.BC1]: RGBA_S3TC_DXT1_Format,
  [TextureFormat.BC1_SRGB]: RGBA_S3TC_DXT1_Format,
  [TextureFormat.BC5]: RED_GREEN_RGTC2_Format,
  [TextureFormat.BC7]: RGBA_BPTC_Format,
  [TextureFormat.BC7_SRGB]: RGBA_BPTC_Format,
  [TextureFormat.ASTC_4x4]: RGBA_ASTC_4x4_Format,
  [TextureFormat.ASTC_4x4_SRGB]: RGBA_ASTC_4x4_Format,
}

const SRGB_FORMATS: ReadonlySet<TextureFormat> = new Set([
  TextureFormat.BC1_SRGB,
  TextureFormat.BC7_SRGB,
  TextureFormat.ASTC_4x4_SRGB,
])

/** Whether a logical format is the sRGB variant of its family. */
function isSrgbFormat(format: TextureFormat): boolean {
  return SRGB_FORMATS.has(format)
}

/** The Three.js `CompressedPixelFormat` constant for a logical `TextureFormat`. */
export function threeFormatFor(format: TextureFormat): CompressedPixelFormat {
  return THREE_FORMAT[format]
}

/**
 * Wrap pre-encoded mip levels into a `CompressedTexture` for the given logical
 * format. `levels[0]` is the base level. The texture's colour space is taken
 * from the format variant (sRGB families tag sRGB, everything else linear), so
 * no separate colour-space argument is needed — pick the format variant you
 * want (e.g. `BC7_SRGB` vs `BC7`).
 */
export function buildCompressedTexture(levels: readonly EncodedLevel[], format: TextureFormat): CompressedTexture {
  return assembleCompressedTexture(levels, threeFormatFor(format), isSrgbFormat(format))
}

export interface EncodeResult {
  width: number
  height: number
  paddedWidth: number
  paddedHeight: number
  data: Uint8Array
  texture: CompressedTexture
  encodeMs: number
}

export interface EncodeToTextureOptions {
  /** Pick the sRGB or linear variant of the encoder's format. Default 'srgb'. */
  colorSpace?: 'srgb' | 'linear'
  /** Encode quality / speed trade-off. Default 'fast'. */
  quality?: EncodeQuality
  /** Flip the image vertically before encoding. Default false. */
  flipY?: boolean
}

/**
 * One-shot: encode a single image with a WebGPU `Encoder` straight to a
 * `CompressedTexture`, plus the raw byte metadata. Convenience wrapper over
 * `encoder.encodeToBytes()` + `buildCompressedTexture()` for callers that want
 * a ready-to-use Three.js texture from one call.
 */
export async function encodeToTexture(
  encoder: Encoder,
  source: EncoderImageSource,
  { colorSpace = 'srgb', quality = 'fast', flipY = false }: EncodeToTextureOptions = {},
): Promise<EncodeResult> {
  // Each concrete encoder exposes its logical formats as a static; pick the
  // sRGB or linear variant the caller asked for (BC5 has only the one).
  const formats = (encoder.constructor as unknown as { textureFormats: readonly TextureFormat[] }).textureFormats
  const wantSrgb = colorSpace === 'srgb' && encoder.supportsSrgb
  const format = formats.find(f => isSrgbFormat(f) === wantSrgb) ?? formats[0]!

  const bytes = await encoder.encodeToBytes(source, { flipY, quality })
  const texture = buildCompressedTexture([bytes], format)
  return { ...bytes, texture }
}
