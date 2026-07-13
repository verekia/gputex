// `gputex/three`'s high-level `compressTexture()` — the engine-agnostic
// `compressTextureToBytes()` pipeline (core, no `three` import) plus Three.js
// texture assembly.
//
//   const { texture } = await compressTexture('/cobblestone.avif', {
//     hint: 'color', colorSpace: 'srgb', mipmaps: true,
//   })
//   material.map = texture
//
// Engines other than three call `compressTextureToBytes()` from the root
// `gputex` entry directly and build their own texture from `result.levels`.

import { ClampToEdgeWrapping, LinearFilter, LinearSRGBColorSpace, SRGBColorSpace, Texture } from 'three'

import { compressTextureToBytes } from '../compressTexture.js'
import { buildCompressedTexture } from './buildTexture.js'

import type { CompressedTexture } from 'three'

import type {
  CompressResult as CompressBytesResult,
  CompressOptions,
  CompressTextureSource,
} from '../compressTexture.js'

export type { CompressOptions, CompressTextureSource } from '../compressTexture.js'

/** A `compressTexture()` result: a ready-to-use texture plus the same encode
 *  metadata `compressTextureToBytes()` returns (minus the raw `levels`). */
export interface CompressResult extends Omit<CompressBytesResult, 'levels' | 'fallbackBitmap'> {
  /** `CompressedTexture` on a compressed path; a plain `Texture` on the RGBA8 fallback. */
  texture: Texture | CompressedTexture
  /** Dispose the texture. The shared encoder/device survive — release those
   *  with `releaseSharedGpuResources()`. */
  destroy(): void
}

/**
 * Wrap a bitmap as a plain RGBA8 Three.js `Texture`. Used when no compressed
 * format is available on either backend; same wrap/filter settings as the
 * compressed tiers so borders render identically.
 */
function wrapUncompressed(bitmap: ImageBitmap, srgb: boolean, flipY: boolean): Texture {
  const tex = new Texture(bitmap)
  tex.colorSpace = srgb ? SRGBColorSpace : LinearSRGBColorSpace
  tex.magFilter = LinearFilter
  tex.minFilter = LinearFilter
  tex.wrapS = tex.wrapT = ClampToEdgeWrapping
  tex.generateMipmaps = false
  tex.flipY = flipY
  tex.needsUpdate = true
  return tex
}

export async function compressTexture(
  source: CompressTextureSource,
  options: CompressOptions = {},
): Promise<CompressResult> {
  const { levels, fallbackBitmap, ...rest } = await compressTextureToBytes(source, options)
  const texture =
    fallbackBitmap !== null
      ? wrapUncompressed(fallbackBitmap, options.colorSpace !== 'linear', options.flipY ?? true)
      : buildCompressedTexture(levels!, rest.format!)
  return { ...rest, texture, destroy: () => texture.dispose() }
}
