// Shared CompressedTexture assembly.
//
// Both the WebGPU `Encoder` and the WebGL fallback encoders end up with the
// same thing — one or more levels of raw compressed bytes plus a Three.js
// `CompressedPixelFormat` — and wrap them into a `CompressedTexture` with
// identical filter / wrap / colour-space settings. That wrapping lives here so
// the two encoder hierarchies can't drift apart.

import {
  ClampToEdgeWrapping,
  CompressedTexture,
  LinearFilter,
  LinearMipmapLinearFilter,
  LinearSRGBColorSpace,
  SRGBColorSpace,
} from 'three'

import type { CompressedPixelFormat, CompressedTextureMipmap } from 'three'

/** One encoded mip level. The fields both encoder backends already produce. */
export interface EncodedLevel {
  /** Logical (pre-padding) dimensions, surfaced on the texture's userData. */
  width: number
  height: number
  /** Block-aligned dimensions the compressed `data` actually covers. */
  paddedWidth: number
  paddedHeight: number
  data: Uint8Array
}

/**
 * Wrap pre-encoded mip levels into a `CompressedTexture`. `levels[0]` is the
 * base level; its padded dimensions become the texture size. Two or more levels
 * → trilinear min filter; one level → bilinear (no chain to sample).
 *
 * sRGB is carried by `texture.colorSpace`; Three.js's WebGL and WebGPU backends
 * both resolve the sRGB vs linear GPU internal format from it, so the same
 * `threeFormat` constant works for both variants.
 */
export function assembleCompressedTexture(
  levels: readonly EncodedLevel[],
  threeFormat: CompressedPixelFormat,
  effectiveSrgb: boolean,
): CompressedTexture {
  if (levels.length === 0) {
    throw new Error('assembleCompressedTexture: no levels provided')
  }
  const mipmaps: CompressedTextureMipmap[] = levels.map(l => ({
    data: l.data,
    width: l.paddedWidth,
    height: l.paddedHeight,
  }))
  const base = levels[0]!
  const texture = new CompressedTexture(mipmaps, base.paddedWidth, base.paddedHeight, threeFormat)
  texture.colorSpace = effectiveSrgb ? SRGBColorSpace : LinearSRGBColorSpace
  texture.magFilter = LinearFilter
  texture.minFilter = levels.length > 1 ? LinearMipmapLinearFilter : LinearFilter
  texture.generateMipmaps = false
  // Clamp, not repeat (matches three.js's own default): with linear filtering,
  // repeat wrapping blends UV-0/1 border samples with the OPPOSITE edge of the
  // image — a 1-px halo of foreign colours around a plane — and on NPOT
  // sources it would bleed the block-padding strip into the far borders too.
  // Callers that tile can still set RepeatWrapping on the returned texture.
  texture.wrapS = texture.wrapT = ClampToEdgeWrapping
  texture.needsUpdate = true
  texture.userData.logicalWidth = base.width
  texture.userData.logicalHeight = base.height
  texture.userData.mipLevels = levels.length
  return texture
}
