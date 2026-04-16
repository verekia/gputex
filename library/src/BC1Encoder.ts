// BC1 (DXT1) GPU encoder. The simplest format in the suite — 4 bpp,
// RGB-only with punch-through alpha. Retained alongside BC5/BC7/ASTC as
// a reference implementation and legacy fallback; the format-selection
// layer never picks it automatically (BC7 is the default desktop
// choice), but it remains available for callers that want BC1 directly.
//
// See `Encoder.ts` for the shared encode pipeline. This file only declares
// the format metadata and loads `bc1.wgsl`.

import { RGBA_S3TC_DXT1_Format } from 'three'

import shaderSource from './bc1.wgsl'
import { Encoder, type FormatVariant } from './Encoder.js'
import { TextureFormat, WebGPUFeature } from './TextureFormat.js'

import type { CompressedPixelFormat } from 'three'

export class BC1Encoder extends Encoder {
  static override readonly requiredFeature: GPUFeatureName = WebGPUFeature.BC
  static readonly textureFormats: readonly TextureFormat[] = [TextureFormat.BC1, TextureFormat.BC1_SRGB]

  override get label(): string {
    return 'bc1'
  }
  override get bytesPerBlock(): number {
    return 8
  }
  override get supportsSrgb(): boolean {
    return true
  }

  override wgslSource(): string {
    return shaderSource
  }

  override gpuTextureFormat({ colorSpace }: FormatVariant): GPUTextureFormat {
    // Three.js's WebGPU backend uses 'bc1-rgba-unorm(-srgb)' for DXT1,
    // even though the shader emits RGB-only blocks with alpha forced to 1.
    return colorSpace === 'srgb' ? 'bc1-rgba-unorm-srgb' : 'bc1-rgba-unorm'
  }

  override threeTextureFormat(): CompressedPixelFormat {
    // Three.js has a single format constant for both sRGB / linear BC1.
    // The sRGB flag is carried separately by `texture.colorSpace`.
    return RGBA_S3TC_DXT1_Format
  }
}
