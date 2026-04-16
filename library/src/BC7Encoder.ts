// BC7 GPU encoder.
//
// BC7 = 16 bytes per block, multi-mode RGBA block compression. This
// implementation produces mode 6 blocks only (single subset, RGBA,
// 4-bit indices, 7777.1 endpoints). Mode 6 is the strongest BC7 mode
// on smooth content — endpoints are near-8-bit and the 16-entry palette
// tracks a single data-line tightly. It degrades on multi-modal blocks
// (sharp colour seams, 4-D white noise) where mode 1's two-subset
// partitioning would help; we accept that trade-off rather than carry
// the 64-entry partition table and mode-selection logic.
//
// Algorithm lives in `bc7.wgsl`; CPU reference + tests are in
// `bc7_ref.ts` and its test file.

import { RGBA_BPTC_Format } from 'three'

import shaderSource from './bc7.wgsl'
import { Encoder, type FormatVariant } from './Encoder.js'
import { TextureFormat, WebGPUFeature } from './TextureFormat.js'

import type { CompressedPixelFormat } from 'three'

export class BC7Encoder extends Encoder {
  static override readonly requiredFeature: GPUFeatureName = WebGPUFeature.BC
  static readonly textureFormats: readonly TextureFormat[] = [TextureFormat.BC7, TextureFormat.BC7_SRGB]

  override get label(): string {
    return 'bc7'
  }
  override get bytesPerBlock(): number {
    return 16
  }
  override get supportsSrgb(): boolean {
    return true
  }

  override wgslSource(): string {
    return shaderSource
  }

  override gpuTextureFormat({ colorSpace }: FormatVariant): GPUTextureFormat {
    return colorSpace === 'srgb' ? 'bc7-rgba-unorm-srgb' : 'bc7-rgba-unorm'
  }

  override threeTextureFormat(): CompressedPixelFormat {
    // Three.js has one BPTC RGBA constant; sRGB is carried by colorSpace.
    return RGBA_BPTC_Format
  }
}
