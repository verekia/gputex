// BC5 GPU encoder.
//
// Two-channel linear block compression. 16 bytes per 4×4 block (two
// BC4 halves, for R and G). Target use: tangent-space normal maps
// where R=normal.x, G=normal.y, and Z is reconstructed in the shader.
//
// No sRGB variant: BC5 is a numeric storage format for 2-channel data.
//
// Algorithm lives in `bc5.wgsl`; CPU reference + tests are in
// `bc4_ref.ts`, `bc5_ref.ts`, and their test files.

import { RED_GREEN_RGTC2_Format } from 'three'

import shaderSource from './bc5.wgsl'
import { Encoder } from './Encoder.js'
import { TextureFormat, WebGPUFeature } from './TextureFormat.js'

import type { CompressedPixelFormat } from 'three'

export class BC5Encoder extends Encoder {
  static override readonly requiredFeature: GPUFeatureName = WebGPUFeature.BC
  static readonly textureFormats: readonly TextureFormat[] = [TextureFormat.BC5]

  override get label(): string {
    return 'bc5'
  }
  override get bytesPerBlock(): number {
    return 16
  }
  override get supportsSrgb(): boolean {
    return false
  }

  override wgslSource(): string {
    return shaderSource
  }

  override gpuTextureFormat(): GPUTextureFormat {
    return 'bc5-rg-unorm'
  }

  override threeTextureFormat(): CompressedPixelFormat {
    // Three.js 0.183 maps RED_GREEN_RGTC2_Format → GPUTextureFormat.BC5RGUnorm
    // in the WebGPU CompressedTexture format switch. (0.170 and earlier
    // had the constant but no WebGPU mapping — uploading silently failed
    // with "WebGPURenderer: Unsupported texture format".)
    return RED_GREEN_RGTC2_Format
  }
}
