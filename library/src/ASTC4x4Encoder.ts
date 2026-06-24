// ASTC 4×4 LDR GPU encoder.
//
// Restricted ASTC subset:
//   • Single partition, no dual-plane
//   • CEM 12 (LDR RGBA direct)
//   • 4×4 weight grid, 2-bit weights (QUANT_4)
//   • 8-bit endpoints (QUANT_256)
// Produces fully valid ASTC 4×4 blocks any conforming decoder accepts;
// much narrower than full ASTC. Target: mobile / iOS WebGPU where the
// `texture-compression-astc` feature is available.
//
// Algorithm lives in `astc4x4.wgsl`; CPU reference + tests are in
// `astc4x4_ref.ts` and its test file. The shader mirrors the CPU ref
// function-for-function so WGSL bugs can be localised by swapping one
// step at a time.

import shaderSource from './astc4x4.wgsl'
import shaderSourceF16 from './astc4x4_fast_f16.wgsl'
import { Encoder, type FormatVariant } from './Encoder.js'
import { TextureFormat, WebGPUFeature } from './TextureFormat.js'

export class ASTC4x4Encoder extends Encoder {
  static override readonly requiredFeature: GPUFeatureName = WebGPUFeature.ASTC
  static readonly textureFormats: readonly TextureFormat[] = [TextureFormat.ASTC_4x4, TextureFormat.ASTC_4x4_SRGB]

  override get label(): string {
    return 'astc4x4'
  }
  override get bytesPerBlock(): number {
    return 16
  }
  override get supportsSrgb(): boolean {
    return true
  }
  override get supportsQuality(): boolean {
    return true
  }

  override wgslSource(): string {
    return shaderSource
  }
  override wgslSourceFastF16(): string {
    return shaderSourceF16
  }

  override gpuTextureFormat({ colorSpace }: FormatVariant): GPUTextureFormat {
    return colorSpace === 'srgb' ? 'astc-4x4-unorm-srgb' : 'astc-4x4-unorm'
  }
}
