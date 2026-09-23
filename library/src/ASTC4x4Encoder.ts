// ASTC 4×4 LDR GPU encoder.
//
// Restricted ASTC subset:
//   • Single partition, no dual-plane
//   • Per-block class by content: CEM 0 (luminance, 5-bit weights) for
//     exactly-grayscale opaque blocks; CEM 8 (RGB) for opaque blocks, with
//     4-bit weights + QUANT_192 endpoints on wide-span blocks and 3-bit
//     weights + 8-bit endpoints on small-span ones; CEM 12 (RGBA, 2-bit
//     weights, 8-bit endpoints) otherwise
//   • 4×4 weight grid, plain-bit weights, trit-ISE QUANT_192 endpoints
// Produces fully valid ASTC 4×4 blocks any conforming decoder accepts;
// much narrower than full ASTC. Target: mobile / iOS WebGPU where the
// `texture-compression-astc` feature is available.
//
// Algorithm lives in `astc4x4_fast_f16.wgsl` (f32 fallback:
// `astc4x4.wgsl`); the CPU reference encoder/decoder in `astc4x4_ref.ts`
// covers the same block layouts and is the quality yardstick the GPU
// suite gates against.

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
