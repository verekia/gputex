// EAC R11 GPU encoder. The mobile-side analogue of BC4 — 4 bpp, one
// channel, unsigned, 11-bit reconstruction. Part of the ETC2 family, so it
// is sampleable wherever ETC2 is (every WebGPU device without BC, plus
// Apple Silicon). The format-selection layer picks it for the
// 'singleChannel' hint on adapters without BC; see selectFormat.ts.
//
// No sRGB variant: EAC R11 is a numeric storage format for 1-channel data.
//
// See `eac_r11.wgsl` for the algorithm and block layout and
// `eac_r11_ref.ts` for the CPU reference. Both WGSL modules run the same
// exact integer arithmetic (the f16 one holds its per-texel vectors in
// f16, losslessly), so they agree byte for byte — and with the WebGL port
// and the CPU reference — wherever the sampler's unorm conversion is exact.

import shaderSource from './eac_r11.wgsl'
import shaderSourceF16 from './eac_r11_fast_f16.wgsl'
import { Encoder } from './Encoder.js'
import { TextureFormat, WebGPUFeature } from './TextureFormat.js'

export class EACR11Encoder extends Encoder {
  static override readonly requiredFeature: GPUFeatureName = WebGPUFeature.ETC2
  static readonly textureFormats: readonly TextureFormat[] = [TextureFormat.EAC_R11]

  override get label(): string {
    return 'eac-r11'
  }
  override get bytesPerBlock(): number {
    return 8
  }
  override get supportsSrgb(): boolean {
    return false
  }
  // Only R is read — see BC4Encoder.
  protected override get srcTextureFormat(): GPUTextureFormat {
    return 'r8unorm'
  }

  override wgslSource(): string {
    return shaderSource
  }
  override wgslSourceFastF16(): string {
    return shaderSourceF16
  }

  override gpuTextureFormat(): GPUTextureFormat {
    return 'eac-r11unorm'
  }
}
