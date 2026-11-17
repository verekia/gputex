// BC4 GPU encoder.
//
// Single-channel linear block compression. 8 bytes per 4×4 block — one BC4
// (RGTC1) block encoding the source's R channel. Target use: grayscale data
// maps (roughness, metalness, ambient occlusion, height, masks). The
// format-selection layer picks it for the 'singleChannel' hint on
// BC-capable adapters; see selectFormat.ts.
//
// No sRGB variant: BC4 is a numeric storage format for 1-channel data.
//
// The algorithm is the single-channel half of the BC5 kernel — a BC4 block
// is byte-identical to the R half of the BC5 block for the same texels. It
// lives in `bc4.wgsl` / `bc4_fast_f16.wgsl`; CPU reference + tests are in
// `bc4_ref.ts`.

import shaderSource from './bc4.wgsl'
import shaderSourceF16 from './bc4_fast_f16.wgsl'
import { Encoder } from './Encoder.js'
import { TextureFormat, WebGPUFeature } from './TextureFormat.js'

export class BC4Encoder extends Encoder {
  static override readonly requiredFeature: GPUFeatureName = WebGPUFeature.BC
  static readonly textureFormats: readonly TextureFormat[] = [TextureFormat.BC4]

  override get label(): string {
    return 'bc4'
  }
  override get bytesPerBlock(): number {
    return 8
  }
  override get supportsSrgb(): boolean {
    return false
  }
  // BC4 reads only R — a one-channel source texture quarters the encode
  // pass's DRAM reads against rgba8 (see BC5Encoder for the measurements
  // behind narrowing the source).
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
    return 'bc4-r-unorm'
  }
}
