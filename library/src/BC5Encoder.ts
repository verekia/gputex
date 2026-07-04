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

import shaderSource from './bc5.wgsl'
import shaderSourceF16 from './bc5_fast_f16.wgsl'
import { Encoder } from './Encoder.js'
import { TextureFormat, WebGPUFeature } from './TextureFormat.js'

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
  // BC5 stores only R and G — a two-channel source texture halves the
  // encode pass's DRAM reads. Measured on the pass (/ab 2026-07): −4% on a
  // real 4K normal map, −8..−12% on procedural 2048²/4096², and the raw
  // read floor itself drops 35%, so the win grows as the kernel's ALU
  // shrinks. Also halves source-texture memory during encodes.
  protected override get srcTextureFormat(): GPUTextureFormat {
    return 'rg8unorm'
  }

  override wgslSource(): string {
    return shaderSource
  }
  override wgslSourceFastF16(): string {
    return shaderSourceF16
  }

  override gpuTextureFormat(): GPUTextureFormat {
    return 'bc5-rg-unorm'
  }
}
