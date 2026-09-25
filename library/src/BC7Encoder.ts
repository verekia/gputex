// BC7 GPU encoder.
//
// BC7 = 16 bytes per block, multi-mode RGBA block compression. This
// implementation picks, per block, between two single-subset modes:
//   • mode 6 — one RGBA line, 16 levels, near-8-bit endpoints: the best
//     single-subset mode when a block's colours lie near one line;
//   • mode 4 — one channel split off into its own scalar plane (a
//     "rotation"), the other three on a line, with 2-bit and 3-bit index
//     sets: decorrelated data (normal maps, channel-packed atlases, noisy
//     photo chroma, independent alpha) where any single 4-D line fails.
// The choice is made from the block's covariance before encoding and both
// modes share the same per-pixel passes, so mixing them costs no extra GPU
// time (×0.98 geomean vs the former mode-6-only kernel on the /eval
// corpus, MSE −33% geomean). Always on — there is no quality toggle. See
// the header of bc7_fast_f16.wgsl for the design and the measurements.
//
// Algorithm lives in `bc7_fast_f16.wgsl` (f16) and `bc7.wgsl` (f32
// fallback); CPU reference + tests are in `bc7_ref.ts` and its test file.

import shaderSource from './bc7.wgsl'
import shaderSourceF16 from './bc7_fast_f16.wgsl'
import { Encoder, type FormatVariant } from './Encoder.js'
import { TextureFormat, WebGPUFeature } from './TextureFormat.js'

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
  override wgslSourceFastF16(): string {
    return shaderSourceF16
  }

  override gpuTextureFormat({ colorSpace }: FormatVariant): GPUTextureFormat {
    return colorSpace === 'srgb' ? 'bc7-rgba-unorm-srgb' : 'bc7-rgba-unorm'
  }
}
