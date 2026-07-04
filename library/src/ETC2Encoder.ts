// ETC2 RGB8 GPU encoder. The mobile-side analogue of BC1 — 4 bpp, opaque
// RGB. The format-selection layer picks it for `quality: 'low'` colour
// textures on ETC2-capable adapters without BC support (most mobile GPUs),
// and as the last-resort compressed format for opaque colour when neither
// BC nor ASTC is available; see selectFormat.ts.
//
// Single-pass: the shader emits ETC1 individual/differential blocks and
// ETC2 planar blocks; T and H modes are never emitted. See `etc2.wgsl` /
// `etc2_ref.ts` for the algorithm and layout. There is no f16 variant: the
// encode is integer-exact 0..255 arithmetic whose error sums overflow f16,
// so the f32 module is the only one. (A two-pass prepared-source variant
// lives in git history — its prep pass is also bandwidth-bound and made
// the per-texture total slower.)
//
// See `Encoder.ts` for the shared encode pipeline. This file only declares
// the format metadata and loads the WGSL source.

import { Encoder, type FormatVariant } from './Encoder.js'
import shaderSource from './etc2.wgsl'
import { TextureFormat, WebGPUFeature } from './TextureFormat.js'

export class ETC2Encoder extends Encoder {
  static override readonly requiredFeature: GPUFeatureName = WebGPUFeature.ETC2
  static readonly textureFormats: readonly TextureFormat[] = [TextureFormat.ETC2_RGB8, TextureFormat.ETC2_RGB8_SRGB]

  override get label(): string {
    return 'etc2'
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
    return colorSpace === 'srgb' ? 'etc2-rgb8unorm-srgb' : 'etc2-rgb8unorm'
  }
}
