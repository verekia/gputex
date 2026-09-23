// ETC2 RGB8 GPU encoder. The mobile-side analogue of BC1 — 4 bpp, opaque
// RGB. The format-selection layer picks it for `quality: 'low'` colour
// textures on ETC2-capable adapters without BC support (most mobile GPUs),
// and as the last-resort compressed format for opaque colour when neither
// BC nor ASTC is available; see selectFormat.ts.
//
// Single-pass: the shader emits ETC1 individual/differential blocks and
// ETC2 planar blocks; T and H modes are never emitted. See `etc2.wgsl` /
// `etc2_ref.ts` for the algorithm and layout. Both modules read the source
// through textureGather, so they declare the @binding(3) clamp sampler the
// shared pipeline binds for such shaders. The f16 variant is EXACT-VALUE
// (every f16 quantity is an integer or half f16 represents exactly; the
// sums stay f32), so its output matches the f32 module byte for byte
// wherever the sampler's unorm conversion is exact — it buys register
// space, not different results.
//
// See `Encoder.ts` for the shared encode pipeline. This file only declares
// the format metadata and loads the WGSL sources.

import { Encoder, type FormatVariant } from './Encoder.js'
import shaderSource from './etc2.wgsl'
import shaderSourceF16 from './etc2_fast_f16.wgsl'
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

  override wgslSourceFastF16(): string {
    return shaderSourceF16
  }

  override gpuTextureFormat({ colorSpace }: FormatVariant): GPUTextureFormat {
    return colorSpace === 'srgb' ? 'etc2-rgb8unorm-srgb' : 'etc2-rgb8unorm'
  }
}
