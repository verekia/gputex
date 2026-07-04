// ETC2 RGB8 GPU encoder. The mobile-side analogue of BC1 — 4 bpp, opaque
// RGB. The format-selection layer picks it for `quality: 'low'` colour
// textures on ETC2-capable adapters without BC support (most mobile GPUs),
// and as the last-resort compressed format for opaque colour when neither
// BC nor ASTC is available; see selectFormat.ts.
//
// The only two-pass encoder: a preparation pass splits the RGBA8 source
// into a packed-luma plane plus a half-resolution quadrant-average plane
// (2 bytes/pixel — the encode pass is DRAM-bound and never needed full-res
// chroma), then the encode pass emits ETC1 individual/differential blocks;
// planar, T and H modes are never emitted. See `etc2_prep.wgsl` /
// `etc2.wgsl` / `etc2_ref.ts` for the algorithm and layout. There is no
// f16 variant: the encode is integer-exact 0..255 arithmetic whose error
// sums overflow f16, so the f32 module is the only one.
//
// See `Encoder.ts` for the shared encode pipeline. This file declares the
// format metadata, the WGSL sources and the prepared-plane geometry.

import { Encoder, type FormatVariant } from './Encoder.js'
import shaderSource from './etc2.wgsl'
import prepSource from './etc2_prep.wgsl'
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

  protected override wgslPrepSource(): string {
    return prepSource
  }

  protected override prepPlanes(
    paddedWidth: number,
    paddedHeight: number,
  ): { format: GPUTextureFormat; width: number; height: number }[] {
    return [
      // Packed luma: four 8-bit round((r+g+b)/3) values per r32uint texel.
      { format: 'r32uint', width: paddedWidth / 4, height: paddedHeight },
      // 2×2 quadrant averages.
      { format: 'rgba8unorm', width: paddedWidth / 2, height: paddedHeight / 2 },
    ]
  }

  protected override prepDispatch(blocksX: number, blocksY: number): [number, number] {
    // One prep invocation per 4×2-pixel tile.
    return [blocksX, blocksY * 2]
  }

  override gpuTextureFormat({ colorSpace }: FormatVariant): GPUTextureFormat {
    return colorSpace === 'srgb' ? 'etc2-rgb8unorm-srgb' : 'etc2-rgb8unorm'
  }
}
