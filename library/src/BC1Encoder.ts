// BC1 (DXT1) GPU encoder. The simplest format in the suite — 4 bpp,
// RGB-only with punch-through alpha. Retained alongside BC5/BC7/ASTC as
// a reference implementation and legacy fallback; the format-selection
// layer never picks it automatically (BC7 is the default desktop
// choice), but it remains available for callers that want BC1 directly.
//
// Like the other block encoders it offers two quality levels: 'fast'
// (default — bbox endpoints + one least-squares refit) and 'high'
// (principal-axis seed + iterative refit). See `bc1.wgsl` / `bc1_ref.ts`.
// On devices with shader-f16 the fast path runs the dedicated f16 module
// (`bc1_fast_f16.wgsl`) — BC1 endpoints are 565-quantised anyway, so f16
// loses nothing.
//
// See `Encoder.ts` for the shared encode pipeline. This file only declares
// the format metadata and loads the WGSL sources.

import shaderSource from './bc1.wgsl'
import shaderSourceFastF16 from './bc1_fast_f16.wgsl'
import { Encoder, type FormatVariant } from './Encoder.js'
import { TextureFormat, WebGPUFeature } from './TextureFormat.js'

export class BC1Encoder extends Encoder {
  static override readonly requiredFeature: GPUFeatureName = WebGPUFeature.BC
  static readonly textureFormats: readonly TextureFormat[] = [TextureFormat.BC1, TextureFormat.BC1_SRGB]

  override get label(): string {
    return 'bc1'
  }
  override get bytesPerBlock(): number {
    return 8
  }
  override get supportsSrgb(): boolean {
    return true
  }
  override get supportsQuality(): boolean {
    // The shader declares a `QUALITY_HIGH` pipeline-overridable constant: 'fast'
    // (bbox + one refit) vs 'high' (principal-axis seed + iterative refit).
    return true
  }

  override wgslSource(): string {
    return shaderSource
  }

  override wgslSourceFastF16(): string | null {
    return shaderSourceFastF16
  }

  override gpuTextureFormat({ colorSpace }: FormatVariant): GPUTextureFormat {
    // Three.js's WebGPU backend uses 'bc1-rgba-unorm(-srgb)' for DXT1,
    // even though the shader emits RGB-only blocks with alpha forced to 1.
    return colorSpace === 'srgb' ? 'bc1-rgba-unorm-srgb' : 'bc1-rgba-unorm'
  }
}
