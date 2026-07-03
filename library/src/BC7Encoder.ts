// BC7 GPU encoder.
//
// BC7 = 16 bytes per block, multi-mode RGBA block compression. This
// implementation produces mode 6 blocks only (single subset, RGBA,
// 4-bit indices, 7777.1 endpoints). Mode 6 is the strongest BC7 mode
// on smooth content — endpoints are near-8-bit and the 16-entry palette
// tracks a single data-line tightly. It degrades on multi-modal blocks
// (sharp colour seams, channel-packed atlases) where mode 1's two-subset
// partitioning would help; a mode 1 candidate was built and measured
// (~+1.3 dB there) but its evaluation cost up to ~3× the compute pass on
// exactly that content, so it was dropped in favour of speed — see the
// postmortem note in bc7_fast_f16.wgsl.
//
// OPT-IN: `adaptiveMode4: true` additionally emits MODE 4 blocks
// (rotation: the worst-explained channel gets its own 3-bit-index scalar
// plane) on decorrelated content — +2.5–2.9 dB on tangent-space normal
// maps, +1.9–2.4 dB on channel-packed atlases, +3 dB on hue-edge-heavy
// content. It is off by default because any GPU warp holding a single
// mode-4 block executes both modes' passes: content that benefits encodes
// 1.4–1.5× slower, and no per-block gate can avoid it (quality and warp
// poisoning scale together — measurements in bc7_fast_f16.wgsl). The
// toggle is a WebGPU pipeline override constant, so the default-off path
// is dead-coded at pipeline creation and costs exactly nothing.
// f16-only: the f32/WebGL fallbacks always encode mode 6.
//
// Algorithm lives in `bc7.wgsl`; CPU reference + tests are in
// `bc7_ref.ts` and its test file.

import shaderSource from './bc7.wgsl'
import shaderSourceF16 from './bc7_fast_f16.wgsl'
import { Encoder, type EncoderOptions, type FormatVariant } from './Encoder.js'
import { TextureFormat, WebGPUFeature } from './TextureFormat.js'

export interface BC7EncoderOptions extends EncoderOptions {
  /**
   * Emit BC7 mode 4 on decorrelated blocks (normal maps, channel-packed
   * atlases): a large quality win there, at up to ~1.5× encode time on
   * exactly that content. See the header note for the measurements.
   */
  adaptiveMode4?: boolean
}

export class BC7Encoder extends Encoder {
  static override readonly requiredFeature: GPUFeatureName = WebGPUFeature.BC
  static readonly textureFormats: readonly TextureFormat[] = [TextureFormat.BC7, TextureFormat.BC7_SRGB]

  private readonly adaptiveMode4: boolean

  constructor(opts: BC7EncoderOptions) {
    // The base constructor builds the pipeline, so stash the flag first
    // via a property on the options object read back in pipelineConstants.
    super(opts)
    this.adaptiveMode4 = opts.adaptiveMode4 === true
    if (this.adaptiveMode4) this._buildPipeline()
  }

  protected override pipelineConstants(): Record<string, number> | undefined {
    // f32 fallback has no mode-4 path (and no such constant).
    if (this.adaptiveMode4 && this._useF16) return { enable_mode4: 1 }
    return undefined
  }

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
