// BC4 (RGTC1) WebGL2 fallback encoder. 8 bytes / block, 1-channel linear.
//
// Needs EXT_texture_compression_rgtc to be sampled. Target use: grayscale
// data maps (the 'singleChannel' hint). No sRGB variant. See
// WebGLBlockEncoder for the pipeline.

import fragSource from './glsl/bc4.frag.glsl'
import { WebGLBlockEncoder } from './WebGLBlockEncoder.js'

export class BC4WebGLEncoder extends WebGLBlockEncoder {
  override get label(): string {
    return 'bc4'
  }
  override get bytesPerBlock(): number {
    return 8
  }
  override get supportsSrgb(): boolean {
    return false
  }
  override fragSource(): string {
    return fragSource
  }
}
