// BC1 (DXT1) WebGL2 fallback encoder. 8 bytes / block.
//
// Needs WEBGL_compressed_texture_s3tc to be sampled (s3tc_srgb for the sRGB
// variant). Used as the broadly-available last-resort colour format when
// neither BPTC nor ASTC is present. See WebGLBlockEncoder for the pipeline.

import fragSource from './glsl/bc1.frag.glsl'
import { WebGLBlockEncoder } from './WebGLBlockEncoder.js'

export class BC1WebGLEncoder extends WebGLBlockEncoder {
  override get label(): string {
    return 'bc1'
  }
  override get bytesPerBlock(): number {
    return 8
  }
  override get supportsSrgb(): boolean {
    return true
  }
  override fragSource(): string {
    return fragSource
  }
}
