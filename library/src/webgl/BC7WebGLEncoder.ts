// BC7 (BPTC) mode-6 WebGL2 fallback encoder. 16 bytes / block, RGBA.
//
// Needs EXT_texture_compression_bptc to be sampled. Default desktop colour
// format on the WebGL path (matches the WebGPU BC7 choice). See
// WebGLBlockEncoder for the pipeline.

import fragSource from './glsl/bc7.frag.glsl'
import { WebGLBlockEncoder } from './WebGLBlockEncoder.js'

export class BC7WebGLEncoder extends WebGLBlockEncoder {
  override get label(): string {
    return 'bc7'
  }
  override get bytesPerBlock(): number {
    return 16
  }
  override get supportsSrgb(): boolean {
    return true
  }
  override fragSource(): string {
    return fragSource
  }
}
