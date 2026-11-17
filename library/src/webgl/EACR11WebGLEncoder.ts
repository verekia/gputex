// EAC R11 WebGL2 fallback encoder. 8 bytes / block, 1-channel linear.
//
// Needs WEBGL_compressed_texture_etc to be sampled (mobile GPUs, and Apple
// Silicon through ANGLE). The 'singleChannel' pick where RGTC is absent,
// mirroring the WebGPU side. No sRGB variant. See WebGLBlockEncoder for the
// pipeline.

import fragSource from './glsl/eac_r11.frag.glsl'
import { WebGLBlockEncoder } from './WebGLBlockEncoder.js'

export class EACR11WebGLEncoder extends WebGLBlockEncoder {
  override get label(): string {
    return 'eac-r11'
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
