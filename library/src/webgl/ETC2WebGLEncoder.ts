// ETC2 RGB8 WebGL2 fallback encoder. 8 bytes / block, opaque RGB.
//
// Needs WEBGL_compressed_texture_etc to be sampled (mobile GPUs, and Apple
// Silicon through ANGLE). The 'low'-quality pick on ETC2-class devices and
// the opaque-colour last resort after BPTC/ASTC, mirroring the WebGPU side.
// See WebGLBlockEncoder for the pipeline.

import fragSource from './glsl/etc2.frag.glsl'
import { WebGLBlockEncoder } from './WebGLBlockEncoder.js'

export class ETC2WebGLEncoder extends WebGLBlockEncoder {
  override get label(): string {
    return 'etc2'
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
