// ASTC 4×4 LDR WebGL2 fallback encoder. 16 bytes / block, RGBA.
//
// Needs WEBGL_compressed_texture_astc to be sampled. Mobile / Apple colour
// format on the WebGL path. As on the WebGPU side, ASTC has no 2-channel mode,
// so a 'normal' hint requires the caller to pre-swizzle (surfaced via
// `astcNormalRemap`). See WebGLBlockEncoder for the pipeline.

import fragSource from './glsl/astc4x4.frag.glsl'
import { WebGLBlockEncoder } from './WebGLBlockEncoder.js'

export class ASTC4x4WebGLEncoder extends WebGLBlockEncoder {
  override get label(): string {
    return 'astc4x4'
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
