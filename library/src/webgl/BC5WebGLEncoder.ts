// BC5 (RGTC2) WebGL2 fallback encoder. 16 bytes / block, 2-channel linear.
//
// Needs EXT_texture_compression_rgtc to be sampled. Target use: tangent-space
// normal maps (R = normal.x, G = normal.y). No sRGB variant. See
// WebGLBlockEncoder for the pipeline.

import { RED_GREEN_RGTC2_Format } from 'three'

import fragSource from './glsl/bc5.frag.glsl'
import { WebGLBlockEncoder } from './WebGLBlockEncoder.js'

import type { CompressedPixelFormat } from 'three'

export class BC5WebGLEncoder extends WebGLBlockEncoder {
  override get label(): string {
    return 'bc5'
  }
  override get bytesPerBlock(): number {
    return 16
  }
  override get supportsSrgb(): boolean {
    return false
  }
  override fragSource(): string {
    return fragSource
  }
  override threeTextureFormat(): CompressedPixelFormat {
    return RED_GREEN_RGTC2_Format
  }
}
