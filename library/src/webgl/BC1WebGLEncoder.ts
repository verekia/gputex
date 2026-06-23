// BC1 (DXT1) WebGL2 fallback encoder. 8 bytes / block.
//
// Needs WEBGL_compressed_texture_s3tc to be sampled (s3tc_srgb for the sRGB
// variant). Used as the broadly-available last-resort colour format when
// neither BPTC nor ASTC is present. See WebGLBlockEncoder for the pipeline.

import { RGBA_S3TC_DXT1_Format } from 'three'

import fragSource from './glsl/bc1.frag.glsl'
import { WebGLBlockEncoder } from './WebGLBlockEncoder.js'

import type { CompressedPixelFormat } from 'three'

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
  override threeTextureFormat(): CompressedPixelFormat {
    return RGBA_S3TC_DXT1_Format
  }
}
