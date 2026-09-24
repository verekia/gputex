// WebGL fallback format selection.
//
// The WebGL analogue of `selectFormat.ts`. Same precedence as the WebGPU side
// — BC7 for colour on desktop, BC5 for normals, ASTC on mobile, ETC2 RGB8 as
// the 'low'-quality pick on ETC2-class devices and the opaque-colour last
// resort — but keyed on WebGL2 compressed-texture extensions, with one
// addition: BC1 (DXT1) is a broadly-available last resort for opaque colour
// when neither BPTC nor ASTC is present (most WebGL2 GPUs expose s3tc even
// when they lack bptc); it ranks ahead of ETC2 there, as on the WebGPU side
// where BC hardware never reaches ETC2. The `preferredFormat: 'bc1'` opt-in
// is honoured here too, keyed on the s3tc extension matching the requested
// colour space.
//
// Degradations vs the WebGPU path:
//   • 'colorWithAlpha' never falls back to BC1 — DXT1 has only 1-bit alpha, so
//     we keep the uncompressed RGBA8 path (which preserves alpha) instead.
//   • 'normal' compresses only via BC5 or ASTC; otherwise uncompressed.

import { TextureFormat } from '../TextureFormat.js'
import { ASTC4x4WebGLEncoder } from './ASTC4x4WebGLEncoder.js'
import { BC1WebGLEncoder } from './BC1WebGLEncoder.js'
import { BC5WebGLEncoder } from './BC5WebGLEncoder.js'
import { BC7WebGLEncoder } from './BC7WebGLEncoder.js'
import { ETC2WebGLEncoder } from './ETC2WebGLEncoder.js'

import type { SelectFormatOptions, TextureHint } from '../selectFormat.js'
import type { WebGLEncoderConstructor } from './WebGLBlockEncoder.js'
import type { WebGLCapabilities } from './webglCapabilities.js'

export interface WebGLFormatSelection {
  /** null = no compressed path available on this WebGL context. */
  format: TextureFormat | null
  /** null when `format` is null. */
  encoderClass: WebGLEncoderConstructor | null
  /** True when the chosen path is ASTC *and* the hint is 'normal' (caller must pre-swizzle). */
  astcNormalRemap: boolean
}

const NONE: WebGLFormatSelection = { format: null, encoderClass: null, astcNormalRemap: false }

export function selectWebGLFormat(
  caps: WebGLCapabilities,
  hint: TextureHint,
  options: SelectFormatOptions = {},
): WebGLFormatSelection {
  const { colorSpace = 'srgb', preferredFormat, quality = 'high' } = options
  const srgb = colorSpace === 'srgb'
  const astc = (astcNormalRemap: boolean): WebGLFormatSelection => ({
    format: srgb ? TextureFormat.ASTC_4x4_SRGB : TextureFormat.ASTC_4x4,
    encoderClass: ASTC4x4WebGLEncoder,
    astcNormalRemap,
  })
  const bc1 = (): WebGLFormatSelection => ({
    format: srgb ? TextureFormat.BC1_SRGB : TextureFormat.BC1,
    encoderClass: BC1WebGLEncoder,
    astcNormalRemap: false,
  })
  const etc2 = (): WebGLFormatSelection => ({
    format: srgb ? TextureFormat.ETC2_RGB8_SRGB : TextureFormat.ETC2_RGB8,
    encoderClass: ETC2WebGLEncoder,
    astcNormalRemap: false,
  })
  const hasBc1 = srgb ? caps.s3tcSrgb : caps.s3tc

  // Explicit BC1 preference — same contract as the WebGPU side: honoured
  // for opaque colour when the s3tc extension matching the colour space is
  // present, ignored (with a warning) for hints BC1 can't carry, and
  // falling through to the normal selection below otherwise.
  if (preferredFormat === 'bc1') {
    if (hint !== 'color') {
      console.warn(
        `[gputex] preferredFormat 'bc1' ignored for hint '${hint}' — BC1 has no real alpha channel and is unsuitable for normal maps.`,
      )
    } else if (hasBc1) {
      return bc1()
    }
  }

  // 'low' quality: the 4-bpp opaque-colour formats — BC1 where s3tc is
  // present, else ETC2 RGB8 — same contract as the WebGPU side.
  if (quality === 'low' && hint === 'color') {
    if (hasBc1) return bc1()
    if (caps.etc) return etc2()
  }

  if (hint === 'normal') {
    // BC5 is linear-only; the colorSpace option is ignored for normals.
    if (caps.rgtc) return { format: TextureFormat.BC5, encoderClass: BC5WebGLEncoder, astcNormalRemap: false }
    if (caps.astc) return astc(true)
    return NONE
  }

  // color / colorWithAlpha
  if (caps.bptc) {
    return {
      format: srgb ? TextureFormat.BC7_SRGB : TextureFormat.BC7,
      encoderClass: BC7WebGLEncoder,
      astcNormalRemap: false,
    }
  }
  if (caps.astc) return astc(false)

  // Last resorts, opaque colour only (neither DXT1's 1-bit alpha nor ETC2
  // RGB8 can carry a real alpha channel): BC1 (needs the s3tc variant
  // matching the colour space), then ETC2 RGB8.
  if (hint === 'color' && hasBc1) return bc1()
  if (hint === 'color' && caps.etc) return etc2()
  return NONE
}
