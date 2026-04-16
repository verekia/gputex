// Format-selection layer.
//
// Given a WebGPU adapter (or Set-like feature provider) and a hint about
// how the texture will be used, pick the best compressed format this
// project can produce:
//
//   • `texture-compression-bc` available, 'normal' hint       → BC5
//   • `texture-compression-bc` available, color/colorWithAlpha → BC7
//   • `texture-compression-astc` available (any hint)          → ASTC 4×4
//   • neither                                                  → null (fall back to RGBA8)
//
// BC1 is never selected automatically — it's retained as a reference /
// legacy encoder for the demo, not a modern choice. Callers that want
// BC1 specifically should construct `BC1Encoder` directly.
//
// ASTC has no 2-channel analogue of BC5. For a normal-map hint on the
// ASTC path, the caller has to pre-swizzle (nx, ny) into (R, R, R, G)
// and apply a (R, W, 0, 0) view swizzle in the shader. We surface this
// via `astcNormalRemap: true` so the caller can apply the swizzle
// instead of silently producing a desaturated-looking normal map.

import { ASTC4x4Encoder } from './ASTC4x4Encoder.js'
import { BC5Encoder } from './BC5Encoder.js'
import { BC7Encoder } from './BC7Encoder.js'
import { detectCapabilities, type FeatureProvider } from './capabilities.js'
import { TextureFormat } from './TextureFormat.js'

import type { EncoderConstructor } from './Encoder.js'

/**
 * How the texture will be used in the renderer. Drives format choice.
 *   • 'color'          — RGB albedo-like data (alpha optional / ignored).
 *   • 'colorWithAlpha' — 4-channel RGBA with meaningful alpha.
 *   • 'normal'         — tangent-space normal map (R=x, G=y, z reconstructed).
 */
export type TextureHint = 'color' | 'colorWithAlpha' | 'normal'

export interface SelectFormatOptions {
  /** Pick the sRGB variant when the format has one. Default 'srgb'. */
  colorSpace?: 'srgb' | 'linear'
}

export interface FormatSelection {
  /** null = no compressed path on this adapter; caller should fall back. */
  format: TextureFormat | null
  /** null when `format` is null. */
  encoderClass: EncoderConstructor | null
  /**
   * True when the chosen path is ASTC *and* the intended use is 'normal'.
   * ASTC has no 2-channel mode, so the caller must pre-swizzle the
   * normal map and apply a matching view swizzle in the shader.
   */
  astcNormalRemap: boolean
}

export function selectFormat(
  adapter: FeatureProvider,
  hint: TextureHint,
  options: SelectFormatOptions = {},
): FormatSelection {
  const { colorSpace = 'srgb' } = options
  const srgb = colorSpace === 'srgb'
  const caps = detectCapabilities(adapter)

  // BC path (desktop: Windows, Linux, most discrete-GPU Macs).
  if (caps.bc) {
    if (hint === 'normal') {
      // BC5 is linear-only. No sRGB variant even if the caller asked.
      return { format: TextureFormat.BC5, encoderClass: BC5Encoder, astcNormalRemap: false }
    }
    return {
      format: srgb ? TextureFormat.BC7_SRGB : TextureFormat.BC7,
      encoderClass: BC7Encoder,
      astcNormalRemap: false,
    }
  }

  // ASTC path (mobile / iOS).
  if (caps.astc) {
    const format = srgb ? TextureFormat.ASTC_4x4_SRGB : TextureFormat.ASTC_4x4
    return {
      format,
      encoderClass: ASTC4x4Encoder,
      astcNormalRemap: hint === 'normal',
    }
  }

  // No compressed path — caller falls back to uncompressed RGBA8.
  return { format: null, encoderClass: null, astcNormalRemap: false }
}
