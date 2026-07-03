// Format-selection layer.
//
// Given a WebGPU adapter (or Set-like feature provider) and a hint about
// how the texture will be used, pick the best compressed format this
// project can produce:
//
//   • `texture-compression-bc` available, 'normal' hint       → BC5
//   • `texture-compression-bc` available, color/colorWithAlpha → BC7
//   • `texture-compression-astc` available (any hint)          → ASTC 4×4
//   • `texture-compression-etc2` available, 'color' hint       → ETC2 RGB8
//   • none of the above                                        → null (fall back to RGBA8)
//
// The `quality` option trades memory for fidelity on opaque colour:
//   • 'high' (default) — the 8-bpp formats above (BC7 / ASTC 4×4).
//   • 'low' — the 4-bpp formats when the adapter has one: BC1 on BC
//     hardware, ETC2 RGB8 on ETC2 hardware (most mobile GPUs). Half the
//     GPU memory of the 'high' choice, at visibly lower quality on smooth
//     content. Only opaque colour can ride these (BC1's punch-through bit
//     and ETC2 RGB8's three channels carry no real alpha, and neither has
//     a 2-channel mode), so 'colorWithAlpha' and 'normal' hints keep the
//     'high' selection — that degradation is documented rather than
//     warned, since `quality` is typically set app-wide.
//
// BC1 is never selected by default — at 0.5 B/px it trades visible
// quality for half of BC7's footprint, a call only the application can
// make via `quality: 'low'` or per-texture via `preferredFormat: 'bc1'`.
// The preference is honoured when the adapter has BC support and the hint
// is 'color' (BC1 carries neither real alpha nor a 2-channel normal map),
// and falls back to the normal selection above otherwise. `BC1Encoder`
// and `ETC2Encoder` also remain directly constructible.
//
// ASTC has no 2-channel analogue of BC5. For a normal-map hint on the
// ASTC path, the caller has to pre-swizzle (nx, ny) into (R, R, R, G)
// and apply a (R, W, 0, 0) view swizzle in the shader. We surface this
// via `astcNormalRemap: true` so the caller can apply the swizzle
// instead of silently producing a desaturated-looking normal map.

import { ASTC4x4Encoder } from './ASTC4x4Encoder.js'
import { BC1Encoder } from './BC1Encoder.js'
import { BC5Encoder } from './BC5Encoder.js'
import { BC7Encoder } from './BC7Encoder.js'
import { detectCapabilities, type FeatureProvider } from './capabilities.js'
import { ETC2Encoder } from './ETC2Encoder.js'
import { TextureFormat } from './TextureFormat.js'

import type { EncoderConstructor } from './Encoder.js'

/**
 * How the texture will be used in the renderer. Drives format choice.
 *   • 'color'          — RGB albedo-like data (alpha optional / ignored).
 *   • 'colorWithAlpha' — 4-channel RGBA with meaningful alpha.
 *   • 'normal'         — tangent-space normal map (R=x, G=y, z reconstructed).
 */
export type TextureHint = 'color' | 'colorWithAlpha' | 'normal'

/**
 * Optional format preference — a wish, not a demand. Applied when the
 * device supports the format and the hint is compatible; otherwise
 * selection proceeds normally (BC7 → ASTC → ETC2 → null). Currently only
 * 'bc1': half the memory of BC7 (0.5 vs 1 byte/pixel) for opaque
 * colour, at visibly lower quality on smooth content.
 */
export type PreferredFormat = 'bc1'

/**
 * Memory/fidelity trade-off for opaque colour textures. 'high' (default)
 * picks the 8-bpp formats (BC7 / ASTC 4×4); 'low' picks the 4-bpp ones
 * (BC1 / ETC2 RGB8) when the adapter supports one. Hints that the 4-bpp
 * formats can't carry ('colorWithAlpha', 'normal') ignore this and keep
 * the 'high' selection.
 */
export type FormatQuality = 'high' | 'low'

export interface SelectFormatOptions {
  /** Pick the sRGB variant when the format has one. Default 'srgb'. */
  colorSpace?: 'srgb' | 'linear'
  /** Prefer a specific format when supported. See `PreferredFormat`. */
  preferredFormat?: PreferredFormat
  /** Memory/fidelity trade-off. See `FormatQuality`. Default 'high'. */
  quality?: FormatQuality
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
  const { colorSpace = 'srgb', preferredFormat, quality = 'high' } = options
  const srgb = colorSpace === 'srgb'
  const caps = detectCapabilities(adapter)

  const bc1: FormatSelection = {
    format: srgb ? TextureFormat.BC1_SRGB : TextureFormat.BC1,
    encoderClass: BC1Encoder,
    astcNormalRemap: false,
  }
  const etc2: FormatSelection = {
    format: srgb ? TextureFormat.ETC2_RGB8_SRGB : TextureFormat.ETC2_RGB8,
    encoderClass: ETC2Encoder,
    astcNormalRemap: false,
  }

  // Explicit BC1 preference: honoured for opaque colour on BC-capable
  // adapters. Non-'color' hints can't ride BC1 (no real alpha, no
  // 2-channel mode), so the preference is ignored with a warning rather
  // than silently degrading the texture. An adapter without BC falls
  // through to the normal selection below.
  if (preferredFormat === 'bc1') {
    if (hint !== 'color') {
      console.warn(
        `[gputex] preferredFormat 'bc1' ignored for hint '${hint}' — BC1 has no real alpha channel and is unsuitable for normal maps.`,
      )
    } else if (caps.bc) {
      return bc1
    }
  }

  // 'low' quality: the 4-bpp opaque-colour formats — BC1 on desktop, ETC2
  // RGB8 on mobile. Alpha and normal hints fall through to the 'high'
  // selection (see the header note).
  if (quality === 'low' && hint === 'color') {
    if (caps.bc) return bc1
    if (caps.etc2) return etc2
  }

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

  // ETC2 last resort: opaque colour only — ETC2 RGB8 carries no alpha and
  // has no 2-channel mode, so other hints fall through to uncompressed
  // (which at least preserves the data).
  if (caps.etc2 && hint === 'color') {
    return etc2
  }

  // No compressed path — caller falls back to uncompressed RGBA8.
  return { format: null, encoderClass: null, astcNormalRemap: false }
}
