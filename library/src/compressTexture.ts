// Public `compressTexture()` entry point.
//
// Ties together:
//   • source loading (URL / Blob / ImageBitmap / HTMLImageElement / ...)
//   • capability-based format selection
//   • single-level or full-mip-chain encoding
//   • uncompressed RGBA8 fallback when no compressed format is available
//
//   const { texture } = await compressTexture('/cobblestone.avif', {
//     hint: 'color', colorSpace: 'srgb', mipmaps: true,
//   })
//   material.map = texture
//
// Device ownership: if the caller passes `device`, we reuse it and never
// destroy it. Otherwise we request our own adapter + device, tag the
// encoder as owning it, and expose a `destroy()` on the result for
// cleanup. (See `CompressResult.destroy`.)

import { LinearFilter, LinearSRGBColorSpace, RepeatWrapping, SRGBColorSpace, Texture } from 'three'

import { Encoder } from './Encoder.js'
import { generateMipChain, padToBlockMultiple, type MipLevel } from './mipgen.js'
import { selectFormat, type TextureHint } from './selectFormat.js'
import { needsWriteTextureWorkaround } from './workarounds.js'

import type { CompressedTexture } from 'three'

import type { TextureFormat } from './TextureFormat.js'

/**
 * Everything `compressTexture()` can take as an image source. A superset
 * of `EncoderImageSource` (see Encoder.ts) that also accepts URL strings
 * and Blob / File objects — the common cases in a web app.
 */
export type CompressTextureSource =
  | string
  | Blob
  | File
  | ImageBitmap
  | HTMLImageElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | ImageData

export interface CompressOptions {
  /** How the texture will be used. Drives format selection. Default 'color'. */
  hint?: TextureHint
  /** Pick the sRGB or linear variant of the chosen format. Default 'srgb'. */
  colorSpace?: 'srgb' | 'linear'
  /** Flip the image vertically before encoding. Default true (matches Three.js convention). */
  flipY?: boolean
  /** Generate a full mip chain down to 1×1 on the CPU, encode every level. */
  mipmaps?: boolean
  /** Reuse an existing device (e.g. Three.js's renderer device) instead
   *  of creating a new one. When provided, the encoder never destroys it. */
  device?: GPUDevice
  adapter?: GPUAdapter
}

export interface CompressResult {
  /** CompressedTexture on the compressed path; Texture on RGBA8 fallback. */
  texture: Texture | CompressedTexture
  /** The compressed format selected, or null when we fell back to RGBA8. */
  format: TextureFormat | null
  /** True iff we returned an uncompressed Texture because no encoder fit. */
  fallbackUncompressed: boolean
  /**
   * True iff the chosen format is ASTC and the hint was 'normal'. The
   * caller must apply the (R, W) → (x, y) swizzle in the material — ASTC
   * has no 2-channel mode, so normal maps ride the RGBA path.
   */
  astcNormalRemap: boolean
  width: number
  height: number
  mipLevels: number
  /** Wall-clock time of GPU encoding, summed across mip levels. */
  encodeMs: number
  /** Release the encoder's internal GPU resources. No-op if `device` was
   *  passed in by the caller. */
  destroy(): void
}

// ---------------------------------------------------------------------------
// Source loading
// ---------------------------------------------------------------------------

/**
 * Normalise any `CompressTextureSource` to an ImageBitmap. Keeps the rest
 * of the pipeline narrow — one input type, one width/height contract.
 *
 * We pass `colorSpaceConversion: 'none'` + `premultiplyAlpha: 'none'` to
 * preserve the source bytes verbatim; the sRGB / alpha handling is done
 * downstream via the texture's `colorSpace` tag.
 */
async function sourceToBitmap(source: CompressTextureSource): Promise<ImageBitmap> {
  const opts: ImageBitmapOptions = {
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
  }
  if (typeof source === 'string') {
    const resp = await fetch(source)
    if (!resp.ok) {
      throw new Error(`compressTexture: fetch ${source} failed (${resp.status})`)
    }
    const blob = await resp.blob()
    return createImageBitmap(blob, opts)
  }
  if (source instanceof Blob) {
    return createImageBitmap(source, opts)
  }
  if (source instanceof ImageBitmap) {
    return source
  }
  // HTMLImageElement / HTMLCanvasElement / OffscreenCanvas / ImageData all
  // satisfy `createImageBitmap`'s ImageBitmapSource type.
  return createImageBitmap(source as ImageBitmapSource, opts)
}

/**
 * Rasterise an ImageBitmap into a level-0 MipLevel (RGBA8 pixel bytes).
 * Needed because `generateMipChain` works on CPU pixel data, not on
 * GPU-hosted bitmaps.
 *
 * Uses an OffscreenCanvas when available (workers + modern browsers) and
 * falls back to a detached HTMLCanvasElement for older contexts.
 */
function bitmapToMipLevel(bitmap: ImageBitmap, flipY: boolean): MipLevel {
  const w = bitmap.width,
    h = bitmap.height
  const canvas: OffscreenCanvas | HTMLCanvasElement =
    typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(w, h)
      : Object.assign(document.createElement('canvas'), { width: w, height: h })
  // `willReadFrequently` keeps Chrome from uploading to the GPU just to
  // immediately `getImageData` back — saves a full bitmap round-trip.
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D
    | null
  if (!ctx) {
    throw new Error('compressTexture: no 2D context available for mip generation')
  }
  if (flipY) {
    ctx.translate(0, h)
    ctx.scale(1, -1)
  }
  ctx.drawImage(bitmap, 0, 0)
  const imageData = ctx.getImageData(0, 0, w, h)
  return { data: imageData.data, width: w, height: h }
}

/**
 * Build an ImageData from a MipLevel. Uses the `ImageData` constructor
 * directly rather than round-tripping through a canvas, so we don't re-
 * quantise values that are already 8-bit.
 *
 * The cast works around a TypeScript DOM-lib quirk: `ImageData`'s
 * constructor overload expects `Uint8ClampedArray<ArrayBuffer>` (not
 * `ArrayBufferLike`), which our plain `Uint8ClampedArray` technically
 * isn't — at runtime the constructor accepts either.
 */
function mipLevelToImageData(level: MipLevel): ImageData {
  return new ImageData(level.data as Uint8ClampedArray<ArrayBuffer>, level.width, level.height)
}

// ---------------------------------------------------------------------------
// Uncompressed fallback
// ---------------------------------------------------------------------------

/**
 * Wrap a bitmap as a plain RGBA8 Three.js `Texture`. Used when no
 * compressed format is available on the adapter. The caller is still
 * expected to dispose this texture like any other.
 */
function wrapUncompressed(bitmap: ImageBitmap, srgb: boolean, flipY: boolean): Texture {
  const tex = new Texture(bitmap)
  tex.colorSpace = srgb ? SRGBColorSpace : LinearSRGBColorSpace
  tex.magFilter = LinearFilter
  tex.minFilter = LinearFilter
  tex.wrapS = tex.wrapT = RepeatWrapping
  tex.generateMipmaps = false
  tex.flipY = flipY
  tex.needsUpdate = true
  return tex
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export async function compressTexture(
  source: CompressTextureSource,
  options: CompressOptions = {},
): Promise<CompressResult> {
  const {
    hint = 'color',
    colorSpace = 'srgb',
    flipY = true,
    mipmaps = false,
    device: providedDevice,
    adapter: providedAdapter,
  } = options

  const srgb = colorSpace === 'srgb'

  const bitmap = await sourceToBitmap(source)

  // Resolve adapter / device. We need an adapter for capability detection
  // regardless of whether the user passed in a device.
  if (!('gpu' in navigator)) {
    // Can't even inspect capabilities; fall straight back to RGBA8.
    console.warn('[compressTexture] WebGPU unavailable; returning uncompressed RGBA8.')
    const tex = wrapUncompressed(bitmap, srgb, flipY)
    return {
      texture: tex,
      format: null,
      fallbackUncompressed: true,
      astcNormalRemap: false,
      width: bitmap.width,
      height: bitmap.height,
      mipLevels: 1,
      encodeMs: 0,
      destroy: () => {
        tex.dispose()
      },
    }
  }

  const adapter = providedAdapter ?? (await navigator.gpu.requestAdapter())
  if (!adapter) {
    console.warn('[compressTexture] No WebGPU adapter; returning uncompressed RGBA8.')
    const tex = wrapUncompressed(bitmap, srgb, flipY)
    return {
      texture: tex,
      format: null,
      fallbackUncompressed: true,
      astcNormalRemap: false,
      width: bitmap.width,
      height: bitmap.height,
      mipLevels: 1,
      encodeMs: 0,
      destroy: () => {
        tex.dispose()
      },
    }
  }

  const selection = selectFormat(adapter, hint, { colorSpace })
  if (!selection.format || !selection.encoderClass) {
    console.warn(
      '[compressTexture] Adapter reports neither texture-compression-bc nor ' +
        'texture-compression-astc; returning uncompressed RGBA8.',
    )
    const tex = wrapUncompressed(bitmap, srgb, flipY)
    return {
      texture: tex,
      format: null,
      fallbackUncompressed: true,
      astcNormalRemap: false,
      width: bitmap.width,
      height: bitmap.height,
      mipLevels: 1,
      encodeMs: 0,
      destroy: () => {
        tex.dispose()
      },
    }
  }

  // Instantiate the encoder. Two paths:
  //   • User provided a device → reuse it.
  //   • No device provided → have the encoder's `create()` request its
  //     own, including the needed feature flag.
  let encoder: Encoder
  if (providedDevice) {
    const EncoderCtor = selection.encoderClass
    encoder = new EncoderCtor({ device: providedDevice, adapter, ownsDevice: false })
  } else {
    encoder = await selection.encoderClass.create()
  }

  try {
    const needsWriteTexture = needsWriteTextureWorkaround(adapter)

    if (!mipmaps) {
      let bytes
      if (needsWriteTexture) {
        const level0 = bitmapToMipLevel(bitmap, flipY)
        const imageData = mipLevelToImageData(level0)
        bytes = await encoder.encodeToBytes(imageData)
      } else {
        bytes = await encoder.encodeToBytes(bitmap, { flipY })
      }
      const tex = encoder.buildMippedTexture([bytes], { colorSpace })
      return {
        texture: tex,
        format: selection.format,
        fallbackUncompressed: false,
        astcNormalRemap: selection.astcNormalRemap,
        width: bytes.width,
        height: bytes.height,
        mipLevels: 1,
        encodeMs: bytes.encodeMs,
        destroy: () => {
          tex.dispose()
          encoder.destroy()
        },
      }
    }

    // Mipped path. Rasterise once, box-filter the chain on CPU, encode
    // each level as a separate compute dispatch. The encoder already
    // knows how to pad non-block-aligned inputs, but sub-4×4 mip levels
    // (where both dims < 4) need an explicit clamp-to-edge pad so the
    // block doesn't gain phantom edge-colour bands.
    const level0 = bitmapToMipLevel(bitmap, flipY)
    const chain = generateMipChain(level0)

    const encodedLevels = []
    let totalEncodeMs = 0
    for (const level of chain) {
      // Pad sub-4 dims up to 4 with clamp-to-edge. `padToBlockMultiple`
      // is a no-op when the level is already block-aligned.
      const padded = padToBlockMultiple(level)
      const imageData = mipLevelToImageData(padded)
      const bytes = await encoder.encodeToBytes(imageData)
      encodedLevels.push(bytes)
      totalEncodeMs += bytes.encodeMs
    }

    const tex = encoder.buildMippedTexture(encodedLevels, { colorSpace })
    return {
      texture: tex,
      format: selection.format,
      fallbackUncompressed: false,
      astcNormalRemap: selection.astcNormalRemap,
      width: level0.width,
      height: level0.height,
      mipLevels: encodedLevels.length,
      encodeMs: totalEncodeMs,
      destroy: () => {
        tex.dispose()
        encoder.destroy()
      },
    }
  } catch (e) {
    // Encoder owns a device when we created it; make sure it's cleaned
    // up on the error path so we don't leak adapters across retries.
    encoder.destroy()
    throw e
  }
}
