// Public `compressTexture()` entry point.
//
// Ties together:
//   • source loading (URL / Blob / ImageBitmap / HTMLImageElement / ...,
//     with SVG sources rasterised to pixels first — see svg.ts)
//   • capability-based format selection
//   • single-level or full-mip-chain encoding
//   • a three-tier path: WebGPU compute → WebGL2 fragment-shader fallback →
//     uncompressed RGBA8 when neither can produce a sampleable compressed format
//
//   const { texture } = await compressTexture('/cobblestone.avif', {
//     hint: 'color', colorSpace: 'srgb', mipmaps: true,
//   })
//   material.map = texture
//
// Device ownership (WebGPU): if the caller passes `device`, we reuse it and
// never destroy it. Otherwise we request our own adapter + device, tag the
// encoder as owning it, and expose a `destroy()` on the result for cleanup.
//
// The WebGL fallback shares one process-wide WebGL2 context (see
// webgl/webglContext.ts) and runs the *fast* encoders only; the `quality`
// option and `device`/`adapter` options apply to the WebGPU path only.

import { LinearFilter, LinearSRGBColorSpace, RepeatWrapping, SRGBColorSpace, Texture } from 'three'

import { Encoder, type EncodeQuality } from '../Encoder.js'
import { generateMipChain, padToBlockMultiple, type MipLevel } from '../mipgen.js'
import { selectFormat, type PreferredFormat, type TextureHint } from '../selectFormat.js'
import { hasSvgExtension, isSvgBlob, isSvgMarkup, rasterizeSvg, type SvgRasterSize } from '../svg.js'
import { selectWebGLFormat } from '../webgl/selectWebGLFormat.js'
import { detectWebGLCapabilities } from '../webgl/webglCapabilities.js'
import { getSharedWebGLContext } from '../webgl/webglContext.js'
import { needsWriteTextureWorkaround } from '../workarounds.js'
import { buildCompressedTexture } from './buildTexture.js'

import type { CompressedTexture } from 'three'

import type { TextureFormat } from '../TextureFormat.js'

/**
 * Everything `compressTexture()` can take as an image source. A superset
 * of `EncoderImageSource` (see Encoder.ts) that also accepts URL strings
 * and Blob / File objects — the common cases in a web app.
 *
 * SVG works through all of these: a URL to an `.svg` file, a string of
 * inline SVG markup (detected by a leading `<`), an SVG Blob/File, or an
 * HTMLImageElement whose src is SVG. Vector sources are rasterised to RGBA
 * before encoding — see the `svgSize` option.
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
  /**
   * Prefer a specific format over the default choice when the device
   * supports it; falls back to the normal selection (BC7 → ASTC → RGBA8)
   * when it doesn't. Currently only 'bc1': half the memory of BC7 for
   * opaque colour textures, at lower quality. Only honoured with
   * `hint: 'color'` — BC1 can't carry real alpha or normal maps.
   */
  preferredFormat?: PreferredFormat
  /** Pick the sRGB or linear variant of the chosen format. Default 'srgb'. */
  colorSpace?: 'srgb' | 'linear'
  /**
   * Rasterisation size for SVG sources. A number scales the SVG so its
   * longest side matches (aspect ratio preserved); `{ width, height }`
   * rasterises at exactly that size. Default: the SVG's intrinsic size
   * (absolute width/height attributes, else the viewBox dimensions).
   * Ignored for non-SVG sources.
   */
  svgSize?: SvgRasterSize
  /** Flip the image vertically before encoding. Default true (matches Three.js convention). */
  flipY?: boolean
  /** Generate a full mip chain down to 1×1 on the CPU, encode every level. */
  mipmaps?: boolean
  /**
   * Encode quality / speed trade-off. 'fast' (default) is ~2–4× faster for a
   * ≤0.36 dB PSNR cost; 'high' runs the exhaustive search (output identical to
   * the CPU reference encoders; for BC1, a principal-axis seed + iterative
   * refit). No effect on the WebGL fallback (which always uses the fast
   * encoders).
   */
  quality?: EncodeQuality
  /** Reuse an existing device (e.g. Three.js's renderer device) instead
   *  of creating a new one. WebGPU path only. When provided, the encoder
   *  never destroys it. */
  device?: GPUDevice
  adapter?: GPUAdapter
}

export interface CompressResult {
  /** CompressedTexture on a compressed path; Texture on RGBA8 fallback. */
  texture: Texture | CompressedTexture
  /** The compressed format selected, or null when we fell back to RGBA8. */
  format: TextureFormat | null
  /** True iff we returned an uncompressed Texture because no encoder fit. */
  fallbackUncompressed: boolean
  /**
   * Which backend produced the result. 'webgpu' = compute path, 'webgl' =
   * fragment-shader fallback, 'none' = uncompressed RGBA8.
   */
  backend: 'webgpu' | 'webgl' | 'none'
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
 *
 * SVG sources take a separate route (`rasterizeSvg`) because
 * `createImageBitmap` can't decode SVG blobs in Chromium/Firefox and the
 * vector needs an explicit raster size anyway.
 */
async function sourceToBitmap(source: CompressTextureSource, svgSize?: SvgRasterSize): Promise<ImageBitmap> {
  const opts: ImageBitmapOptions = {
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
  }
  if (typeof source === 'string') {
    // A string starting with `<` is inline SVG markup, not a URL.
    if (isSvgMarkup(source)) {
      return rasterizeSvg(source, { size: svgSize })
    }
    const resp = await fetch(source)
    if (!resp.ok) {
      throw new Error(`compressTexture: fetch ${source} failed (${resp.status})`)
    }
    const blob = await resp.blob()
    // Trust the Content-Type, with the URL extension as a fallback for
    // servers that mislabel `.svg` files (text/plain, octet-stream, …).
    if (isSvgBlob(blob) || (!isImageMimeType(blob.type) && hasSvgExtension(source))) {
      return rasterizeSvg(blob, { size: svgSize })
    }
    return createImageBitmap(blob, opts)
  }
  if (source instanceof Blob) {
    if (isSvgBlob(source)) {
      return rasterizeSvg(source, { size: svgSize })
    }
    return createImageBitmap(source, opts)
  }
  if (source instanceof ImageBitmap) {
    return source
  }
  // An <img> holding an SVG must be re-fetched and rasterised: Firefox's
  // `createImageBitmap` rejects SVG image elements outright.
  if (typeof HTMLImageElement !== 'undefined' && source instanceof HTMLImageElement) {
    const src = source.currentSrc || source.src
    if (src && (hasSvgExtension(src) || /^data:image\/svg\+xml/i.test(src))) {
      const resp = await fetch(src)
      if (!resp.ok) {
        throw new Error(`compressTexture: fetch ${src} failed (${resp.status})`)
      }
      return rasterizeSvg(await resp.blob(), { size: svgSize })
    }
  }
  // HTMLImageElement / HTMLCanvasElement / OffscreenCanvas / ImageData all
  // satisfy `createImageBitmap`'s ImageBitmapSource type.
  return createImageBitmap(source as ImageBitmapSource, opts)
}

/** True for MIME types `createImageBitmap` could plausibly decode. */
function isImageMimeType(type: string): boolean {
  return /^image\//i.test(type) && !/svg/i.test(type)
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
 * compressed format is available on either backend. The caller is still
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
    preferredFormat,
    colorSpace = 'srgb',
    svgSize,
    flipY = true,
    mipmaps = false,
    quality = 'fast',
    device: providedDevice,
    adapter: providedAdapter,
  } = options

  const srgb = colorSpace === 'srgb'
  const bitmap = await sourceToBitmap(source, svgSize)

  // Tier 1: WebGPU compute path. Tier 2: WebGL2 fragment-shader fallback.
  // Tier 3: uncompressed RGBA8.
  const viaWebGPU = await encodeViaWebGPU()
  if (viaWebGPU) return viaWebGPU
  const viaWebGL = encodeViaWebGL()
  if (viaWebGL) return viaWebGL

  console.warn(
    '[compressTexture] No compressed path available (WebGPU and WebGL2 both ' +
      'lack a usable compressed-texture format); returning uncompressed RGBA8.',
  )
  const tex = wrapUncompressed(bitmap, srgb, flipY)
  return {
    texture: tex,
    format: null,
    fallbackUncompressed: true,
    backend: 'none',
    astcNormalRemap: false,
    width: bitmap.width,
    height: bitmap.height,
    mipLevels: 1,
    encodeMs: 0,
    destroy: () => {
      tex.dispose()
    },
  }

  // ----------------------------- WebGPU ------------------------------ //

  /** Returns a compressed result, or null if WebGPU can't produce one. */
  async function encodeViaWebGPU(): Promise<CompressResult | null> {
    if (!('gpu' in navigator)) return null
    const adapter = providedAdapter ?? (await navigator.gpu.requestAdapter())
    if (!adapter) return null

    const selection = selectFormat(adapter, hint, { colorSpace, preferredFormat })
    if (!selection.format || !selection.encoderClass) return null

    // Instantiate the encoder. Reuse a caller-provided device, otherwise
    // have the encoder's `create()` request its own (with the needed feature).
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
          bytes = await encoder.encodeToBytes(imageData, { quality })
        } else {
          bytes = await encoder.encodeToBytes(bitmap, { flipY, quality })
        }
        const tex = buildCompressedTexture([bytes], selection.format)
        return {
          texture: tex,
          format: selection.format,
          fallbackUncompressed: false,
          backend: 'webgpu',
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
      // each level as a separate compute dispatch.
      const level0 = bitmapToMipLevel(bitmap, flipY)
      const chain = generateMipChain(level0)

      const encodedLevels = []
      let totalEncodeMs = 0
      for (const level of chain) {
        const padded = padToBlockMultiple(level)
        const imageData = mipLevelToImageData(padded)
        const bytes = await encoder.encodeToBytes(imageData, { quality })
        encodedLevels.push(bytes)
        totalEncodeMs += bytes.encodeMs
      }

      const tex = buildCompressedTexture(encodedLevels, selection.format)
      return {
        texture: tex,
        format: selection.format,
        fallbackUncompressed: false,
        backend: 'webgpu',
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
      // Encoder owns a device when we created it; clean up on the error path
      // so we don't leak adapters across retries.
      encoder.destroy()
      throw e
    }
  }

  // ------------------------------ WebGL ------------------------------ //

  /**
   * Returns a compressed result, or null if WebGL2 can't produce one. Any
   * encode failure degrades to null (→ uncompressed) rather than throwing —
   * the fallback's job is to keep producing a working texture.
   */
  function encodeViaWebGL(): CompressResult | null {
    const gl = getSharedWebGLContext()
    if (!gl) return null

    const caps = detectWebGLCapabilities(gl)
    const selection = selectWebGLFormat(caps, hint, { colorSpace, preferredFormat })
    if (!selection.format || !selection.encoderClass) return null

    const encoder = selection.encoderClass.create(gl)
    try {
      if (!mipmaps) {
        const bytes = encoder.encodeToBytes(bitmap, { flipY })
        const tex = buildCompressedTexture([bytes], selection.format)
        return {
          texture: tex,
          format: selection.format,
          fallbackUncompressed: false,
          backend: 'webgl',
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

      // Mipped path mirrors the WebGPU one: flip baked into level 0, each
      // CPU-box-filtered level padded to a whole block then encoded.
      const level0 = bitmapToMipLevel(bitmap, flipY)
      const chain = generateMipChain(level0)

      const encodedLevels = []
      let totalEncodeMs = 0
      for (const level of chain) {
        const padded = padToBlockMultiple(level)
        const bytes = encoder.encodeToBytes(padded)
        encodedLevels.push(bytes)
        totalEncodeMs += bytes.encodeMs
      }

      const tex = buildCompressedTexture(encodedLevels, selection.format)
      return {
        texture: tex,
        format: selection.format,
        fallbackUncompressed: false,
        backend: 'webgl',
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
      encoder.destroy()
      console.warn('[compressTexture] WebGL fallback encode failed; returning uncompressed RGBA8.', e)
      return null
    }
  }
}
