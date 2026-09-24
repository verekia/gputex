// Engine-agnostic `compressTextureToBytes()` entry point — the full pipeline,
// with no `three` (or any engine) import. `gputex/three`'s `compressTexture()`
// wraps this and turns the result into a `CompressedTexture`; other engines
// (manacore, Babylon.js, …) build their own texture from `result.levels`.
//
// Ties together:
//   • source loading (URL / Blob / ImageBitmap / HTMLImageElement / ...,
//     with SVG sources rasterised to pixels first — see svg.ts)
//   • capability-based format selection
//   • single-level or full-mip-chain encoding
//   • a three-tier path: WebGPU compute → WebGL2 fragment-shader fallback →
//     uncompressed RGBA8 when neither can produce a sampleable compressed format
//
//   const { levels, format } = await compressTextureToBytes('/cobblestone.avif', {
//     hint: 'color', colorSpace: 'srgb', mipmaps: true,
//   })
//   // build a CompressedTexture from `levels` in your engine
//
// Device ownership (WebGPU): if the caller passes `device`, we reuse it and
// never destroy it. Otherwise consecutive calls share one module-level
// adapter + device + per-format encoder (see the shared-GPU section below);
// `releaseSharedGpuResources()` tears that down. The result's `destroy()`
// only releases resources owned by that call.
//
// The WebGL fallback shares one process-wide WebGL2 context (see
// webgl/webglContext.ts) and runs the *fast* encoders only; the `quality`
// option and `device`/`adapter` options apply to the WebGPU path only.

import { Encoder, type EncodedLevelBytes, type EncoderConstructor } from './Encoder.js'
import { generateGpuMipChain } from './gpuMipgen.js'
import { generateMipChain, padToBlockMultiple, type MipLevel } from './mipgen.js'
import {
  selectFormat,
  type FormatQuality,
  type FormatSelection,
  type PreferredFormat,
  type TextureHint,
} from './selectFormat.js'
import { hasSvgExtension, isSvgBlob, isSvgMarkup, rasterizeSvg, type SvgRasterSize } from './svg.js'
import { buildTranscodeKey, readTranscodeCache, writeTranscodeCache } from './transcodeCache.js'
import { selectWebGLFormat, type WebGLFormatSelection } from './webgl/selectWebGLFormat.js'
import { detectWebGLCapabilities } from './webgl/webglCapabilities.js'
import { getSharedWebGLContext } from './webgl/webglContext.js'
import { needsWriteTextureWorkaround } from './workarounds.js'

import type { TextureFormat } from './TextureFormat.js'
import type { WebGLBlockEncoder, WebGLEncoderConstructor } from './webgl/WebGLBlockEncoder.js'

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
   * supports it; falls back to the normal selection (BC7 → ASTC → ETC2 →
   * RGBA8) when it doesn't. Currently only 'bc1': half the memory of BC7
   * for opaque colour textures, at lower quality. Only honoured with
   * `hint: 'color'` — BC1 can't carry real alpha or normal maps.
   */
  preferredFormat?: PreferredFormat
  /**
   * Memory/fidelity trade-off for opaque colour textures. Default 'high'
   * (BC7 / ASTC 4×4, 1 byte/pixel). 'low' picks the 4-bpp formats when the
   * device has one — BC1 on desktop-class GPUs, ETC2 RGB8 on mobile-class
   * ones — halving GPU memory at visibly lower quality on smooth content.
   * Ignored for `hint: 'colorWithAlpha'` and `hint: 'normal'` (the 4-bpp
   * formats can't carry them). On the WebGL fallback tier only BC1 is
   * available at 'low'.
   */
  quality?: FormatQuality
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
  /** Reuse an existing device (e.g. Three.js's renderer device) instead
   *  of creating a new one. WebGPU path only. When provided, the encoder
   *  never destroys it. */
  device?: GPUDevice
  adapter?: GPUAdapter
  /**
   * Skip the WebGPU tier and encode on the WebGL2 fallback even when WebGPU
   * is available — for testing the fallback path on WebGPU-capable
   * browsers (pair it with three's `new WebGPURenderer({ forceWebGL: true })`
   * to render through WebGL2 as well). `device`/`adapter` are ignored.
   * Default false.
   */
  forceWebGL?: boolean
  /**
   * Keep the compressed bytes in a session-scoped in-memory LRU and reuse
   * them on repeat calls, skipping BOTH the image decode and the encode —
   * the dominant costs. Re-loading a texture later in the session (e.g.
   * two worlds sharing an atlas) becomes a few ms. Keyed by source
   * identity + selected format + encode options; capped at 256 MiB of
   * compressed bytes by default (`setTranscodeCacheLimit()` to tune) and
   * never touches persistent storage. Default false.
   *
   * URL and Blob/File sources get an identity automatically (URL string or
   * content hash). Pixel sources (ImageBitmap, canvas, ImageData) are only
   * cached when `cacheKey` is provided.
   */
  cache?: boolean
  /**
   * Explicit cache identity for the source, overriding the derived one.
   * Use when you already know a stable name (e.g. an asset path) and want
   * to skip content hashing, or to make pixel sources cacheable.
   */
  cacheKey?: string
}

export interface CompressResult {
  /**
   * Encoded compressed mip levels (`levels[0]` is the base level), ready to
   * upload to a compressed texture. Null on the RGBA8 fallback path — use
   * `fallbackBitmap` instead.
   */
  levels: EncodedLevelBytes[] | null
  /**
   * Decoded RGBA8 bitmap, set only when `fallbackUncompressed` (no compressed
   * format was available on either backend). Upload it as a plain RGBA8
   * texture; the caller applies colour space / flipY at the texture level.
   */
  fallbackBitmap: ImageBitmap | null
  /** The compressed format selected, or null when we fell back to RGBA8. */
  format: TextureFormat | null
  /** True iff we fell back to an uncompressed RGBA8 bitmap because no encoder fit. */
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
  /**
   * Wall-clock time to turn the source into decoded RGBA pixels: fetch /
   * base64 decode, image decode, SVG rasterisation. Usually the dominant
   * cost for large images — when a load feels slower than `encodeMs`
   * suggests, this is where the time went.
   */
  decodeMs: number
  /** Wall-clock time of the whole `compressTexture()` call: decode + CPU
   *  mip generation + encode + texture assembly. */
  totalMs: number
  /** True when the result came from the in-memory transcode cache (the
   *  `cache` option) — no decode or encode ran; decodeMs/encodeMs are 0. */
  cacheHit: boolean
}

// ---------------------------------------------------------------------------
// Shared WebGPU device + encoder cache
// ---------------------------------------------------------------------------
//
// When the caller passes neither `device` nor `adapter`, consecutive
// `compressTexture()` calls share one adapter/device and one encoder per
// format. Requesting an adapter + device costs tens of ms and a fresh
// encoder recompiles its shader; paying that once per texture dominated the
// encode for small/medium images. Sharing also lets the encoders' internal
// GPU-resource caches (see Encoder.ts) carry across textures.
//
// The WebGL tier has always worked this way (`getSharedWebGLContext()`);
// this is the WebGPU counterpart.

interface SharedGpu {
  adapter: GPUAdapter
  device: GPUDevice
  encoders: Map<EncoderConstructor, Encoder>
}

/** A resolved WebGPU tier: adapter + a selection guaranteed usable. */
interface GpuTier {
  adapter: GPUAdapter
  shared: SharedGpu | null
  selection: FormatSelection & { format: TextureFormat; encoderClass: EncoderConstructor }
}

/** A resolved WebGL2 tier: context + a selection guaranteed usable. */
interface GlTier {
  gl: WebGL2RenderingContext
  selection: WebGLFormatSelection & {
    format: TextureFormat
    encoderClass: NonNullable<WebGLFormatSelection['encoderClass']>
  }
}

let sharedGpuPromise: Promise<SharedGpu | null> | null = null

/** Features worth having on the shared device when the adapter offers them.
 *  Superset of what any one encoder's `create()` would request, since the
 *  device is shared across formats. */
const SHARED_DEVICE_FEATURES: readonly GPUFeatureName[] = [
  'texture-compression-bc',
  'texture-compression-astc',
  'texture-compression-etc2',
  'shader-f16',
  'timestamp-query',
]

async function createSharedGpu(): Promise<SharedGpu | null> {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) return null
  const requiredFeatures = SHARED_DEVICE_FEATURES.filter(f => adapter.features.has(f))
  const device = await adapter.requestDevice({ requiredFeatures })
  return { adapter, device, encoders: new Map() }
}

/** The shared adapter/device/encoders, created on first use. Null when the
 *  platform has no usable adapter. */
function getSharedGpu(): Promise<SharedGpu | null> {
  if (!sharedGpuPromise) {
    const p = createSharedGpu()
    sharedGpuPromise = p
    p.then(shared => {
      if (!shared) return
      // A lost device (GPU reset, `releaseSharedGpuResources()`, browser
      // reclaim) can't encode; drop this generation so the next call
      // recreates, unless a newer generation already replaced it.
      void shared.device.lost.then(() => {
        shared.encoders.forEach(encoder => encoder.destroy())
        shared.encoders.clear()
        if (sharedGpuPromise === p) sharedGpuPromise = null
      })
    }).catch(() => {
      // Device request failed; allow the next call to retry. The failure
      // itself still propagates to whoever awaited this generation.
      if (sharedGpuPromise === p) sharedGpuPromise = null
    })
  }
  return sharedGpuPromise
}

// One WebGL2 encoder per class on the shared context, reused across
// `compressTexture()` calls like the WebGPU encoders above: creating one
// compiles and links its fragment program (tens of ms), and a live encoder
// keeps its source/target textures for the next same-sized encode. Rebuilt
// when the shared context is replaced (context loss).
let sharedGl: { gl: WebGL2RenderingContext; encoders: Map<unknown, WebGLBlockEncoder> } | null = null

function sharedWebGLEncoder(gl: WebGL2RenderingContext, cls: WebGLEncoderConstructor): WebGLBlockEncoder {
  if (!sharedGl || sharedGl.gl !== gl) {
    sharedGl?.encoders.forEach(e => e.destroy())
    sharedGl = { gl, encoders: new Map() }
  }
  let enc = sharedGl.encoders.get(cls)
  if (!enc) {
    enc = cls.create(gl)
    sharedGl.encoders.set(cls, enc)
  }
  return enc
}

function dropSharedWebGLEncoder(cls: WebGLEncoderConstructor): void {
  const enc = sharedGl?.encoders.get(cls)
  if (!enc) return
  enc.destroy()
  sharedGl!.encoders.delete(cls)
}

/**
 * Destroy the WebGPU device and the WebGPU/WebGL2 encoders that
 * `compressTexture()` shares across calls (the WebGPU ones are created
 * lazily when neither the `device` nor the `adapter` option is passed).
 * Safe to call at any time — in-flight encodes on the shared device will
 * fail, and the next `compressTexture()` call recreates everything. No-op
 * when nothing is cached.
 */
export function releaseSharedGpuResources(): void {
  sharedGl?.encoders.forEach(e => e.destroy())
  sharedGl = null
  const p = sharedGpuPromise
  sharedGpuPromise = null
  void p
    ?.then(shared => {
      if (!shared) return
      shared.encoders.forEach(encoder => encoder.destroy())
      shared.encoders.clear()
      shared.device.destroy()
    })
    .catch(() => {})
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
    // data: URLs are decoded by hand, NOT fetched — Chrome's fetch() of a
    // multi-MB data URL is pathologically slow (>1 s for a base64'd 4K PNG
    // vs ~60 ms decoding it ourselves). Common source in drag-drop / paste
    // flows via FileReader.readAsDataURL and canvas.toDataURL.
    if (/^data:/i.test(source)) {
      const blob = dataUrlToBlob(source)
      if (isSvgBlob(blob)) {
        return rasterizeSvg(blob, { size: svgSize })
      }
      return createImageBitmap(blob, opts)
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
 * Decode an RFC 2397 `data:` URL to a Blob without going through `fetch`.
 * Uses the native `Uint8Array.fromBase64` where available (Chrome 140+,
 * ~4× faster than the `atob` loop on multi-MB payloads).
 */
function dataUrlToBlob(url: string): Blob {
  const comma = url.indexOf(',')
  if (comma < 0) {
    throw new Error('compressTexture: malformed data: URL (no comma)')
  }
  // `data:[<mediatype>][;base64],<data>` — `;base64` is always last.
  const header = url.slice(5, comma)
  const isBase64 = /;base64$/i.test(header)
  const type = header.replace(/;base64$/i, '')
  if (!isBase64) {
    return new Blob([decodeURIComponent(url.slice(comma + 1))], { type })
  }
  const payload = url.slice(comma + 1)
  const fromBase64 = (Uint8Array as unknown as { fromBase64?: (s: string) => Uint8Array<ArrayBuffer> }).fromBase64
  if (fromBase64) {
    return new Blob([fromBase64(payload)], { type })
  }
  const bin = atob(payload)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return new Blob([bytes], { type })
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
// Main entry point
// ---------------------------------------------------------------------------

export async function compressTextureToBytes(
  source: CompressTextureSource,
  options: CompressOptions = {},
): Promise<CompressResult> {
  const {
    hint = 'color',
    preferredFormat,
    quality = 'high',
    colorSpace = 'srgb',
    svgSize,
    flipY = true,
    mipmaps = false,
    cache = false,
    cacheKey,
    device: providedDevice,
    adapter: providedAdapter,
    forceWebGL = false,
  } = options

  const t0 = performance.now()

  // Resolve backend + format BEFORE touching pixels: format selection only
  // needs capabilities, and knowing it first lets a transcode-cache hit
  // skip the image decode and the encode entirely.
  const gpu = forceWebGL ? null : await resolveWebGPU()
  const gl = gpu ? null : resolveWebGL()

  // Transcode cache lookup (opt-in). The key includes the selected format,
  // so entries never cross device classes (BC vs ASTC).
  const activeFormat = gpu?.selection.format ?? gl?.selection.format ?? null
  let transcodeKey: string | null = null
  if (cache && activeFormat) {
    transcodeKey = await buildTranscodeKey(source, cacheKey, {
      format: activeFormat,
      backend: gpu ? 'webgpu' : 'webgl',
      colorSpace,
      flipY,
      mipmaps,
      svgSize,
    })
    if (transcodeKey) {
      const hit = readTranscodeCache(transcodeKey)
      if (hit) {
        return {
          levels: hit.levels as EncodedLevelBytes[],
          fallbackBitmap: null,
          format: hit.format,
          fallbackUncompressed: false,
          backend: gpu ? 'webgpu' : 'webgl',
          astcNormalRemap: (gpu ?? gl)!.selection.astcNormalRemap,
          width: hit.width,
          height: hit.height,
          mipLevels: hit.levels.length,
          encodeMs: 0,
          decodeMs: 0,
          totalMs: performance.now() - t0,
          cacheHit: true,
        }
      }
    }
  }

  const tDecode = performance.now()
  const bitmap = await sourceToBitmap(source, svgSize)
  const decodeMs = performance.now() - tDecode

  // Tier 1: WebGPU compute path. Tier 2: WebGL2 fragment-shader fallback.
  // Tier 3: uncompressed RGBA8.
  if (gpu) return encodeViaWebGPU(gpu)
  const viaWebGL = gl ? encodeViaWebGL(gl) : null
  if (viaWebGL) return viaWebGL

  console.warn(
    '[compressTextureToBytes] No compressed path available (WebGPU and WebGL2 both ' +
      'lack a usable compressed-texture format); returning uncompressed RGBA8.',
  )
  return {
    levels: null,
    fallbackBitmap: bitmap,
    format: null,
    fallbackUncompressed: true,
    backend: 'none',
    astcNormalRemap: false,
    width: bitmap.width,
    height: bitmap.height,
    mipLevels: 1,
    encodeMs: 0,
    decodeMs,
    totalMs: performance.now() - t0,
    cacheHit: false,
  }

  // ----------------------------- WebGPU ------------------------------ //

  /** Resolve the WebGPU tier: adapter + a usable format selection, or null
   *  when this device can't take the compute path. */
  async function resolveWebGPU(): Promise<GpuTier | null> {
    if (!('gpu' in navigator)) return null

    // Resolve the adapter: caller-provided, else the shared one. A provided
    // device without an adapter still needs one for capability detection.
    let shared: SharedGpu | null = null
    let adapter: GPUAdapter | null
    if (providedAdapter) {
      adapter = providedAdapter
    } else if (providedDevice) {
      adapter = await navigator.gpu.requestAdapter()
    } else {
      shared = await getSharedGpu()
      adapter = shared?.adapter ?? null
    }
    if (!adapter) return null

    const selection = selectFormat(adapter, hint, { colorSpace, preferredFormat, quality })
    if (!selection.format || !selection.encoderClass) return null
    return {
      adapter,
      shared,
      selection: { ...selection, format: selection.format, encoderClass: selection.encoderClass },
    }
  }

  /** Encode on the WebGPU tier. The selection is already validated, so any
   *  failure from here throws rather than falling back. */
  async function encodeViaWebGPU({ adapter, shared, selection }: GpuTier): Promise<CompressResult> {
    // Instantiate the encoder. Reuse a caller-provided device; on the shared
    // path reuse (or create and cache) the per-format shared encoder;
    // otherwise (adapter-only callers) have the encoder's `create()` request
    // its own device.
    const EncoderCtor = selection.encoderClass
    let encoder: Encoder
    let sharedEncoder = false
    if (providedDevice) {
      encoder = new EncoderCtor({ device: providedDevice, adapter, ownsDevice: false })
    } else if (shared) {
      sharedEncoder = true
      let cached = shared.encoders.get(EncoderCtor)
      if (!cached) {
        cached = new EncoderCtor({ device: shared.device, adapter: shared.adapter, ownsDevice: false })
        shared.encoders.set(EncoderCtor, cached)
      }
      encoder = cached
    } else {
      encoder = await EncoderCtor.create()
    }
    // Shared encoders outlive this call; their GPU resources are released
    // via releaseSharedGpuResources(), not per-result destroy().
    const destroyEncoder = sharedEncoder ? () => {} : () => encoder.destroy()

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
        if (transcodeKey) {
          writeTranscodeCache(transcodeKey, {
            format: selection.format,
            width: bytes.width,
            height: bytes.height,
            levels: [bytes],
          })
        }
        // bytes are read back and self-contained, so a non-shared encoder is
        // done; shared encoders survive (destroyEncoder is a no-op for those)
        destroyEncoder()
        return {
          levels: [bytes],
          fallbackBitmap: null,
          format: selection.format,
          fallbackUncompressed: false,
          backend: 'webgpu',
          astcNormalRemap: selection.astcNormalRemap,
          width: bytes.width,
          height: bytes.height,
          mipLevels: 1,
          encodeMs: bytes.encodeMs,
          decodeMs,
          totalMs: performance.now() - t0,
          cacheHit: false,
        }
      }

      // Mipped path. On healthy devices the whole chain stays on the GPU:
      // upload the bitmap once, box-filter the levels in a compute pass
      // (gpuMipgen.ts), and encode every level straight from the texture's
      // mip views — no getImageData readback, no CPU filter, no per-level
      // re-uploads. Devices with the broken copyExternalImageToTexture
      // can't take that road and keep the CPU chain; both produce identical
      // bytes (the GPU box filter is integer-exact vs mipgen.ts).
      let chainResult
      if (needsWriteTexture) {
        const level0 = bitmapToMipLevel(bitmap, flipY)
        chainResult = await encoder.encodeMipChainToBytes(generateMipChain(level0).map(padToBlockMultiple))
      } else {
        const chainTex = await generateGpuMipChain(encoder.device, bitmap, { flipY })
        try {
          chainResult = await encoder.encodeMipChainFromTexture(chainTex)
        } finally {
          chainTex.destroy()
        }
      }
      const { levels, encodeMs } = chainResult

      if (transcodeKey) {
        writeTranscodeCache(transcodeKey, {
          format: selection.format,
          width: bitmap.width,
          height: bitmap.height,
          levels,
        })
      }
      destroyEncoder()
      return {
        levels,
        fallbackBitmap: null,
        format: selection.format,
        fallbackUncompressed: false,
        backend: 'webgpu',
        astcNormalRemap: selection.astcNormalRemap,
        width: bitmap.width,
        height: bitmap.height,
        mipLevels: levels.length,
        encodeMs,
        decodeMs,
        totalMs: performance.now() - t0,
        cacheHit: false,
      }
    } catch (e) {
      // Encoder owns a device when we created it; clean up on the error path
      // so we don't leak adapters across retries. (Shared encoders stay —
      // a lost shared device resets itself via its `lost` handler.)
      destroyEncoder()
      throw e
    }
  }

  // ------------------------------ WebGL ------------------------------ //

  /** Resolve the WebGL2 tier: shared context + a usable format selection,
   *  or null when the fallback can't produce a compressed texture. */
  function resolveWebGL(): GlTier | null {
    const gl = getSharedWebGLContext()
    if (!gl) return null

    const caps = detectWebGLCapabilities(gl)
    const selection = selectWebGLFormat(caps, hint, { colorSpace, preferredFormat, quality })
    if (!selection.format || !selection.encoderClass) return null
    return { gl, selection: { ...selection, format: selection.format, encoderClass: selection.encoderClass } }
  }

  /**
   * Encode on the WebGL2 tier. Any encode failure degrades to null
   * (→ uncompressed) rather than throwing — the fallback's job is to keep
   * producing a working texture.
   */
  function encodeViaWebGL({ gl, selection }: GlTier): CompressResult | null {
    let encoder: WebGLBlockEncoder | null = null
    try {
      encoder = sharedWebGLEncoder(gl, selection.encoderClass)
      if (!mipmaps) {
        const bytes = encoder.encodeToBytes(bitmap, { flipY })
        if (transcodeKey) {
          writeTranscodeCache(transcodeKey, {
            format: selection.format,
            width: bytes.width,
            height: bytes.height,
            levels: [bytes],
          })
        }
        return {
          levels: [bytes],
          fallbackBitmap: null,
          format: selection.format,
          fallbackUncompressed: false,
          backend: 'webgl',
          astcNormalRemap: selection.astcNormalRemap,
          width: bytes.width,
          height: bytes.height,
          mipLevels: 1,
          encodeMs: bytes.encodeMs,
          decodeMs,
          totalMs: performance.now() - t0,
          cacheHit: false,
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

      if (transcodeKey) {
        writeTranscodeCache(transcodeKey, {
          format: selection.format,
          width: level0.width,
          height: level0.height,
          levels: encodedLevels,
        })
      }
      return {
        levels: encodedLevels,
        fallbackBitmap: null,
        format: selection.format,
        fallbackUncompressed: false,
        backend: 'webgl',
        astcNormalRemap: selection.astcNormalRemap,
        width: level0.width,
        height: level0.height,
        mipLevels: encodedLevels.length,
        encodeMs: totalEncodeMs,
        decodeMs,
        totalMs: performance.now() - t0,
        cacheHit: false,
      }
    } catch (e) {
      // Don't keep an encoder whose program or state may be broken.
      if (encoder) dropSharedWebGLEncoder(selection.encoderClass)
      console.warn('[compressTextureToBytes] WebGL fallback encode failed; returning uncompressed RGBA8.', e)
      return null
    }
  }
}
