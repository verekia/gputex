// SVG rasterisation.
//
// SVGs are vectors, so before they can be block-compressed they must be
// rasterised to RGBA pixels at some concrete size. The obvious route —
// `createImageBitmap(svgBlob)` — is unsupported in Chromium and Firefox,
// and `createImageBitmap(imgElement)` with an SVG src throws in Firefox.
// The one path that works everywhere is: load the SVG into an
// `HTMLImageElement`, draw it onto a canvas, and read the canvas back.
//
// Two extra wrinkles handled here:
//   • An SVG may have no intrinsic pixel size (only a viewBox, or nothing).
//     Firefox refuses to draw such an image to a canvas at all, and every
//     browser would otherwise pick an arbitrary default (300×150). We
//     therefore rewrite the root <svg> tag with explicit width/height
//     before loading it — which also guarantees the browser rasterises
//     the vector at exactly the requested size (crisp upscaling) instead
//     of scaling a smaller raster.
//   • The caller may want a different raster size than the SVG's intrinsic
//     one (`SvgRasterSize`): a number scales the longest side with aspect
//     preserved; `{ width, height }` rasterises at exactly that size
//     (aspect mismatches follow normal SVG `preserveAspectRatio` rules).
//
// Rasterisation needs `Image`, so this module is main-thread-DOM only —
// workers can't rasterise SVGs (no browser API for it without a DOM).
// The parsing/sizing helpers below are pure string/number functions and
// are unit-tested in `__tests__/svg.test.ts`.

/**
 * Target raster size for an SVG source. A number scales the SVG so its
 * longest side matches (aspect ratio preserved); an object rasterises at
 * exactly that size. When omitted, the SVG's intrinsic size is used
 * (absolute width/height attributes, else the viewBox dimensions).
 */
export type SvgRasterSize = number | { width: number; height: number }

export interface RasterizeSvgOptions {
  /** Target raster size. Default: the SVG's intrinsic size. */
  size?: SvgRasterSize
}

/** Size information extracted from the root `<svg>` tag. */
export interface SvgDimensions {
  /** Absolute `width` attribute in px, or null (missing / relative units). */
  width: number | null
  /** Absolute `height` attribute in px, or null (missing / relative units). */
  height: number | null
  viewBoxWidth: number | null
  viewBoxHeight: number | null
}

// ---------------------------------------------------------------------------
// Source detection
// ---------------------------------------------------------------------------

/** True when a source string is inline SVG markup rather than a URL. */
export function isSvgMarkup(source: string): boolean {
  return source.trimStart().startsWith('<')
}

/** True when a URL's path (query/hash ignored) ends in `.svg`. */
export function hasSvgExtension(url: string): boolean {
  return /\.svg$/i.test(url.split(/[?#]/, 1)[0]!)
}

/**
 * True when a Blob/File holds SVG. Decided by MIME type when the blob has
 * one; a `File` with no type falls back to its filename extension.
 */
export function isSvgBlob(blob: Blob): boolean {
  if (blob.type) {
    return blob.type.split(';', 1)[0]!.trim().toLowerCase() === 'image/svg+xml'
  }
  return typeof File !== 'undefined' && blob instanceof File && hasSvgExtension(blob.name)
}

// ---------------------------------------------------------------------------
// Root-tag parsing (pure, unit-tested)
// ---------------------------------------------------------------------------

// First `<svg` followed by whitespace, `>` or `/` — skips `<svga>`-style
// false positives. Attribute values containing `>` on the root tag would
// break this, but never occur in practice.
const ROOT_TAG_RE = /<svg(?=[\s/>])[^>]*>/

function getAttr(tag: string, name: string): string | null {
  // `\s` before the name keeps `width` from matching `stroke-width`.
  const m = new RegExp(`\\s${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`).exec(tag)
  return m ? (m[1] ?? m[2] ?? '') : null
}

function removeAttr(tag: string, name: string): string {
  return tag.replace(new RegExp(`\\s${name}\\s*=\\s*(?:"[^"]*"|'[^']*')`, 'g'), '')
}

/**
 * Parse a CSS length as absolute pixels. Unitless and `px` values qualify;
 * anything else (`%`, `em`, `pt`, `cm`, …) returns null so the caller falls
 * back to the viewBox.
 */
function parseAbsoluteLength(value: string | null): number | null {
  if (value == null) return null
  const m = /^\s*\+?(\d+(?:\.\d+)?|\.\d+)(?:px)?\s*$/i.exec(value)
  if (!m) return null
  const n = Number(m[1])
  return n > 0 ? n : null
}

/**
 * Extract the size-relevant attributes from an SVG document's root tag.
 * Returns null when the text has no `<svg>` root element at all.
 */
export function parseSvgDimensions(svgText: string): SvgDimensions | null {
  const m = ROOT_TAG_RE.exec(svgText)
  if (!m) return null
  const tag = m[0]
  const dims: SvgDimensions = {
    width: parseAbsoluteLength(getAttr(tag, 'width')),
    height: parseAbsoluteLength(getAttr(tag, 'height')),
    viewBoxWidth: null,
    viewBoxHeight: null,
  }
  const viewBox = getAttr(tag, 'viewBox')
  if (viewBox) {
    const parts = viewBox
      .trim()
      .split(/[\s,]+/)
      .map(Number)
    if (parts.length === 4 && parts.every(Number.isFinite) && parts[2]! > 0 && parts[3]! > 0) {
      dims.viewBoxWidth = parts[2]!
      dims.viewBoxHeight = parts[3]!
    }
  }
  return dims
}

/**
 * Decide the pixel size to rasterise at, from the SVG's parsed dimensions
 * and the caller's `SvgRasterSize`. Intrinsic sizing follows the SVG rules:
 * absolute width/height attributes win; a missing one is derived from the
 * viewBox aspect ratio; with neither attribute the viewBox size is used.
 * Throws when no size can be determined and none was provided.
 */
export function resolveSvgRasterSize(dims: SvgDimensions, size?: SvgRasterSize): { width: number; height: number } {
  if (size !== undefined && typeof size === 'object') {
    const width = Math.round(size.width)
    const height = Math.round(size.height)
    if (!(width >= 1) || !(height >= 1)) {
      throw new Error(`rasterizeSvg: svgSize must be ≥1×1 (got ${size.width}×${size.height})`)
    }
    return { width, height }
  }

  let w = dims.width
  let h = dims.height
  if (dims.viewBoxWidth != null && dims.viewBoxHeight != null) {
    if (w == null && h != null) w = (h * dims.viewBoxWidth) / dims.viewBoxHeight
    else if (h == null && w != null) h = (w * dims.viewBoxHeight) / dims.viewBoxWidth
    else if (w == null && h == null) {
      w = dims.viewBoxWidth
      h = dims.viewBoxHeight
    }
  }

  if (typeof size === 'number') {
    if (!(size >= 1)) {
      throw new Error(`rasterizeSvg: svgSize must be ≥1 (got ${size})`)
    }
    // Aspect from the intrinsic size when known, square otherwise.
    const aspect = w != null && h != null ? w / h : 1
    return aspect >= 1
      ? { width: Math.round(size), height: Math.max(1, Math.round(size / aspect)) }
      : { width: Math.max(1, Math.round(size * aspect)), height: Math.round(size) }
  }

  if (w == null || h == null) {
    throw new Error(
      'rasterizeSvg: the SVG has no intrinsic size (no absolute width/height attributes and no viewBox) — ' +
        'pass svgSize to choose a rasterisation size',
    )
  }
  return { width: Math.max(1, Math.round(w)), height: Math.max(1, Math.round(h)) }
}

/**
 * Rewrite the root `<svg>` tag with explicit pixel width/height. When the
 * tag has no viewBox but did declare an absolute size, a matching viewBox
 * is added so the resize scales the content instead of cropping it.
 */
export function setSvgRootSize(svgText: string, width: number, height: number): string {
  const m = ROOT_TAG_RE.exec(svgText)
  if (!m) {
    throw new Error('rasterizeSvg: no <svg> root element found in source')
  }
  let tag = m[0]
  const origWidth = parseAbsoluteLength(getAttr(tag, 'width'))
  const origHeight = parseAbsoluteLength(getAttr(tag, 'height'))
  const hasViewBox = getAttr(tag, 'viewBox') != null
  tag = removeAttr(removeAttr(tag, 'width'), 'height')
  let inject = ` width="${width}" height="${height}"`
  if (!hasViewBox && origWidth != null && origHeight != null) {
    inject += ` viewBox="0 0 ${origWidth} ${origHeight}"`
  }
  tag = `<svg${inject}${tag.slice('<svg'.length)}`
  return svgText.slice(0, m.index) + tag + svgText.slice(m.index + m[0].length)
}

// ---------------------------------------------------------------------------
// Rasterisation (DOM only)
// ---------------------------------------------------------------------------

/**
 * Rasterise an SVG (markup string or Blob/File) to an `ImageBitmap`.
 *
 * Used automatically by `compressTexture()` for SVG sources; exported for
 * callers driving the core encoders directly — the returned bitmap is a
 * valid `EncoderImageSource`. Main-thread only (needs `Image`).
 */
export async function rasterizeSvg(source: string | Blob, options: RasterizeSvgOptions = {}): Promise<ImageBitmap> {
  if (typeof Image === 'undefined') {
    throw new Error(
      'rasterizeSvg: SVG rasterisation needs a DOM Image element and cannot run in this environment (e.g. a worker)',
    )
  }

  const svgText = typeof source === 'string' ? source : await source.text()
  const dims = parseSvgDimensions(svgText)
  if (!dims) {
    throw new Error('rasterizeSvg: no <svg> root element found in source')
  }
  const { width, height } = resolveSvgRasterSize(dims, options.size)
  const sized = setSvgRootSize(svgText, width, height)

  const url = URL.createObjectURL(new Blob([sized], { type: 'image/svg+xml;charset=utf-8' }))
  try {
    const img = new Image()
    img.decoding = 'async'
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve()
      img.onerror = () => reject(new Error('rasterizeSvg: the browser failed to decode the SVG'))
      img.src = url
    })
    // Flush any async decode before drawing; post-onload failures are benign.
    await img.decode().catch(() => {})

    const canvas: OffscreenCanvas | HTMLCanvasElement =
      typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(width, height)
        : Object.assign(document.createElement('canvas'), { width, height })
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null
    if (!ctx) {
      throw new Error('rasterizeSvg: no 2D context available')
    }
    ctx.drawImage(img, 0, 0, width, height)
    return await createImageBitmap(canvas, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  } finally {
    URL.revokeObjectURL(url)
  }
}
