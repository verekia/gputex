// GPU encoder test + benchmark suite (WebGPU only).
//
// Runs in a real browser against real GPU hardware — this is the validation
// harness for the WGSL encoding shaders that the bun unit tests (CPU-only)
// cannot provide. Driven by example/pages/test.tsx and automatable: the page
// exposes `window.__GPUTEX_TESTS__` with a status flag and the full results.
//
// Three groups:
//   1. Correctness — `quality: 'high'` GPU output is compared byte-for-byte
//      against the CPU reference encoders (gputex/testing) on real image
//      content, including a non-multiple-of-4 size to cover the clamp-to-edge
//      padding path, plus determinism (same input twice → same bytes).
//   2. Quality — `quality: 'fast'` (and 'high') output is CPU-decoded and its
//      PSNR vs the source must beat a per-format threshold. Thresholds are
//      set ~0.1 dB under the measured baseline so any perf tweak that hurts
//      quality fails the suite.
//   3. Performance — median wall-clock encode time and (via timestamp
//      queries) GPU compute-pass time per format × quality on a 2048×2048
//      image, plus the f32 fast fallback (forced with `disableF16`) so both
//      shader variants stay measured on f16 hardware.

import { ASTC4x4Encoder, BC1Encoder, BC5Encoder, BC7Encoder } from 'gputex'
import type { EncodeQuality, Encoder } from 'gputex'

import {
  decodeASTC4x4Block,
  decodeBC1Block,
  decodeBC5Block,
  decodeBC7Block,
  encodeASTC4x4Block,
  encodeBC1Block,
  encodeBC5Block,
  encodeBC7Mode6Block,
} from 'gputex/testing'

export type FormatKey = 'bc1' | 'bc5' | 'bc7' | 'astc'

export interface CorrectnessResult {
  name: string
  format: FormatKey
  pass: boolean
  detail: string
}

export interface QualityResult {
  format: FormatKey
  quality: EncodeQuality
  variant: 'f16' | 'f32'
  image: string
  psnrDb: number
  thresholdDb: number
  /**
   * Worst EASY block: max over blocks that 'high' encodes near-losslessly
   * (SSE ≤ 0.05) of SSE(this encode) − SSE(high), decoded, normalised
   * units. Localized artifacts on easy content (e.g. a flat tile turning
   * the wrong colour) barely move aggregate PSNR but explode this metric;
   * genuinely hard blocks (noise), where fast may legitimately trail high,
   * are excluded. Also reported: the overall worst excess, for context.
   */
  worstEasyBlockExcess: number
  worstBlockExcess: number
  excessLimit: number | null
  pass: boolean
}

export interface PerfResult {
  format: FormatKey
  quality: EncodeQuality
  variant: 'f16' | 'f32'
  image: string
  wallMsMedian: number
  gpuMsMedian: number | null
  mpixPerSec: number
}

export interface SuiteResults {
  env: {
    vendor: string
    architecture: string
    description: string
    features: string[]
    hasF16: boolean
    userAgent: string
  }
  correctness: CorrectnessResult[]
  quality: QualityResult[]
  perf: PerfResult[]
  failures: number
}

export type ProgressFn = (message: string) => void

// ---------------------------------------------------------------------------
// PSNR thresholds (dB), measured on the committed test textures. Set ~0.1 dB
// under the observed baseline so quality regressions fail loudly while normal
// cross-GPU float jitter passes. `null` = record only (used while baselining).
// ---------------------------------------------------------------------------
const PSNR_THRESHOLDS: Record<string, number | null> = {
  // `${format}:${quality}:${image}` — measured on the FULL 512² committed test
  // textures (2026-07, Apple/metal-3: 27.36/32.02, 45.99/46.60, 31.27/33.69,
  // 28.38/33.20, alpha 34.58/32.84) minus ~0.15 dB. `null` = record only
  // (used while baselining a change).
  'bc1:fast:color': 27.2,
  'bc1:high:color': 31.85,
  'bc5:fast:normal': 45.8,
  'bc5:high:normal': 46.45,
  'bc7:fast:color': 31.1,
  'bc7:high:color': 33.5,
  'astc:fast:color': 28.2,
  'astc:high:color': 33.05,
  'bc7:fast:alpha': 34.4,
  'astc:fast:alpha': 32.7,
}

// Worst-EASY-block gate for the fast paths: over blocks that 'high' encodes
// near-losslessly (SSE ≤ EASY_BLOCK_SSE), max of SSE(fast) − SSE(high). A
// fast path must never lose badly on content that is easy to encode — the
// 2026-07 BC1 rank-1 refit bug turned FLAT tiles (high SSE ≈ 0.001) into the
// wrong colour entirely (fast SSE ≈ 3.0). Hard blocks (the noise-checker
// quadrant) are excluded: there the bbox-seeded fast path legitimately
// trails the exhaustive search by ~2+ SSE, same as the pre-rewrite encoder.
// `null` = record only.
const EASY_BLOCK_SSE = 0.05
const EXCESS_LIMITS: Record<string, number | null> = {
  // `${format}:${image}` — ~2–3× the observed values (2026-07, Apple/metal-3:
  // 0.026, 0.010, 0.066–0.078, 0.071–0.109, alpha 0.011/0.027), still 10×+
  // below catastrophic-artifact level.
  'bc1:color': 0.1,
  'bc5:normal': 0.05,
  'bc7:color': 0.2,
  'astc:color': 0.25,
  'bc7:alpha': 0.05,
  'astc:alpha': 0.1,
}

// ------------------------------------------------------------------ helpers

const median = (xs: number[]): number => {
  const s = xs.toSorted((a, b) => a - b)
  const mid = s.length >> 1
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2
}

async function loadImageData(url: string): Promise<ImageData> {
  const res = await fetch(url)
  const blob = await res.blob()
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  const canvas = new OffscreenCanvas(bmp.width, bmp.height)
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(bmp, 0, 0)
  bmp.close()
  return ctx.getImageData(0, 0, canvas.width, canvas.height)
}

function cropImageData(src: ImageData, x: number, y: number, w: number, h: number): ImageData {
  const out = new ImageData(w, h)
  for (let row = 0; row < h; row++) {
    const from = ((y + row) * src.width + x) * 4
    out.data.set(src.data.subarray(from, from + w * 4), row * w * 4)
  }
  return out
}

/** Procedural RGBA test image: gradients + sine detail + hash noise. */
function makeProceduralImage(w: number, h: number, withAlpha: boolean): ImageData {
  const img = new ImageData(w, h)
  const d = img.data
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4
      // Deterministic integer hash for a noise component.
      let n = (x * 374761393 + y * 668265263) | 0
      n = (n ^ (n >> 13)) | 0
      n = (n * 1274126177) | 0
      const noise = ((n >>> 24) & 0xff) / 255
      const gx = x / (w - 1)
      const gy = y / (h - 1)
      const sines = 0.5 + 0.5 * Math.sin(x * 0.35) * Math.cos(y * 0.23)
      d[i] = Math.round(255 * Math.min(1, gx * 0.8 + noise * 0.2))
      d[i + 1] = Math.round(255 * Math.min(1, gy * 0.7 + sines * 0.3))
      d[i + 2] = Math.round(255 * Math.min(1, (1 - gx) * 0.6 + noise * 0.25 + sines * 0.15))
      d[i + 3] = withAlpha ? Math.round(255 * Math.min(1, 0.3 + 0.7 * gx + 0.15 * noise)) : 255
    }
  }
  return img
}

/**
 * Extract the 16 texels of block (bx,by) with clamp-to-edge semantics — the
 * same padding rule the shaders apply — as normalised [0,1] floats.
 * Returns `channels` values per texel, channel-interleaved.
 */
function extractBlock(img: ImageData, bx: number, by: number, channels: number): Float64Array {
  const out = new Float64Array(16 * channels)
  for (let ly = 0; ly < 4; ly++) {
    for (let lx = 0; lx < 4; lx++) {
      const sx = Math.min(bx * 4 + lx, img.width - 1)
      const sy = Math.min(by * 4 + ly, img.height - 1)
      const src = (sy * img.width + sx) * 4
      const dst = (ly * 4 + lx) * channels
      for (let c = 0; c < channels; c++) out[dst + c] = img.data[src + c]! / 255
    }
  }
  return out
}

type BlockDecoder = (block: Uint8Array) => { values: Float64Array; channels: number }

const DECODERS: Record<FormatKey, { bytesPerBlock: number; decode: BlockDecoder }> = {
  bc1: {
    bytesPerBlock: 8,
    decode: block => ({ values: Float64Array.from(decodeBC1Block(block)), channels: 3 }),
  },
  bc5: {
    bytesPerBlock: 16,
    decode: block => {
      const { r, g } = decodeBC5Block(block)
      const values = new Float64Array(32)
      for (let k = 0; k < 16; k++) {
        values[k * 2] = r[k]!
        values[k * 2 + 1] = g[k]!
      }
      return { values, channels: 2 }
    },
  },
  bc7: {
    bytesPerBlock: 16,
    decode: block => ({ values: Float64Array.from(decodeBC7Block(block)), channels: 4 }),
  },
  astc: {
    bytesPerBlock: 16,
    decode: block => ({ values: Float64Array.from(decodeASTC4x4Block(block)), channels: 4 }),
  },
}

/** CPU-encode every block of `img` with the reference encoder for `format`. */
function referenceEncode(format: FormatKey, img: ImageData): Uint8Array {
  const blocksX = (img.width + 3) >> 2
  const blocksY = (img.height + 3) >> 2
  const bpb = DECODERS[format].bytesPerBlock
  const out = new Uint8Array(blocksX * blocksY * bpb)
  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      let block: Uint8Array
      if (format === 'bc1') {
        block = encodeBC1Block(extractBlock(img, bx, by, 3), { quality: 'high' })
      } else if (format === 'bc5') {
        const rgba = extractBlock(img, bx, by, 4)
        const r = new Float64Array(16)
        const g = new Float64Array(16)
        for (let k = 0; k < 16; k++) {
          r[k] = rgba[k * 4]!
          g[k] = rgba[k * 4 + 1]!
        }
        block = encodeBC5Block(r, g)
      } else if (format === 'bc7') {
        block = encodeBC7Mode6Block(extractBlock(img, bx, by, 4))
      } else {
        block = encodeASTC4x4Block(extractBlock(img, bx, by, 4))
      }
      out.set(block, (by * blocksX + bx) * bpb)
    }
  }
  return out
}

/**
 * PSNR (dB) between the source image and the CPU-decoded compressed bytes,
 * over the channels the format actually stores, source-visible pixels only.
 */
function computePsnr(format: FormatKey, img: ImageData, data: Uint8Array): number {
  const { bytesPerBlock, decode } = DECODERS[format]
  const blocksX = (img.width + 3) >> 2
  const blocksY = (img.height + 3) >> 2
  let sum = 0
  let count = 0
  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      const off = (by * blocksX + bx) * bytesPerBlock
      const { values, channels } = decode(data.subarray(off, off + bytesPerBlock))
      for (let ly = 0; ly < 4; ly++) {
        for (let lx = 0; lx < 4; lx++) {
          const sx = bx * 4 + lx
          const sy = by * 4 + ly
          if (sx >= img.width || sy >= img.height) continue
          const src = (sy * img.width + sx) * 4
          const k = (ly * 4 + lx) * channels
          for (let c = 0; c < channels; c++) {
            const d = values[k + c]! * 255 - img.data[src + c]!
            sum += d * d
            count++
          }
        }
      }
    }
  }
  if (sum === 0) return Infinity
  return 10 * Math.log10((255 * 255 * count) / sum)
}

/**
 * Per-block decoded squared error vs the source (clamp-padded), normalised
 * units, over the channels the format stores.
 */
function perBlockSse(format: FormatKey, img: ImageData, data: Uint8Array): Float64Array {
  const { bytesPerBlock, decode } = DECODERS[format]
  const blocksX = (img.width + 3) >> 2
  const blocksY = (img.height + 3) >> 2
  const out = new Float64Array(blocksX * blocksY)
  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      const bi = by * blocksX + bx
      const off = bi * bytesPerBlock
      const { values, channels } = decode(data.subarray(off, off + bytesPerBlock))
      const src = extractBlock(img, bx, by, channels)
      let sse = 0
      for (let i = 0; i < values.length; i++) {
        const d = values[i]! - src[i]!
        sse += d * d
      }
      out[bi] = sse
    }
  }
  return out
}

function diffBytes(a: Uint8Array, b: Uint8Array, bytesPerBlock: number): { blocks: number; first: number } {
  let blocks = 0
  let first = -1
  for (let off = 0; off < a.length; off += bytesPerBlock) {
    for (let i = 0; i < bytesPerBlock; i++) {
      if (a[off + i] !== b[off + i]) {
        blocks++
        if (first < 0) first = off / bytesPerBlock
        break
      }
    }
  }
  return { blocks, first }
}

/**
 * Compare GPU output against the CPU reference. Bytes are NOT required to be
 * identical: the GPU runs f32 (vs f64 on the CPU), so ties and least-squares
 * refits can land on different-but-equally-good encodings for a small
 * fraction of blocks. What must hold for every differing block is that the
 * GPU's decoded squared error is no worse than the reference's beyond a
 * few-LSB tolerance — that catches real bugs (bad packing, wrong palette,
 * broken search) while tolerating FP tie-breaks.
 */
function compareToReference(
  format: FormatKey,
  img: ImageData,
  gpuData: Uint8Array,
  refData: Uint8Array,
): { pass: boolean; detail: string } {
  const { bytesPerBlock, decode } = DECODERS[format]
  const blocksX = (img.width + 3) >> 2
  const blocksY = (img.height + 3) >> 2
  const total = blocksX * blocksY
  const channels = format === 'bc1' ? 3 : format === 'bc5' ? 2 : 4

  const blockSse = (decoded: Float64Array, src: Float64Array): number => {
    let sse = 0
    for (let i = 0; i < decoded.length; i++) {
      const d = decoded[i]! - src[i]!
      sse += d * d
    }
    return sse
  }

  let identical = 0
  let worstExcess = 0
  let worstBlock = -1
  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      const off = (by * blocksX + bx) * bytesPerBlock
      let same = true
      for (let i = 0; i < bytesPerBlock; i++) {
        if (gpuData[off + i] !== refData[off + i]) {
          same = false
          break
        }
      }
      if (same) {
        identical++
        continue
      }
      const src = extractBlock(img, bx, by, channels)
      const gpuSse = blockSse(decode(gpuData.subarray(off, off + bytesPerBlock)).values, src)
      const refSse = blockSse(decode(refData.subarray(off, off + bytesPerBlock)).values, src)
      const excess = gpuSse - refSse
      if (excess > worstExcess) {
        worstExcess = excess
        worstBlock = by * blocksX + bx
      }
    }
  }

  // Per-block tolerance: 5e-4 (≈32 LSB², (1/255)² ≈ 1.54e-5 each) — a GPU-f32
  // vs CPU-f64 flip of a refit acceptance can legitimately move one block this
  // much (both candidates are near-equal minima). Real encoder bugs (bad
  // packing, wrong palette) overshoot by orders of magnitude, and the
  // aggregate PSNR gate below catches any systematic drift.
  const gpuPsnr = computePsnr(format, img, gpuData)
  const refPsnr = computePsnr(format, img, refData)
  const psnrDelta = gpuPsnr - refPsnr
  const pass = worstExcess <= 5e-4 && psnrDelta >= -0.05
  const pct = ((identical / total) * 100).toFixed(1)
  return {
    pass,
    detail:
      `${identical}/${total} blocks identical (${pct}%), ΔPSNR ${psnrDelta >= 0 ? '+' : ''}${psnrDelta.toFixed(3)} dB` +
      (identical === total ? '' : `, worst block excess ${worstExcess.toExponential(1)} (#${worstBlock})`),
  }
}

// ------------------------------------------------------------------- suite

export async function runSuite(onProgress: ProgressFn): Promise<SuiteResults> {
  if (!('gpu' in navigator)) throw new Error('WebGPU not available in this browser')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('No WebGPU adapter')

  const requestable: GPUFeatureName[] = [
    'texture-compression-bc',
    'texture-compression-astc',
    'shader-f16',
    'timestamp-query',
  ]
  const features = requestable.filter(f => adapter.features.has(f))
  const device = await adapter.requestDevice({ requiredFeatures: features })
  const hasF16 = device.features.has('shader-f16')

  const env = {
    vendor: adapter.info?.vendor ?? '?',
    architecture: adapter.info?.architecture ?? '?',
    description: adapter.info?.description ?? '',
    features: [...device.features].toSorted(),
    hasF16,
    userAgent: navigator.userAgent,
  }

  const encoders: Record<FormatKey, Encoder> = {
    bc1: new BC1Encoder({ device, adapter }),
    bc5: new BC5Encoder({ device, adapter }),
    bc7: new BC7Encoder({ device, adapter }),
    astc: new ASTC4x4Encoder({ device, adapter }),
  }
  // f32-forced twins, for validating/benchmarking the fallback fast path on
  // f16 hardware. Identical to `encoders` when the device lacks f16.
  const encodersF32: Record<FormatKey, Encoder> = {
    bc1: new BC1Encoder({ device, adapter, disableF16: true }),
    bc5: new BC5Encoder({ device, adapter, disableF16: true }),
    bc7: new BC7Encoder({ device, adapter, disableF16: true }),
    astc: new ASTC4x4Encoder({ device, adapter, disableF16: true }),
  }

  onProgress('Loading test images…')
  const colorFull = await loadImageData('/textures/color.png')
  const normalFull = await loadImageData('/textures/normal.png')
  const color = cropImageData(colorFull, 0, 0, 256, 256)
  const normal = cropImageData(normalFull, 0, 0, 256, 256)
  // Odd size → exercises the clamp-to-edge padding path (not multiples of 4).
  const colorOdd = cropImageData(colorFull, 17, 9, 133, 61)
  const normalOdd = cropImageData(normalFull, 17, 9, 133, 61)
  const alpha = makeProceduralImage(128, 128, true)

  const correctness: CorrectnessResult[] = []
  const quality: QualityResult[] = []
  const perf: PerfResult[] = []

  // ---------------------------------------------------------- correctness
  const FORMATS: FormatKey[] = ['bc1', 'bc5', 'bc7', 'astc']
  const highImages: Record<FormatKey, Array<[string, ImageData]>> = {
    bc1: [
      ['color 256²', color],
      ['color 133×61 (odd)', colorOdd],
    ],
    bc5: [
      ['normal 256²', normal],
      ['normal 133×61 (odd)', normalOdd],
    ],
    bc7: [
      ['color 256²', color],
      ['color 133×61 (odd)', colorOdd],
      ['alpha 128²', alpha],
    ],
    astc: [
      ['color 256²', color],
      ['color 133×61 (odd)', colorOdd],
      ['alpha 128²', alpha],
    ],
  }

  for (const format of FORMATS) {
    const enc = encoders[format]
    for (const [name, img] of highImages[format]) {
      onProgress(`Correctness: ${format} high vs CPU reference — ${name}`)
      const gpu = await enc.encodeToBytes(img, { quality: 'high' })
      const ref = referenceEncode(format, img)
      if (gpu.data.length !== ref.length) {
        correctness.push({
          name: `${format} high ≈ reference (${name})`,
          format,
          pass: false,
          detail: `size mismatch gpu=${gpu.data.length} ref=${ref.length}`,
        })
        continue
      }
      const cmp = compareToReference(format, img, gpu.data, ref)
      correctness.push({ name: `${format} high ≈ reference (${name})`, format, ...cmp })
    }

    onProgress(`Correctness: ${format} determinism`)
    const a = await enc.encodeToBytes(color, { quality: 'fast' })
    const b = await enc.encodeToBytes(color, { quality: 'fast' })
    const det = diffBytes(a.data, b.data, DECODERS[format].bytesPerBlock)
    correctness.push({
      name: `${format} fast deterministic`,
      format,
      pass: det.blocks === 0,
      detail: det.blocks === 0 ? 'two runs identical' : `${det.blocks} blocks differ between runs`,
    })
  }

  // -------------------------------------------------------------- quality
  // Quality runs on the FULL committed test cards — every quadrant (smooth
  // gradients, flat saturated tiles, per-channel ramps, noise + radial disc)
  // stresses a different failure mode, and a crop would hide localized bugs.
  const qualityCases: Array<{ format: FormatKey; image: string; img: ImageData; qualities: EncodeQuality[] }> = [
    { format: 'bc1', image: 'color', img: colorFull, qualities: ['fast', 'high'] },
    { format: 'bc5', image: 'normal', img: normalFull, qualities: ['fast', 'high'] },
    { format: 'bc7', image: 'color', img: colorFull, qualities: ['fast', 'high'] },
    { format: 'astc', image: 'color', img: colorFull, qualities: ['fast', 'high'] },
    { format: 'bc7', image: 'alpha', img: alpha, qualities: ['fast'] },
    { format: 'astc', image: 'alpha', img: alpha, qualities: ['fast'] },
  ]

  for (const { format, image, img, qualities } of qualityCases) {
    // Per-block error of the 'high' encode — the yardstick for the fast
    // paths' worst-block gate. Aggregate PSNR alone is insensitive to a
    // handful of catastrophically wrong blocks.
    onProgress(`Quality: ${format} high baseline on ${image}`)
    const high = await encoders[format].encodeToBytes(img, { quality: 'high' })
    const highSse = perBlockSse(format, img, high.data)

    for (const q of qualities) {
      // 'high' is always f32; for 'fast' measure the default (f16 when
      // available) and, on f16 hardware, the forced-f32 fallback too.
      const variants: Array<['f16' | 'f32', Encoder]> =
        q === 'fast' && hasF16
          ? [
              ['f16', encoders[format]],
              ['f32', encodersF32[format]],
            ]
          : [[hasF16 && q === 'fast' ? 'f16' : 'f32', encoders[format]]]
      for (const [variant, enc] of variants) {
        onProgress(`Quality: ${format} ${q} (${variant}) PSNR on ${image}`)
        const { data } = q === 'high' ? high : await enc.encodeToBytes(img, { quality: q })
        const psnrDb = computePsnr(format, img, data)
        const threshold = PSNR_THRESHOLDS[`${format}:${q}:${image}`] ?? null

        let worstBlockExcess = 0
        let worstEasyBlockExcess = 0
        if (q === 'fast') {
          const sse = perBlockSse(format, img, data)
          for (let i = 0; i < sse.length; i++) {
            const excess = sse[i]! - highSse[i]!
            if (excess > worstBlockExcess) worstBlockExcess = excess
            if (highSse[i]! <= EASY_BLOCK_SSE && excess > worstEasyBlockExcess) worstEasyBlockExcess = excess
          }
        }
        const excessLimit = q === 'fast' ? (EXCESS_LIMITS[`${format}:${image}`] ?? null) : null
        quality.push({
          format,
          quality: q,
          variant,
          image,
          psnrDb,
          thresholdDb: threshold ?? 0,
          worstEasyBlockExcess,
          worstBlockExcess,
          excessLimit,
          pass:
            (threshold === null ? true : psnrDb >= threshold) &&
            (excessLimit === null ? true : worstEasyBlockExcess <= excessLimit),
        })
      }
    }
  }

  // ---------------------------------------------------------------- perf
  onProgress('Preparing 2048×2048 benchmark image…')
  const benchImg = makeProceduralImage(2048, 2048, false)
  const benchBitmap = await createImageBitmap(benchImg, {
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
  })
  const WARMUP = 4
  const RUNS = 20
  const mpix = (benchBitmap.width * benchBitmap.height) / 1e6

  // Ramp the GPU out of its idle clock state before timing anything — the
  // correctness/quality phases above are CPU-heavy and leave the GPU cold,
  // which otherwise skews the first benchmark cases by 2×+.
  onProgress('Warming up GPU clocks…')
  for (let i = 0; i < 15; i++) {
    await encoders.bc7.encodeToBytes(benchBitmap, { quality: 'high' })
  }

  const perfCases: Array<{ format: FormatKey; quality: EncodeQuality; variant: 'f16' | 'f32'; enc: Encoder }> = []
  for (const format of FORMATS) {
    perfCases.push({ format, quality: 'fast', variant: hasF16 ? 'f16' : 'f32', enc: encoders[format] })
    if (hasF16) perfCases.push({ format, quality: 'fast', variant: 'f32', enc: encodersF32[format] })
    perfCases.push({ format, quality: 'high', variant: 'f32', enc: encoders[format] })
  }

  for (const { format, quality: q, variant, enc } of perfCases) {
    onProgress(`Benchmark: ${format} ${q} (${variant}) — 2048²`)
    for (let i = 0; i < WARMUP; i++) {
      await enc.encodeToBytes(benchBitmap, { quality: q })
    }
    const wall: number[] = []
    const gpu: number[] = []
    for (let i = 0; i < RUNS; i++) {
      const r = await enc.encodeToBytes(benchBitmap, { quality: q, withGpuTime: true })
      wall.push(r.encodeMs)
      if (r.gpuMs !== undefined) gpu.push(r.gpuMs)
    }
    const wallMsMedian = median(wall)
    perf.push({
      format,
      quality: q,
      variant,
      image: '2048×2048',
      wallMsMedian,
      gpuMsMedian: gpu.length ? median(gpu) : null,
      mpixPerSec: mpix / (wallMsMedian / 1000),
    })
  }
  benchBitmap.close()

  const failures = correctness.filter(c => !c.pass).length + quality.filter(qr => !qr.pass).length

  device.destroy()

  return { env, correctness, quality, perf, failures }
}
