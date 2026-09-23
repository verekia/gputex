// GPU encoder test + benchmark suite (WebGPU only).
//
// Runs in a real browser against real GPU hardware — this is the validation
// harness for the WGSL encoding shaders that the bun unit tests (CPU-only)
// cannot provide. Driven by example/pages/test.tsx and automatable: the page
// exposes `window.__GPUTEX_TESTS__` with a status flag and the full results.
//
// Three groups:
//   1. Correctness — determinism (same input twice → identical bytes) and the
//      clamp-to-edge padding path (a non-multiple-of-4 image must land within
//      a couple of dB of the CPU reference encode — a padding bug craters it).
//   2. Quality — GPU output is CPU-decoded and validated on the FULL 1024²
//      test cards against per-format PSNR floors, and per block against the
//      exhaustive CPU reference encoders (gputex/testing): over blocks the
//      reference encodes near-losslessly, the GPU encoder must not lose by
//      more than a small per-format limit. Both the f16 and
//      (force-disabled-f16) f32 shader variants are gated.
//   3. Performance — median wall-clock encode time and (via timestamp
//      queries) GPU compute-pass time per format on a 2048×2048 image, for
//      both shader variants.

import { ASTC4x4Encoder, BC1Encoder, BC5Encoder, BC7Encoder, ETC2Encoder } from 'gputex'
import type { Encoder } from 'gputex'

import {
  decodeASTC4x4Block,
  decodeBC1Block,
  decodeBC5Block,
  decodeBC7Block,
  decodeETC2Block,
  encodeASTC4x4Block,
  encodeBC1Block,
  encodeBC5Block,
  encodeBC7Mode6Block,
  encodeETC2Block,
} from 'gputex/testing'

export type FormatKey = 'bc1' | 'bc5' | 'bc7' | 'astc' | 'etc2'

export interface CorrectnessResult {
  name: string
  format: FormatKey
  pass: boolean
  detail: string
}

export interface QualityResult {
  format: FormatKey
  variant: 'f16' | 'f32'
  image: string
  psnrDb: number
  /**
   * PSNR of the exhaustive CPU reference encode on the same image, or null
   * for the PSNR-floor-only rows (2K/4K real textures) where the reference
   * encode is prohibitively slow (~16 s per format at 4096²).
   */
  refPsnrDb: number | null
  thresholdDb: number
  /**
   * Worst EASY block: max over blocks that the CPU reference encodes
   * near-losslessly (SSE ≤ 0.05) of SSE(gpu) − SSE(reference), decoded,
   * normalised units. Localized artifacts on easy content (e.g. a flat tile
   * turning the wrong colour) barely move aggregate PSNR but explode this
   * metric; genuinely hard blocks (noise), where a single-line encoder may
   * legitimately trail the exhaustive search, are excluded. Also reported:
   * the overall worst excess, for context. Null on PSNR-floor-only rows.
   */
  worstEasyBlockExcess: number | null
  worstBlockExcess: number | null
  excessLimit: number | null
  pass: boolean
}

export interface PerfResult {
  format: FormatKey
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
// PSNR thresholds (dB), measured on the committed test textures. Set ~0.15 dB
// under the observed baseline so quality regressions fail loudly while normal
// cross-GPU float jitter passes. `null` = record only.
// ---------------------------------------------------------------------------
const PSNR_THRESHOLDS: Record<string, number | null> = {
  // `${format}:${image}` — measured on the committed test images (2026-07,
  // Apple/metal-3, mode-6-only BC7 with the 8-step power iteration) minus
  // ~0.15 dB. bc5 rows re-pinned 2026-07 for the moment-form kernel (pass-2
  // reprojection returned at parity cost; +0.09..0.22 dB, ≥ the exhaustive
  // reference on every ref-gated row). BC7 rows on multi-modal content (packed-*, rock-color) sit at
  // the mode-6 exhaustive reference level — the mode 1 candidate that
  // lifted them ~+1.3 dB was dropped for speed (see bc7_fast_f16.wgsl).
  // Unlisted rows and the `:normal` colour-format cross-card entries are
  // record-only.
  // BC1 rows re-pinned 2026-09 (solid-colour path for near-flat blocks +
  // moment-form refit passes): measured color 29.32, rock-ao 41.63, rock-
  // displacement 45.85, wood-roughness 40.64, wood-displacement 43.36 (min
  // f16/f32).
  'bc1:color': 29.17,
  'bc5:normal': 53.0,
  'bc7:color': 31.9,
  // ASTC colour rows re-pinned 2026-09 for the opaque CEM 8 bit budgets
  // (QUANT_192 endpoints + 4-bit weights on wide blocks): measured
  // 32.03 / 37.83 / 38.16 / 50.88 (color / packed / rock / wood, min
  // f16/f32), up from 31.97 / 37.59 / 37.52 / 49.60.
  'astc:color': 31.88,
  // 1024² committed alpha card (2026-07): bc7 38.43/38.38 f16/f32, astc
  // 37.28/37.27 — both above the exhaustive reference on this content.
  'bc7:alpha': 38.2,
  'astc:alpha': 37.1,
  'bc1:normal': null,
  'bc7:normal': null,
  'astc:normal': null,
  // Real textures (2026-07 baselines, min over f16/f32, minus ~0.15 dB).
  'bc7:packed-256': 32.75,
  'bc7:packed-512': 35.0,
  'bc1:packed-1024': 34.75,
  'bc7:packed-1024': 37.65,
  'astc:packed-1024': 37.68,
  'bc7:packed-2048': 45.15,
  'bc7:packed-4096': 49.95,
  'bc1:rock-color-1k': 33.9,
  'bc7:rock-color-1k': 38.1,
  'astc:rock-color-1k': 38.0,
  'bc5:rock-normal-1k': 46.35,
  'bc1:rock-roughness-1k': 39.1,
  'bc1:rock-ao-1k': 41.48,
  'bc1:rock-displacement-1k': 45.7,
  // BC7 on exact-grayscale maps (analytic luma axis) — measured
  // 51.62 / 53.63 / 56.68 / 52.10 / 55.09 (2026-07, min f16/f32).
  'bc7:rock-roughness-1k': 51.45,
  'bc7:rock-ao-1k': 53.45,
  'bc7:rock-displacement-1k': 56.5,
  'bc7:wood-roughness-1k': 51.95,
  'bc7:wood-displacement-1k': 54.9,
  // ASTC luminance path (CEM 0, 5-bit weights) on exact-grayscale maps —
  // measured 59.73 / 62.45 / 70.59 / 60.04 / 65.27 (2026-07, f16 ≡ f32).
  'astc:rock-roughness-1k': 59.55,
  'astc:rock-ao-1k': 62.3,
  'astc:rock-displacement-1k': 70.4,
  'astc:wood-roughness-1k': 59.85,
  'astc:wood-displacement-1k': 65.1,
  'bc7:rock-color-2k': 38.75,
  'bc5:rock-normal-2k': 44.9,
  'bc7:rock-color-4k': 39.15,
  'bc5:rock-normal-4k': 43.75,
  'bc1:wood-color-1k': 41.9,
  'bc7:wood-color-1k': 49.4,
  'astc:wood-color-1k': 50.7,
  // ETC2 (2026-07, minus ~0.15 dB; SETTLED at the two-candidate scored
  // search + planar + no refit — the hedged O(1) table pick saved ~3% GPU
  // for −0.5 dB and was reverted, the two-pass prepared source lost
  // per-texture; both live in git history. f16 and f32 modules are
  // byte-identical (exact-value f16), so both rows share these pins. The
  // low 'color'-card number is the format, not the encoder: ETC1-family
  // blocks modulate only luma per pixel, so the card's per-pixel chroma
  // checkers crater without the unimplemented T/H modes.
  'etc2:color': 19.82,
  'etc2:packed-1024': 31.86,
  'etc2:rock-color-1k': 33.64,
  'etc2:rock-roughness-1k': 39.99,
  'etc2:wood-color-1k': 39.06,
  'bc5:wood-normal-1k': 48.0,
  'bc1:wood-roughness-1k': 40.49,
  'bc1:wood-displacement-1k': 43.21,
  'bc7:wood-color-2k': 50.65,
  'bc5:wood-normal-2k': 48.95,
  'bc7:wood-color-4k': 51.7,
  'bc5:wood-normal-4k': 47.65,
}

// Worst-EASY-block gate: over blocks that the CPU reference encodes
// near-losslessly (SSE ≤ EASY_BLOCK_SSE), max of SSE(gpu) − SSE(reference).
// The encoder must never lose badly on content that is easy to encode — a
// 2026-07 BC1 rank-1 refit bug turned FLAT tiles (reference SSE ≈ 0.001)
// into the wrong colour entirely (SSE ≈ 3.0). Hard blocks (the noise tile)
// are excluded: there a single-line seed legitimately trails the exhaustive
// search. `null` = record only.
const EASY_BLOCK_SSE = 0.05
const EXCESS_LIMITS: Record<string, number | null> = {
  // `${format}:${image}` — ~2–3× the observed values (2026-07, Apple/metal-3:
  // 0.061, 0.006, 0.016, 0.050, alpha 0.002/0.004, packed-materials
  // 0.061/0.054/0.055), still 10×+ below catastrophic-artifact level.
  'bc1:color': 0.15,
  'bc5:normal': 0.05,
  'bc7:color': 0.12,
  'astc:color': 0.15,
  // 1024² alpha card: observed easy-block excess 0.014 / 0.052.
  'bc7:alpha': 0.05,
  'astc:alpha': 0.15,
  // Real textures (observed 0.001–0.06).
  'bc7:packed-256': 0.05,
  'bc7:packed-512': 0.1,
  'bc1:packed-1024': 0.15,
  'bc7:packed-1024': 0.15,
  'astc:packed-1024': 0.15,
  'bc1:rock-color-1k': 0.15,
  'bc7:rock-color-1k': 0.05,
  'astc:rock-color-1k': 0.15,
  'bc5:rock-normal-1k': 0.05,
  'bc1:rock-roughness-1k': 0.05,
  'bc1:rock-ao-1k': 0.1,
  'bc1:rock-displacement-1k': 0.05,
  // BC7 gray rows: observed ≤ 0.005.
  'bc7:rock-roughness-1k': 0.05,
  'bc7:rock-ao-1k': 0.05,
  'bc7:rock-displacement-1k': 0.05,
  'bc7:wood-roughness-1k': 0.05,
  'bc7:wood-displacement-1k': 0.05,
  // ASTC luminance rows: observed ≤ 0.001 — the scalar path tracks the
  // exhaustive reference almost block-for-block.
  'astc:rock-roughness-1k': 0.05,
  'astc:rock-ao-1k': 0.05,
  'astc:rock-displacement-1k': 0.05,
  'astc:wood-roughness-1k': 0.05,
  'astc:wood-displacement-1k': 0.05,
  'bc1:wood-color-1k': 0.05,
  'bc7:wood-color-1k': 0.05,
  'astc:wood-color-1k': 0.05,
  // ETC2 (2026-07 scored-search shader, observed 0.291 / 0.029 / 0.112 /
  // 0.023 / 0.038 — estimate-based selection trails the exact reference a
  // little more per block than the other formats' exact searches do).
  'etc2:color': 0.45,
  'etc2:packed-1024': 0.05,
  'etc2:rock-color-1k': 0.2,
  'etc2:rock-roughness-1k': 0.05,
  'etc2:wood-color-1k': 0.08,
  'bc5:wood-normal-1k': 0.05,
  'bc1:wood-roughness-1k': 0.1,
  'bc1:wood-displacement-1k': 0.05,
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
  etc2: {
    bytesPerBlock: 8,
    decode: block => ({ values: Float64Array.from(decodeETC2Block(block)), channels: 3 }),
  },
}

/**
 * CPU-encode every block of `img` with the exhaustive reference encoder for
 * `format` — the quality yardstick the GPU encoder is gated against.
 */
function referenceEncode(format: FormatKey, img: ImageData, quality: 'fast' | 'high' = 'high'): Uint8Array {
  const blocksX = (img.width + 3) >> 2
  const blocksY = (img.height + 3) >> 2
  const bpb = DECODERS[format].bytesPerBlock
  const out = new Uint8Array(blocksX * blocksY * bpb)
  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      let block: Uint8Array
      if (format === 'bc1') {
        block = encodeBC1Block(extractBlock(img, bx, by, 3), { quality: 'high' })
      } else if (format === 'etc2') {
        block = encodeETC2Block(extractBlock(img, bx, by, 3), { quality })
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

// ------------------------------------------------------------------- suite

export async function runSuite(onProgress: ProgressFn): Promise<SuiteResults> {
  if (!('gpu' in navigator)) throw new Error('WebGPU not available in this browser')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('No WebGPU adapter')

  const requestable: GPUFeatureName[] = [
    'texture-compression-bc',
    'texture-compression-astc',
    'texture-compression-etc2',
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
    etc2: new ETC2Encoder({ device, adapter }),
  }
  // f32-forced twins, for validating/benchmarking the fallback shader on
  // f16 hardware. Identical to `encoders` when the device lacks f16.
  const encodersF32: Record<FormatKey, Encoder> = {
    bc1: new BC1Encoder({ device, adapter, disableF16: true }),
    bc5: new BC5Encoder({ device, adapter, disableF16: true }),
    bc7: new BC7Encoder({ device, adapter, disableF16: true }),
    astc: new ASTC4x4Encoder({ device, adapter, disableF16: true }),
    etc2: new ETC2Encoder({ device, adapter, disableF16: true }),
  }
  // ETC2 (and any future integer-domain encoder) ships no f16 module — its
  // only variant is f32, so the f16/f32 twin logic below collapses for it.
  const hasF16Variant = (format: FormatKey): boolean => hasF16 && encoders[format].wgslSourceFastF16() !== null

  onProgress('Loading test images…')
  const colorFull = await loadImageData('/textures/color.png')
  const normalFull = await loadImageData('/textures/normal.png')
  const color = cropImageData(colorFull, 0, 0, 256, 256)
  // Odd size → exercises the clamp-to-edge padding path (not multiples of 4).
  const colorOdd = cropImageData(colorFull, 17, 9, 133, 61)
  const normalOdd = cropImageData(normalFull, 17, 9, 133, 61)

  const correctness: CorrectnessResult[] = []
  const quality: QualityResult[] = []
  const perf: PerfResult[] = []

  // ---------------------------------------------------------- correctness
  const FORMATS: FormatKey[] = ['bc1', 'bc5', 'bc7', 'astc', 'etc2']
  for (const format of FORMATS) {
    const enc = encoders[format]

    onProgress(`Correctness: ${format} determinism`)
    const a = await enc.encodeToBytes(color)
    const b = await enc.encodeToBytes(color)
    const det = diffBytes(a.data, b.data, DECODERS[format].bytesPerBlock)
    correctness.push({
      name: `${format} deterministic`,
      format,
      pass: det.blocks === 0,
      detail: det.blocks === 0 ? 'two runs identical' : `${det.blocks} blocks differ between runs`,
    })

    // Clamp-to-edge padding: a non-multiple-of-4 image must land within a
    // couple of dB of the CPU reference — a padding bug (reading the zeroed
    // padding strip, mis-clamped coordinates) craters this by far more than
    // the encoder's normal gap to the reference. ETC2 gates against its
    // 'fast' mirror instead of 'high': the fast path no longer emits planar,
    // so on smooth content the fast-vs-high gap alone can exceed the bug
    // threshold — and the mirror comparison is the stronger contract anyway
    // (the GPU pipeline matches it byte-for-byte, prepared source included).
    onProgress(`Correctness: ${format} odd-size padding vs CPU reference`)
    const oddImg = format === 'bc5' ? normalOdd : colorOdd
    const gpuOdd = await enc.encodeToBytes(oddImg)
    const refOdd = referenceEncode(format, oddImg, format === 'etc2' ? 'fast' : 'high')
    const gpuPsnr = computePsnr(format, oddImg, gpuOdd.data)
    const refPsnr = computePsnr(format, oddImg, refOdd)
    const delta = gpuPsnr - refPsnr
    correctness.push({
      name: `${format} odd-size padding (133×61)`,
      format,
      pass: delta >= -2.0,
      detail: `gpu ${gpuPsnr.toFixed(2)} dB vs reference ${refPsnr.toFixed(2)} dB (Δ ${delta.toFixed(2)})`,
    })
  }

  // -------------------------------------------------------------- quality
  // Quality runs on the FULL synthetic test cards — every tile (smooth
  // gradients, hard edges, Nyquist checkers, zone plate, noise, disc-over-
  // checker probe, natural-ish content) stresses a different failure mode,
  // and a crop would hide localized bugs — plus a set of REAL textures: the
  // packed-materials game atlas (channel-packed monochrome maps, the
  // multi-modal probe that synthetic images hid) and the Rock064 /
  // WoodFloor004 PBR sets (photographic colour, tangent-space normals, and
  // grayscale roughness/AO/displacement maps).
  //
  // Rows with `ref: true` are gated against the exhaustive CPU reference
  // (PSNR floor + worst-easy-block excess). The reference encode costs ~1 s
  // per format at 1024² but ~16 s at 4096², so the 2K/4K rows are
  // PSNR-floor-only and run just the default shader variant — the f16↔f32
  // and per-block gates are already covered by the 1K row of the same
  // content. The `:normal` colour-format rows are record-only (null
  // thresholds): anti-correlated R/G is exactly where a bbox-diagonal
  // endpoint seed collapses, so they track the PCA seeding's headline win
  // without gating.
  //
  // Images load lazily per spec — a 4096² ImageData is 64 MB, so holding
  // the whole set at once would cost ~½ GB.
  const rock = (size: string, map: string) => `/textures/Rock064_${size}-JPG/Rock064_${size}-JPG_${map}.jpg`
  const wood = (size: string, map: string) => `/textures/WoodFloor004_${size}-JPG/WoodFloor004_${size}-JPG_${map}.jpg`
  const packed = (size: number) => `/textures/packed-materials/packed-materials-${size}.png`
  interface QualitySpec {
    image: string
    formats: FormatKey[]
    src: string | ImageData
    /** Run the exhaustive CPU reference + per-block gates. */
    ref: boolean
    /** Also run the f32-forced variant (default true for ref'd rows). */
    bothVariants: boolean
  }
  const gated = (image: string, formats: FormatKey[], src: string | ImageData): QualitySpec => ({
    image,
    formats,
    src,
    ref: true,
    bothVariants: true,
  })
  const floorOnly = (image: string, formats: FormatKey[], src: string): QualitySpec => ({
    image,
    formats,
    src,
    ref: false,
    bothVariants: false,
  })
  const qualitySpecs: QualitySpec[] = [
    // Synthetic cards.
    gated('color', ['bc1', 'bc7', 'astc', 'etc2'], colorFull),
    gated('normal', ['bc5', 'bc1', 'bc7', 'astc'], normalFull),
    // Committed 1024² alpha card (gen-test-textures.mjs): smooth alpha
    // ramps, cutout edges, Nyquist/noise alpha, low-alpha precision, plus
    // opaque and exactly-gray tiles so the per-block class selection and
    // its transitions are gated on one image.
    gated('alpha', ['bc7', 'astc'], '/textures/alpha.png'),
    // Packed-materials game atlas (channel-packed, has alpha at 1024).
    gated('packed-256', ['bc7'], packed(256)),
    gated('packed-512', ['bc7'], packed(512)),
    gated('packed-1024', ['bc1', 'bc7', 'astc', 'etc2'], packed(1024)),
    floorOnly('packed-2048', ['bc7'], packed(2048)),
    floorOnly('packed-4096', ['bc7'], packed(4096)),
    // Rock064 PBR set (photographic).
    gated('rock-color-1k', ['bc1', 'bc7', 'astc', 'etc2'], rock('1K', 'Color')),
    gated('rock-normal-1k', ['bc5'], rock('1K', 'NormalGL')),
    gated('rock-roughness-1k', ['bc1', 'bc7', 'astc', 'etc2'], rock('1K', 'Roughness')),
    gated('rock-ao-1k', ['bc1', 'bc7', 'astc'], rock('1K', 'AmbientOcclusion')),
    gated('rock-displacement-1k', ['bc1', 'bc7', 'astc'], rock('1K', 'Displacement')),
    floorOnly('rock-color-2k', ['bc7'], rock('2K', 'Color')),
    floorOnly('rock-normal-2k', ['bc5'], rock('2K', 'NormalGL')),
    floorOnly('rock-color-4k', ['bc7'], rock('4K', 'Color')),
    floorOnly('rock-normal-4k', ['bc5'], rock('4K', 'NormalGL')),
    // WoodFloor004 PBR set (photographic, strong plank seams).
    gated('wood-color-1k', ['bc1', 'bc7', 'astc', 'etc2'], wood('1K', 'Color')),
    gated('wood-normal-1k', ['bc5'], wood('1K', 'NormalGL')),
    gated('wood-roughness-1k', ['bc1', 'bc7', 'astc'], wood('1K', 'Roughness')),
    gated('wood-displacement-1k', ['bc1', 'bc7', 'astc'], wood('1K', 'Displacement')),
    floorOnly('wood-color-2k', ['bc7'], wood('2K', 'Color')),
    floorOnly('wood-normal-2k', ['bc5'], wood('2K', 'NormalGL')),
    floorOnly('wood-color-4k', ['bc7'], wood('4K', 'Color')),
    floorOnly('wood-normal-4k', ['bc5'], wood('4K', 'NormalGL')),
  ]

  for (const spec of qualitySpecs) {
    const { image, ref } = spec
    onProgress(`Quality: loading ${image}…`)
    const img = typeof spec.src === 'string' ? await loadImageData(spec.src) : spec.src

    for (const format of spec.formats) {
      // Per-block error of the exhaustive CPU reference encode — the
      // yardstick for the worst-block gate. Aggregate PSNR alone is
      // insensitive to a handful of catastrophically wrong blocks.
      let refSse: Float64Array | null = null
      let refPsnrDb: number | null = null
      if (ref) {
        onProgress(`Quality: ${format} CPU reference baseline on ${image}`)
        const refData = referenceEncode(format, img)
        refSse = perBlockSse(format, img, refData)
        refPsnrDb = computePsnr(format, img, refData)
      }

      const variants: Array<['f16' | 'f32', Encoder]> =
        hasF16Variant(format) && spec.bothVariants
          ? [
              ['f16', encoders[format]],
              ['f32', encodersF32[format]],
            ]
          : [[hasF16Variant(format) ? 'f16' : 'f32', encoders[format]]]
      for (const [variant, enc] of variants) {
        onProgress(`Quality: ${format} (${variant}) PSNR on ${image}`)
        const { data } = await enc.encodeToBytes(img)
        const psnrDb = computePsnr(format, img, data)
        const threshold = PSNR_THRESHOLDS[`${format}:${image}`] ?? null

        let worstBlockExcess: number | null = null
        let worstEasyBlockExcess: number | null = null
        if (refSse) {
          worstBlockExcess = 0
          worstEasyBlockExcess = 0
          const sse = perBlockSse(format, img, data)
          for (let i = 0; i < sse.length; i++) {
            const excess = sse[i]! - refSse[i]!
            if (excess > worstBlockExcess) worstBlockExcess = excess
            if (refSse[i]! <= EASY_BLOCK_SSE && excess > worstEasyBlockExcess) worstEasyBlockExcess = excess
          }
        }
        const excessLimit = ref ? (EXCESS_LIMITS[`${format}:${image}`] ?? null) : null
        quality.push({
          format,
          variant,
          image,
          psnrDb,
          refPsnrDb,
          thresholdDb: threshold ?? 0,
          worstEasyBlockExcess,
          worstBlockExcess,
          excessLimit,
          pass:
            (threshold === null ? true : psnrDb >= threshold) &&
            (excessLimit === null || worstEasyBlockExcess === null ? true : worstEasyBlockExcess <= excessLimit),
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
  // which otherwise skews the first benchmark cases by 2×+. The encodes are
  // cheap, so it takes a burst of them to move the clocks.
  onProgress('Warming up GPU clocks…')
  for (let i = 0; i < 40; i++) {
    await encoders.bc7.encodeToBytes(benchBitmap)
  }

  const perfCases: Array<{ format: FormatKey; variant: 'f16' | 'f32'; enc: Encoder }> = []
  for (const format of FORMATS) {
    perfCases.push({ format, variant: hasF16Variant(format) ? 'f16' : 'f32', enc: encoders[format] })
    if (hasF16Variant(format)) perfCases.push({ format, variant: 'f32', enc: encodersF32[format] })
  }

  for (const { format, variant, enc } of perfCases) {
    onProgress(`Benchmark: ${format} (${variant}) — 2048²`)
    for (let i = 0; i < WARMUP; i++) {
      await enc.encodeToBytes(benchBitmap)
    }
    const wall: number[] = []
    const gpu: number[] = []
    for (let i = 0; i < RUNS; i++) {
      const r = await enc.encodeToBytes(benchBitmap, { withGpuTime: true })
      wall.push(r.encodeMs)
      if (r.gpuMs !== undefined) gpu.push(r.gpuMs)
    }
    const wallMsMedian = median(wall)
    perf.push({
      format,
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
