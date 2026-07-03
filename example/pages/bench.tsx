// Per-size encode benchmark (WebGPU only). Complements /test (which times a
// single 2048² image): measures median end-to-end encodeToBytes() wall time
// for each format × image size — the numbers that matter for
// runtime texture streaming, where per-encode host overhead dominates small
// and medium sizes.
//
// Automation hook: window.__GPUTEX_BENCH__ = { status, results?, error? }
// results = [{ format, size, wallMsMedian, mpixPerSec }]

import { useEffect, useRef, useState } from 'react'

import {
  ASTC4x4Encoder,
  ASTC4x4WebGLEncoder,
  BC1Encoder,
  BC1WebGLEncoder,
  BC5Encoder,
  BC5WebGLEncoder,
  BC7Encoder,
  BC7WebGLEncoder,
  generateMipChain,
  padToBlockMultiple,
} from 'gputex'
import type { Encoder } from 'gputex'

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
import { clearTranscodeCache, compressTexture, releaseSharedGpuResources, setTranscodeCacheLimit } from 'gputex/three'

type Status = 'running' | 'done' | 'error'

interface BenchResult {
  format: string
  size: number
  wallMsMedian: number
  mpixPerSec: number
}

declare global {
  interface Window {
    __GPUTEX_BENCH__?: { status: Status; results?: BenchResult[]; error?: string }
    /** Encoder classes, exposed for ad-hoc console/automation experiments. */
    __GPUTEX_ENCODERS__?: Record<string, unknown>
  }
}

if (typeof window !== 'undefined') {
  window.__GPUTEX_ENCODERS__ = {
    BC1Encoder,
    BC5Encoder,
    BC7Encoder,
    ASTC4x4Encoder,
    BC1WebGLEncoder,
    BC5WebGLEncoder,
    BC7WebGLEncoder,
    ASTC4x4WebGLEncoder,
    decodeBC1Block,
    decodeBC5Block,
    decodeBC7Block,
    decodeASTC4x4Block,
    encodeBC1Block,
    encodeBC5Block,
    encodeBC7Mode6Block,
    encodeASTC4x4Block,
    compressTexture,
    releaseSharedGpuResources,
    clearTranscodeCache,
    setTranscodeCacheLimit,
    generateMipChain,
    padToBlockMultiple,
  }
}

const SIZES = [256, 512, 1024, 2048, 4096]
const WARMUP = 5
const RUNS = 30

const median = (xs: number[]): number => {
  const s = xs.toSorted((a, b) => a - b)
  const mid = s.length >> 1
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2
}

/** Procedural RGBA test image: gradients + sine detail + hash noise. */
function makeProceduralImage(w: number, h: number): ImageData {
  const img = new ImageData(w, h)
  const d = img.data
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4
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
      d[i + 3] = 255
    }
  }
  return img
}

async function runBench(onProgress: (msg: string) => void): Promise<BenchResult[]> {
  if (!('gpu' in navigator)) throw new Error('WebGPU not available in this browser')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('No WebGPU adapter')
  const requestable: GPUFeatureName[] = ['texture-compression-bc', 'texture-compression-astc', 'shader-f16']
  const device = await adapter.requestDevice({
    requiredFeatures: requestable.filter(f => adapter.features.has(f)),
  })

  const encoders: Array<[string, Encoder]> = [
    ['bc1', new BC1Encoder({ device, adapter })],
    ['bc5', new BC5Encoder({ device, adapter })],
    ['bc7', new BC7Encoder({ device, adapter })],
    ['astc', new ASTC4x4Encoder({ device, adapter })],
  ]

  onProgress('Preparing images…')
  const bitmaps = new Map<number, ImageBitmap>()
  for (const size of SIZES) {
    bitmaps.set(
      size,
      await createImageBitmap(makeProceduralImage(size, size), {
        colorSpaceConversion: 'none',
        premultiplyAlpha: 'none',
      }),
    )
  }

  // Ramp the GPU out of its idle clock state before timing anything. The
  // encodes are cheap, so it takes a burst of them to move the clocks.
  onProgress('Warming up GPU clocks…')
  const bigBitmap = bitmaps.get(4096)!
  for (let i = 0; i < 20; i++) {
    await encoders[2]![1].encodeToBytes(bigBitmap)
  }

  const results: BenchResult[] = []
  for (const [format, enc] of encoders) {
    for (const size of SIZES) {
      onProgress(`Benchmark: ${format} fast — ${size}²`)
      const bitmap = bitmaps.get(size)!
      for (let i = 0; i < WARMUP; i++) {
        await enc.encodeToBytes(bitmap)
      }
      const wall: number[] = []
      for (let i = 0; i < RUNS; i++) {
        const r = await enc.encodeToBytes(bitmap)
        wall.push(r.encodeMs)
      }
      const wallMsMedian = median(wall)
      results.push({ format, size, wallMsMedian, mpixPerSec: (size * size) / 1e6 / (wallMsMedian / 1000) })
    }
  }

  for (const [, enc] of encoders) enc.destroy()
  for (const b of bitmaps.values()) b.close()
  device.destroy()
  return results
}

const BenchPage = () => {
  const [status, setStatus] = useState<Status>('running')
  const [progress, setProgress] = useState('Starting…')
  const [results, setResults] = useState<BenchResult[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const started = useRef(false)

  useEffect(() => {
    if (started.current) return
    started.current = true
    window.__GPUTEX_BENCH__ = { status: 'running' }
    runBench(setProgress)
      .then(r => {
        setResults(r)
        setStatus('done')
        window.__GPUTEX_BENCH__ = { status: 'done', results: r }
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? (e.stack ?? e.message) : String(e)
        setError(msg)
        setStatus('error')
        window.__GPUTEX_BENCH__ = { status: 'error', error: msg }
      })
  }, [])

  const formats = results ? [...new Set(results.map(r => r.format))] : []

  return (
    <div className="min-h-screen bg-neutral-900 p-6 text-sm text-gray-200">
      <h1 className="text-lg font-semibold text-white">GPUtex — per-size encode benchmark</h1>
      <p className="mb-4 text-xs text-gray-400">
        Median end-to-end <code>encodeToBytes()</code> wall time, {RUNS} runs per cell.
      </p>

      {status === 'running' && (
        <div className="rounded-lg border border-blue-500/30 bg-blue-500/10 p-3 font-mono text-xs text-blue-200">
          Running… {progress}
        </div>
      )}
      {status === 'error' && (
        <pre className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-xs whitespace-pre-wrap text-red-300">
          {error}
        </pre>
      )}

      {results && (
        <table className="border-collapse">
          <thead>
            <tr>
              <th className="border-b border-white/10 px-2 py-1 text-left text-[11px] font-semibold text-gray-300">
                format
              </th>
              {SIZES.map(s => (
                <th
                  key={s}
                  className="border-b border-white/10 px-2 py-1 text-left text-[11px] font-semibold text-gray-300"
                >
                  {s}²
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {formats.map(f => (
              <tr key={f}>
                <td className="border-b border-white/5 px-2 py-1 font-mono text-[11px] text-gray-200">{f}</td>
                {SIZES.map(s => {
                  const r = results.find(x => x.format === f && x.size === s)
                  return (
                    <td key={s} className="border-b border-white/5 px-2 py-1 font-mono text-[11px] text-gray-200">
                      {r ? `${r.wallMsMedian.toFixed(2)} ms` : '—'}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default BenchPage
