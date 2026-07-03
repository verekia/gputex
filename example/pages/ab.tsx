// Raw-WGSL A/B shader benchmark (WebGPU only, dev tool).
//
// Times encoder compute shaders the same way the external comparison harness
// does: each shader gets its own pipeline (layout 'auto') and its own
// @workgroup_size mapping, all shaders read one shared rgba8unorm source
// texture, and a GPU timestamp query brackets a compute pass of many
// back-to-back dispatches (GPU saturated, clocks boosted). Variants are
// INTERLEAVED sample-by-sample in one session, so clock-state bimodality
// cancels out of the comparison — cross-session timings are not comparable.
// Reported per shader: min of samples (peak sustained throughput) and median.
//
// Shader sources are fetched from /ab/<name>.wgsl — run
// example/scripts/ab-sync.sh to snapshot the working tree (…_work.wgsl) and
// git HEAD (…_head.wgsl) into public/ab/ after every edit.
//
// Query params:
//   shaders=astc_head,astc_work   (required, comma-separated /ab/ names)
//   size=4096      source texture size (default 4096)
//   batch=10       dispatches per timed pass (default 10)
//   samples=25     timed samples per shader (default 25, +8 warmup)
//   image=proc     proc | gray | alpha | /textures/... URL (default proc)
//
// Automation hook: window.__GPUTEX_AB__ = { status, results?, error? }
// results = [{ name, minMs, medianMs, ratioVsFirst }] (per-dispatch ms)

import { useEffect, useRef, useState } from 'react'

type Status = 'running' | 'done' | 'error'

interface AbResult {
  name: string
  minMs: number
  medianMs: number
  ratioVsFirst: number
  samplesMs: number[]
}

declare global {
  interface Window {
    __GPUTEX_AB__?: { status: Status; results?: AbResult[]; error?: string }
  }
}

const median = (xs: number[]): number => {
  const s = xs.toSorted((a, b) => a - b)
  const mid = s.length >> 1
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2
}

/** Procedural RGBA test image: gradients + sine detail + hash noise. */
function makeProceduralImage(w: number, h: number, kind: 'proc' | 'gray' | 'alpha'): ImageData {
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
      if (kind === 'gray') {
        const v = Math.round(255 * Math.min(1, gx * 0.5 + gy * 0.2 + sines * 0.2 + noise * 0.1))
        d[i] = v
        d[i + 1] = v
        d[i + 2] = v
        d[i + 3] = 255
      } else {
        d[i] = Math.round(255 * Math.min(1, gx * 0.8 + noise * 0.2))
        d[i + 1] = Math.round(255 * Math.min(1, gy * 0.7 + sines * 0.3))
        d[i + 2] = Math.round(255 * Math.min(1, (1 - gx) * 0.6 + noise * 0.25 + sines * 0.15))
        d[i + 3] = kind === 'alpha' ? Math.round(255 * Math.min(1, 0.3 + 0.7 * gx + 0.15 * noise)) : 255
      }
    }
  }
  return img
}

async function loadImageData(url: string): Promise<ImageData> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`)
  const blob = await res.blob()
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  const canvas = new OffscreenCanvas(bmp.width, bmp.height)
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(bmp, 0, 0)
  bmp.close()
  return ctx.getImageData(0, 0, canvas.width, canvas.height)
}

interface ShaderCase {
  name: string
  pipeline: GPUComputePipeline
  bindGroup: GPUBindGroup
  wgX: number
  wgY: number
}

async function runAb(onProgress: (msg: string) => void): Promise<AbResult[]> {
  const q = new URLSearchParams(window.location.search)
  const names = (q.get('shaders') ?? '').split(',').filter(Boolean)
  if (names.length === 0) throw new Error('missing ?shaders=a,b (files under /ab/<name>.wgsl)')
  const size = Number(q.get('size') ?? 4096)
  const batch = Number(q.get('batch') ?? 10)
  const samples = Number(q.get('samples') ?? 25)
  const WARMUP = 8
  const imageParam = q.get('image') ?? 'proc'

  if (!('gpu' in navigator)) throw new Error('WebGPU not available')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('no adapter')
  for (const f of ['shader-f16', 'timestamp-query'] as GPUFeatureName[]) {
    if (!adapter.features.has(f)) throw new Error(`adapter lacks ${f}`)
  }
  const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16', 'timestamp-query'] })

  onProgress('Preparing source texture…')
  const img = imageParam.startsWith('/')
    ? await loadImageData(imageParam)
    : makeProceduralImage(size, size, imageParam as 'proc' | 'gray' | 'alpha')
  const w = img.width
  const h = img.height
  const blocksX = (w + 3) >> 2
  const blocksY = (h + 3) >> 2

  const tex = device.createTexture({
    size: [w, h],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  })
  device.queue.writeTexture({ texture: tex }, img.data, { bytesPerRow: w * 4 }, [w, h])

  const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(uniform, 0, new Uint32Array([blocksX, blocksY, w, h]))

  const cases: ShaderCase[] = []
  for (const name of names) {
    onProgress(`Compiling ${name}…`)
    const res = await fetch(`/ab/${name}.wgsl`, { cache: 'no-store' })
    if (!res.ok) throw new Error(`fetch /ab/${name}.wgsl: ${res.status} — run example/scripts/ab-sync.sh`)
    const code = await res.text()
    const wg = /@workgroup_size\((\d+)\s*,\s*(\d+)/.exec(code)
    if (!wg) throw new Error(`${name}: no @workgroup_size`)
    const bpb = name.includes('bc1') ? 8 : 16
    const dst = device.createBuffer({ size: blocksX * blocksY * bpb, usage: GPUBufferUsage.STORAGE })
    const module = device.createShaderModule({ code })
    const info = await module.getCompilationInfo()
    const errors = info.messages.filter(m => m.type === 'error')
    if (errors.length) throw new Error(`${name}: ${errors.map(m => `${m.lineNum}: ${m.message}`).join('\n')}`)
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'encode' } })
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: tex.createView() },
      { binding: 1, resource: { buffer: dst } },
      { binding: 2, resource: { buffer: uniform } },
    ]
    if (/:\s*sampler\s*;/.test(code)) {
      entries.push({
        binding: 3,
        resource: device.createSampler({ addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' }),
      })
    }
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries })
    cases.push({ name, pipeline, bindGroup, wgX: Number(wg[1]), wgY: Number(wg[2]) })
  }

  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 })
  const resolveBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
  const readBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })

  /** One timed sample: `batch` back-to-back dispatches in a single pass. */
  const runSample = async (c: ShaderCase, timed: boolean): Promise<number> => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass(
      timed ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined,
    )
    pass.setPipeline(c.pipeline)
    pass.setBindGroup(0, c.bindGroup)
    for (let i = 0; i < batch; i++) {
      pass.dispatchWorkgroups(Math.ceil(blocksX / c.wgX), Math.ceil(blocksY / c.wgY), 1)
    }
    pass.end()
    if (timed) {
      enc.resolveQuerySet(querySet, 0, 2, resolveBuf, 0)
      enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, 16)
    }
    device.queue.submit([enc.finish()])
    if (!timed) {
      await device.queue.onSubmittedWorkDone()
      return 0
    }
    await readBuf.mapAsync(GPUMapMode.READ)
    const t = new BigUint64Array(readBuf.getMappedRange())
    const ms = Number(t[1]! - t[0]!) / 1e6 / batch
    readBuf.unmap()
    return ms
  }

  // Warmup: boost clocks and shake out first-dispatch compilation, all
  // variants equally.
  for (let s = 0; s < WARMUP; s++) {
    onProgress(`Warmup ${s + 1}/${WARMUP}…`)
    for (const c of cases) await runSample(c, false)
  }

  // Timed, interleaved round-robin.
  const times = new Map<string, number[]>(cases.map(c => [c.name, []]))
  for (let s = 0; s < samples; s++) {
    onProgress(`Sample ${s + 1}/${samples}…`)
    for (const c of cases) {
      times.get(c.name)!.push(await runSample(c, true))
    }
  }

  const mins = cases.map(c => Math.min(...times.get(c.name)!))
  const results: AbResult[] = cases.map((c, i) => ({
    name: c.name,
    minMs: mins[i]!,
    medianMs: median(times.get(c.name)!),
    ratioVsFirst: mins[i]! / mins[0]!,
    samplesMs: times.get(c.name)!,
  }))

  device.destroy()
  return results
}

const AbPage = () => {
  const [status, setStatus] = useState<Status>('running')
  const [progress, setProgress] = useState('Starting…')
  const [results, setResults] = useState<AbResult[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const started = useRef(false)

  useEffect(() => {
    if (started.current) return
    started.current = true
    window.__GPUTEX_AB__ = { status: 'running' }
    runAb(setProgress)
      .then(r => {
        setResults(r)
        setStatus('done')
        window.__GPUTEX_AB__ = { status: 'done', results: r }
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? (e.stack ?? e.message) : String(e)
        setError(msg)
        setStatus('error')
        window.__GPUTEX_AB__ = { status: 'error', error: msg }
      })
  }, [])

  return (
    <div className="min-h-screen bg-neutral-900 p-6 text-sm text-gray-200">
      <h1 className="text-lg font-semibold text-white">GPUtex — raw-WGSL A/B shader benchmark</h1>
      <p className="mb-4 text-xs text-gray-400">
        Per-dispatch GPU time via timestamp queries, batched dispatches, variants interleaved in-session.
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
        <table className="border-collapse font-mono text-[11px]">
          <thead>
            <tr>
              {['shader', 'min ms', 'median ms', 'vs first'].map(hd => (
                <th key={hd} className="border-b border-white/10 px-2 py-1 text-left font-semibold text-gray-300">
                  {hd}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {results.map(r => (
              <tr key={r.name}>
                <td className="border-b border-white/5 px-2 py-1">{r.name}</td>
                <td className="border-b border-white/5 px-2 py-1">{r.minMs.toFixed(4)}</td>
                <td className="border-b border-white/5 px-2 py-1">{r.medianMs.toFixed(4)}</td>
                <td className="border-b border-white/5 px-2 py-1">{r.ratioVsFirst.toFixed(3)}×</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default AbPage
