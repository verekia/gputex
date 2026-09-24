// Shader A/B evaluation: GPU time AND hardware-decoded quality, per image,
// over a real-texture corpus (WebGPU only, dev tool).
//
// /ab answers "which variant is faster"; this page answers "is the change a
// clear win" — every variant is timed AND scored on the same images in one
// session:
//
//   • Speed: per-dispatch compute time from timestamp queries, `batch`
//     back-to-back dispatches per timed pass, variants interleaved
//     sample-by-sample so clock-state drift cancels out (same method as
//     /ab). Sources are uploaded the way the library does it (rgba8unorm;
//     rg8unorm for bc5 shaders — BC5Encoder.srcTextureFormat).
//   • Quality: the encoded blocks are copied into a real compressed texture
//     (bc1/bc5/bc7/astc/etc2) and a compute pass textureLoad()s it — the
//     GPU's own hardware decoder — against the source, writing per-block
//     squared error. PSNR is over the channels the format stores, visible
//     pixels only. Because decoding runs on the GPU this costs milliseconds
//     even at 4096², so the whole corpus (1K/2K/4K) is scored every run.
//   • Per-block comparison against the FIRST shader: blocks that got worse /
//     better, the worst single-block regression, and the worst regression
//     among blocks the first shader encodes near-losslessly ("easy" blocks —
//     localized artifacts on easy content barely move aggregate PSNR but
//     explode this), plus the byte-identical block fraction.
//
// Shader sources are fetched from /ab/<name>.wgsl (example/scripts/ab-sync.sh
// snapshots the working tree as <fmt>_work and git HEAD as <fmt>_head). The
// format is inferred from the name prefix (bc1, bc5, bc7, astc, etc2).
//
// Query params:
//   shaders=bc7_f16_head,bc7_f16_work   (required)
//   images=default | all | key,key,…    (see IMAGES below; default = the
//                                        format's standard corpus)
//   batch=N  samples=15                  timing (samples=0 skips timing;
//                                        batch defaults to ~10 4K-equivalents)
//   src=rgba8                            bind bc5 shaders to the rgba8 source
//                                        (default: rg8, like BC5Encoder)
//   easy=16                              easy-block threshold: block SSE in
//                                        8-bit units² summed over its pixels
//                                        and stored channels
//
// Automation hook: window.__GPUTEX_EVAL__ = { status, results?, summary?, error? }

import { useEffect, useRef, useState } from 'react'

type Status = 'running' | 'done' | 'error'
type Fmt = 'bc1' | 'bc5' | 'bc7' | 'astc' | 'etc2'

interface Row {
  shader: string
  minMs: number | null
  medianMs: number | null
  /** PSNR over the format's stored channels (bc1/etc2: RGB, bc5: RG, bc7/astc: RGBA). */
  psnr: number
  psnrRgb: number
  psnrA: number
  /** vs the first shader (0 / null for the first). */
  dPsnr: number
  timeRatio: number | null
  worseBlocks: number
  betterBlocks: number
  maxBlockWorse: number
  maxEasyWorse: number
  /** The block behind maxEasyWorse: source RGBA bytes (row-major) and both encodings. */
  maxEasyBlock?: { index: number; bx: number; by: number; src: number[]; base: number[]; mine: number[] }
  identicalPct: number
}

interface ImageResult {
  image: string
  width: number
  height: number
  rows: Row[]
}

interface Summary {
  shader: string
  /** Mean ΔPSNR vs the first shader over all images. */
  meanDPsnr: number
  minDPsnr: number
  maxDPsnr: number
  /** Geometric mean of per-image min-time ratios vs the first shader. */
  geoTimeRatio: number | null
  totalWorse: number
  worstEasy: number
}

declare global {
  interface Window {
    __GPUTEX_EVAL__?: { status: Status; results?: ImageResult[]; summary?: Summary[]; text?: string; error?: string }
  }
}

const rock = (size: string, map: string) => `/textures/Rock064_${size}-JPG/Rock064_${size}-JPG_${map}.jpg`
const wood = (size: string, map: string) => `/textures/WoodFloor004_${size}-JPG/WoodFloor004_${size}-JPG_${map}.jpg`
const packed = (size: number) => `/textures/packed-materials/packed-materials-${size}.png`

const IMAGES: Record<string, string> = {
  color: '/textures/color.png',
  normal: '/textures/normal.png',
  alpha: '/textures/alpha.png',
  'packed-256': packed(256),
  'packed-512': packed(512),
  'packed-1024': packed(1024),
  'packed-2048': packed(2048),
  'packed-4096': packed(4096),
  'proc-color': 'proc:color',
  'proc-alpha': 'proc:alpha',
  'proc-translucent': 'proc:translucent',
  'cutout-rock-1k': `cutout:${rock('1K', 'Color')}`,
  'cutout-wood-4k': `cutout:${wood('4K', 'Color')}`,
}
for (const [size, key] of [
  ['1K', '1k'],
  ['2K', '2k'],
  ['4K', '4k'],
] as const) {
  IMAGES[`rock-color-${key}`] = rock(size, 'Color')
  IMAGES[`rock-normal-${key}`] = rock(size, 'NormalGL')
  IMAGES[`rock-roughness-${key}`] = rock(size, 'Roughness')
  IMAGES[`rock-ao-${key}`] = rock(size, 'AmbientOcclusion')
  IMAGES[`rock-displacement-${key}`] = rock(size, 'Displacement')
  IMAGES[`wood-color-${key}`] = wood(size, 'Color')
  IMAGES[`wood-normal-${key}`] = wood(size, 'NormalGL')
  IMAGES[`wood-roughness-${key}`] = wood(size, 'Roughness')
  IMAGES[`wood-displacement-${key}`] = wood(size, 'Displacement')
}

const COLOR_SET = [
  'color',
  'packed-1024',
  'packed-4096',
  'rock-color-1k',
  'rock-color-4k',
  'rock-roughness-1k',
  'rock-ao-1k',
  'rock-displacement-4k',
  'wood-color-1k',
  'wood-color-4k',
  'wood-roughness-1k',
  'wood-displacement-1k',
  'rock-normal-1k',
  'normal',
]
const DEFAULT_SETS: Record<Fmt, string[]> = {
  bc1: COLOR_SET,
  etc2: COLOR_SET,
  bc7: [...COLOR_SET, 'alpha'],
  astc: [...COLOR_SET, 'alpha'],
  bc5: [
    'normal',
    'rock-normal-1k',
    'rock-normal-2k',
    'rock-normal-4k',
    'wood-normal-1k',
    'wood-normal-4k',
    'rock-displacement-4k',
  ],
}

const FORMAT_INFO: Record<Fmt, { tex: GPUTextureFormat; bpb: number; channels: number[] }> = {
  bc1: { tex: 'bc1-rgba-unorm', bpb: 8, channels: [0, 1, 2] },
  etc2: { tex: 'etc2-rgb8unorm', bpb: 8, channels: [0, 1, 2] },
  bc5: { tex: 'bc5-rg-unorm', bpb: 16, channels: [0, 1] },
  bc7: { tex: 'bc7-rgba-unorm', bpb: 16, channels: [0, 1, 2, 3] },
  astc: { tex: 'astc-4x4-unorm', bpb: 16, channels: [0, 1, 2, 3] },
}

const fmtOf = (name: string): Fmt => {
  for (const f of ['bc1', 'bc5', 'bc7', 'astc', 'etc2'] as const) if (name.startsWith(f)) return f
  throw new Error(`${name}: cannot infer the format from the name (want a bc1/bc5/bc7/astc/etc2 prefix)`)
}

const psnrOf = (sse: number, n: number): number => (sse === 0 ? Infinity : 10 * Math.log10((255 * 255 * n) / sse))

const median = (xs: number[]): number => {
  const s = xs.toSorted((a, b) => a - b)
  const mid = s.length >> 1
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2
}

/**
 * Procedural 2048² images (keys 'proc-*'): gradients + sine detail + hash
 * noise — the /ab and /bench benchmark image — in three alpha flavours:
 * opaque, 'alpha' (ramp that saturates to opaque on the right fifth, so
 * warps mix opaque and translucent blocks) and 'translucent' (never opaque).
 */
function makeProceduralImage(kind: 'color' | 'alpha' | 'translucent', w = 2048, h = 2048): ImageData {
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
      d[i + 3] =
        kind === 'color'
          ? 255
          : kind === 'alpha'
            ? Math.round(255 * Math.min(1, 0.3 + 0.7 * gx + 0.15 * noise))
            : Math.round(255 * (0.2 + 0.5 * gy + 0.2 * noise))
    }
  }
  return img
}

async function loadImageData(url: string): Promise<ImageData> {
  if (url.startsWith('proc:')) return makeProceduralImage(url.slice(5) as 'color' | 'alpha' | 'translucent')
  if (url.startsWith('cutout:')) {
    // Alpha-tested foliage/decal stand-in: the image with a hard mask of
    // scattered discs (alpha 255 inside, 0 outside) — opaque interiors,
    // fully transparent exteriors, mixed blocks along every edge.
    const img = await loadImageData(url.slice(7))
    const { width: w, height: h, data } = img
    const r = w / 10
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const cx = (Math.floor(x / (2.5 * r)) + 0.5) * 2.5 * r
        const cy = (Math.floor(y / (2.5 * r)) + 0.5) * 2.5 * r
        data[(y * w + x) * 4 + 3] = (x - cx) ** 2 + (y - cy) ** 2 < r * r ? 255 : 0
      }
    }
    return img
  }
  const res = await fetch(url)
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`)
  const bmp = await createImageBitmap(await res.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  const canvas = new OffscreenCanvas(bmp.width, bmp.height)
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(bmp, 0, 0)
  bmp.close()
  return ctx.getImageData(0, 0, canvas.width, canvas.height)
}

// Per-block squared error of the hardware-decoded texture vs the source, in
// 8-bit units², per channel. Pixels outside the source (padding) are skipped.
const ERR_WGSL = /* wgsl */ `
struct P { bx: u32, by: u32, w: u32, h: u32 };
@group(0) @binding(0) var dec: texture_2d<f32>;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.bx || g.y >= p.by) { return; }
  var e = vec4<f32>(0.0);
  for (var i = 0u; i < 16u; i++) {
    let c = vec2<u32>(g.x * 4u + (i & 3u), g.y * 4u + (i >> 2u));
    if (c.x < p.w && c.y < p.h) {
      let d = (textureLoad(dec, c, 0) - textureLoad(src, c, 0)) * 255.0;
      e += d * d;
    }
  }
  out[g.y * p.bx + g.x] = e;
}`

interface ShaderCase {
  name: string
  fmt: Fmt
  code: string
  pipeline: GPUComputePipeline
  wgX: number
  wgY: number
  usesSampler: boolean
  rg8: boolean
}

async function runEval(onProgress: (msg: string) => void): Promise<{ results: ImageResult[]; summary: Summary[] }> {
  const q = new URLSearchParams(window.location.search)
  const names = (q.get('shaders') ?? '').split(',').filter(Boolean)
  if (names.length === 0) throw new Error('missing ?shaders=a,b (files under /ab/<name>.wgsl)')
  const fmts = new Set(names.map(fmtOf))
  if (fmts.size !== 1) throw new Error('all shaders must target the same format')
  const fmt = [...fmts][0]!
  const info = FORMAT_INFO[fmt]
  const imagesParam = q.get('images') ?? 'default'
  const imageKeys =
    imagesParam === 'default'
      ? DEFAULT_SETS[fmt]
      : imagesParam === 'all'
        ? Object.keys(IMAGES)
        : imagesParam.split(',').filter(Boolean)
  for (const k of imageKeys) if (!IMAGES[k]) throw new Error(`unknown image '${k}'`)
  // Dispatches per timed pass: default scales with the image so every pass
  // runs a few ms (timestamps quantise to ~6.5 µs; a lone 1K dispatch is
  // ~50 µs).
  const batchParam = q.get('batch')
  const batchFor = (w: number, h: number): number =>
    batchParam ? Number(batchParam) : Math.min(160, Math.max(10, Math.round((10 * 4096 * 4096) / (w * h))))
  const samples = Number(q.get('samples') ?? 15)
  const easy = Number(q.get('easy') ?? 16)
  const WARMUP = 6

  if (!('gpu' in navigator)) throw new Error('WebGPU not available')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('no adapter')
  const wanted: GPUFeatureName[] = [
    'shader-f16',
    'timestamp-query',
    'texture-compression-bc',
    'texture-compression-astc',
    'texture-compression-etc2',
    'subgroups',
  ]
  const device = await adapter.requestDevice({ requiredFeatures: wanted.filter(f => adapter.features.has(f)) })
  if (!device.features.has('timestamp-query') && samples > 0) throw new Error('adapter lacks timestamp-query')

  const cases: ShaderCase[] = []
  for (const name of names) {
    onProgress(`Compiling ${name}…`)
    const res = await fetch(`/ab/${name}.wgsl`, { cache: 'no-store' })
    if (!res.ok) throw new Error(`fetch /ab/${name}.wgsl: ${res.status} — run example/scripts/ab-sync.sh`)
    const code = await res.text()
    const wg = /@workgroup_size\((\d+)\s*,\s*(\d+)/.exec(code)
    if (!wg) throw new Error(`${name}: no @workgroup_size`)
    const module = device.createShaderModule({ code, label: name })
    const msgs = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error')
    if (msgs.length) throw new Error(`${name}: ${msgs.map(m => `${m.lineNum}: ${m.message}`).join('\n')}`)
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'encode' },
    })
    cases.push({
      name,
      fmt,
      code,
      pipeline,
      wgX: Number(wg[1]),
      wgY: Number(wg[2]),
      usesSampler: /:\s*sampler\s*;/.test(code),
      rg8: fmt === 'bc5' && q.get('src') !== 'rgba8',
    })
  }

  const errModule = device.createShaderModule({ code: ERR_WGSL, label: 'eval-err' })
  const errPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: errModule, entryPoint: 'main' },
  })
  const sampler = device.createSampler({ addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' })
  const querySet = samples > 0 ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null
  const resolveBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
  const tsRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })

  const results: ImageResult[] = []
  for (const key of imageKeys) {
    onProgress(`Loading ${key}…`)
    const img = await loadImageData(IMAGES[key]!)
    const w = img.width
    const h = img.height
    const bx = (w + 3) >> 2
    const by = (h + 3) >> 2
    const pw = bx * 4
    const ph = by * 4
    const blocks = bx * by
    const outBytes = blocks * info.bpb

    // Source textures sized to the padded block grid (the library does the
    // same; shaders clamp reads to width-1/height-1).
    const srcRgba = device.createTexture({
      size: [pw, ph],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    })
    device.queue.writeTexture({ texture: srcRgba }, img.data, { bytesPerRow: w * 4 }, [w, h])
    let srcRg: GPUTexture | null = null
    if (cases.some(c => c.rg8)) {
      srcRg = device.createTexture({
        size: [pw, ph],
        format: 'rg8unorm',
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
      })
      const rg = new Uint8Array(w * h * 2)
      for (let i = 0; i < w * h; i++) {
        rg[i * 2] = img.data[i * 4]!
        rg[i * 2 + 1] = img.data[i * 4 + 1]!
      }
      device.queue.writeTexture({ texture: srcRg }, rg, { bytesPerRow: w * 2 }, [w, h])
    }

    // 32-byte params: the canonical 4 × u32 header, zero-padded for shaders
    // that declare extra (zero-default) fields.
    const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(params, 0, new Uint32Array([bx, by, w, h, 0, 0, 0, 0]))

    const batch = batchFor(w, h)
    const perCase = cases.map(c => {
      const dst = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: (c.rg8 ? srcRg! : srcRgba).createView() },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: params } },
      ]
      if (c.usesSampler) entries.push({ binding: 3, resource: sampler })
      const bindGroup = device.createBindGroup({ layout: c.pipeline.getBindGroupLayout(0), entries })
      return { c, dst, bindGroup }
    })

    const dispatch = (pass: GPUComputePassEncoder, pc: (typeof perCase)[number], n: number): void => {
      pass.setPipeline(pc.c.pipeline)
      pass.setBindGroup(0, pc.bindGroup)
      for (let i = 0; i < n; i++) pass.dispatchWorkgroups(Math.ceil(bx / pc.c.wgX), Math.ceil(by / pc.c.wgY), 1)
    }

    // ---- quality: one encode per shader, hardware decode, per-block error
    onProgress(`${key}: encoding + hardware decode…`)
    const blockErr: Float32Array[] = []
    const blockBytes: Uint8Array[] = []
    for (const pc of perCase) {
      const enc = device.createCommandEncoder()
      const pass = enc.beginComputePass()
      dispatch(pass, pc, 1)
      pass.end()
      const comp = device.createTexture({
        size: [pw, ph],
        format: info.tex,
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
      })
      const rowBytes = bx * info.bpb
      if (rowBytes % 256 === 0) {
        enc.copyBufferToTexture({ buffer: pc.dst, bytesPerRow: rowBytes }, { texture: comp }, [pw, ph])
      } else {
        // Re-pitch rows to the 256-byte alignment copyBufferToTexture needs.
        const pitch = Math.ceil(rowBytes / 256) * 256
        const tmp = device.createBuffer({ size: pitch * by, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
        for (let r = 0; r < by; r++) enc.copyBufferToBuffer(pc.dst, r * rowBytes, tmp, r * pitch, rowBytes)
        enc.copyBufferToTexture({ buffer: tmp, bytesPerRow: pitch }, { texture: comp }, [pw, ph])
      }
      const errBuf = device.createBuffer({ size: blocks * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
      const errBg = device.createBindGroup({
        layout: errPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: comp.createView() },
          { binding: 1, resource: srcRgba.createView() },
          { binding: 2, resource: { buffer: errBuf } },
          { binding: 3, resource: { buffer: params } },
        ],
      })
      const ep = enc.beginComputePass()
      ep.setPipeline(errPipeline)
      ep.setBindGroup(0, errBg)
      ep.dispatchWorkgroups(Math.ceil(bx / 8), Math.ceil(by / 8), 1)
      ep.end()
      const errRead = device.createBuffer({
        size: blocks * 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
      const bytesRead = device.createBuffer({
        size: outBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
      enc.copyBufferToBuffer(errBuf, 0, errRead, 0, blocks * 16)
      enc.copyBufferToBuffer(pc.dst, 0, bytesRead, 0, outBytes)
      device.queue.submit([enc.finish()])
      await Promise.all([errRead.mapAsync(GPUMapMode.READ), bytesRead.mapAsync(GPUMapMode.READ)])
      blockErr.push(new Float32Array(errRead.getMappedRange().slice(0)))
      blockBytes.push(new Uint8Array(bytesRead.getMappedRange().slice(0)))
      errRead.destroy()
      bytesRead.destroy()
      errBuf.destroy()
      comp.destroy()
    }

    // ---- speed: interleaved batched timing
    const times: number[][] = perCase.map(() => [])
    if (samples > 0 && querySet) {
      const run = async (pc: (typeof perCase)[number], timed: boolean): Promise<number> => {
        const enc = device.createCommandEncoder()
        const pass = enc.beginComputePass(
          timed ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined,
        )
        dispatch(pass, pc, batch)
        pass.end()
        if (timed) {
          enc.resolveQuerySet(querySet, 0, 2, resolveBuf, 0)
          enc.copyBufferToBuffer(resolveBuf, 0, tsRead, 0, 16)
        }
        device.queue.submit([enc.finish()])
        if (!timed) {
          await device.queue.onSubmittedWorkDone()
          return 0
        }
        await tsRead.mapAsync(GPUMapMode.READ)
        const t = new BigUint64Array(tsRead.getMappedRange())
        const ms = Number(t[1]! - t[0]!) / 1e6 / batch
        tsRead.unmap()
        return ms
      }
      for (let s = 0; s < WARMUP; s++) for (const pc of perCase) await run(pc, false)
      for (let s = 0; s < samples; s++) {
        onProgress(`${key}: timing sample ${s + 1}/${samples}`)
        for (let i = 0; i < perCase.length; i++) times[i]!.push(await run(perCase[i]!, true))
      }
    }

    // ---- stats
    const channels = info.channels
    const visible = w * h
    const blockSse = blockErr.map(e => {
      const out = new Float64Array(blocks)
      for (let b = 0; b < blocks; b++) {
        let s = 0
        for (const ch of channels) s += e[b * 4 + ch]!
        out[b] = s
      }
      return out
    })
    const rows: Row[] = perCase.map((pc, i) => {
      const e = blockErr[i]!
      const tot = [0, 0, 0, 0]
      for (let b = 0; b < blocks; b++) for (let ch = 0; ch < 4; ch++) tot[ch]! += e[b * 4 + ch]!
      const stored = channels.reduce((s, ch) => s + tot[ch]!, 0)
      const psnr = psnrOf(stored, visible * channels.length)
      const psnrRgb = psnrOf(
        tot[0]! + tot[1]! + (channels.length >= 3 ? tot[2]! : 0),
        visible * Math.min(3, channels.length),
      )
      const psnrA = channels.includes(3) ? psnrOf(tot[3]!, visible) : NaN
      let worse = 0
      let better = 0
      let maxWorse = 0
      let maxEasyWorse = 0
      let maxEasyIdx = -1
      let identical = 0
      const base = blockSse[0]!
      const mine = blockSse[i]!
      const b0 = blockBytes[0]!
      const bi = blockBytes[i]!
      for (let b = 0; b < blocks; b++) {
        const d = mine[b]! - base[b]!
        if (d > 1e-3) worse++
        else if (d < -1e-3) better++
        if (d > maxWorse) maxWorse = d
        if (base[b]! <= easy && d > maxEasyWorse) {
          maxEasyWorse = d
          maxEasyIdx = b
        }
        let same = true
        for (let k = 0; k < info.bpb; k++) {
          if (b0[b * info.bpb + k] !== bi[b * info.bpb + k]) {
            same = false
            break
          }
        }
        if (same) identical++
      }
      const t = times[i]!
      return {
        shader: pc.c.name,
        minMs: t.length ? Math.min(...t) : null,
        medianMs: t.length ? median(t) : null,
        psnr,
        psnrRgb,
        psnrA,
        dPsnr: 0,
        timeRatio: null,
        worseBlocks: worse,
        betterBlocks: better,
        maxBlockWorse: maxWorse,
        maxEasyWorse,
        maxEasyBlock:
          maxEasyIdx < 0
            ? undefined
            : (() => {
                const bxi = maxEasyIdx % bx
                const byi = Math.floor(maxEasyIdx / bx)
                const src: number[] = []
                for (let y = 0; y < 4; y++)
                  for (let x = 0; x < 4; x++) {
                    const px = Math.min(bxi * 4 + x, w - 1)
                    const py = Math.min(byi * 4 + y, h - 1)
                    for (let c = 0; c < 4; c++) src.push(img.data[(py * w + px) * 4 + c]!)
                  }
                const slice = (u: Uint8Array): number[] => [
                  ...u.subarray(maxEasyIdx * info.bpb, (maxEasyIdx + 1) * info.bpb),
                ]
                return { index: maxEasyIdx, bx: bxi, by: byi, src, base: slice(b0), mine: slice(bi) }
              })(),
        identicalPct: (100 * identical) / blocks,
      }
    })
    for (const r of rows) {
      r.dPsnr = r.psnr - rows[0]!.psnr
      r.timeRatio = r.minMs !== null && rows[0]!.minMs ? r.minMs / rows[0]!.minMs : null
    }
    results.push({ image: key, width: w, height: h, rows })

    for (const pc of perCase) pc.dst.destroy()
    srcRgba.destroy()
    srcRg?.destroy()
    params.destroy()
  }

  const summary: Summary[] = cases.map((c, i) => {
    const d = results.map(r => r.rows[i]!.dPsnr)
    const ratios = results.map(r => r.rows[i]!.timeRatio).filter((x): x is number => x !== null)
    return {
      shader: c.name,
      meanDPsnr: d.reduce((a, b) => a + b, 0) / d.length,
      minDPsnr: Math.min(...d),
      maxDPsnr: Math.max(...d),
      geoTimeRatio: ratios.length ? Math.exp(ratios.reduce((a, b) => a + Math.log(b), 0) / ratios.length) : null,
      totalWorse: results.reduce((a, r) => a + r.rows[i]!.worseBlocks, 0),
      worstEasy: Math.max(...results.map(r => r.rows[i]!.maxEasyWorse)),
    }
  })

  device.destroy()
  return { results, summary }
}

/** Plain-text table for automation logs. */
function toText(results: ImageResult[], summary: Summary[]): string {
  const lines: string[] = []
  for (const r of results) {
    lines.push(`${r.image} (${r.width}×${r.height})`)
    for (const row of r.rows) {
      lines.push(
        `  ${row.shader.padEnd(22)} ${row.minMs?.toFixed(4) ?? '   -  '} ms ×${row.timeRatio?.toFixed(3) ?? ' -  '}  ` +
          `${row.psnr.toFixed(3)} dB Δ${row.dPsnr >= 0 ? '+' : ''}${row.dPsnr.toFixed(3)}` +
          (Number.isNaN(row.psnrA) ? '' : ` (rgb ${row.psnrRgb.toFixed(2)} a ${row.psnrA.toFixed(2)})`) +
          `  worse ${row.worseBlocks} better ${row.betterBlocks} maxWorse ${row.maxBlockWorse.toFixed(0)} ` +
          `easyWorse ${row.maxEasyWorse.toFixed(1)} same ${row.identicalPct.toFixed(1)}%`,
      )
    }
  }
  lines.push('SUMMARY vs first')
  for (const s of summary) {
    lines.push(
      `  ${s.shader.padEnd(22)} ΔdB mean ${s.meanDPsnr.toFixed(3)} [${s.minDPsnr.toFixed(3)}, ${s.maxDPsnr.toFixed(3)}]  ` +
        `time ×${s.geoTimeRatio?.toFixed(3) ?? '-'}  worse blocks ${s.totalWorse}  worst easy ${s.worstEasy.toFixed(1)}`,
    )
  }
  return lines.join('\n')
}

const EvalPage = () => {
  const [status, setStatus] = useState<Status>('running')
  const [progress, setProgress] = useState('Starting…')
  const [text, setText] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const started = useRef(false)

  useEffect(() => {
    if (started.current) return
    started.current = true
    window.__GPUTEX_EVAL__ = { status: 'running' }
    runEval(setProgress)
      .then(({ results, summary }) => {
        const t = toText(results, summary)
        setText(t)
        setStatus('done')
        window.__GPUTEX_EVAL__ = { status: 'done', results, summary, text: t }
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? (e.stack ?? e.message) : String(e)
        setError(msg)
        setStatus('error')
        window.__GPUTEX_EVAL__ = { status: 'error', error: msg }
      })
  }, [])

  return (
    <div className="min-h-screen bg-neutral-900 p-6 text-sm text-gray-200">
      <h1 className="text-lg font-semibold text-white">GPUtex — shader A/B: speed + hardware-decoded quality</h1>
      <p className="mb-4 text-xs text-gray-400">
        Per-dispatch GPU time (interleaved, batched) and per-block error of the GPU-decoded output, per image.
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
      {text && <pre className="font-mono text-[11px] whitespace-pre">{text}</pre>}
    </div>
  )
}

export default EvalPage
