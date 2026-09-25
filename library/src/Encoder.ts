// Abstract base class for all block-compression encoders.
//
// Each concrete encoder (BC1, BC5, BC7, ASTC_4x4) overrides a handful of
// hooks — shader source, block size, format strings — and inherits the shared
// `encodeToBytes()` pipeline: pad → upload → dispatch → readback into raw
// compressed bytes. The base class is deliberately Three.js-free; wrapping the
// bytes into a `CompressedTexture` lives behind the `gputex/three` entry
// (see ./three/buildTexture.ts), so engines other than Three.js can consume
// the bytes directly.
//
// Every supported format follows the same pattern:
//   • 4×4-pixel blocks
//   • one compute-shader invocation per block, no cross-thread cooperation
//   • 8 or 16 bytes of output per block written to a storage buffer
//   • uniform params buffer shape (at minimum): { blocks_x, blocks_y,
//     width, height } as four u32s
//
// That's enough shared structure to factor out everything except the shader
// and the few format metadata getters.
//
import { uploadSourceTexture } from './workarounds.js'

import type { MipLevel } from './mipgen.js'
import type { TextureFormat } from './TextureFormat.js'

/**
 * Anything `GPUQueue.copyExternalImageToTexture` accepts. Matches the
 * WebGPU spec's CopyExternalImageSource set.
 */
export type EncoderImageSource =
  | ImageBitmap
  | ImageData
  | HTMLImageElement
  | HTMLVideoElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | VideoFrame

export interface EncoderOptions {
  device: GPUDevice
  adapter?: GPUAdapter
  ownsDevice?: boolean
  /**
   * Force the f32 'fast' shader even when the device supports shader-f16.
   * For tests/benchmarks that need to exercise the f32 fallback path on
   * f16-capable hardware. Default false.
   */
  disableF16?: boolean
}

export interface EncodeCallOptions {
  /** Tags the output color space. Forced 'linear' for encoders with supportsSrgb=false. */
  colorSpace?: 'srgb' | 'linear'
}

/**
 * One level of encoded output: the compressed block bytes plus logical and
 * block-aligned dimensions. This is the encoder's native output shape, with
 * no Three.js (or any engine) involvement — feed `data` into whatever
 * renderer's compressed-texture upload you like, or use
 * `buildCompressedTexture()` from `gputex/three`.
 */
export interface EncodedLevelBytes {
  width: number
  height: number
  paddedWidth: number
  paddedHeight: number
  data: Uint8Array
}

/** Result of a raw bytes-only single-image encode. */
export interface EncodeBytesResult extends EncodedLevelBytes {
  encodeMs: number
  /**
   * GPU-side compute-pass time in ms, measured with timestamp queries.
   * Present only when the encode was called with `withGpuTime: true` and the
   * device has the 'timestamp-query' feature (requested automatically by
   * `create()` when available). Browsers quantise timestamps (Chrome: 100µs),
   * so treat small values as approximate.
   */
  gpuMs?: number
}

/** Result of a whole-chain encode — see `Encoder.encodeMipChainToBytes()`. */
export interface EncodeMipChainResult {
  /** Encoded levels in the order given; `levels[0]` is the base level. */
  levels: EncodedLevelBytes[]
  /** Wall-clock time for the whole chain: uploads → dispatches → readback. */
  encodeMs: number
  /** GPU compute time for the whole chain's single compute pass (see
   *  `EncodeBytesResult.gpuMs` for caveats). */
  gpuMs?: number
}

/** GPU-timing plumbing for one submission (timestamp-query based). */
interface GpuTiming {
  querySet: GPUQuerySet
  resolve: GPUBuffer
  staging: GPUBuffer
}

// Chain-encode buffer slices start at 256-byte offsets: bind-group buffer
// offsets must honour min*BufferOffsetAlignment, and 256 is the spec ceiling
// for both the storage and uniform limits.
const CHAIN_ALIGN = 256

// Every shader's Params: { blocks_x, blocks_y, width, height, y0 } — 20
// bytes, bound as a 32-byte slot. y0 is the first block row a dispatch
// covers (row-band encodes); whole-grid dispatches pass 0.
const PARAMS_SIZE = 32
const PARAMS_ALIGN = 256
const MAX_BANDS = 8
// Target output bytes per band. Interleaved A/B on Apple M3 (16 MB BC7/ASTC
// 4096² outputs): 4 bands −25%, 8 bands (2 MB) a further −22%, 16 bands
// (1 MB) +13% back — per-submission/map overhead takes over.
const BAND_BYTES = 2 << 20

/**
 * Split a block grid into row bands for a banded encode: about one band per
 * BAND_BYTES of output (1 below 3 MB, at most MAX_BANDS). Band heights are
 * multiples of the workgroup height, so a band's dispatch never spills
 * into the next band's rows.
 */
function bandRows(blocksY: number, outByteLen: number, wgY: number): { y0: number; rows: number }[] {
  const n = Math.min(MAX_BANDS, Math.max(1, Math.round(outByteLen / BAND_BYTES)))
  const per = Math.ceil(blocksY / n / wgY) * wgY
  const bands: { y0: number; rows: number }[] = []
  for (let y0 = 0; y0 < blocksY; y0 += per) bands.push({ y0, rows: Math.min(per, blocksY - y0) })
  return bands
}

/**
 * Allocate a result array and touch every page of it. A fresh large
 * ArrayBuffer is backed by untouched zero pages, so the first write to each
 * 4 KB page faults — copying a mapped readback into it (or `slice()`-ing
 * the mapped range, which allocates the same way) pays those faults inside
 * the copy: at 16 MB about half of its ~1.3 ms. Called right after
 * `queue.submit()`, the faults are taken while the GPU is still encoding
 * and the copy afterwards runs at memcpy speed.
 */
function allocPrefaulted(byteLength: number): Uint8Array {
  const out = new Uint8Array(byteLength)
  for (let i = 0; i < byteLength; i += 4096) out[i] = 0
  return out
}

/** Per-level geometry for a chain encode: block grid + the level's slice of
 *  the shared output buffer. */
interface ChainGeom {
  width: number
  height: number
  paddedWidth: number
  paddedHeight: number
  blocksX: number
  blocksY: number
  byteLen: number
  dstOffset: number
}

export interface FormatVariant {
  colorSpace: 'srgb' | 'linear'
}

/**
 * Constructor shape for concrete encoder subclasses; used by the polymorphic
 * `Encoder.create()` so the static method's return type narrows to the
 * subclass when you call e.g. `BC1Encoder.create()`.
 *
 * `create` is included so generic code holding an `EncoderConstructor`
 * (like `compressTexture()`'s selected-format branch) can still call
 * `.create()` without a widening cast.
 */
export type EncoderConstructor<T extends Encoder = Encoder> = {
  new (opts: EncoderOptions): T
  requiredFeature: GPUFeatureName | null
  /** Logical formats this encoder can emit (linear first, then sRGB variant). */
  readonly textureFormats: readonly TextureFormat[]
  create(): Promise<T>
}

export abstract class Encoder {
  /**
   * Subclasses set this to the WebGPU feature string the output texture
   * needs for sampling ('texture-compression-bc' / 'texture-compression-astc').
   * `null` means no feature is required (e.g. a pure-storage debug pipeline).
   */
  static readonly requiredFeature: GPUFeatureName | null = null

  /**
   * Create an encoder that owns its own WebGPU device. Requests the
   * subclass's `requiredFeature` if the adapter reports it — missing the
   * feature is non-fatal at encode time (the storage buffer is still
   * written), it only prevents the resulting CompressedTexture from being
   * sampled.
   *
   * The `this: EncoderConstructor<T>` annotation lets `BC1Encoder.create()`
   * return `Promise<BC1Encoder>` instead of `Promise<Encoder>`.
   */
  static async create<T extends Encoder>(this: EncoderConstructor<T>): Promise<T> {
    if (!('gpu' in navigator)) {
      throw new Error('WebGPU not available in this browser')
    }
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) throw new Error('No WebGPU adapter')

    const requiredFeatures: GPUFeatureName[] = []
    if (this.requiredFeature && adapter.features.has(this.requiredFeature)) {
      requiredFeatures.push(this.requiredFeature)
    }
    // f16 powers the ~2× faster 'fast' path. Non-fatal if absent — the encoder
    // falls back to the f32 fast shader.
    if (adapter.features.has('shader-f16')) {
      requiredFeatures.push('shader-f16')
    }
    // Timestamp queries power the opt-in `withGpuTime` shader timing used by
    // the GPU test/benchmark suite. Zero cost unless an encode asks for it.
    if (adapter.features.has('timestamp-query')) {
      requiredFeatures.push('timestamp-query')
    }
    const device = await adapter.requestDevice({ requiredFeatures })
    return new this({ device, adapter, ownsDevice: true })
  }

  readonly device: GPUDevice
  readonly adapter?: GPUAdapter
  readonly ownsDevice: boolean
  readonly disableF16: boolean
  // The single compute pipeline: built from the f16 module when the device
  // supports shader-f16 and the subclass provides an f16 source, from the
  // f32 module otherwise. Both implement the same algorithm. Created with
  // `createComputePipelineAsync` so shader compilation overlaps whatever
  // follows construction (image decode, first upload) instead of stalling
  // the first dispatch; encodes await readiness.
  protected _pipelineReady!: Promise<GPUComputePipeline>
  // Source-preparation pipeline, when the subclass declares one (ETC2's
  // packed-luma + quadrant-average split). Null for direct-source encoders.
  protected _prepPipelineReady: Promise<GPUComputePipeline> | null = null

  // -------------------------------------------------------------------- //
  // Per-encoder GPU resource cache. Creating the source texture, output/
  // staging buffers and bind group on every encode costs ~1ms of host time
  // per call — for small/medium images that overhead dominates the encode
  // (the compute pass itself is tens of µs at 512²). Sequential encodes
  // (the common case: one texture after another, or a mip chain) reuse
  // these; concurrent encodes on the same encoder see `_resourcesBusy` and
  // fall back to transient resources, keeping the API contract unchanged.
  // Buffers are grow-only, the texture is recreated on size change, and the
  // bind group is kept until any bound resource is recreated.
  private _cachedSrcTex: GPUTexture | null = null
  private _cachedSrcW = 0
  private _cachedSrcH = 0
  // Upload memoisation: the ImageBitmap whose pixels the cached source
  // texture currently holds. ImageBitmaps are immutable, so encoding the
  // same bitmap again (benchmark loops, quality-ladder re-encodes, format
  // A/B) can skip the copyExternalImageToTexture entirely — at 4096² that
  // upload is ~9 ms, dominating the whole encode. Mutable sources
  // (ImageData, canvases, video) are never memoised.
  private _cachedSrcSource: ImageBitmap | null = null
  private _cachedSrcFlipY = false
  private _cachedDst: GPUBuffer | null = null
  // One MAP_READ staging buffer and bind group per row band (see bandRows).
  private _cachedStagings: GPUBuffer[] = []
  private _cachedParams: GPUBuffer | null = null
  private _lastParamsKey: string | null = null
  private _cachedBindGroups: GPUBindGroup[] = []
  private _cachedPrepPlanes: GPUTexture[] | null = null
  private _cachedPrepBindGroup: GPUBindGroup | null = null
  private _resourcesBusy = false
  /** Set when the active shader declares a @binding(3) sampler (the BC5
   * kernels read texels through textureGather + clamp-to-edge). */
  private _usesSampler = false
  private _sampler: GPUSampler | null = null

  // Mip-chain cache — the `encodeMipChainToBytes()` counterpart of the
  // single-shot cache above: per-level source textures + bind groups keyed
  // on the exact level-size signature, one params buffer holding every
  // level's uniforms at 256-byte offsets, grow-only output/staging buffers.
  // Pays off when consecutive chains share dimensions (bulk-loading
  // same-sized textures through `compressTexture()`).
  private _chainSig: string | null = null
  private _chainTextures: GPUTexture[] = []
  private _chainPrepPlanes: GPUTexture[][] = []
  private _chainPrepBindGroups: GPUBindGroup[] = []
  private _chainParams: GPUBuffer | null = null
  private _chainBindGroups: GPUBindGroup[] = []
  // Level-0 bind groups of the base-level row bands (see bandRows).
  private _chainBandBindGroups: GPUBindGroup[] = []
  private _chainDst: GPUBuffer | null = null
  // One MAP_READ staging buffer per chain submission (base-level band).
  private _chainStagings: GPUBuffer[] = []
  private _chainBusy = false

  constructor({ device, adapter, ownsDevice = false, disableF16 = false }: EncoderOptions) {
    this.device = device
    this.adapter = adapter
    this.ownsDevice = ownsDevice
    this.disableF16 = disableF16
    this._buildPipeline()
  }

  protected _buildPipeline(): void {
    const device = this.device
    // subclasses override the wgsl source hooks; stubs throw here
    const useF16 = this._useF16
    const code = useF16 ? this.wgslSourceFastF16()! : this.wgslSource()
    this._usesSampler = /@binding\(3\)\s+var\s+\w+\s*:\s*sampler\s*;/.test(code)
    const module = device.createShaderModule({
      label: `${this.label}-encoder${useF16 ? '-f16' : ''}`,
      code,
    })
    this._pipelineReady = device.createComputePipelineAsync({
      label: `${this.label}-encoder-pipeline${useF16 ? '-f16' : ''}`,
      layout: 'auto',
      compute: { module, entryPoint: 'encode' },
    })
    const prepCode = this.wgslPrepSource()
    if (prepCode) {
      const prepModule = device.createShaderModule({ label: `${this.label}-prep`, code: prepCode })
      this._prepPipelineReady = device.createComputePipelineAsync({
        label: `${this.label}-prep-pipeline`,
        layout: 'auto',
        compute: { module: prepModule, entryPoint: 'encode' },
      })
      this._prepPipelineReady.catch(() => {})
    }
    // An encoder may be constructed and never used — don't let an orphaned
    // promise surface as an unhandled rejection. Encodes await the live
    // promise and still observe compile errors.
    this._pipelineReady.catch(() => {})
  }

  /**
   * Resolves once the encoder's compute pipeline(s) have compiled. Encodes
   * await this themselves; call it to compile ahead of first use (see
   * `prewarmCompressTexture()`). Rejects with the compile error, if any.
   */
  async ready(): Promise<void> {
    await this._pipelineReady
    if (this._prepPipelineReady) await this._prepPipelineReady
  }

  destroy(): void {
    this._cachedSrcTex?.destroy()
    if (this._cachedPrepPlanes) for (const t of this._cachedPrepPlanes) t.destroy()
    this._cachedPrepPlanes = null
    this._cachedPrepBindGroup = null
    this._cachedDst?.destroy()
    for (const st of this._cachedStagings) st.destroy()
    this._cachedParams?.destroy()
    this._cachedSrcTex = null
    this._cachedSrcSource = null
    this._cachedDst = null
    this._cachedStagings = []
    this._cachedParams = null
    this._cachedBindGroups = []
    for (const tex of this._chainTextures) tex.destroy()
    this._chainTextures = []
    for (const planes of this._chainPrepPlanes) for (const t of planes) t.destroy()
    this._chainPrepPlanes = []
    this._chainPrepBindGroups = []
    this._chainParams?.destroy()
    this._chainDst?.destroy()
    for (const st of this._chainStagings) st.destroy()
    this._chainParams = null
    this._chainDst = null
    this._chainStagings = []
    this._chainBindGroups = []
    this._chainBandBindGroups = []
    this._chainSig = null
    if (this.ownsDevice) this.device.destroy()
  }

  // ------------------------------------------------------------------ //
  // Abstract hooks — subclasses override                               //
  // ------------------------------------------------------------------ //

  /** Short lowercase identifier used in GPU object labels and errors. */
  abstract get label(): string

  /** 8 for BC1/BC4, 16 for BC5/BC7/ASTC 4×4. */
  abstract get bytesPerBlock(): number

  /** WGSL `@workgroup_size` dimensions. Default 8×8×1. */
  get workgroupSize(): readonly [number, number, number] {
    return [8, 8, 1]
  }

  /** Whether this format has an sRGB variant. Default true. */
  get supportsSrgb(): boolean {
    return true
  }

  /**
   * Format of the single-shot source texture the encode pass samples.
   * Encoders that read only a channel subset can narrow it to cut DRAM
   * traffic on the bandwidth-bound compute pass (BC5 reads rg8unorm — half
   * the bytes of rgba8). Must be valid as a copyExternalImageToTexture
   * destination and renderable. Chain encodes keep rgba8unorm regardless
   * (their inputs are RGBA mip levels / texture views); the WGSL is
   * format-agnostic (`texture_2d<f32>`), so mixing is byte-identical.
   */
  protected get srcTextureFormat(): GPUTextureFormat {
    return 'rgba8unorm'
  }

  /**
   * Optional f16 WGSL variant. Used only when the device reports the
   * `shader-f16` feature; the format's f32 `wgslSource()` is the automatic
   * fallback. Returns null when there's no f16 variant.
   */
  wgslSourceFastF16(): string | null {
    return null
  }

  /** Whether the f16 shader is both available and supported on this device. */
  protected get _useF16(): boolean {
    return !this.disableF16 && this.wgslSourceFastF16() !== null && this.device.features.has('shader-f16')
  }

  /** WGSL compute-shader source (f32; the fallback when f16 is unavailable). */
  abstract wgslSource(): string

  // ------------------------------------------------------------------ //
  // Optional source-preparation pass (null/empty for direct encoders). //
  // A subclass returning a prep shader gets a two-pass encode: the prep
  // pass reads the RGBA8 source (binding 0) and writes the prepared
  // planes as storage textures (plane 0 at binding 1, plane 1 at
  // binding 3, params at binding 2); the encode pass then reads plane 0
  // at binding 0 and plane 1 at binding 3 instead of the source.
  // ------------------------------------------------------------------ //

  /** WGSL for the preparation pass, or null when the encoder reads the
   *  RGBA8 source directly. */
  protected wgslPrepSource(): string | null {
    return null
  }

  /** Formats and sizes of the prepared planes for a padded source size. */
  protected prepPlanes(
    paddedWidth: number,
    paddedHeight: number,
  ): { format: GPUTextureFormat; width: number; height: number }[] {
    void paddedWidth
    void paddedHeight
    return []
  }

  /** Workgroup counts for the prep dispatch. */
  protected prepDispatch(blocksX: number, blocksY: number): [number, number] {
    return [blocksX, blocksY]
  }

  /** Create the prepared-plane textures for one padded source size. */
  private _createPrepPlanes(paddedWidth: number, paddedHeight: number): GPUTexture[] {
    return this.prepPlanes(paddedWidth, paddedHeight).map((p, i) =>
      this.device.createTexture({
        label: `${this.label}-prep-${i}`,
        size: [p.width, p.height, 1],
        format: p.format,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      }),
    )
  }

  /** Bind group for one prep dispatch. */
  private _createPrepBindGroup(
    prepPipeline: GPUComputePipeline,
    srcView: GPUTextureView,
    planes: readonly GPUTexture[],
    params: GPUBuffer,
    paramsOffset = 0,
  ): GPUBindGroup {
    return this.device.createBindGroup({
      label: `${this.label}-prep-bg`,
      layout: prepPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srcView },
        { binding: 1, resource: planes[0]!.createView() },
        { binding: 2, resource: { buffer: params, offset: paramsOffset, size: PARAMS_SIZE } },
        { binding: 3, resource: planes[1]!.createView() },
      ],
    })
  }

  /** e.g. 'bc1-rgba-unorm-srgb'. */
  abstract gpuTextureFormat(opts: FormatVariant): GPUTextureFormat

  /**
   * True if the device reports the feature the output texture needs.
   * The encoder itself only writes to a storage buffer, so this is about
   * whether the result can actually be sampled.
   */
  get supportsSampling(): boolean {
    const feat = (this.constructor as typeof Encoder).requiredFeature
    return !feat || this.device.features.has(feat)
  }

  // ------------------------------------------------------------------ //
  // Shared encodeToBytes() — pad, upload, dispatch, readback.          //
  // ------------------------------------------------------------------ //

  /**
   * Encode one image source to raw compressed bytes. This is the encoder's
   * native, engine-agnostic output. `compressTexture()` and
   * `encodeToTexture()` (both in `gputex/three`) call it and then wrap the
   * bytes into a `CompressedTexture`; callers targeting another engine feed
   * `data` into that engine's compressed-texture upload directly.
   */
  async encodeToBytes(
    source: EncoderImageSource,
    { flipY = false, withGpuTime = false }: { flipY?: boolean; withGpuTime?: boolean } = {},
  ): Promise<EncodeBytesResult> {
    const device = this.device
    // ImageBitmap/VideoFrame/etc. all expose width/height numerically;
    // narrow via the structural type.
    const width = (source as { width: number }).width
    const height = (source as { height: number }).height
    if (!width || !height) {
      throw new Error(`${this.label}Encoder: source has no dimensions`)
    }

    // Pad to the 4×4 block grid. Block-compressed texture dimensions must
    // be multiples of the block size; non-conforming sources get clamp-to-
    // edge shading inside the shader.
    const paddedWidth = (width + 3) & ~3
    const paddedHeight = (height + 3) & ~3
    const blocksX = paddedWidth >> 2
    const blocksY = paddedHeight >> 2
    const blockCount = blocksX * blocksY
    const outByteLen = blockCount * this.bytesPerBlock

    // Resolves instantly after the first encode; only the first one can
    // actually wait on shader compilation.
    const pipeline = await this._pipelineReady
    const prepPipeline = this._prepPipelineReady ? await this._prepPipelineReady : null

    // Acquire GPU resources — from the per-encoder cache when it's free (the
    // common sequential case), transiently when another encode on this
    // encoder is still in flight.
    const useCache = !this._resourcesBusy
    if (useCache) this._resourcesBusy = true

    let srcTex: GPUTexture | undefined
    let dstBuffer: GPUBuffer | undefined
    let paramsBuffer: GPUBuffer | undefined
    const stagings: GPUBuffer[] = []
    const transientStagings: GPUBuffer[] = []
    let transientPrepPlanes: GPUTexture[] | null = null

    try {
      // 1. Source texture sized to the padded block grid. Reused while the
      //    padded size is stable (repeated encodes, same-sized textures).
      let srcTexIsNew = true
      if (useCache && this._cachedSrcTex && this._cachedSrcW === paddedWidth && this._cachedSrcH === paddedHeight) {
        srcTex = this._cachedSrcTex
        srcTexIsNew = false
      } else {
        srcTex = device.createTexture({
          label: `${this.label}-src`,
          size: [paddedWidth, paddedHeight, 1],
          format: this.srcTextureFormat,
          // RENDER_ATTACHMENT is required by copyExternalImageToTexture
          // (internally a blit) even though we never render into this texture.
          usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        })
        if (useCache) {
          this._cachedSrcTex?.destroy()
          this._cachedSrcTex = srcTex
          this._cachedSrcW = paddedWidth
          this._cachedSrcH = paddedHeight
        }
      }
      // Prepared planes track the source texture's size exactly.
      let prepPlanes: GPUTexture[] | null = null
      let prepPlanesNew = false
      if (prepPipeline) {
        if (useCache && !srcTexIsNew && this._cachedPrepPlanes) {
          prepPlanes = this._cachedPrepPlanes
        } else {
          prepPlanes = this._createPrepPlanes(paddedWidth, paddedHeight)
          prepPlanesNew = true
          if (useCache) {
            if (this._cachedPrepPlanes) for (const t of this._cachedPrepPlanes) t.destroy()
            this._cachedPrepPlanes = prepPlanes
            this._cachedPrepBindGroup = null
          } else {
            transientPrepPlanes = prepPlanes
          }
        }
      }
      const uploadSkippable =
        !srcTexIsNew &&
        source instanceof ImageBitmap &&
        this._cachedSrcSource === source &&
        this._cachedSrcFlipY === flipY
      // (Uploading band by band, interleaved with the band dispatches,
      // measured −5..+8% at 2048–4096² — no clear win, not done.)
      if (!uploadSkippable) {
        uploadSourceTexture(device, srcTex, source, width, height, flipY)
      }
      if (useCache) {
        this._cachedSrcSource = source instanceof ImageBitmap ? source : null
        this._cachedSrcFlipY = flipY
      }

      // 2. Output storage buffer (grow-only: a larger cached buffer serves
      //    smaller encodes, e.g. mip levels).
      let dstIsNew = true
      if (useCache && this._cachedDst && this._cachedDst.size >= outByteLen) {
        dstBuffer = this._cachedDst
        dstIsNew = false
      } else {
        dstBuffer = device.createBuffer({
          label: `${this.label}-dst`,
          size: outByteLen,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        })
        if (useCache) {
          this._cachedDst?.destroy()
          this._cachedDst = dstBuffer
        }
      }

      // Large outputs are encoded in ROW BANDS: one submission + MAP_READ
      // staging buffer per band, so the readback of band i (a GPU-process
      // copy that dominates big encodes) overlaps the compute of band i+1.
      const [wgX, wgY] = this.workgroupSize
      const bands = bandRows(blocksY, outByteLen, wgY)
      const rowBytes = blocksX * this.bytesPerBlock
      for (let b = 0; b < bands.length; b++) {
        const len = bands[b]!.rows * rowBytes
        let st = useCache ? this._cachedStagings[b] : undefined
        if (!st || st.size < len) {
          st = device.createBuffer({
            label: `${this.label}-staging-${b}`,
            size: len,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          })
          if (useCache) {
            this._cachedStagings[b]?.destroy()
            this._cachedStagings[b] = st
          } else {
            transientStagings.push(st)
          }
        }
        stagings.push(st)
      }

      // 3. Uniform params — one 32-byte slot per band at PARAMS_ALIGN
      // offsets: { blocksX, blocksY, width, height, y0 }. width/height are
      // the SOURCE dimensions, not the padded ones: the shaders clamp texel
      // reads to (width-1, height-1), which must be the last real texel.
      // Clamping to the padded size would read the zero-initialized padding
      // strip and bleed black into the edge blocks' palettes.
      const paramsData = new Uint32Array((MAX_BANDS * PARAMS_ALIGN) / 4)
      bands.forEach((band, b) => {
        paramsData.set([blocksX, blocksY, width, height, band.y0], (b * PARAMS_ALIGN) / 4)
      })
      const paramsKey = `${blocksX},${blocksY},${width},${height},${bands.length}`
      if (useCache) {
        if (!this._cachedParams) {
          this._cachedParams = device.createBuffer({
            label: `${this.label}-params`,
            size: MAX_BANDS * PARAMS_ALIGN,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          })
          this._lastParamsKey = null
        }
        paramsBuffer = this._cachedParams
        if (this._lastParamsKey !== paramsKey) {
          device.queue.writeBuffer(paramsBuffer, 0, paramsData)
          this._lastParamsKey = paramsKey
        }
      } else {
        paramsBuffer = device.createBuffer({
          label: `${this.label}-params`,
          size: MAX_BANDS * PARAMS_ALIGN,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        device.queue.writeBuffer(paramsBuffer, 0, paramsData)
      }

      // 4. Bind groups (one per band slot), kept until a bound resource is
      //    recreated. With a prep pass the encode pass reads the prepared
      //    planes; without one it reads the source texture directly.
      if (useCache && (srcTexIsNew || dstIsNew || prepPlanesNew)) this._cachedBindGroups = []
      const bindGroups: GPUBindGroup[] = []
      for (let b = 0; b < bands.length; b++) {
        let bg = useCache ? this._cachedBindGroups[b] : undefined
        if (!bg) {
          const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: prepPlanes ? prepPlanes[0]!.createView() : srcTex.createView() },
            { binding: 1, resource: { buffer: dstBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer, offset: b * PARAMS_ALIGN, size: PARAMS_SIZE } },
          ]
          if (prepPlanes) {
            entries.push({ binding: 3, resource: prepPlanes[1]!.createView() })
          } else if (this._usesSampler) {
            this._sampler ??= device.createSampler({
              label: `${this.label}-clamp-sampler`,
              addressModeU: 'clamp-to-edge',
              addressModeV: 'clamp-to-edge',
            })
            entries.push({ binding: 3, resource: this._sampler })
          }
          bg = device.createBindGroup({
            label: `${this.label}-bg-${b}`,
            layout: pipeline.getBindGroupLayout(0),
            entries,
          })
          if (useCache) this._cachedBindGroups[b] = bg
        }
        bindGroups.push(bg)
      }
      let prepBindGroup = useCache && !prepPlanesNew && !srcTexIsNew ? this._cachedPrepBindGroup : null
      if (prepPipeline && prepPlanes && !prepBindGroup) {
        prepBindGroup = this._createPrepBindGroup(prepPipeline, srcTex.createView(), prepPlanes, paramsBuffer)
        if (useCache) this._cachedPrepBindGroup = prepBindGroup
      }

      // 5. Dispatch + readback, one submission per band. When asked (and
      //    the device has 'timestamp-query'), the first pass's start and the
      //    last pass's end are timestamped: shader-only GPU time across all
      //    bands. The readback goes through MAP_READ staging buffers rather
      //    than copyBufferToTexture: the encoder may run on its own device,
      //    separate from the renderer's, so the compressed bytes transit the
      //    CPU anyway before the renderer uploads them.
      const timing = withGpuTime ? this._createTiming() : null
      const t0 = performance.now()
      const maps: Promise<void>[] = []
      for (let b = 0; b < bands.length; b++) {
        const { y0, rows } = bands[b]!
        const first = b === 0
        const last = b === bands.length - 1
        const enc = device.createCommandEncoder({ label: `${this.label}-encode-${b}` })
        // The prep pass writes storage textures the encode pass samples, so
        // they cannot share a compute pass; it runs once, before band 0.
        if (first && prepPipeline && prepBindGroup) {
          const prepPass = enc.beginComputePass(
            timing ? { timestampWrites: { querySet: timing.querySet, beginningOfPassWriteIndex: 0 } } : undefined,
          )
          prepPass.setPipeline(prepPipeline)
          prepPass.setBindGroup(0, prepBindGroup)
          const [px, py] = this.prepDispatch(blocksX, blocksY)
          prepPass.dispatchWorkgroups(Math.ceil(px / wgX), Math.ceil(py / wgY), 1)
          prepPass.end()
        }
        const tsw: GPUComputePassTimestampWrites | undefined =
          timing && (first || last)
            ? {
                querySet: timing.querySet,
                ...(first && !prepPipeline ? { beginningOfPassWriteIndex: 0 } : {}),
                ...(last ? { endOfPassWriteIndex: 1 } : {}),
              }
            : undefined
        const pass = enc.beginComputePass(tsw ? { timestampWrites: tsw } : undefined)
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bindGroups[b]!)
        pass.dispatchWorkgroups(Math.ceil(blocksX / wgX), Math.ceil(rows / wgY), 1)
        pass.end()
        enc.copyBufferToBuffer(dstBuffer, y0 * rowBytes, stagings[b]!, 0, rows * rowBytes)
        if (timing && last) {
          enc.resolveQuerySet(timing.querySet, 0, 2, timing.resolve, 0)
          enc.copyBufferToBuffer(timing.resolve, 0, timing.staging, 0, 16)
        }
        device.queue.submit([enc.finish()])
        maps.push(stagings[b]!.mapAsync(GPUMapMode.READ, 0, rows * rowBytes))
      }

      // 6. Readback. Map only the bytes each band produced — the cached
      //    staging buffers may be larger. The result array is allocated and
      //    pre-faulted while the GPU works (see allocPrefaulted).
      const data = allocPrefaulted(outByteLen)
      for (let b = 0; b < bands.length; b++) {
        const { y0, rows } = bands[b]!
        await maps[b]
        data.set(new Uint8Array(stagings[b]!.getMappedRange(0, rows * rowBytes)), y0 * rowBytes)
        stagings[b]!.unmap()
      }
      const encodeMs = performance.now() - t0

      const gpuMs = timing ? await this._readTimingMs(timing) : undefined

      return { width, height, paddedWidth, paddedHeight, data, encodeMs, gpuMs }
    } finally {
      if (useCache) {
        this._resourcesBusy = false
      } else {
        srcTex?.destroy()
        dstBuffer?.destroy()
        for (const st of transientStagings) st.destroy()
        paramsBuffer?.destroy()
        if (transientPrepPlanes) for (const t of transientPrepPlanes) t.destroy()
      }
    }
  }

  // ------------------------------------------------------------------ //
  // Whole-chain encode — every level in one submission.                //
  // ------------------------------------------------------------------ //

  /**
   * Encode a whole mip chain in ONE GPU submission. A per-level
   * `encodeToBytes()` loop costs a full CPU↔GPU round trip per level (a
   * 1024² chain is 11 levels → 11 `mapAsync` waits with the GPU idle in
   * between); this path uploads every level, records one dispatch per level
   * into a single compute pass, copies all outputs into one staging buffer
   * and maps it once.
   *
   * `levels` are raw RGBA8 pixels in base-to-tail order; sizes don't have to
   * halve level-to-level (each level is padded and clamped independently,
   * exactly like `encodeToBytes`). Uploads use `writeTexture` — the direct
   * raw-bytes path, which also sidesteps the broken
   * `copyExternalImageToTexture` devices (see workarounds.ts). There is no
   * flip option: bake any vertical flip into level 0 before generating the
   * chain, as `compressTexture()` does.
   */
  async encodeMipChainToBytes(
    levels: readonly MipLevel[],
    { withGpuTime = false }: { withGpuTime?: boolean } = {},
  ): Promise<EncodeMipChainResult> {
    const device = this.device
    if (levels.length === 0) {
      throw new Error(`${this.label}Encoder: encodeMipChainToBytes needs at least one level`)
    }
    const pipeline = await this._pipelineReady
    const prepPipeline = this._prepPipelineReady ? await this._prepPipelineReady : null
    const t0 = performance.now()

    levels.forEach((level, i) => {
      if (!level.width || !level.height) {
        throw new Error(`${this.label}Encoder: mip level ${i} has no dimensions`)
      }
      if (level.data.length < level.width * level.height * 4) {
        throw new Error(
          `${this.label}Encoder: mip level ${i} has ${level.data.length} bytes, ` +
            `expected ${level.width * level.height * 4}`,
        )
      }
    })
    const { geoms, byteSpan, bands } = this._chainGeometry(levels)
    const sig = geoms.map(g => `${g.width}x${g.height}`).join()

    const useCache = !this._chainBusy
    if (useCache) this._chainBusy = true

    let textures: GPUTexture[] | undefined
    let prepPlaneSets: GPUTexture[][] | undefined
    let prepBindGroups: GPUBindGroup[] | undefined
    let params: GPUBuffer | undefined
    let bindGroups: GPUBindGroup[] | undefined
    let bandBindGroups: GPUBindGroup[] | undefined
    let dst: GPUBuffer | undefined
    const transientStagings: GPUBuffer[] = []
    // Set when this call built the level-set resources without caching them
    // (another chain encode was in flight) — destroy them on the way out.
    let transientLevelSet = false

    try {
      // Output buffer, grow-only. Recreating it invalidates the cached bind
      // groups (they bind slices of the old buffer).
      let dstIsNew = true
      if (useCache && this._chainDst && this._chainDst.size >= byteSpan) {
        dst = this._chainDst
        dstIsNew = false
      } else {
        dst = device.createBuffer({
          label: `${this.label}-chain-dst`,
          size: byteSpan,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        })
        if (useCache) {
          this._chainDst?.destroy()
          this._chainDst = dst
        }
      }

      // Level-set resources: source textures, the packed params buffer and
      // the bind groups. Valid together only for an exact level-size
      // signature bound to the current dst buffer — reuse needs both.
      if (useCache && !dstIsNew && this._chainSig === sig && this._chainParams) {
        textures = this._chainTextures
        params = this._chainParams
        bindGroups = this._chainBindGroups
        bandBindGroups = this._chainBandBindGroups
        prepPlaneSets = this._chainPrepPlanes
        prepBindGroups = this._chainPrepBindGroups
      } else {
        const texs = geoms.map((g, i) =>
          device.createTexture({
            label: `${this.label}-chain-src-${i}`,
            size: [g.paddedWidth, g.paddedHeight, 1],
            format: 'rgba8unorm',
            // No RENDER_ATTACHMENT: writeTexture is a plain copy, not the
            // copyExternalImageToTexture blit.
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
          }),
        )
        textures = texs

        params = this._createChainParams(geoms, bands)

        if (this._usesSampler) {
          this._sampler ??= device.createSampler({
            label: `${this.label}-clamp-sampler`,
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
          })
        }
        const dstBuf = dst
        const paramsBuf = params
        const planeSets = prepPipeline ? geoms.map(g => this._createPrepPlanes(g.paddedWidth, g.paddedHeight)) : []
        prepPlaneSets = planeSets
        prepBindGroups = prepPipeline
          ? geoms.map((g, i) =>
              this._createPrepBindGroup(prepPipeline, texs[i]!.createView(), planeSets[i]!, paramsBuf, i * CHAIN_ALIGN),
            )
          : []
        const makeBg = (i: number, slot: number): GPUBindGroup => {
          const g = geoms[i]!
          const entries: GPUBindGroupEntry[] = [
            {
              binding: 0,
              resource: prepPipeline ? planeSets[i]![0]!.createView() : texs[i]!.createView(),
            },
            { binding: 1, resource: { buffer: dstBuf, offset: g.dstOffset, size: g.byteLen } },
            { binding: 2, resource: { buffer: paramsBuf, offset: slot * CHAIN_ALIGN, size: PARAMS_SIZE } },
          ]
          if (prepPipeline) {
            entries.push({ binding: 3, resource: planeSets[i]![1]!.createView() })
          } else if (this._usesSampler) {
            entries.push({ binding: 3, resource: this._sampler! })
          }
          return device.createBindGroup({
            label: `${this.label}-chain-bg-${i}`,
            layout: pipeline.getBindGroupLayout(0),
            entries,
          })
        }
        bindGroups = geoms.map((_, i) => makeBg(i, i))
        bandBindGroups = bands.map((_, b) => makeBg(0, geoms.length + b))

        if (useCache) {
          for (const tex of this._chainTextures) tex.destroy()
          for (const planes of this._chainPrepPlanes) for (const t of planes) t.destroy()
          this._chainParams?.destroy()
          this._chainTextures = textures
          this._chainPrepPlanes = planeSets
          this._chainPrepBindGroups = prepBindGroups
          this._chainParams = params
          this._chainBindGroups = bindGroups
          this._chainBandBindGroups = bandBindGroups
          this._chainSig = sig
        } else {
          transientLevelSet = true
        }
      }

      // Upload every level's pixels.
      for (let i = 0; i < levels.length; i++) {
        const level = levels[i]!
        device.queue.writeTexture({ texture: textures[i]! }, level.data, { bytesPerRow: level.width * 4 }, [
          level.width,
          level.height,
          1,
        ])
      }

      // One submission per base-level band (a prep pass first, when the
      // encoder has one; the tail levels ride with the last band).
      return await this._submitChainAndRead(
        pipeline,
        geoms,
        byteSpan,
        bands,
        bindGroups,
        bandBindGroups,
        dst,
        useCache,
        transientStagings,
        withGpuTime,
        t0,
        prepPipeline && prepBindGroups ? { pipeline: prepPipeline, bindGroups: prepBindGroups } : null,
      )
    } finally {
      if (useCache) {
        this._chainBusy = false
      } else {
        dst?.destroy()
      }
      for (const st of transientStagings) st.destroy()
      if (transientLevelSet) {
        if (textures) for (const tex of textures) tex.destroy()
        if (prepPlaneSets) for (const planes of prepPlaneSets) for (const t of planes) t.destroy()
        params?.destroy()
      }
    }
  }

  /**
   * Encode every mip level of a GPU-resident texture in one submission —
   * the zero-CPU-pixels counterpart of `encodeMipChainToBytes()`. Pair it
   * with `generateGpuMipChain()` (gpuMipgen.ts): upload the image once,
   * box-filter the chain on the GPU, then encode straight from the
   * texture's mip views. Pixels never transit the CPU between the source
   * image and the compressed-bytes readback.
   *
   * `srcTex` must be `rgba8unorm` with TEXTURE_BINDING usage; level 0's
   * dimensions are taken from the texture and lower levels follow the
   * standard floor-halving chain. Encoded output is identical to feeding
   * the equivalent CPU chain to `encodeMipChainToBytes()`.
   */
  async encodeMipChainFromTexture(
    srcTex: GPUTexture,
    { withGpuTime = false }: { withGpuTime?: boolean } = {},
  ): Promise<EncodeMipChainResult> {
    const device = this.device
    const pipeline = await this._pipelineReady
    const prepPipeline = this._prepPipelineReady ? await this._prepPipelineReady : null
    const t0 = performance.now()

    const dims: { width: number; height: number }[] = []
    for (let i = 0; i < srcTex.mipLevelCount; i++) {
      dims.push({ width: Math.max(1, srcTex.width >> i), height: Math.max(1, srcTex.height >> i) })
    }
    const { geoms, byteSpan, bands } = this._chainGeometry(dims)

    const useCache = !this._chainBusy
    if (useCache) this._chainBusy = true

    let dst: GPUBuffer | undefined
    const transientStagings: GPUBuffer[] = []
    let params: GPUBuffer | undefined
    let planeSets: GPUTexture[][] | null = null
    try {
      // Grow-only dst (and staging buffers), shared with
      // encodeMipChainToBytes()'s cache slots. Recreating dst invalidates
      // that path's cached bind groups (they bind slices of the old
      // buffer), so drop its signature too.
      if (useCache && this._chainDst && this._chainDst.size >= byteSpan) {
        dst = this._chainDst
      } else {
        dst = device.createBuffer({
          label: `${this.label}-chain-dst`,
          size: byteSpan,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        })
        if (useCache) {
          this._chainDst?.destroy()
          this._chainDst = dst
          this._chainSig = null
        }
      }

      // Params + bind groups are transient: the source texture is new each
      // call, so nothing level-specific is reusable. (~0.1 ms per chain.)
      params = this._createChainParams(geoms, bands)

      if (this._usesSampler) {
        this._sampler ??= device.createSampler({
          label: `${this.label}-clamp-sampler`,
          addressModeU: 'clamp-to-edge',
          addressModeV: 'clamp-to-edge',
        })
      }
      const dstBuf = dst
      const paramsBuf = params
      planeSets = prepPipeline ? geoms.map(g => this._createPrepPlanes(g.paddedWidth, g.paddedHeight)) : null
      const planes = planeSets
      const prepBindGroups =
        prepPipeline && planes
          ? geoms.map((_, i) =>
              this._createPrepBindGroup(
                prepPipeline,
                srcTex.createView({ baseMipLevel: i, mipLevelCount: 1 }),
                planes[i]!,
                paramsBuf,
                i * CHAIN_ALIGN,
              ),
            )
          : null
      const makeBg = (i: number, slot: number): GPUBindGroup => {
        const g = geoms[i]!
        const entries: GPUBindGroupEntry[] = [
          {
            binding: 0,
            resource: planes ? planes[i]![0]!.createView() : srcTex.createView({ baseMipLevel: i, mipLevelCount: 1 }),
          },
          { binding: 1, resource: { buffer: dstBuf, offset: g.dstOffset, size: g.byteLen } },
          { binding: 2, resource: { buffer: paramsBuf, offset: slot * CHAIN_ALIGN, size: PARAMS_SIZE } },
        ]
        if (planes) {
          entries.push({ binding: 3, resource: planes[i]![1]!.createView() })
        } else if (this._usesSampler) {
          entries.push({ binding: 3, resource: this._sampler! })
        }
        return device.createBindGroup({
          label: `${this.label}-chain-bg-${i}`,
          layout: pipeline.getBindGroupLayout(0),
          entries,
        })
      }
      const bindGroups = geoms.map((_, i) => makeBg(i, i))
      const bandBindGroups = bands.map((_, b) => makeBg(0, geoms.length + b))

      return await this._submitChainAndRead(
        pipeline,
        geoms,
        byteSpan,
        bands,
        bindGroups,
        bandBindGroups,
        dst,
        useCache,
        transientStagings,
        withGpuTime,
        t0,
        prepPipeline && prepBindGroups ? { pipeline: prepPipeline, bindGroups: prepBindGroups } : null,
      )
    } finally {
      if (useCache) {
        this._chainBusy = false
      } else {
        dst?.destroy()
      }
      for (const st of transientStagings) st.destroy()
      // Safe immediately after submit: destruction is deferred until the
      // GPU is done with the buffer/textures.
      params?.destroy()
      if (planeSets) for (const planes of planeSets) for (const t of planes) t.destroy()
    }
  }

  /** Block-grid geometry + packed output offsets for a chain of levels.
   *  `byteSpan` is both the dst buffer size and the readback copy size (a
   *  multiple of 4: byteLen is a multiple of bytesPerBlock ≥ 8, offsets are
   *  CHAIN_ALIGN-ed). `bands` splits the base level into row bands. */
  private _chainGeometry(dims: readonly { width: number; height: number }[]): {
    geoms: ChainGeom[]
    byteSpan: number
    bands: { y0: number; rows: number }[]
  } {
    let dstCursor = 0
    const geoms = dims.map(({ width, height }) => {
      const paddedWidth = (width + 3) & ~3
      const paddedHeight = (height + 3) & ~3
      const blocksX = paddedWidth >> 2
      const blocksY = paddedHeight >> 2
      const byteLen = blocksX * blocksY * this.bytesPerBlock
      const dstOffset = dstCursor
      dstCursor = Math.ceil((dstCursor + byteLen) / CHAIN_ALIGN) * CHAIN_ALIGN
      return { width, height, paddedWidth, paddedHeight, blocksX, blocksY, byteLen, dstOffset }
    })
    const last = geoms[geoms.length - 1]!
    const base = geoms[0]!
    return {
      geoms,
      byteSpan: last.dstOffset + last.byteLen,
      bands: bandRows(base.blocksY, base.byteLen, this.workgroupSize[1]),
    }
  }

  /** One uniform buffer, one write: level i's { blocksX, blocksY, width,
   *  height, y0 = 0 } at slot i, then base-level band b at slot
   *  levels + b (y0 = the band's first block row). Slots are CHAIN_ALIGN
   *  apart. */
  private _createChainParams(geoms: readonly ChainGeom[], bands: readonly { y0: number }[]): GPUBuffer {
    const slots = geoms.length + bands.length
    const params = this.device.createBuffer({
      label: `${this.label}-chain-params`,
      size: slots * CHAIN_ALIGN,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const data = new Uint32Array((slots * CHAIN_ALIGN) / 4)
    geoms.forEach((g, i) => data.set([g.blocksX, g.blocksY, g.width, g.height, 0], (i * CHAIN_ALIGN) / 4))
    const g0 = geoms[0]!
    bands.forEach((band, b) =>
      data.set([g0.blocksX, g0.blocksY, g0.width, g0.height, band.y0], ((geoms.length + b) * CHAIN_ALIGN) / 4),
    )
    this.device.queue.writeBuffer(params, 0, data)
    return params
  }

  /** Shared chain-encode tail: one submission per base-level row band —
   *  band b dispatches its rows of level 0, the last band also every tail
   *  level — each read back through its own staging buffer as it lands,
   *  then sliced into per-level byte arrays. */
  private async _submitChainAndRead(
    pipeline: GPUComputePipeline,
    geoms: readonly ChainGeom[],
    byteSpan: number,
    bands: readonly { y0: number; rows: number }[],
    bindGroups: readonly GPUBindGroup[],
    bandBindGroups: readonly GPUBindGroup[],
    dst: GPUBuffer,
    useCache: boolean,
    transientStagings: GPUBuffer[],
    withGpuTime: boolean,
    t0: number,
    prep: { pipeline: GPUComputePipeline; bindGroups: readonly GPUBindGroup[] } | null = null,
  ): Promise<EncodeMipChainResult> {
    const device = this.device
    const timing = withGpuTime ? this._createTiming() : null
    const [wgX, wgY] = this.workgroupSize
    const base = geoms[0]!
    const rowBytes = base.blocksX * this.bytesPerBlock

    // Readback ranges of dst: band b's rows of level 0; the last band runs
    // on to the end of the chain (the tail levels follow level 0 in dst).
    const ranges = bands.map((band, b) => {
      const start = base.dstOffset + band.y0 * rowBytes
      const end = b === bands.length - 1 ? byteSpan : start + band.rows * rowBytes
      return { start, len: end - start }
    })
    const stagings = ranges.map(({ len }, b) => {
      let st = useCache ? this._chainStagings[b] : undefined
      if (!st || st.size < len) {
        st = device.createBuffer({
          label: `${this.label}-chain-staging-${b}`,
          size: len,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
        if (useCache) {
          this._chainStagings[b]?.destroy()
          this._chainStagings[b] = st
        } else {
          transientStagings.push(st)
        }
      }
      return st
    })

    const maps: Promise<void>[] = []
    for (let b = 0; b < bands.length; b++) {
      const first = b === 0
      const last = b === bands.length - 1
      const enc = device.createCommandEncoder({ label: `${this.label}-encode-chain-${b}` })
      // Prep writes storage textures the encode pass samples — separate
      // passes; it covers every level, before band 0.
      if (first && prep) {
        const prepPass = enc.beginComputePass(
          timing ? { timestampWrites: { querySet: timing.querySet, beginningOfPassWriteIndex: 0 } } : undefined,
        )
        prepPass.setPipeline(prep.pipeline)
        for (let i = 0; i < geoms.length; i++) {
          const g = geoms[i]!
          prepPass.setBindGroup(0, prep.bindGroups[i]!)
          const [px, py] = this.prepDispatch(g.blocksX, g.blocksY)
          prepPass.dispatchWorkgroups(Math.ceil(px / wgX), Math.ceil(py / wgY), 1)
        }
        prepPass.end()
      }
      const tsw: GPUComputePassTimestampWrites | undefined =
        timing && (first || last)
          ? {
              querySet: timing.querySet,
              ...(first && !prep ? { beginningOfPassWriteIndex: 0 } : {}),
              ...(last ? { endOfPassWriteIndex: 1 } : {}),
            }
          : undefined
      const pass = enc.beginComputePass(tsw ? { timestampWrites: tsw } : undefined)
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bandBindGroups[b]!)
      pass.dispatchWorkgroups(Math.ceil(base.blocksX / wgX), Math.ceil(bands[b]!.rows / wgY), 1)
      if (last) {
        for (let i = 1; i < geoms.length; i++) {
          const g = geoms[i]!
          pass.setBindGroup(0, bindGroups[i]!)
          pass.dispatchWorkgroups(Math.ceil(g.blocksX / wgX), Math.ceil(g.blocksY / wgY), 1)
        }
      }
      pass.end()
      enc.copyBufferToBuffer(dst, ranges[b]!.start, stagings[b]!, 0, ranges[b]!.len)
      if (timing && last) {
        enc.resolveQuerySet(timing.querySet, 0, 2, timing.resolve, 0)
        enc.copyBufferToBuffer(timing.resolve, 0, timing.staging, 0, 16)
      }
      device.queue.submit([enc.finish()])
      maps.push(stagings[b]!.mapAsync(GPUMapMode.READ, 0, ranges[b]!.len))
    }

    // The per-level result arrays are allocated and pre-faulted while the
    // GPU works (see allocPrefaulted).
    const out: EncodedLevelBytes[] = geoms.map(g => ({
      width: g.width,
      height: g.height,
      paddedWidth: g.paddedWidth,
      paddedHeight: g.paddedHeight,
      data: allocPrefaulted(g.byteLen),
    }))
    for (let b = 0; b < bands.length; b++) {
      const { start, len } = ranges[b]!
      await maps[b]
      const mapped = stagings[b]!.getMappedRange(0, len)
      if (b < bands.length - 1) {
        out[0]!.data.set(new Uint8Array(mapped), start - base.dstOffset)
      } else {
        // Rest of level 0, then every tail level.
        const lvl0Rest = base.dstOffset + base.byteLen - start
        out[0]!.data.set(new Uint8Array(mapped, 0, lvl0Rest), start - base.dstOffset)
        for (let i = 1; i < geoms.length; i++) {
          const g = geoms[i]!
          out[i]!.data.set(new Uint8Array(mapped, g.dstOffset - start, g.byteLen))
        }
      }
      stagings[b]!.unmap()
    }
    const encodeMs = performance.now() - t0
    const gpuMs = timing ? await this._readTimingMs(timing) : undefined

    return { levels: out, encodeMs, gpuMs }
  }

  // ------------------------------------------------------------------ //
  // GPU timing plumbing (timestamp queries)                            //
  // ------------------------------------------------------------------ //

  /** Create the query set + resolve/staging buffers for one timed
   *  submission, or null when the device lacks 'timestamp-query'. */
  private _createTiming(): GpuTiming | null {
    const device = this.device
    if (!device.features.has('timestamp-query')) return null
    return {
      querySet: device.createQuerySet({ type: 'timestamp', count: 2 }),
      resolve: device.createBuffer({
        label: `${this.label}-ts-resolve`,
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      }),
      staging: device.createBuffer({
        label: `${this.label}-ts-staging`,
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
    }
  }

  /** Read back a timed submission's pass duration (ms) and destroy the
   *  timing objects. Timestamps are u64 nanoseconds. */
  private async _readTimingMs(timing: GpuTiming): Promise<number | undefined> {
    await timing.staging.mapAsync(GPUMapMode.READ)
    const [begin, end] = new BigUint64Array(timing.staging.getMappedRange().slice(0))
    timing.staging.unmap()
    timing.staging.destroy()
    timing.resolve.destroy()
    timing.querySet.destroy()
    if (end !== undefined && begin !== undefined && end > begin) {
      return Number(end - begin) / 1e6
    }
    return undefined
  }
}
