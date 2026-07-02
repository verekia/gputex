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

/**
 * Encoder quality level. 'fast' (default) uses the projection-based paths in
 * the shaders — an order of magnitude faster for a ≤0.65 dB PSNR cost. 'high'
 * runs the exhaustive search, matching the CPU reference encoders
 * block-for-block (byte-identical up to FP tie-breaks with equal error).
 * BC1's 'high' adds a principal-axis endpoint seed and iterative refit.
 */
export type EncodeQuality = 'fast' | 'high'

export interface EncodeCallOptions {
  /** Tags the output color space. Forced 'linear' for encoders with supportsSrgb=false. */
  colorSpace?: 'srgb' | 'linear'
  /** Encode quality / speed trade-off. Default 'fast'. */
  quality?: EncodeQuality
}

/**
 * Result of a raw bytes-only encode. This is the encoder's native output: the
 * compressed block bytes plus dimensions, with no Three.js (or any engine)
 * involvement. Feed `data` into whatever renderer's compressed-texture upload
 * you like, or use `buildCompressedTexture()` from `gputex/three`.
 */
export interface EncodeBytesResult {
  width: number
  height: number
  paddedWidth: number
  paddedHeight: number
  data: Uint8Array
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
  // `!:` because these are set in `_buildPipeline()` which the constructor
  // calls; TypeScript's flow analysis doesn't see through method calls.
  protected _module!: GPUShaderModule
  // f16 'fast' module — built only when the device supports shader-f16 and the
  // subclass provides an f16 source. null otherwise (falls back to _module).
  protected _moduleF16: GPUShaderModule | null = null
  protected _pipelineF16: GPUComputePipeline | null = null
  // Default pipeline (fast). Kept as a field for back-compat; the per-quality
  // cache below holds the specialised pipelines for encoders that support it.
  protected _pipeline!: GPUComputePipeline
  protected _pipelineCache = new Map<EncodeQuality, GPUComputePipeline>()

  constructor({ device, adapter, ownsDevice = false, disableF16 = false }: EncoderOptions) {
    this.device = device
    this.adapter = adapter
    this.ownsDevice = ownsDevice
    this.disableF16 = disableF16
    this._buildPipeline()
  }

  protected _buildPipeline(): void {
    const device = this.device
    const code = this.wgslSource() // subclasses override; stubs throw here
    this._module = device.createShaderModule({
      label: `${this.label}-encoder`,
      code,
    })
    if (this._useF16) {
      this._moduleF16 = device.createShaderModule({
        label: `${this.label}-encoder-f16`,
        code: this.wgslSourceFastF16()!,
      })
    }
    if (this.supportsQuality) {
      // Eagerly build the default (fast) pipeline so shader compile errors
      // still surface at construction time, as before.
      this._pipeline = this._getPipeline('fast')
    } else {
      this._pipeline = device.createComputePipeline({
        label: `${this.label}-encoder-pipeline`,
        layout: 'auto',
        compute: { module: this._module, entryPoint: 'encode' },
      })
    }
  }

  /**
   * Pipeline for a given quality level. Encoders that don't declare a
   * `QUALITY_HIGH` override (`supportsQuality === false`, e.g. BC1) ignore the
   * argument and reuse the single pipeline. Specialised pipelines are cached.
   */
  protected _getPipeline(quality: EncodeQuality): GPUComputePipeline {
    if (!this.supportsQuality) return this._pipeline
    // 'fast' uses the dedicated f16 module when available — it's a standalone
    // fast-only shader (no QUALITY_HIGH override). 'high' always uses the f32
    // module so its output keeps matching the CPU reference.
    if (quality === 'fast' && this._moduleF16) {
      if (!this._pipelineF16) {
        this._pipelineF16 = this.device.createComputePipeline({
          label: `${this.label}-encoder-pipeline-fast-f16`,
          layout: 'auto',
          compute: { module: this._moduleF16, entryPoint: 'encode' },
        })
      }
      return this._pipelineF16
    }
    const cached = this._pipelineCache.get(quality)
    if (cached) return cached
    const pipeline = this.device.createComputePipeline({
      label: `${this.label}-encoder-pipeline-${quality}`,
      layout: 'auto',
      compute: {
        module: this._module,
        entryPoint: 'encode',
        constants: { QUALITY_HIGH: quality === 'high' ? 1 : 0 },
      },
    })
    this._pipelineCache.set(quality, pipeline)
    return pipeline
  }

  destroy(): void {
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
   * Whether the shader declares a `QUALITY_HIGH` pipeline-overridable constant
   * (i.e. has distinct fast/high search paths). Default false (e.g. a stub or a
   * format with a single path); BC1/BC5/BC7/ASTC override it to true.
   */
  get supportsQuality(): boolean {
    return false
  }

  /**
   * Optional f16 WGSL for the 'fast' path. Used only when the device reports the
   * `shader-f16` feature; the format's f32 `wgslSource()` is the fallback and
   * `'high'` always uses it. Returns null when there's no f16 variant.
   */
  wgslSourceFastF16(): string | null {
    return null
  }

  /** Whether the f16 fast path is both available and supported on this device. */
  protected get _useF16(): boolean {
    return !this.disableF16 && this.wgslSourceFastF16() !== null && this.device.features.has('shader-f16')
  }

  /** WGSL compute-shader source. */
  abstract wgslSource(): string

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
    {
      flipY = false,
      quality = 'fast',
      withGpuTime = false,
    }: { flipY?: boolean; quality?: EncodeQuality; withGpuTime?: boolean } = {},
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

    // 1. Source texture sized to the padded block grid.
    const srcTex = device.createTexture({
      label: `${this.label}-src`,
      size: [paddedWidth, paddedHeight, 1],
      format: 'rgba8unorm',
      // RENDER_ATTACHMENT is required by copyExternalImageToTexture
      // (internally a blit) even though we never render into this texture.
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    uploadSourceTexture(device, srcTex, source, width, height, flipY, source instanceof ImageData)

    // 2. Output storage buffer.
    const dstBuffer = device.createBuffer({
      label: `${this.label}-dst`,
      size: outByteLen,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    // 3. Uniform params buffer — 4 × u32 canonical header.
    const paramsBuffer = device.createBuffer({
      label: `${this.label}-params`,
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // width/height are the SOURCE dimensions, not the padded ones: the shaders
    // clamp texel reads to (width-1, height-1), which must be the last real
    // texel. Clamping to the padded size would read the zero-initialized
    // padding strip and bleed black into the edge blocks' palettes.
    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([blocksX, blocksY, width, height]))

    // 4. Pipeline + bind group. The pipeline is specialised for the requested
    //    quality level; its `layout: 'auto'` bind group layout is identical
    //    across levels (same bindings), but we source it from the selected
    //    pipeline to keep them provably compatible.
    const pipeline = this._getPipeline(quality)
    const bindGroup = device.createBindGroup({
      label: `${this.label}-bg`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srcTex.createView() },
        { binding: 1, resource: { buffer: dstBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    })

    // 5. Dispatch — one workgroup tile per (workgroupSize) blocks. When asked
    //    (and the device has 'timestamp-query'), bracket the pass with
    //    timestamps so the caller gets shader-only GPU time.
    const useTimestamps = withGpuTime && device.features.has('timestamp-query')
    const querySet = useTimestamps ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null
    const queryBuffer = useTimestamps
      ? device.createBuffer({
          label: `${this.label}-ts-resolve`,
          size: 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        })
      : null

    const [wgX, wgY] = this.workgroupSize
    const t0 = performance.now()
    const enc = device.createCommandEncoder({ label: `${this.label}-encode` })
    const pass = enc.beginComputePass(
      querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined,
    )
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(Math.ceil(blocksX / wgX), Math.ceil(blocksY / wgY), 1)
    pass.end()
    if (querySet && queryBuffer) enc.resolveQuerySet(querySet, 0, 2, queryBuffer, 0)

    // 6. Buffer readback. We go through a MAP_READ staging buffer rather
    //    than copyBufferToTexture: the encoder owns its own adapter/device,
    //    separate from the Three.js renderer's, so the compressed bytes
    //    have to transit CPU anyway before the renderer uploads them.
    const staging = device.createBuffer({
      label: `${this.label}-staging`,
      size: outByteLen,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    enc.copyBufferToBuffer(dstBuffer, 0, staging, 0, outByteLen)
    const tsStaging =
      querySet && queryBuffer
        ? device.createBuffer({
            label: `${this.label}-ts-staging`,
            size: 16,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          })
        : null
    if (tsStaging && queryBuffer) enc.copyBufferToBuffer(queryBuffer, 0, tsStaging, 0, 16)
    device.queue.submit([enc.finish()])

    await staging.mapAsync(GPUMapMode.READ)
    const data = new Uint8Array(staging.getMappedRange().slice(0))
    staging.unmap()
    const encodeMs = performance.now() - t0

    let gpuMs: number | undefined
    if (tsStaging) {
      await tsStaging.mapAsync(GPUMapMode.READ)
      const [begin, end] = new BigUint64Array(tsStaging.getMappedRange().slice(0))
      tsStaging.unmap()
      tsStaging.destroy()
      // Timestamps are u64 nanoseconds.
      if (end !== undefined && begin !== undefined && end > begin) {
        gpuMs = Number(end - begin) / 1e6
      }
    }
    querySet?.destroy()
    queryBuffer?.destroy()

    srcTex.destroy()
    dstBuffer.destroy()
    staging.destroy()
    paramsBuffer.destroy()

    return { width, height, paddedWidth, paddedHeight, data, encodeMs, gpuMs }
  }
}
