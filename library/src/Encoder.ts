// Abstract base class for all block-compression encoders.
//
// Each concrete encoder (BC1, BC5, BC7, ASTC_4x4) overrides a handful of
// hooks — shader source, block size, format strings — and inherits the shared
// `encode()` pipeline: pad → upload → dispatch → readback → wrap as a
// Three.js CompressedTexture.
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
import {
  CompressedTexture,
  LinearFilter,
  LinearMipmapLinearFilter,
  LinearSRGBColorSpace,
  SRGBColorSpace,
  RepeatWrapping,
} from 'three'

import { uploadSourceTexture } from './workarounds.js'

import type { CompressedTextureMipmap, CompressedPixelFormat } from 'three'

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
}

export interface EncodeCallOptions {
  /** Tags the output color space. Forced 'linear' for encoders with supportsSrgb=false. */
  colorSpace?: 'srgb' | 'linear'
}

export interface EncodeResult {
  width: number
  height: number
  paddedWidth: number
  paddedHeight: number
  data: Uint8Array
  texture: CompressedTexture
  encodeMs: number
}

/**
 * Result of a raw bytes-only encode (no wrapping in a `CompressedTexture`).
 * Used by the mipped encode path to bundle multiple levels into a single
 * `CompressedTexture` at the end.
 */
export interface EncodeBytesResult {
  width: number
  height: number
  paddedWidth: number
  paddedHeight: number
  data: Uint8Array
  encodeMs: number
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
    const device = await adapter.requestDevice({ requiredFeatures })
    return new this({ device, adapter, ownsDevice: true })
  }

  readonly device: GPUDevice
  readonly adapter?: GPUAdapter
  readonly ownsDevice: boolean
  // `!:` because these are set in `_buildPipeline()` which the constructor
  // calls; TypeScript's flow analysis doesn't see through method calls.
  protected _module!: GPUShaderModule
  protected _pipeline!: GPUComputePipeline

  constructor({ device, adapter, ownsDevice = false }: EncoderOptions) {
    this.device = device
    this.adapter = adapter
    this.ownsDevice = ownsDevice
    this._buildPipeline()
  }

  protected _buildPipeline(): void {
    const device = this.device
    const code = this.wgslSource() // subclasses override; stubs throw here
    this._module = device.createShaderModule({
      label: `${this.label}-encoder`,
      code,
    })
    this._pipeline = device.createComputePipeline({
      label: `${this.label}-encoder-pipeline`,
      layout: 'auto',
      compute: { module: this._module, entryPoint: 'encode' },
    })
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

  /** WGSL compute-shader source. */
  abstract wgslSource(): string

  /** e.g. 'bc1-rgba-unorm-srgb'. */
  abstract gpuTextureFormat(opts: FormatVariant): GPUTextureFormat

  /** Three.js `CompressedPixelFormat` constant for `CompressedTexture`. */
  abstract threeTextureFormat(opts: FormatVariant): CompressedPixelFormat

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
  // Shared encode() — pad, upload, dispatch, readback, wrap.           //
  // ------------------------------------------------------------------ //

  async encode(source: EncoderImageSource, { colorSpace = 'srgb' }: EncodeCallOptions = {}): Promise<EncodeResult> {
    const effectiveSrgb = colorSpace === 'srgb' && this.supportsSrgb
    const bytes = await this.encodeToBytes(source)

    const threeFormat = this.threeTextureFormat({ colorSpace: effectiveSrgb ? 'srgb' : 'linear' })
    const mip: CompressedTextureMipmap = {
      data: bytes.data,
      width: bytes.paddedWidth,
      height: bytes.paddedHeight,
    }
    const texture = new CompressedTexture([mip], bytes.paddedWidth, bytes.paddedHeight, threeFormat)
    texture.colorSpace = effectiveSrgb ? SRGBColorSpace : LinearSRGBColorSpace
    texture.magFilter = LinearFilter
    // Single mip level: using a trilinear min filter would sample a
    // mip level that doesn't exist.
    texture.minFilter = LinearFilter
    texture.generateMipmaps = false
    texture.wrapS = texture.wrapT = RepeatWrapping
    texture.needsUpdate = true
    texture.userData.logicalWidth = bytes.width
    texture.userData.logicalHeight = bytes.height

    return {
      width: bytes.width,
      height: bytes.height,
      paddedWidth: bytes.paddedWidth,
      paddedHeight: bytes.paddedHeight,
      data: bytes.data,
      texture,
      encodeMs: bytes.encodeMs,
    }
  }

  /**
   * Encode one image source to raw compressed bytes, skipping the
   * `CompressedTexture` wrap. Used by the public `encode()` above and by
   * the mipped encode path in `compressTexture()` so N mip levels end up
   * in a single `CompressedTexture` instead of N throwaway wrappers.
   *
   * Public (not protected) because `compressTexture()` calls it across the
   * encoder boundary. Still safe to call from outside — it just does
   * less work than `encode()` and the caller assembles the texture.
   */
  async encodeToBytes(
    source: EncoderImageSource,
    { flipY = false }: { flipY?: boolean } = {},
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
    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([blocksX, blocksY, paddedWidth, paddedHeight]))

    // 4. Bind group — layout comes from `layout: 'auto'` on the pipeline.
    const bindGroup = device.createBindGroup({
      label: `${this.label}-bg`,
      layout: this._pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srcTex.createView() },
        { binding: 1, resource: { buffer: dstBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    })

    // 5. Dispatch — one workgroup tile per (workgroupSize) blocks.
    const [wgX, wgY] = this.workgroupSize
    const t0 = performance.now()
    const enc = device.createCommandEncoder({ label: `${this.label}-encode` })
    const pass = enc.beginComputePass()
    pass.setPipeline(this._pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(Math.ceil(blocksX / wgX), Math.ceil(blocksY / wgY), 1)
    pass.end()

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
    device.queue.submit([enc.finish()])

    await staging.mapAsync(GPUMapMode.READ)
    const data = new Uint8Array(staging.getMappedRange().slice(0))
    staging.unmap()
    const encodeMs = performance.now() - t0

    srcTex.destroy()
    dstBuffer.destroy()
    staging.destroy()
    paramsBuffer.destroy()

    return { width, height, paddedWidth, paddedHeight, data, encodeMs }
  }

  /**
   * Assemble a `CompressedTexture` from pre-encoded mip levels. Called
   * by `compressTexture()` after it has run each level through
   * `encodeToBytes()`. Centralised here so the single-level and mipped
   * paths share the same format / colour-space / wrap settings.
   *
   * `levels[0]` is the base level; its padded dimensions become the
   * texture's overall size. Filter setup assumes at least 2 levels →
   * trilinear; 1 level → bilinear.
   */
  buildMippedTexture(
    levels: readonly EncodeBytesResult[],
    { colorSpace = 'srgb' }: EncodeCallOptions = {},
  ): CompressedTexture {
    if (levels.length === 0) {
      throw new Error(`${this.label}Encoder.buildMippedTexture: no levels provided`)
    }
    const effectiveSrgb = colorSpace === 'srgb' && this.supportsSrgb
    const threeFormat = this.threeTextureFormat({ colorSpace: effectiveSrgb ? 'srgb' : 'linear' })
    const mipmaps: CompressedTextureMipmap[] = levels.map(l => ({
      data: l.data,
      width: l.paddedWidth,
      height: l.paddedHeight,
    }))
    const base = levels[0]!
    const texture = new CompressedTexture(mipmaps, base.paddedWidth, base.paddedHeight, threeFormat)
    texture.colorSpace = effectiveSrgb ? SRGBColorSpace : LinearSRGBColorSpace
    texture.magFilter = LinearFilter
    texture.minFilter = levels.length > 1 ? LinearMipmapLinearFilter : LinearFilter
    // We provide the full mip chain explicitly; Three.js must not try to
    // generate its own (which would clobber our compressed levels).
    texture.generateMipmaps = false
    texture.wrapS = texture.wrapT = RepeatWrapping
    texture.needsUpdate = true
    texture.userData.logicalWidth = base.width
    texture.userData.logicalHeight = base.height
    texture.userData.mipLevels = levels.length
    return texture
  }
}
