// Abstract base for the WebGL2 fallback block encoders.
//
// WebGL2 has no compute shaders, so each encoder runs as a fragment shader that
// computes one compressed 4×4 block per fragment, writing the block words into
// an RGBA32UI render target. The shared pipeline here mirrors the WebGPU
// `Encoder`: pad → upload source → draw one fragment per block → read back →
// repack into the format's byte layout. Subclasses supply only the GLSL
// fragment source and the format metadata.
//
//   • One fragment ↔ one block. The viewport is sized (blocks_x × blocks_y) and
//     a fullscreen triangle covers every block texel exactly once.
//   • Output is always RGBA32UI (128 bits/fragment). 16-byte formats use all
//     four words; BC1 (8 bytes) uses the low two and the rest is discarded.
//   • Readback via readPixels(RGBA_INTEGER, UNSIGNED_INT) — the one integer
//     combination ES 3.0 guarantees — so the path is portable across drivers.
//
// Block bytes are produced byte-identical to the WebGPU encoders for the same
// input + flipY (same algorithms, same little-endian word order), so the
// resulting CompressedTexture displays identically under either renderer.

import vertSource from './glsl/fullscreen.vert.glsl'
import { getSharedWebGLContext } from './webglContext.js'

/** Raw RGBA8 pixel data (e.g. a CPU-generated mip level). */
export interface RawPixelSource {
  data: ArrayBufferView
  width: number
  height: number
}

/** Anything the WebGL encoders can upload as the source image. */
export type WebGLEncoderImageSource =
  | ImageBitmap
  | ImageData
  | HTMLImageElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | RawPixelSource

export interface WebGLEncodeBytesResult {
  width: number
  height: number
  paddedWidth: number
  paddedHeight: number
  data: Uint8Array
  encodeMs: number
}

export interface WebGLEncoderOptions {
  gl: WebGL2RenderingContext
}

/**
 * Constructor shape for the concrete subclasses; lets the static `create()`
 * narrow its return type to the subclass (as the WebGPU `EncoderConstructor`
 * does).
 */
export type WebGLEncoderConstructor<T extends WebGLBlockEncoder = WebGLBlockEncoder> = {
  new (opts: WebGLEncoderOptions): T
  create(gl?: WebGL2RenderingContext | null): T
}

// Compile without querying COMPILE_STATUS: that query blocks until the
// driver finishes, which would defeat KHR_parallel_shader_compile. Errors
// surface when the program's link status is checked (_ensureLinked).
function compileShader(gl: WebGL2RenderingContext, type: number, source: string, label: string): WebGLShader {
  const shader = gl.createShader(type)
  if (!shader) throw new Error(`${label}: gl.createShader failed`)
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  return shader
}

interface ParallelShaderCompile {
  readonly COMPLETION_STATUS_KHR: number
}

export abstract class WebGLBlockEncoder {
  /**
   * Create an encoder on the shared process-wide context (or a caller-supplied
   * one). Throws when WebGL2 is unavailable. The `this:` annotation lets
   * `BC7WebGLEncoder.create()` return `BC7WebGLEncoder`.
   */
  static create<T extends WebGLBlockEncoder>(this: WebGLEncoderConstructor<T>, gl?: WebGL2RenderingContext | null): T {
    const ctx = gl ?? getSharedWebGLContext()
    if (!ctx) throw new Error('WebGL2 not available in this environment')
    return new this({ gl: ctx })
  }

  readonly gl: WebGL2RenderingContext
  // Set in _buildProgram(), which the constructor calls.
  protected _program!: WebGLProgram
  protected _vao!: WebGLVertexArrayObject
  protected _uSrc: WebGLUniformLocation | null = null
  protected _uSrcSize: WebGLUniformLocation | null = null
  protected _uFlipY: WebGLUniformLocation | null = null
  // Program compile/link runs asynchronously in the driver where
  // KHR_parallel_shader_compile is available; the link status (and the
  // uniform lookups, which would block on it) are deferred to first use.
  private _parallel: ParallelShaderCompile | null = null
  private _shaders: WebGLShader[] = []
  private _linked = false

  // Per-encoder resource cache (mirrors the WebGPU Encoder). Sequential
  // encodes reuse the source texture while its size is stable and the
  // RGBA32UI target + framebuffer while the block grid is stable, so a
  // same-sized encode allocates nothing. encodeToBytes() is synchronous, so
  // there is no concurrent-encode case to guard.
  private _srcTex: WebGLTexture | null = null
  private _srcW = 0
  private _srcH = 0
  // Upload memoisation: the ImageBitmap whose pixels _srcTex currently
  // holds. ImageBitmaps are immutable, so re-encoding the same bitmap
  // (format A/B, quality-ladder re-encodes, benchmark loops) skips the
  // upload — ~14-19 ms of a ~22-28 ms 4096² encode. Mutable sources
  // (ImageData, raw pixels, canvases, images) are always re-uploaded. flipY
  // is applied in the shader, so it doesn't affect the texture contents.
  private _srcBitmap: ImageBitmap | null = null
  private _outTex: WebGLTexture | null = null
  private _fbo: WebGLFramebuffer | null = null
  private _outBX = 0
  private _outBY = 0

  constructor({ gl }: WebGLEncoderOptions) {
    this.gl = gl
    this._buildProgram()
  }

  // ------------------------------------------------------------------ //
  // Abstract hooks — subclasses override                               //
  // ------------------------------------------------------------------ //

  /** Short lowercase identifier for labels / errors. */
  abstract get label(): string
  /** 8 for BC1, 16 for BC5/BC7/ASTC 4×4. */
  abstract get bytesPerBlock(): number
  /** Whether this format has an sRGB variant (false for BC5). */
  abstract get supportsSrgb(): boolean
  /** GLSL ES 3.00 fragment-shader source. */
  abstract fragSource(): string

  protected _buildProgram(): void {
    const gl = this.gl
    const program = gl.createProgram()
    const vao = gl.createVertexArray()
    if (!program || !vao) throw new Error(`${this.label}: failed to allocate WebGL program/VAO`)

    // Enabling the extension (getExtension does) lets the driver compile and
    // link in the background; ready() polls its completion flag.
    this._parallel = gl.getExtension('KHR_parallel_shader_compile') as ParallelShaderCompile | null
    const vert = compileShader(gl, gl.VERTEX_SHADER, vertSource, this.label)
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, this.fragSource(), this.label)
    gl.attachShader(program, vert)
    gl.attachShader(program, frag)
    gl.linkProgram(program)
    this._shaders = [vert, frag]
    this._program = program
    this._vao = vao
  }

  /** Check the link (blocking until it completes), then look up uniforms. */
  private _ensureLinked(): void {
    if (this._linked) return
    const gl = this.gl
    const program = this._program
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const logs = [gl.getProgramInfoLog(program), ...this._shaders.map(sh => gl.getShaderInfoLog(sh))]
      throw new Error(`${this.label}: WebGL program link failed: ${logs.filter(Boolean).join('\n')}`)
    }
    // Shader objects are no longer needed once linked.
    for (const sh of this._shaders) gl.deleteShader(sh)
    this._shaders = []
    this._uSrc = gl.getUniformLocation(program, 'uSrc')
    this._uSrcSize = gl.getUniformLocation(program, 'uSrcSize')
    this._uFlipY = gl.getUniformLocation(program, 'uFlipY')
    this._linked = true
  }

  /**
   * Resolves once the fragment program has compiled and linked, without
   * blocking the main thread where the driver exposes
   * KHR_parallel_shader_compile (elsewhere the final status check waits for
   * the driver). Encodes check this themselves; call it to compile ahead of
   * first use (see `prewarmCompressTexture()`). Rejects on a compile/link
   * error.
   */
  async ready(): Promise<void> {
    const gl = this.gl
    const ext = this._parallel
    while (
      ext &&
      !this._linked &&
      !gl.isContextLost() &&
      !gl.getProgramParameter(this._program, ext.COMPLETION_STATUS_KHR)
    ) {
      await new Promise(resolve => setTimeout(resolve, 4))
    }
    if (gl.isContextLost()) throw new Error(`${this.label}WebGLEncoder: WebGL context lost`)
    this._ensureLinked()
  }

  /** Release the GL program, VAO and cached textures. The shared context itself is left intact. */
  destroy(): void {
    const gl = this.gl
    if (!gl.isContextLost()) {
      for (const sh of this._shaders) gl.deleteShader(sh)
      gl.deleteProgram(this._program)
      gl.deleteVertexArray(this._vao)
      if (this._srcTex) gl.deleteTexture(this._srcTex)
      if (this._outTex) gl.deleteTexture(this._outTex)
      if (this._fbo) gl.deleteFramebuffer(this._fbo)
    }
    this._shaders = []
    this._srcTex = null
    this._srcBitmap = null
    this._outTex = null
    this._fbo = null
  }

  /**
   * Upload the source image to the cached RGBA8 texture (recreated when the
   * image size changes), bound on unit 0. Raw pixel sources (ImageData / mip
   * levels) go through the typed-array overload; DOM sources (ImageBitmap /
   * canvas / image) through the element overload. No flip / premultiply /
   * colour conversion — flipY is applied in the shader so each mip level
   * flips by its own height. Re-encoding the ImageBitmap already held by the
   * texture skips the upload.
   */
  protected _uploadSource(source: WebGLEncoderImageSource, width: number, height: number): WebGLTexture {
    const gl = this.gl
    gl.activeTexture(gl.TEXTURE0)
    const isBitmap = typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap
    const sameSize = this._srcTex !== null && this._srcW === width && this._srcH === height
    if (sameSize && isBitmap && this._srcBitmap === source) {
      gl.bindTexture(gl.TEXTURE_2D, this._srcTex)
      return this._srcTex!
    }
    if (!sameSize) {
      if (this._srcTex) gl.deleteTexture(this._srcTex)
      const tex = gl.createTexture()
      if (!tex) throw new Error(`${this.label}: gl.createTexture failed`)
      gl.bindTexture(gl.TEXTURE_2D, tex)
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, width, height)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      this._srcTex = tex
      this._srcW = width
      this._srcH = height
    } else {
      gl.bindTexture(gl.TEXTURE_2D, this._srcTex)
    }
    // Invalidate before uploading: a throwing upload must not leave a stale
    // bitmap identity pointing at half-written pixels.
    this._srcBitmap = null
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false)
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)

    const raw = source as Partial<RawPixelSource>
    if (raw.data && ArrayBuffer.isView(raw.data)) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, raw.data)
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, source as TexImageSource)
    }
    if (isBitmap) this._srcBitmap = source as ImageBitmap
    return this._srcTex!
  }

  /**
   * Encode one image source to raw compressed bytes. `flipY` samples the source
   * bottom-up (matching Three.js's convention) and is applied in the shader;
   * the high-level mipped path bakes the flip into level 0 and passes false.
   */
  encodeToBytes(source: WebGLEncoderImageSource, { flipY = false }: { flipY?: boolean } = {}): WebGLEncodeBytesResult {
    const gl = this.gl
    if (gl.isContextLost()) throw new Error(`${this.label}WebGLEncoder: WebGL context lost`)
    this._ensureLinked()

    const width = (source as { width: number }).width
    const height = (source as { height: number }).height
    if (!width || !height) {
      throw new Error(`${this.label}WebGLEncoder: source has no dimensions`)
    }

    const paddedWidth = (width + 3) & ~3
    const paddedHeight = (height + 3) & ~3
    const blocksX = paddedWidth >> 2
    const blocksY = paddedHeight >> 2
    const blockCount = blocksX * blocksY
    const outByteLen = blockCount * this.bytesPerBlock

    const t0 = performance.now()

    // 1. Source texture (sized to the unpadded image; the shader clamps reads).
    const srcTex = this._uploadSource(source, width, height)

    // 2. Output integer texture (RGBA32UI) + framebuffer, reused while the
    //    block grid is unchanged.
    if (!this._outTex || !this._fbo || this._outBX !== blocksX || this._outBY !== blocksY) {
      if (this._outTex) gl.deleteTexture(this._outTex)
      if (this._fbo) gl.deleteFramebuffer(this._fbo)
      this._outTex = null
      this._fbo = null
      const outTex = gl.createTexture()
      const fbo = gl.createFramebuffer()
      if (!outTex || !fbo) {
        if (outTex) gl.deleteTexture(outTex)
        if (fbo) gl.deleteFramebuffer(fbo)
        throw new Error(`${this.label}: failed to allocate output texture/framebuffer`)
      }
      gl.bindTexture(gl.TEXTURE_2D, outTex)
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32UI, blocksX, blocksY)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0)
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null)
        gl.bindTexture(gl.TEXTURE_2D, null)
        gl.deleteFramebuffer(fbo)
        gl.deleteTexture(outTex)
        throw new Error(`${this.label}: integer framebuffer incomplete (0x${status.toString(16)})`)
      }
      this._outTex = outTex
      this._fbo = fbo
      this._outBX = blocksX
      this._outBY = blocksY
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo)
    }

    // 3. Draw exactly one fragment per block.
    gl.useProgram(this._program)
    gl.bindVertexArray(this._vao)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, srcTex)
    gl.uniform1i(this._uSrc, 0)
    gl.uniform2i(this._uSrcSize, width, height)
    gl.uniform1i(this._uFlipY, flipY ? 1 : 0)
    gl.disable(gl.BLEND)
    gl.disable(gl.DEPTH_TEST)
    gl.disable(gl.SCISSOR_TEST)
    gl.viewport(0, 0, blocksX, blocksY)
    gl.drawArrays(gl.TRIANGLES, 0, 3)

    // 4. Read back the block words. RGBA_INTEGER/UNSIGNED_INT is the integer
    //    readback combo ES 3.0 guarantees; rows are blocksX*16 bytes (a
    //    multiple of 4) so the default PACK_ALIGNMENT needs no adjustment.
    const words = new Uint32Array(blockCount * 4)
    gl.readPixels(0, 0, blocksX, blocksY, gl.RGBA_INTEGER, gl.UNSIGNED_INT, words)

    // 5. Repack into the format byte layout. readPixels returns texel (x,y) at
    //    index (y*blocksX+x)*4 — exactly block index by*blocksX+bx in word
    //    order — so 16-byte formats are already contiguous. BC1 keeps the low
    //    two words per block.
    let data: Uint8Array
    if (this.bytesPerBlock === 16) {
      data = new Uint8Array(words.buffer, 0, outByteLen)
    } else {
      const packed = new Uint32Array(blockCount * 2)
      for (let k = 0; k < blockCount; k++) {
        packed[k * 2] = words[k * 4]
        packed[k * 2 + 1] = words[k * 4 + 1]
      }
      data = new Uint8Array(packed.buffer, 0, outByteLen)
    }

    const encodeMs = performance.now() - t0

    // 6. Unbind; the program, VAO and cached textures persist on the encoder.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.bindTexture(gl.TEXTURE_2D, null)
    gl.bindVertexArray(null)

    return { width, height, paddedWidth, paddedHeight, data, encodeMs }
  }
}
