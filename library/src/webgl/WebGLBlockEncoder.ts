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

import { assembleCompressedTexture, type EncodedLevel } from '../textureAssembly.js'
import vertSource from './glsl/fullscreen.vert.glsl'
import { getSharedWebGLContext } from './webglContext.js'

import type { CompressedPixelFormat, CompressedTexture } from 'three'

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

function compileShader(gl: WebGL2RenderingContext, type: number, source: string, label: string): WebGLShader {
  const shader = gl.createShader(type)
  if (!shader) throw new Error(`${label}: gl.createShader failed`)
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader)
    gl.deleteShader(shader)
    const kind = type === gl.VERTEX_SHADER ? 'vertex' : 'fragment'
    throw new Error(`${label}: ${kind} shader compile failed: ${log}`)
  }
  return shader
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
  /** Three.js `CompressedPixelFormat` constant; sRGB is carried by colorSpace. */
  abstract threeTextureFormat(): CompressedPixelFormat

  protected _buildProgram(): void {
    const gl = this.gl
    const program = gl.createProgram()
    const vao = gl.createVertexArray()
    if (!program || !vao) throw new Error(`${this.label}: failed to allocate WebGL program/VAO`)

    const vert = compileShader(gl, gl.VERTEX_SHADER, vertSource, this.label)
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, this.fragSource(), this.label)
    gl.attachShader(program, vert)
    gl.attachShader(program, frag)
    gl.linkProgram(program)
    // Shader objects are no longer needed once linked.
    gl.deleteShader(vert)
    gl.deleteShader(frag)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program)
      gl.deleteProgram(program)
      throw new Error(`${this.label}: WebGL program link failed: ${log}`)
    }

    this._program = program
    this._vao = vao
    this._uSrc = gl.getUniformLocation(program, 'uSrc')
    this._uSrcSize = gl.getUniformLocation(program, 'uSrcSize')
    this._uFlipY = gl.getUniformLocation(program, 'uFlipY')
  }

  /** Release the GL program + VAO. The shared context itself is left intact. */
  destroy(): void {
    const gl = this.gl
    if (gl.isContextLost()) return
    gl.deleteProgram(this._program)
    gl.deleteVertexArray(this._vao)
  }

  /**
   * Upload the source image to a freshly created RGBA8 texture bound on unit 0.
   * Raw pixel sources (ImageData / mip levels) go through the typed-array
   * overload; DOM sources (ImageBitmap / canvas / image) through the element
   * overload. No flip / premultiply / colour conversion — flipY is applied in
   * the shader so each mip level flips by its own height.
   */
  protected _uploadSource(source: WebGLEncoderImageSource, width: number, height: number): WebGLTexture {
    const gl = this.gl
    const tex = gl.createTexture()
    if (!tex) throw new Error(`${this.label}: gl.createTexture failed`)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, tex)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false)
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)

    const raw = source as Partial<RawPixelSource>
    if (raw.data && ArrayBuffer.isView(raw.data)) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, raw.data)
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, source as TexImageSource)
    }

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    return tex
  }

  /**
   * Encode one image source to raw compressed bytes. `flipY` samples the source
   * bottom-up (matching Three.js's convention) and is applied in the shader;
   * the high-level mipped path bakes the flip into level 0 and passes false.
   */
  encodeToBytes(source: WebGLEncoderImageSource, { flipY = false }: { flipY?: boolean } = {}): WebGLEncodeBytesResult {
    const gl = this.gl
    if (gl.isContextLost()) throw new Error(`${this.label}WebGLEncoder: WebGL context lost`)

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

    // 2. Output integer texture (RGBA32UI) + framebuffer.
    const outTex = gl.createTexture()
    const fbo = gl.createFramebuffer()
    if (!outTex || !fbo) {
      gl.deleteTexture(srcTex)
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
      gl.deleteFramebuffer(fbo)
      gl.deleteTexture(outTex)
      gl.deleteTexture(srcTex)
      throw new Error(`${this.label}: integer framebuffer incomplete (0x${status.toString(16)})`)
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

    // 6. Release per-encode resources; the program + VAO persist on the encoder.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.bindTexture(gl.TEXTURE_2D, null)
    gl.bindVertexArray(null)
    gl.deleteFramebuffer(fbo)
    gl.deleteTexture(outTex)
    gl.deleteTexture(srcTex)

    return { width, height, paddedWidth, paddedHeight, data, encodeMs }
  }

  /** Wrap pre-encoded levels into a CompressedTexture. Shared with the WebGPU path. */
  buildMippedTexture(
    levels: readonly EncodedLevel[],
    { colorSpace = 'srgb' }: { colorSpace?: 'srgb' | 'linear' } = {},
  ): CompressedTexture {
    if (levels.length === 0) {
      throw new Error(`${this.label}WebGLEncoder.buildMippedTexture: no levels provided`)
    }
    const effectiveSrgb = colorSpace === 'srgb' && this.supportsSrgb
    return assembleCompressedTexture(levels, this.threeTextureFormat(), effectiveSrgb)
  }
}
