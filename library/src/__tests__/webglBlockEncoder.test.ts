// `WebGLBlockEncoder.encodeToBytes()` against a recording fake WebGL2 context.
//
// The real pipeline needs a GPU, so the browser suite (WebGPU-only) never
// touches this path at all — these cover the host-side contract around the
// draw: block-grid geometry for non-block-aligned sources, the byte layout
// the readback is repacked into, and that per-encode GL objects are released
// rather than leaked.
//
// The fake records every call, hands out identity-tagged objects for the GL
// "objects", and fills readPixels with a known pattern so the 8-byte repack
// can be asserted.

import { WebGLBlockEncoder, type WebGLEncoderImageSource } from '../webgl/WebGLBlockEncoder.js'

interface GLObject {
  kind: string
  id: number
}

interface FakeGL {
  calls: string[]
  live: Set<GLObject>
  created: (kind: string) => number
  deleted: (kind: string) => number
  countCall: (name: string) => number
}

/**
 * A WebGL2RenderingContext stand-in covering exactly the surface
 * `WebGLBlockEncoder` touches. Constants are arbitrary but distinct.
 */
function makeFakeGL(): FakeGL & Record<string, unknown> {
  let nextId = 1
  const calls: string[] = []
  const live = new Set<GLObject>()
  const createdBy: Record<string, number> = {}
  const deletedBy: Record<string, number> = {}

  const make = (kind: string): GLObject => {
    createdBy[kind] = (createdBy[kind] ?? 0) + 1
    const o = { kind, id: nextId++ }
    live.add(o)
    return o
  }
  const drop = (kind: string, o: GLObject | null): void => {
    deletedBy[kind] = (deletedBy[kind] ?? 0) + 1
    if (o) live.delete(o)
  }
  const rec =
    (name: string) =>
    (...args: unknown[]): void => {
      calls.push(`${name}(${args.map(a => (typeof a === 'object' && a !== null ? 'obj' : String(a))).join(',')})`)
    }

  const gl: Record<string, unknown> = {
    // --- constants ---
    VERTEX_SHADER: 1,
    FRAGMENT_SHADER: 2,
    COMPILE_STATUS: 3,
    LINK_STATUS: 4,
    TEXTURE_2D: 5,
    TEXTURE0: 6,
    RGBA8: 7,
    RGBA: 8,
    UNSIGNED_BYTE: 9,
    RGBA32UI: 10,
    NEAREST: 11,
    CLAMP_TO_EDGE: 12,
    TEXTURE_MIN_FILTER: 13,
    TEXTURE_MAG_FILTER: 14,
    TEXTURE_WRAP_S: 15,
    TEXTURE_WRAP_T: 16,
    FRAMEBUFFER: 17,
    COLOR_ATTACHMENT0: 18,
    FRAMEBUFFER_COMPLETE: 19,
    RGBA_INTEGER: 20,
    UNSIGNED_INT: 21,
    BLEND: 22,
    DEPTH_TEST: 23,
    SCISSOR_TEST: 24,
    TRIANGLES: 25,
    UNPACK_FLIP_Y_WEBGL: 26,
    UNPACK_PREMULTIPLY_ALPHA_WEBGL: 27,
    UNPACK_COLORSPACE_CONVERSION_WEBGL: 28,
    UNPACK_ALIGNMENT: 29,
    NONE: 30,

    // --- program build ---
    isContextLost: () => false,
    createShader: () => make('shader'),
    shaderSource: rec('shaderSource'),
    compileShader: rec('compileShader'),
    getShaderParameter: () => true,
    getShaderInfoLog: () => '',
    deleteShader: (o: GLObject) => drop('shader', o),
    createProgram: () => make('program'),
    attachShader: rec('attachShader'),
    linkProgram: rec('linkProgram'),
    getProgramParameter: () => true,
    getProgramInfoLog: () => '',
    deleteProgram: (o: GLObject) => drop('program', o),
    createVertexArray: () => make('vao'),
    deleteVertexArray: (o: GLObject) => drop('vao', o),
    getUniformLocation: () => ({ kind: 'uniform', id: nextId++ }),

    // --- textures / framebuffers ---
    createTexture: () => make('texture'),
    deleteTexture: (o: GLObject) => drop('texture', o),
    createFramebuffer: () => make('framebuffer'),
    deleteFramebuffer: (o: GLObject) => drop('framebuffer', o),
    activeTexture: rec('activeTexture'),
    bindTexture: rec('bindTexture'),
    bindFramebuffer: rec('bindFramebuffer'),
    framebufferTexture2D: rec('framebufferTexture2D'),
    checkFramebufferStatus: () => 19,
    texStorage2D: rec('texStorage2D'),
    texImage2D: rec('texImage2D'),
    texParameteri: rec('texParameteri'),
    pixelStorei: rec('pixelStorei'),

    // --- draw ---
    useProgram: rec('useProgram'),
    bindVertexArray: rec('bindVertexArray'),
    uniform1i: rec('uniform1i'),
    uniform2i: rec('uniform2i'),
    disable: rec('disable'),
    viewport: rec('viewport'),
    drawArrays: rec('drawArrays'),

    // Fill each block's four words with a recognisable pattern so the
    // 8-byte repack (keep words 0 and 1, drop 2 and 3) is checkable.
    readPixels: (_x: number, _y: number, w: number, h: number, _f: number, _t: number, out: Uint32Array) => {
      calls.push(`readPixels(${w},${h})`)
      for (let k = 0; k < w * h; k++) {
        out[k * 4] = 0x10000000 + k
        out[k * 4 + 1] = 0x20000000 + k
        out[k * 4 + 2] = 0xdead0000 + k
        out[k * 4 + 3] = 0xbeef0000 + k
      }
    },

    // --- assertions surface ---
    calls,
    live,
    created: (kind: string) => createdBy[kind] ?? 0,
    deleted: (kind: string) => deletedBy[kind] ?? 0,
    countCall: (name: string) => calls.filter(c => c.startsWith(`${name}(`)).length,
  }
  return gl as FakeGL & Record<string, unknown>
}

class Fake16Encoder extends WebGLBlockEncoder {
  override get label(): string {
    return 'fake16'
  }
  override get bytesPerBlock(): number {
    return 16
  }
  override get supportsSrgb(): boolean {
    return true
  }
  override fragSource(): string {
    return '#version 300 es\nout uvec4 outColor;\nvoid main() {}'
  }
}

class Fake8Encoder extends Fake16Encoder {
  override get label(): string {
    return 'fake8'
  }
  override get bytesPerBlock(): number {
    return 8
  }
}

/** A raw-pixel source of the given size (the RawPixelSource branch). */
function source(width: number, height: number): WebGLEncoderImageSource {
  return { data: new Uint8Array(width * height * 4), width, height }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyGL = any

describe('WebGLBlockEncoder geometry', () => {
  it('reports padded dimensions and a correctly sized 16-byte payload', () => {
    const gl = makeFakeGL()
    const encoder = new Fake16Encoder({ gl: gl as AnyGL })
    const r = encoder.encodeToBytes(source(10, 6))

    expect(r.width).toBe(10)
    expect(r.height).toBe(6)
    expect(r.paddedWidth).toBe(12)
    expect(r.paddedHeight).toBe(8)
    // 3×2 blocks × 16 bytes.
    expect(r.data.byteLength).toBe(3 * 2 * 16)
    // The draw covers exactly the block grid, not the pixel grid.
    expect(gl.calls).toContain('viewport(0,0,3,2)')
    encoder.destroy()
  })

  it('sizes a sub-block source up to one whole block', () => {
    const gl = makeFakeGL()
    const encoder = new Fake16Encoder({ gl: gl as AnyGL })
    const r = encoder.encodeToBytes(source(1, 1))
    expect(r.paddedWidth).toBe(4)
    expect(r.paddedHeight).toBe(4)
    expect(r.data.byteLength).toBe(16)
    encoder.destroy()
  })

  it('rejects a source with no dimensions', () => {
    const gl = makeFakeGL()
    const encoder = new Fake16Encoder({ gl: gl as AnyGL })
    expect(() => encoder.encodeToBytes(source(0, 8))).toThrow(/no dimensions/)
    encoder.destroy()
  })
})

describe('WebGLBlockEncoder readback repack', () => {
  it('keeps the low two words per block for 8-byte formats', () => {
    const gl = makeFakeGL()
    const encoder = new Fake8Encoder({ gl: gl as AnyGL })
    const r = encoder.encodeToBytes(source(8, 8))

    const blocks = 4
    expect(r.data.byteLength).toBe(blocks * 8)
    const words = new Uint32Array(r.data.buffer, r.data.byteOffset, blocks * 2)
    for (let k = 0; k < blocks; k++) {
      expect(words[k * 2]).toBe(0x10000000 + k)
      expect(words[k * 2 + 1]).toBe(0x20000000 + k)
    }
    encoder.destroy()
  })

  it('passes all four words straight through for 16-byte formats', () => {
    const gl = makeFakeGL()
    const encoder = new Fake16Encoder({ gl: gl as AnyGL })
    const r = encoder.encodeToBytes(source(8, 8))
    const words = new Uint32Array(r.data.buffer, r.data.byteOffset, 4 * 4)
    expect(words[0]).toBe(0x10000000)
    expect(words[2]).toBe(0xdead0000)
    expect(words[3]).toBe(0xbeef0000)
    encoder.destroy()
  })
})

describe('WebGLBlockEncoder resource lifetime', () => {
  // The encoder is shared across compressTexture() calls, so anything it
  // allocates per encode and fails to release accumulates for the lifetime
  // of the page — on the tier that runs on the weakest hardware.
  it('leaves nothing per-encode alive after a mip chain', () => {
    const gl = makeFakeGL()
    const encoder = new Fake16Encoder({ gl: gl as AnyGL })
    for (let size = 64; size >= 1; size >>= 1) encoder.encodeToBytes(source(size, size))

    expect(gl.countCall('drawArrays')).toBe(7)
    // Every texture and framebuffer the encodes made is gone again; only the
    // program and VAO (built once, in the constructor) remain.
    expect(gl.created('texture')).toBe(gl.deleted('texture'))
    expect(gl.created('framebuffer')).toBe(gl.deleted('framebuffer'))
    expect(gl.live.size).toBe(2)
    encoder.destroy()
    expect(gl.live.size).toBe(0)
  })

  it('releases the program and VAO on destroy()', () => {
    const gl = makeFakeGL()
    const encoder = new Fake16Encoder({ gl: gl as AnyGL })
    encoder.encodeToBytes(source(16, 16))
    encoder.destroy()

    expect(gl.deleted('program')).toBe(1)
    expect(gl.deleted('vao')).toBe(1)
    expect(gl.live.size).toBe(0)
  })
})
