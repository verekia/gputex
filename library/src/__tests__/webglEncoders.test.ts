import { _internal as _eacInternal } from '../eac_r11_ref.js'
import { ASTC4x4WebGLEncoder } from '../webgl/ASTC4x4WebGLEncoder.js'
import { BC1WebGLEncoder } from '../webgl/BC1WebGLEncoder.js'
import { BC4WebGLEncoder } from '../webgl/BC4WebGLEncoder.js'
import { BC5WebGLEncoder } from '../webgl/BC5WebGLEncoder.js'
import { BC7WebGLEncoder } from '../webgl/BC7WebGLEncoder.js'
import { EACR11WebGLEncoder } from '../webgl/EACR11WebGLEncoder.js'
import { ETC2WebGLEncoder } from '../webgl/ETC2WebGLEncoder.js'

// Inspect each encoder's metadata + GLSL source without constructing it
// (construction needs a real WebGL2 context to compile the program). As in
// encoders.test.ts, `Object.create(prototype)` yields an object that inherits
// the getters/methods but skips the constructor. These tests also confirm the
// `.glsl` text-loader resolves the fragment sources at import time.
//
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyProto = any

function expectValidFragSource(src: string) {
  expect(typeof src).toBe('string')
  expect(src).toContain('#version 300 es')
  // Integer render target output — the crux of the readback-based pipeline.
  expect(src).toContain('out uvec4 outColor')
  expect(src).toContain('void main()')
}

describe('BC7WebGLEncoder', () => {
  it('declares its metadata and loads bc7.frag.glsl', () => {
    const view: AnyProto = Object.create(BC7WebGLEncoder.prototype)
    expect(view.label).toBe('bc7')
    expect(view.bytesPerBlock).toBe(16)
    expect(view.supportsSrgb).toBe(true)
    expectValidFragSource(view.fragSource())
  })
})

describe('BC5WebGLEncoder', () => {
  it('declares its metadata and loads bc5.frag.glsl', () => {
    const view: AnyProto = Object.create(BC5WebGLEncoder.prototype)
    expect(view.label).toBe('bc5')
    expect(view.bytesPerBlock).toBe(16)
    expect(view.supportsSrgb).toBe(false)
    const src: string = view.fragSource()
    expectValidFragSource(src)
    expect(src).toContain('lvlToIdx')
  })
})

describe('BC4WebGLEncoder', () => {
  it('declares its metadata and loads bc4.frag.glsl', () => {
    const view: AnyProto = Object.create(BC4WebGLEncoder.prototype)
    expect(view.label).toBe('bc4')
    expect(view.bytesPerBlock).toBe(8)
    expect(view.supportsSrgb).toBe(false)
    const src: string = view.fragSource()
    expectValidFragSource(src)
    expect(src).toContain('lvlToIdx')
  })
})

describe('EACR11WebGLEncoder', () => {
  it('declares its metadata and loads eac_r11.frag.glsl', () => {
    const view: AnyProto = Object.create(EACR11WebGLEncoder.prototype)
    expect(view.label).toBe('eac-r11')
    expect(view.bytesPerBlock).toBe(8)
    expect(view.supportsSrgb).toBe(false)
    expectValidFragSource(view.fragSource())
  })

  it('carries the same candidate table as the CPU reference', () => {
    const view: AnyProto = Object.create(EACR11WebGLEncoder.prototype)
    const src: string = view.fragSource()
    const lut = src.slice(src.indexOf('const uint LUT[164] = uint[164]('))
    const want = _eacInternal.CANDIDATES.map(c => `0x${c.toString(16).padStart(3, '0')}u`)
    expect(lut.slice(0, lut.indexOf(');')).match(/0x[0-9a-f]{3}u/g)).toEqual(want)
  })
})

describe('ASTC4x4WebGLEncoder', () => {
  it('declares its metadata and loads astc4x4.frag.glsl', () => {
    const view: AnyProto = Object.create(ASTC4x4WebGLEncoder.prototype)
    expect(view.label).toBe('astc4x4')
    expect(view.bytesPerBlock).toBe(16)
    expect(view.supportsSrgb).toBe(true)
    expectValidFragSource(view.fragSource())
  })
})

describe('BC1WebGLEncoder', () => {
  it('declares its metadata and loads bc1.frag.glsl', () => {
    const view: AnyProto = Object.create(BC1WebGLEncoder.prototype)
    expect(view.label).toBe('bc1')
    expect(view.bytesPerBlock).toBe(8)
    expect(view.supportsSrgb).toBe(true)
    expectValidFragSource(view.fragSource())
  })

  it('ETC2: declares its metadata and loads etc2.frag.glsl', () => {
    const view: AnyProto = Object.create(ETC2WebGLEncoder.prototype)
    expect(view.label).toBe('etc2')
    expect(view.bytesPerBlock).toBe(8)
    expect(view.supportsSrgb).toBe(true)
    const src: string = view.fragSource()
    expectValidFragSource(src)
    expect(src).toContain('sbPair')
  })

  it('uses 8 bytes per block; the others use 16', () => {
    // bytesPerBlock drives the readback repack in WebGLBlockEncoder (BC1 keeps
    // the low two of the four RGBA32UI words per block; the rest use all four).
    const bc1: AnyProto = Object.create(BC1WebGLEncoder.prototype)
    const bc7: AnyProto = Object.create(BC7WebGLEncoder.prototype)
    expect(bc1.bytesPerBlock).toBe(8)
    expect(bc7.bytesPerBlock).toBe(16)
  })
})
