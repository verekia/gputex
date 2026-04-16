import { ASTC4x4Encoder } from '../ASTC4x4Encoder.js'
import { BC1Encoder } from '../BC1Encoder.js'
import { BC5Encoder } from '../BC5Encoder.js'
import { BC7Encoder } from '../BC7Encoder.js'
import { TextureFormat, WebGPUFeature } from '../TextureFormat.js'

// Meta-inspect an encoder class without instantiating it (instantiation
// needs a real GPUDevice and compiles the shader). These tests assert the
// format-selection layer can read everything it needs statically.
//
// `Object.create(prototype)` gives an object that inherits the class's
// getters and methods but skips the constructor. We cast to `any` so the
// test can invoke prototype methods without TypeScript tracking the
// (deliberately incomplete) instance shape.
//
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyProto = any

describe('BC1Encoder metadata', () => {
  it('declares its required feature and formats', () => {
    expect(BC1Encoder.requiredFeature).toBe(WebGPUFeature.BC)
    expect(BC1Encoder.textureFormats).toEqual([TextureFormat.BC1, TextureFormat.BC1_SRGB])
  })

  it('loads its WGSL shader source (bc1.wgsl imported via ?raw)', () => {
    const view: AnyProto = Object.create(BC1Encoder.prototype)
    expect(view.label).toBe('bc1')
    expect(view.bytesPerBlock).toBe(8)
    expect(view.supportsSrgb).toBe(true)
    const src: string = view.wgslSource()
    expect(typeof src).toBe('string')
    expect(src).toContain('@compute')
    expect(src).toContain('@workgroup_size')
  })
})

describe('BC5Encoder', () => {
  it('declares its metadata', () => {
    expect(BC5Encoder.requiredFeature).toBe(WebGPUFeature.BC)
    expect(BC5Encoder.textureFormats).toEqual([TextureFormat.BC5])
    const view: AnyProto = Object.create(BC5Encoder.prototype)
    expect(view.label).toBe('bc5')
    expect(view.bytesPerBlock).toBe(16)
    expect(view.supportsSrgb).toBe(false)
    expect(view.gpuTextureFormat({ colorSpace: 'linear' })).toBe('bc5-rg-unorm')
  })

  it('loads its WGSL shader source (bc5.wgsl imported via ?raw)', () => {
    const view: AnyProto = Object.create(BC5Encoder.prototype)
    const src: string = view.wgslSource()
    expect(typeof src).toBe('string')
    expect(src).toContain('@compute')
    expect(src).toContain('@workgroup_size')
    // Sanity: the shader mentions the BC5-specific concepts.
    expect(src).toContain('encode_bc4')
    expect(src).toMatch(/6-interp/i)
  })
})

describe('BC7Encoder', () => {
  it('declares its metadata', () => {
    expect(BC7Encoder.requiredFeature).toBe(WebGPUFeature.BC)
    expect(BC7Encoder.textureFormats).toEqual([TextureFormat.BC7, TextureFormat.BC7_SRGB])
    const view: AnyProto = Object.create(BC7Encoder.prototype)
    expect(view.label).toBe('bc7')
    expect(view.bytesPerBlock).toBe(16)
    expect(view.supportsSrgb).toBe(true)
    expect(view.gpuTextureFormat({ colorSpace: 'srgb' })).toBe('bc7-rgba-unorm-srgb')
    expect(view.gpuTextureFormat({ colorSpace: 'linear' })).toBe('bc7-rgba-unorm')
  })

  it('loads its WGSL shader source (bc7.wgsl imported via ?raw)', () => {
    const view: AnyProto = Object.create(BC7Encoder.prototype)
    const src: string = view.wgslSource()
    expect(typeof src).toBe('string')
    expect(src).toContain('@compute')
    expect(src).toContain('@workgroup_size')
    // Sanity: the shader exposes the BC7-specific helpers by name.
    expect(src).toContain('farthest_pair')
    expect(src).toContain('build_palette_6')
    expect(src).toMatch(/Mode 6/i)
  })
})

describe('ASTC4x4Encoder', () => {
  it('declares its metadata', () => {
    expect(ASTC4x4Encoder.requiredFeature).toBe(WebGPUFeature.ASTC)
    expect(ASTC4x4Encoder.textureFormats).toEqual([TextureFormat.ASTC_4x4, TextureFormat.ASTC_4x4_SRGB])
    const view: AnyProto = Object.create(ASTC4x4Encoder.prototype)
    expect(view.label).toBe('astc4x4')
    expect(view.bytesPerBlock).toBe(16)
    expect(view.supportsSrgb).toBe(true)
    expect(view.gpuTextureFormat({ colorSpace: 'srgb' })).toBe('astc-4x4-unorm-srgb')
    expect(view.gpuTextureFormat({ colorSpace: 'linear' })).toBe('astc-4x4-unorm')
  })

  it('loads its WGSL shader source (astc4x4.wgsl imported via ?raw)', () => {
    const view: AnyProto = Object.create(ASTC4x4Encoder.prototype)
    const src: string = view.wgslSource()
    expect(typeof src).toBe('string')
    expect(src).toContain('@compute')
    expect(src).toContain('@workgroup_size')
    // Sanity: the shader exposes the ASTC-specific helpers by name.
    expect(src).toContain('weight_unq')
    expect(src).toContain('build_palette')
    expect(src).toMatch(/CEM 12/i)
  })
})
