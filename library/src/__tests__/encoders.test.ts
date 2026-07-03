import { ASTC4x4Encoder } from '../ASTC4x4Encoder.js'
import { BC1Encoder } from '../BC1Encoder.js'
import { BC5Encoder } from '../BC5Encoder.js'
import { BC7Encoder } from '../BC7Encoder.js'
import { ETC2Encoder } from '../ETC2Encoder.js'
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
    // Sanity: the shader exposes the BC1-specific helpers by name.
    expect(src).toContain('fn project_stats')
    expect(src).toContain('fn principal_axis')
  })
})

describe('f16 fast shader variants', () => {
  it('every BC/ASTC encoder ships an f16 fast module (BC1 included)', () => {
    for (const cls of [BC1Encoder, BC5Encoder, BC7Encoder, ASTC4x4Encoder]) {
      const view: AnyProto = Object.create(cls.prototype)
      const src: string | null = view.wgslSourceFastF16()
      expect(typeof src).toBe('string')
      expect(src).toContain('enable f16')
      expect(src).toContain('@compute')
    }
  })

  it('ETC2 has no f16 variant (integer-exact f32 algorithm)', () => {
    const view: AnyProto = Object.create(ETC2Encoder.prototype)
    expect(view.wgslSourceFastF16()).toBe(null)
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
    expect(src).toContain('BC4')
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
    expect(src).toContain('principal_axis4')
    expect(src).toContain('pick_ep')
    expect(src).toMatch(/Mode 6/i)
  })
})

describe('ETC2Encoder', () => {
  it('declares its metadata', () => {
    expect(ETC2Encoder.requiredFeature).toBe(WebGPUFeature.ETC2)
    expect(ETC2Encoder.textureFormats).toEqual([TextureFormat.ETC2_RGB8, TextureFormat.ETC2_RGB8_SRGB])
    const view: AnyProto = Object.create(ETC2Encoder.prototype)
    expect(view.label).toBe('etc2')
    expect(view.bytesPerBlock).toBe(8)
    expect(view.supportsSrgb).toBe(true)
    expect(view.gpuTextureFormat({ colorSpace: 'srgb' })).toBe('etc2-rgb8unorm-srgb')
    expect(view.gpuTextureFormat({ colorSpace: 'linear' })).toBe('etc2-rgb8unorm')
  })

  it('loads its WGSL shader source (etc2.wgsl imported via ?raw)', () => {
    const view: AnyProto = Object.create(ETC2Encoder.prototype)
    const src: string = view.wgslSource()
    expect(typeof src).toBe('string')
    expect(src).toContain('@compute')
    expect(src).toContain('@workgroup_size')
    // Sanity: the shader exposes the ETC2-specific helpers by name.
    expect(src).toContain('fn sb_search')
    expect(src).toContain('fn quantise_bases')
    expect(src).toMatch(/planar/i)
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
    expect(src).toContain('principal_axis4')
    expect(src).toContain('proj_fit')
    // All three block classes are packed by the shader.
    expect(src).toContain('0x253u')
    expect(src).toContain('0x053u')
    expect(src).toContain('0x042u')
  })
})
