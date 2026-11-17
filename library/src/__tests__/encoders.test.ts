import { ASTC4x4Encoder } from '../ASTC4x4Encoder.js'
import { BC1Encoder } from '../BC1Encoder.js'
import { BC4Encoder } from '../BC4Encoder.js'
import { BC5Encoder } from '../BC5Encoder.js'
import { BC7Encoder } from '../BC7Encoder.js'
import { _internal as _eacInternal } from '../eac_r11_ref.js'
import { EACR11Encoder } from '../EACR11Encoder.js'
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
    expect(src).toContain('fn moments')
    expect(src).toContain('fn principal_axis')
  })
})

describe('f16 fast shader variants', () => {
  it('every encoder ships an f16 fast module', () => {
    for (const cls of [BC1Encoder, BC4Encoder, BC5Encoder, BC7Encoder, ASTC4x4Encoder, ETC2Encoder, EACR11Encoder]) {
      const view: AnyProto = Object.create(cls.prototype)
      const src: string | null = view.wgslSourceFastF16()
      expect(typeof src).toBe('string')
      expect(src).toContain('enable f16')
      expect(src).toContain('@compute')
    }
  })

  it("ETC2's f16 module is the exact-value port (big sums stay f32)", () => {
    const view: AnyProto = Object.create(ETC2Encoder.prototype)
    const src: string = view.wgslSourceFastF16()
    expect(src).toContain('array<vec4<f16>, 4>')
    expect(src).toContain('EXACT-VALUE f16')
  })
})

describe('opaque-source kernels', () => {
  it('ASTC ships one: three-channel gathers, no translucent class', () => {
    const view: AnyProto = Object.create(ASTC4x4Encoder.prototype)
    const src: string = view.wgslSourceOpaqueF16()
    expect(src).toContain('enable f16')
    expect(src).toContain('@binding(3) var smp: sampler;')
    expect(src).toContain('textureGather(2, src_tex')
    expect(src).not.toContain('textureGather(3, src_tex')
    // CEM 0 and CEM 8 block modes only — no CEM 12 (0x042) layout.
    expect(src).toContain('0x253u')
    expect(src).toContain('0x242u')
    expect(src).toContain('0x053u')
    expect(src).not.toContain('0x042u')
  })

  it('the other encoders have none (the promise is ignored there)', () => {
    for (const cls of [BC1Encoder, BC4Encoder, BC5Encoder, BC7Encoder, ETC2Encoder, EACR11Encoder]) {
      const view: AnyProto = Object.create(cls.prototype)
      expect(view.wgslSourceOpaqueF16()).toBeNull()
    }
  })
})

describe('BC4Encoder', () => {
  it('declares its metadata', () => {
    expect(BC4Encoder.requiredFeature).toBe(WebGPUFeature.BC)
    expect(BC4Encoder.textureFormats).toEqual([TextureFormat.BC4])
    const view: AnyProto = Object.create(BC4Encoder.prototype)
    expect(view.label).toBe('bc4')
    expect(view.bytesPerBlock).toBe(8)
    expect(view.supportsSrgb).toBe(false)
    expect(view.srcTextureFormat).toBe('r8unorm')
    expect(view.gpuTextureFormat({ colorSpace: 'linear' })).toBe('bc4-r-unorm')
  })

  it('loads both WGSL modules, reading R through four gathers', () => {
    const view: AnyProto = Object.create(BC4Encoder.prototype)
    for (const src of [view.wgslSource(), view.wgslSourceFastF16()] as string[]) {
      expect(src).toContain('@compute')
      expect(src).toContain('@binding(3) var smp: sampler;')
      expect(src.match(/textureGather\(0, src_tex/g)?.length).toBe(4)
      expect(src).not.toContain('textureGather(1, src_tex')
      // 8-byte blocks: two u32 per block.
      expect(src).toContain('let o = bi * 2u;')
    }
  })
})

describe('EACR11Encoder', () => {
  it('declares its metadata', () => {
    expect(EACR11Encoder.requiredFeature).toBe(WebGPUFeature.ETC2)
    expect(EACR11Encoder.textureFormats).toEqual([TextureFormat.EAC_R11])
    const view: AnyProto = Object.create(EACR11Encoder.prototype)
    expect(view.label).toBe('eac-r11')
    expect(view.bytesPerBlock).toBe(8)
    expect(view.supportsSrgb).toBe(false)
    expect(view.srcTextureFormat).toBe('r8unorm')
    expect(view.gpuTextureFormat({ colorSpace: 'linear' })).toBe('eac-r11unorm')
  })

  it('both WGSL modules carry the same candidate table as the CPU reference', () => {
    const view: AnyProto = Object.create(EACR11Encoder.prototype)
    const want = _eacInternal.CANDIDATES.map(c => `0x${c.toString(16).padStart(5, '0')}u`)
    for (const src of [view.wgslSource(), view.wgslSourceFastF16()] as string[]) {
      expect(src).toContain('@compute')
      const lut = src.slice(src.indexOf('const LUT = array<u32, 41>('))
      expect(lut.slice(0, lut.indexOf(');')).match(/0x[0-9a-f]{5}u/g)).toEqual(want)
    }
  })

  it('the f16 module is the exact-value port (error sums stay f32)', () => {
    const view: AnyProto = Object.create(EACR11Encoder.prototype)
    const src: string = view.wgslSourceFastF16()
    expect(src).toContain('EXACT-VALUE f16')
    expect(src).toContain('var t: array<h4, 4>;')
    expect(src).toContain('var e = 0.0;')
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
    expect(src).toContain('compact3')
    expect(src).toMatch(/Mode 6/i)
    expect(src).toMatch(/Mode 4/i)
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
    expect(src).toContain('fn sb_pair')
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
