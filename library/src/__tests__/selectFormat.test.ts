import { ASTC4x4Encoder } from '../ASTC4x4Encoder.js'
import { BC5Encoder } from '../BC5Encoder.js'
import { BC7Encoder } from '../BC7Encoder.js'
import { selectFormat } from '../selectFormat.js'
import { TextureFormat, WebGPUFeature } from '../TextureFormat.js'

// Build a minimal FeatureProvider stub. `detectCapabilities` only touches
// `features.has(name)`, matching the structural type.
function stubAdapter(features: readonly GPUFeatureName[]) {
  const set = new Set(features)
  return { features: { has: (name: string) => set.has(name as GPUFeatureName) } }
}

describe('selectFormat: BC path', () => {
  it('returns BC7_SRGB + BC7Encoder for color hint (sRGB default)', () => {
    const sel = selectFormat(stubAdapter([WebGPUFeature.BC]), 'color')
    expect(sel.format).toBe(TextureFormat.BC7_SRGB)
    expect(sel.encoderClass).toBe(BC7Encoder)
    expect(sel.astcNormalRemap).toBe(false)
  })

  it('returns BC7 (linear) when colorSpace=linear is requested', () => {
    const sel = selectFormat(stubAdapter([WebGPUFeature.BC]), 'color', { colorSpace: 'linear' })
    expect(sel.format).toBe(TextureFormat.BC7)
    expect(sel.encoderClass).toBe(BC7Encoder)
  })

  it('returns BC7_SRGB for colorWithAlpha — same path as color', () => {
    const sel = selectFormat(stubAdapter([WebGPUFeature.BC]), 'colorWithAlpha')
    expect(sel.format).toBe(TextureFormat.BC7_SRGB)
    expect(sel.encoderClass).toBe(BC7Encoder)
  })

  it('returns BC5 for normal hint regardless of colorSpace option', () => {
    // BC5 has no sRGB variant; the `colorSpace` option is ignored for normals.
    const withSrgb = selectFormat(stubAdapter([WebGPUFeature.BC]), 'normal', { colorSpace: 'srgb' })
    const withoutSrgb = selectFormat(stubAdapter([WebGPUFeature.BC]), 'normal', { colorSpace: 'linear' })
    expect(withSrgb.format).toBe(TextureFormat.BC5)
    expect(withSrgb.encoderClass).toBe(BC5Encoder)
    expect(withoutSrgb.format).toBe(TextureFormat.BC5)
    expect(withSrgb.astcNormalRemap).toBe(false)
  })
})

describe('selectFormat: ASTC path', () => {
  it('returns ASTC_4x4_SRGB + ASTC4x4Encoder for color hint', () => {
    const sel = selectFormat(stubAdapter([WebGPUFeature.ASTC]), 'color')
    expect(sel.format).toBe(TextureFormat.ASTC_4x4_SRGB)
    expect(sel.encoderClass).toBe(ASTC4x4Encoder)
    expect(sel.astcNormalRemap).toBe(false)
  })

  it('returns ASTC_4x4 (linear) when colorSpace=linear', () => {
    const sel = selectFormat(stubAdapter([WebGPUFeature.ASTC]), 'color', { colorSpace: 'linear' })
    expect(sel.format).toBe(TextureFormat.ASTC_4x4)
  })

  it('returns ASTC_4x4 with astcNormalRemap=true for normal hint', () => {
    // ASTC has no 2-channel mode; the caller must apply a pre-swizzle.
    // The flag makes that requirement visible instead of silently
    // producing a degraded normal map.
    const sel = selectFormat(stubAdapter([WebGPUFeature.ASTC]), 'normal')
    expect(sel.format).toBe(TextureFormat.ASTC_4x4_SRGB)
    expect(sel.encoderClass).toBe(ASTC4x4Encoder)
    expect(sel.astcNormalRemap).toBe(true)
  })
})

describe('selectFormat: BC takes precedence over ASTC', () => {
  it('picks BC when both features are reported', () => {
    // Unusual but possible on a driver that exposes both. BC is what
    // our desktop path wants; ASTC stays reserved for mobile.
    const sel = selectFormat(stubAdapter([WebGPUFeature.BC, WebGPUFeature.ASTC]), 'color')
    expect(sel.encoderClass).toBe(BC7Encoder)
  })
})

describe('selectFormat: fallback', () => {
  it('returns null format + null encoder when neither feature is present', () => {
    const sel = selectFormat(stubAdapter([]), 'color')
    expect(sel.format).toBe(null)
    expect(sel.encoderClass).toBe(null)
    expect(sel.astcNormalRemap).toBe(false)
  })

  it('ignores unrelated features like ETC2', () => {
    // ETC2 is a real `GPUFeatureName` but we have no ETC2 encoder.
    const sel = selectFormat(stubAdapter([WebGPUFeature.ETC2]), 'color')
    expect(sel.format).toBe(null)
    expect(sel.encoderClass).toBe(null)
  })
})
