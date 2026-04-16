import { detectCapabilities, type FeatureProvider } from '../capabilities.js'
import { TextureFormat, WebGPUFeature } from '../TextureFormat.js'

// Adapter stand-in: features is a Set-like with a .has() method, the same
// shape `GPUAdapter.features` exposes.
const fakeAdapter = (...features: string[]): FeatureProvider => ({
  features: new Set(features),
})

describe('TextureFormat', () => {
  it('lists every logical format we support', () => {
    expect(Object.keys(TextureFormat).sort()).toEqual([
      'ASTC_4x4',
      'ASTC_4x4_SRGB',
      'BC1',
      'BC1_SRGB',
      'BC5',
      'BC7',
      'BC7_SRGB',
    ])
  })

  // Runtime immutability is ensured by `as const` at the type layer; we don't
  // assert Object.isFrozen because the constant is a plain literal object.
})

describe('detectCapabilities', () => {
  it('reports nothing on an adapter with no compressed-texture features', () => {
    const caps = detectCapabilities(fakeAdapter())
    expect(caps).toEqual({
      bc: false,
      astc: false,
      etc2: false,
      supportedFormats: [],
    })
  })

  it('exposes every BC format when texture-compression-bc is present', () => {
    const caps = detectCapabilities(fakeAdapter(WebGPUFeature.BC))
    expect(caps.bc).toBe(true)
    expect(caps.astc).toBe(false)
    expect(caps.supportedFormats).toEqual([
      TextureFormat.BC1,
      TextureFormat.BC1_SRGB,
      TextureFormat.BC5,
      TextureFormat.BC7,
      TextureFormat.BC7_SRGB,
    ])
  })

  it('exposes ASTC formats when texture-compression-astc is present', () => {
    const caps = detectCapabilities(fakeAdapter(WebGPUFeature.ASTC))
    expect(caps.astc).toBe(true)
    expect(caps.supportedFormats).toEqual([TextureFormat.ASTC_4x4, TextureFormat.ASTC_4x4_SRGB])
  })

  it('unions BC and ASTC formats when both are present', () => {
    const caps = detectCapabilities(fakeAdapter(WebGPUFeature.BC, WebGPUFeature.ASTC))
    expect(caps.bc).toBe(true)
    expect(caps.astc).toBe(true)
    // BC formats come first, then ASTC.
    expect(caps.supportedFormats).toContain(TextureFormat.BC7)
    expect(caps.supportedFormats).toContain(TextureFormat.ASTC_4x4)
    expect(caps.supportedFormats.length).toBe(7)
  })

  it('ignores ETC2 presence for supportedFormats (no encoder yet)', () => {
    const caps = detectCapabilities(fakeAdapter(WebGPUFeature.ETC2))
    expect(caps.etc2).toBe(true)
    expect(caps.supportedFormats).toEqual([])
  })

  it('throws on an adapter missing a features Set', () => {
    // Cast through `unknown` — we explicitly want to validate the runtime
    // guard even though the types would reject these calls.
    expect(() => detectCapabilities({} as unknown as FeatureProvider)).toThrow(TypeError)
    expect(() => detectCapabilities(null as unknown as FeatureProvider)).toThrow(TypeError)
  })
})
