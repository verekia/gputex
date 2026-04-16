// Public identifiers for the compressed texture formats this project can emit.
//
// These are the *logical* formats the format-selection layer picks from; the
// concrete WebGPU and Three.js format strings are resolved per-encoder via
// `Encoder.gpuTextureFormat()` / `Encoder.threeTextureFormat()`.
//
// BC5 has no sRGB variant because it is a 2-channel linear format (tangent-
// space normals). The other three families have matched UNORM + sRGB pairs.
//
// Shape note: `as const` + derived type union keeps the identifiers usable
// both as runtime strings and as TypeScript literal types (`TextureFormat`
// is both the frozen object and the union type). `Object.freeze` previously
// provided the same guarantee at runtime; `as const` is enforced at the
// type level and any accidental mutation still produces a TS error.

export const TextureFormat = {
  BC1: 'BC1',
  BC1_SRGB: 'BC1_SRGB',
  BC5: 'BC5',
  BC7: 'BC7',
  BC7_SRGB: 'BC7_SRGB',
  ASTC_4x4: 'ASTC_4x4',
  ASTC_4x4_SRGB: 'ASTC_4x4_SRGB',
} as const

export type TextureFormat = (typeof TextureFormat)[keyof typeof TextureFormat]

// WebGPU feature strings. Naming them once here keeps `has(feature)` checks
// consistent across the capability layer and each encoder's static metadata.
// `satisfies Record<string, GPUFeatureName>` constrains the values to real
// WebGPU feature names (typos caught at compile time) without widening the
// literal types.
export const WebGPUFeature = {
  BC: 'texture-compression-bc',
  ASTC: 'texture-compression-astc',
  ETC2: 'texture-compression-etc2',
} as const satisfies Record<string, GPUFeatureName>

export type WebGPUFeature = (typeof WebGPUFeature)[keyof typeof WebGPUFeature]
