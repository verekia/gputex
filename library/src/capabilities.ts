// Capability detection.
//
// Given a GPUAdapter (or a minimal stand-in with `.features` as a Set),
// return which TextureFormat values are usable on this hardware.
//
// The encoder itself only needs `texture-compression-bc` /
// `-astc` to sample the result — writing the storage buffer never
// requires the feature. So "supported" here means "the output texture
// will actually be sampleable", which is the user-facing distinction.

import { TextureFormat, WebGPUFeature } from './TextureFormat.js'

/**
 * Minimal structural type for the adapter argument: only `.features.has()`
 * is touched. Lets tests pass a `{ features: new Set(...) }` stub without
 * constructing a full `GPUAdapter`.
 */
export interface FeatureProvider {
  features: { has(name: string): boolean }
}

export interface Capabilities {
  bc: boolean
  astc: boolean
  etc2: boolean
  supportedFormats: TextureFormat[]
}

const FORMATS_BY_FEATURE: Record<WebGPUFeature, readonly TextureFormat[]> = {
  [WebGPUFeature.BC]: [
    TextureFormat.BC1,
    TextureFormat.BC1_SRGB,
    TextureFormat.BC5,
    TextureFormat.BC7,
    TextureFormat.BC7_SRGB,
  ],
  [WebGPUFeature.ASTC]: [TextureFormat.ASTC_4x4, TextureFormat.ASTC_4x4_SRGB],
  // ETC2 has no encoder yet — explicit empty keeps exhaustiveness check.
  [WebGPUFeature.ETC2]: [],
}

export function detectCapabilities(adapter: FeatureProvider): Capabilities {
  if (!adapter || !adapter.features || typeof adapter.features.has !== 'function') {
    throw new TypeError('detectCapabilities: adapter.features (Set-like) is required')
  }
  const has = (f: string): boolean => adapter.features.has(f)
  const bc = has(WebGPUFeature.BC)
  const astc = has(WebGPUFeature.ASTC)
  const etc2 = has(WebGPUFeature.ETC2)

  const supportedFormats: TextureFormat[] = []
  if (bc) supportedFormats.push(...FORMATS_BY_FEATURE[WebGPUFeature.BC])
  if (astc) supportedFormats.push(...FORMATS_BY_FEATURE[WebGPUFeature.ASTC])

  return { bc, astc, etc2, supportedFormats }
}
