// Public entry point for the encoder package.
//
// Two surfaces:
//   • High-level — `compressTexture()` + `GputexLoader`
//     for drop-in use in Three.js apps.
//   • Low-level  — the individual `*Encoder` classes and capability /
//     format utilities for callers that want direct control.

// Format / feature identifiers.
export { TextureFormat, WebGPUFeature } from './TextureFormat.js'
export type { Capabilities, FeatureProvider } from './capabilities.js'
export { detectCapabilities } from './capabilities.js'

// Base encoder machinery.
export {
  Encoder,
  type EncoderImageSource,
  type EncoderOptions,
  type EncoderConstructor,
  type EncodeCallOptions,
  type EncodeQuality,
  type EncodeResult,
  type EncodeBytesResult,
  type FormatVariant,
} from './Encoder.js'

// Concrete WebGPU encoders.
export { BC1Encoder } from './BC1Encoder.js'
export { BC5Encoder } from './BC5Encoder.js'
export { BC7Encoder } from './BC7Encoder.js'
export { ASTC4x4Encoder } from './ASTC4x4Encoder.js'

// WebGL2 fallback encoders + capability / selection utilities. Used
// automatically by `compressTexture()` when WebGPU is unavailable; exported
// for callers that want to drive the WebGL path directly.
export {
  WebGLBlockEncoder,
  BC1WebGLEncoder,
  BC5WebGLEncoder,
  BC7WebGLEncoder,
  ASTC4x4WebGLEncoder,
  createWebGLContext,
  getSharedWebGLContext,
  isWebGLAvailable,
  detectWebGLCapabilities,
  selectWebGLFormat,
  type WebGLEncoderImageSource,
  type WebGLEncoderOptions,
  type WebGLEncoderConstructor,
  type WebGLEncodeBytesResult,
  type RawPixelSource,
  type WebGLCapabilities,
  type ExtensionProvider,
  type WebGLFormatSelection,
} from './webgl/index.js'

// Format selection + high-level API.
export { selectFormat, type TextureHint, type SelectFormatOptions, type FormatSelection } from './selectFormat.js'
export {
  compressTexture,
  type CompressTextureSource,
  type CompressOptions,
  type CompressResult,
} from './compressTexture.js'
export { GputexLoader } from './GputexLoader.js'

// Mip-chain helpers (exported so callers can pre-generate a chain once
// and feed it into `Encoder.encodeToBytes` themselves if they want
// finer control over the pipeline than `compressTexture` provides).
export { generateMipChain, padToBlockMultiple, type MipLevel } from './mipgen.js'
