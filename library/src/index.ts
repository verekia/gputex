// Public entry point — the engine-agnostic core.
//
// Nothing reachable from here imports `three`, so this entry works in any
// renderer (Babylon.js, raw WebGPU/WebGL, a worker, …). Encoders produce raw
// compressed block bytes via `encodeToBytes()`; feed them into whatever
// compressed-texture upload your engine exposes.
//
// Three.js users want `gputex/three` instead: it re-exports everything here
// plus the Three.js-coupled helpers (`compressTexture()`, `GputexLoader`,
// `buildCompressedTexture()`). That entry is the only one that pulls in
// `three`, which is why `three` is an optional peer dependency.

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

// Format selection.
export {
  selectFormat,
  type TextureHint,
  type PreferredFormat,
  type SelectFormatOptions,
  type FormatSelection,
} from './selectFormat.js'

// Mip-chain helpers (exported so callers can pre-generate a chain once
// and feed it into `Encoder.encodeToBytes` themselves if they want
// finer control over the pipeline than `compressTexture` provides).
export { generateMipChain, padToBlockMultiple, type MipLevel } from './mipgen.js'
