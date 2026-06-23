// WebGL2 fallback encoder surface.
//
// Used when WebGPU is unavailable: the same block-compression formats produced
// via fragment shaders instead of compute shaders. Re-exported from the package
// root so callers can drive the WebGL path directly if they want to.

export {
  WebGLBlockEncoder,
  type WebGLEncoderImageSource,
  type WebGLEncoderOptions,
  type WebGLEncoderConstructor,
  type WebGLEncodeBytesResult,
  type RawPixelSource,
} from './WebGLBlockEncoder.js'

export { BC1WebGLEncoder } from './BC1WebGLEncoder.js'
export { BC5WebGLEncoder } from './BC5WebGLEncoder.js'
export { BC7WebGLEncoder } from './BC7WebGLEncoder.js'
export { ASTC4x4WebGLEncoder } from './ASTC4x4WebGLEncoder.js'

export { createWebGLContext, getSharedWebGLContext, isWebGLAvailable } from './webglContext.js'
export { detectWebGLCapabilities, type WebGLCapabilities, type ExtensionProvider } from './webglCapabilities.js'
export { selectWebGLFormat, type WebGLFormatSelection } from './selectWebGLFormat.js'
