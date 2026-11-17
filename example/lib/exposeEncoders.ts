// Puts the encoder classes, CPU reference codecs and the high-level API on
// `window.__GPUTEX_ENCODERS__` for console / automation experiments (the
// /bench and /lab pages call this; example/scripts/device-run.mjs --js
// drives it on real devices).

import {
  ASTC4x4Encoder,
  ASTC4x4WebGLEncoder,
  BC1Encoder,
  BC1WebGLEncoder,
  BC5Encoder,
  BC5WebGLEncoder,
  BC7Encoder,
  BC7WebGLEncoder,
  ETC2Encoder,
  ETC2WebGLEncoder,
  generateGpuMipChain,
  generateMipChain,
  padToBlockMultiple,
} from 'gputex'

import {
  decodeASTC4x4Block,
  decodeBC1Block,
  decodeBC5Block,
  decodeBC7Block,
  decodeETC2Block,
  encodeASTC4x4Block,
  encodeBC1Block,
  encodeBC5Block,
  encodeBC7Mode6Block,
  encodeETC2Block,
} from 'gputex/testing'
import { clearTranscodeCache, compressTexture, releaseSharedGpuResources, setTranscodeCacheLimit } from 'gputex/three'

declare global {
  interface Window {
    /** Encoder classes, exposed for ad-hoc console/automation experiments. */
    __GPUTEX_ENCODERS__?: Record<string, unknown>
  }
}

export function exposeEncoders(): void {
  if (typeof window === 'undefined') return
  window.__GPUTEX_ENCODERS__ = {
    BC1Encoder,
    BC5Encoder,
    BC7Encoder,
    ASTC4x4Encoder,
    ETC2Encoder,
    BC1WebGLEncoder,
    BC5WebGLEncoder,
    BC7WebGLEncoder,
    ASTC4x4WebGLEncoder,
    ETC2WebGLEncoder,
    decodeBC1Block,
    decodeBC5Block,
    decodeBC7Block,
    decodeASTC4x4Block,
    decodeETC2Block,
    encodeBC1Block,
    encodeBC5Block,
    encodeBC7Mode6Block,
    encodeASTC4x4Block,
    encodeETC2Block,
    compressTexture,
    releaseSharedGpuResources,
    clearTranscodeCache,
    setTranscodeCacheLimit,
    generateMipChain,
    generateGpuMipChain,
    padToBlockMultiple,
  }
}
