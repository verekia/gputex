// Test-support entry point (`gputex/testing`).
//
// Re-exports the CPU reference encoders/decoders that the GPU shaders are
// validated against. These are NOT part of the runtime API surface — they are
// slow, block-at-a-time TypeScript implementations meant for test suites and
// benchmarks (e.g. the browser GPU test page in the example app) that need to:
//
//   • gate GPU output against the exhaustive reference encode, per block and
//     in aggregate PSNR (the reference is the quality yardstick)
//   • decode compressed blocks on the CPU to compute PSNR
//
// Keeping them behind a dedicated entry keeps `gputex` itself lean while
// letting any consumer verify encoder output without re-deriving the formats.

export { encodeBC1Block, decodeBC1Block, type BC1Pixels, type BC1Block, type BC1Quality } from './bc1_ref.js'
export { encodeBC4Block, decodeBC4Block, type BC4Values, type BC4Block } from './bc4_ref.js'
export { encodeBC5Block, decodeBC5Block, type BC5Block, type BC5Decoded } from './bc5_ref.js'
export {
  encodeBC7Block,
  decodeBC7Block,
  encodeBC7Mode6Block,
  decodeBC7Mode6Block,
  decodeBC7Mode1Block,
  decodeBC7Mode4Block,
  readBC7Mode,
  type BC7Pixels,
  type BC7Block,
} from './bc7_ref.js'
export { encodeASTC4x4Block, decodeASTC4x4Block, type ASTC4x4Pixels, type ASTC4x4Block } from './astc4x4_ref.js'
export {
  encodeETC2Block,
  decodeETC2Block,
  readETC2Mode,
  type ETC2Pixels,
  type ETC2Block,
  type ETC2Quality,
  type ETC2Mode,
} from './etc2_ref.js'
