// `gputex/three` — the Three.js-coupled entry point.
//
// Re-exports the entire three-free core surface (so Three.js apps need only one
// import) and adds the helpers that depend on `three`:
//   • `compressTexture()` + `GputexLoader` — drop-in high-level API returning a
//     `CompressedTexture`.
//   • `buildCompressedTexture()` / `encodeToTexture()` / `threeFormatFor()` —
//     turn the core encoders' raw bytes into a `CompressedTexture`.
//
// The root `gputex` entry stays free of any `three` import; engines other than
// Three.js consume the core `compressTextureToBytes()` / encoders directly and
// never load this module. `compressTexture` / `CompressResult` here are the
// texture-returning variants (they shadow the core byte-oriented
// `compressTextureToBytes` / `CompressResult` re-exported just above).

export * from '../index.js'

export { compressTexture, type CompressResult } from './compressTexture.js'
export { GputexLoader } from './GputexLoader.js'
export {
  buildCompressedTexture,
  encodeToTexture,
  threeFormatFor,
  type EncodeResult,
  type EncodeToTextureOptions,
} from './buildTexture.js'
