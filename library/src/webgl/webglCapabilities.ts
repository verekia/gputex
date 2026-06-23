// WebGL fallback capability detection.
//
// Mirrors `capabilities.ts` (the WebGPU side) but keyed on WebGL2 compressed-
// texture extensions instead of `GPUAdapter` features. Each flag answers "can a
// texture in this compressed format actually be sampled by a WebGL2 renderer on
// this machine" — which is what gates whether the matching encoder is useful.
//
// We probe the *encoding* context's extensions as a proxy for the renderer's:
// both run on the same GPU, and Three.js's WebGL backend enables these
// extensions on demand when it meets a format it recognises (see
// WebGLUtils.convert → extensions.get).

/**
 * Minimal structural type for the extension source: only `getExtension` is
 * touched, so tests can pass a `{ getExtension }` stub instead of a real
 * `WebGL2RenderingContext`.
 */
export interface ExtensionProvider {
  getExtension(name: string): unknown
}

export interface WebGLCapabilities {
  /** EXT_texture_compression_bptc → BC7 (BPTC). */
  bptc: boolean
  /** EXT_texture_compression_rgtc → BC5 (RGTC2). */
  rgtc: boolean
  /** WEBGL_compressed_texture_s3tc → BC1 (DXT1). */
  s3tc: boolean
  /** WEBGL_compressed_texture_s3tc_srgb → sRGB DXT1 (separate extension). */
  s3tcSrgb: boolean
  /** WEBGL_compressed_texture_astc → ASTC 4×4 (and other footprints). */
  astc: boolean
}

export function detectWebGLCapabilities(gl: ExtensionProvider): WebGLCapabilities {
  if (!gl || typeof gl.getExtension !== 'function') {
    throw new TypeError('detectWebGLCapabilities: a WebGL2 context (or { getExtension }) is required')
  }
  const has = (name: string): boolean => gl.getExtension(name) != null
  return {
    bptc: has('EXT_texture_compression_bptc'),
    rgtc: has('EXT_texture_compression_rgtc'),
    s3tc: has('WEBGL_compressed_texture_s3tc'),
    s3tcSrgb: has('WEBGL_compressed_texture_s3tc_srgb'),
    astc: has('WEBGL_compressed_texture_astc'),
  }
}
