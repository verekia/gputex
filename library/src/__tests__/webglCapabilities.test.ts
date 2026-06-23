import { detectWebGLCapabilities, type ExtensionProvider } from '../webgl/webglCapabilities.js'

// Stub the only method detectWebGLCapabilities touches: getExtension returns a
// truthy object for "supported" extension names, null otherwise — exactly what
// a real WebGL2 context does.
function stubGL(supported: readonly string[]): ExtensionProvider {
  const set = new Set(supported)
  return { getExtension: (name: string) => (set.has(name) ? {} : null) }
}

describe('detectWebGLCapabilities', () => {
  it('reports all formats absent on a bare context', () => {
    const caps = detectWebGLCapabilities(stubGL([]))
    expect(caps).toEqual({ bptc: false, rgtc: false, s3tc: false, s3tcSrgb: false, astc: false })
  })

  it('maps each extension string to its capability flag', () => {
    expect(detectWebGLCapabilities(stubGL(['EXT_texture_compression_bptc'])).bptc).toBe(true)
    expect(detectWebGLCapabilities(stubGL(['EXT_texture_compression_rgtc'])).rgtc).toBe(true)
    expect(detectWebGLCapabilities(stubGL(['WEBGL_compressed_texture_s3tc'])).s3tc).toBe(true)
    expect(detectWebGLCapabilities(stubGL(['WEBGL_compressed_texture_s3tc_srgb'])).s3tcSrgb).toBe(true)
    expect(detectWebGLCapabilities(stubGL(['WEBGL_compressed_texture_astc'])).astc).toBe(true)
  })

  it('detects a typical desktop set (bptc + rgtc + s3tc)', () => {
    const caps = detectWebGLCapabilities(
      stubGL([
        'EXT_texture_compression_bptc',
        'EXT_texture_compression_rgtc',
        'WEBGL_compressed_texture_s3tc',
        'WEBGL_compressed_texture_s3tc_srgb',
      ]),
    )
    expect(caps).toEqual({ bptc: true, rgtc: true, s3tc: true, s3tcSrgb: true, astc: false })
  })

  it('throws on an object without getExtension', () => {
    // @ts-expect-error — intentionally wrong shape
    expect(() => detectWebGLCapabilities({})).toThrow(TypeError)
  })
})
