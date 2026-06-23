import { TextureFormat } from '../TextureFormat.js'
import { ASTC4x4WebGLEncoder } from '../webgl/ASTC4x4WebGLEncoder.js'
import { BC1WebGLEncoder } from '../webgl/BC1WebGLEncoder.js'
import { BC5WebGLEncoder } from '../webgl/BC5WebGLEncoder.js'
import { BC7WebGLEncoder } from '../webgl/BC7WebGLEncoder.js'
import { selectWebGLFormat } from '../webgl/selectWebGLFormat.js'

import type { WebGLCapabilities } from '../webgl/webglCapabilities.js'

function caps(partial: Partial<WebGLCapabilities>): WebGLCapabilities {
  return { bptc: false, rgtc: false, s3tc: false, s3tcSrgb: false, astc: false, ...partial }
}

describe('selectWebGLFormat: BPTC (BC7) path', () => {
  it('returns BC7_SRGB for color (sRGB default)', () => {
    const sel = selectWebGLFormat(caps({ bptc: true }), 'color')
    expect(sel.format).toBe(TextureFormat.BC7_SRGB)
    expect(sel.encoderClass).toBe(BC7WebGLEncoder)
    expect(sel.astcNormalRemap).toBe(false)
  })

  it('returns BC7 (linear) when colorSpace=linear', () => {
    const sel = selectWebGLFormat(caps({ bptc: true }), 'color', { colorSpace: 'linear' })
    expect(sel.format).toBe(TextureFormat.BC7)
  })

  it('returns BC7_SRGB for colorWithAlpha', () => {
    const sel = selectWebGLFormat(caps({ bptc: true }), 'colorWithAlpha')
    expect(sel.format).toBe(TextureFormat.BC7_SRGB)
    expect(sel.encoderClass).toBe(BC7WebGLEncoder)
  })

  it('prefers BC7 over ASTC when both are present', () => {
    const sel = selectWebGLFormat(caps({ bptc: true, astc: true }), 'color')
    expect(sel.encoderClass).toBe(BC7WebGLEncoder)
  })
})

describe('selectWebGLFormat: normal hint', () => {
  it('returns BC5 when RGTC is present', () => {
    const sel = selectWebGLFormat(caps({ rgtc: true }), 'normal', { colorSpace: 'srgb' })
    expect(sel.format).toBe(TextureFormat.BC5)
    expect(sel.encoderClass).toBe(BC5WebGLEncoder)
    expect(sel.astcNormalRemap).toBe(false)
  })

  it('prefers BC5 over ASTC for normals when both present', () => {
    const sel = selectWebGLFormat(caps({ rgtc: true, astc: true }), 'normal')
    expect(sel.encoderClass).toBe(BC5WebGLEncoder)
  })

  it('falls back to ASTC with astcNormalRemap=true when only ASTC is present', () => {
    const sel = selectWebGLFormat(caps({ astc: true }), 'normal')
    expect(sel.format).toBe(TextureFormat.ASTC_4x4_SRGB)
    expect(sel.encoderClass).toBe(ASTC4x4WebGLEncoder)
    expect(sel.astcNormalRemap).toBe(true)
  })

  it('returns no format for normals when only s3tc is present (BC1 is unsuitable)', () => {
    const sel = selectWebGLFormat(caps({ s3tc: true, s3tcSrgb: true }), 'normal')
    expect(sel.format).toBe(null)
    expect(sel.encoderClass).toBe(null)
  })
})

describe('selectWebGLFormat: ASTC path', () => {
  it('returns ASTC_4x4_SRGB for color', () => {
    const sel = selectWebGLFormat(caps({ astc: true }), 'color')
    expect(sel.format).toBe(TextureFormat.ASTC_4x4_SRGB)
    expect(sel.encoderClass).toBe(ASTC4x4WebGLEncoder)
    expect(sel.astcNormalRemap).toBe(false)
  })

  it('returns ASTC_4x4 (linear) when colorSpace=linear', () => {
    const sel = selectWebGLFormat(caps({ astc: true }), 'color', { colorSpace: 'linear' })
    expect(sel.format).toBe(TextureFormat.ASTC_4x4)
  })
})

describe('selectWebGLFormat: BC1 last-resort (s3tc)', () => {
  it('returns BC1_SRGB for opaque color when sRGB s3tc is present', () => {
    const sel = selectWebGLFormat(caps({ s3tc: true, s3tcSrgb: true }), 'color')
    expect(sel.format).toBe(TextureFormat.BC1_SRGB)
    expect(sel.encoderClass).toBe(BC1WebGLEncoder)
  })

  it('returns BC1 (linear) for opaque color when only linear s3tc is present', () => {
    const sel = selectWebGLFormat(caps({ s3tc: true }), 'color', { colorSpace: 'linear' })
    expect(sel.format).toBe(TextureFormat.BC1)
    expect(sel.encoderClass).toBe(BC1WebGLEncoder)
  })

  it('does NOT use BC1 for sRGB color when the sRGB s3tc extension is missing', () => {
    // DXT1 sRGB needs WEBGL_compressed_texture_s3tc_srgb specifically.
    const sel = selectWebGLFormat(caps({ s3tc: true }), 'color', { colorSpace: 'srgb' })
    expect(sel.format).toBe(null)
  })

  it('does NOT use BC1 for colorWithAlpha (DXT1 has only 1-bit alpha)', () => {
    const sel = selectWebGLFormat(caps({ s3tc: true, s3tcSrgb: true }), 'colorWithAlpha')
    expect(sel.format).toBe(null)
    expect(sel.encoderClass).toBe(null)
  })
})

describe('selectWebGLFormat: no compressed path', () => {
  it('returns null format + null encoder when nothing is supported', () => {
    const sel = selectWebGLFormat(caps({}), 'color')
    expect(sel.format).toBe(null)
    expect(sel.encoderClass).toBe(null)
    expect(sel.astcNormalRemap).toBe(false)
  })
})
