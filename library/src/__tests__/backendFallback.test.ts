import { afterEach, beforeEach, describe, expect, it, spyOn } from 'bun:test'

import { compressTextureToBytes, releaseSharedGpuResources } from '../compressTexture.js'

// A broken or missing WebGPU stack must degrade to the next tier, never
// reject: renderers fall back to WebGL2 in exactly these situations, and a
// rejected encode leaves the app with no texture at all. Bun has no WebGL2,
// so "the next tier" is the uncompressed RGBA8 result here.

const nav = navigator as unknown as { gpu?: unknown }
const g = globalThis as unknown as { createImageBitmap?: unknown }

function setGpu(descriptor: PropertyDescriptor): void {
  Object.defineProperty(nav, 'gpu', { ...descriptor, configurable: true })
}

function lost(): never {
  throw new Error('device lost')
}

/** An adapter whose device builds pipelines but can't run an encode (lost). */
function deadDeviceAdapter() {
  const device = {
    features: new Set<string>(),
    lost: new Promise(() => {}),
    createShaderModule: () => ({}),
    createComputePipelineAsync: async () => ({}),
    createTexture: lost,
    createBuffer: lost,
    destroy: () => {},
  }
  return {
    features: new Set(['texture-compression-bc']),
    info: { vendor: 'apple', architecture: 'metal-3' },
    requestDevice: async () => device,
  }
}

const OPTS = { hint: 'color', colorSpace: 'linear', quality: 'low', flipY: false } as const

const source = () => new Blob([new Uint8Array(16)], { type: 'image/png' })

describe('compressTextureToBytes backend fallback', () => {
  let prevGpu: PropertyDescriptor | undefined
  let prevCreateImageBitmap: unknown
  let warn: ReturnType<typeof spyOn>

  beforeEach(() => {
    prevGpu = Object.getOwnPropertyDescriptor(nav, 'gpu')
    prevCreateImageBitmap = g.createImageBitmap
    g.createImageBitmap = async () => ({ width: 8, height: 8, close: () => {} })
    warn = spyOn(console, 'warn').mockImplementation(() => {})
  })

  afterEach(() => {
    releaseSharedGpuResources()
    if (prevGpu) Object.defineProperty(nav, 'gpu', prevGpu)
    else delete nav.gpu
    g.createImageBitmap = prevCreateImageBitmap
    warn.mockRestore()
  })

  it('treats a navigator.gpu that reads undefined as no WebGPU', async () => {
    setGpu({ get: () => undefined })
    const r = await compressTextureToBytes(source(), OPTS)
    expect(r.backend).toBe('none')
    expect(r.fallbackUncompressed).toBe(true)
  })

  it('falls back when the adapter request rejects', async () => {
    setGpu({
      value: {
        requestAdapter: async () => {
          throw new Error('adapter request failed')
        },
      },
    })
    const r = await compressTextureToBytes(source(), OPTS)
    expect(r.backend).toBe('none')
  })

  it('falls back when the device request rejects', async () => {
    const adapter = {
      features: new Set(['texture-compression-bc']),
      requestDevice: async () => {
        throw new Error('device request failed')
      },
    }
    setGpu({ value: { requestAdapter: async () => adapter } })
    const r = await compressTextureToBytes(source(), OPTS)
    expect(r.backend).toBe('none')
  })

  it('falls back when the WebGPU encode itself fails', async () => {
    const adapter = deadDeviceAdapter()
    setGpu({ value: { requestAdapter: async () => adapter } })
    const r = await compressTextureToBytes(source(), OPTS)
    expect(r.backend).toBe('none')
    expect(warn.mock.calls.some(args => String(args[0]).includes('WebGPU encode failed'))).toBe(true)
  })
})
