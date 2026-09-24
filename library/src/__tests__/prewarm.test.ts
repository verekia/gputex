import { afterEach, describe, expect, it } from 'bun:test'

import { prewarmCompressTexture, releaseSharedGpuResources } from '../compressTexture.js'
import { TextureFormat } from '../TextureFormat.js'

// A stub WebGPU adapter/device that records which shader modules get built,
// so the tests can check prewarming compiles exactly the pipelines the
// capability-based selection picks — and nothing else.
function stubGpu(adapterFeatures: readonly string[]): { modules: string[]; restore: () => void } {
  const modules: string[] = []
  const device = {
    features: new Set(adapterFeatures),
    lost: new Promise(() => {}),
    createShaderModule: ({ label }: { label: string }) => {
      modules.push(label)
      return {}
    },
    createComputePipelineAsync: async () => ({}),
    destroy: () => {},
  }
  const adapter = {
    features: new Set(adapterFeatures),
    info: { vendor: 'apple', architecture: 'metal-3' },
    requestDevice: async () => device,
  }
  const nav = navigator as unknown as { gpu?: unknown }
  const prev = Object.getOwnPropertyDescriptor(nav, 'gpu')
  Object.defineProperty(nav, 'gpu', { value: { requestAdapter: async () => adapter }, configurable: true })
  return {
    modules,
    restore: () => {
      if (prev) Object.defineProperty(nav, 'gpu', prev)
      else delete nav.gpu
    },
  }
}

describe('prewarmCompressTexture', () => {
  let restore: (() => void) | null = null
  afterEach(() => {
    releaseSharedGpuResources()
    restore?.()
    restore = null
  })

  it('reports no backend when neither WebGPU nor WebGL2 exists', async () => {
    const r = await prewarmCompressTexture([{ hint: 'color' }, { hint: 'normal' }])
    expect(r.targets).toEqual([
      { backend: 'none', format: null },
      { backend: 'none', format: null },
    ])
  })

  it('compiles only the encoders selected for a BC-class adapter', async () => {
    const gpu = stubGpu(['texture-compression-bc', 'shader-f16'])
    restore = gpu.restore
    const r = await prewarmCompressTexture([
      { hint: 'color', quality: 'low', colorSpace: 'linear', mipmaps: true },
      { hint: 'normal' },
      { hint: 'color', quality: 'low', colorSpace: 'linear' },
    ])
    expect(r.targets).toEqual([
      { backend: 'webgpu', format: TextureFormat.BC1 },
      { backend: 'webgpu', format: TextureFormat.BC5 },
      { backend: 'webgpu', format: TextureFormat.BC1 },
    ])
    // BC1 once (shared across targets), BC5, and the mip pipeline — no BC7,
    // ASTC or ETC2.
    expect(gpu.modules.toSorted()).toEqual(['bc1-encoder-f16', 'bc5-encoder-f16', 'gputex-mipgen'])
  })

  it('follows the ETC2 selection on an ETC2-only adapter', async () => {
    const gpu = stubGpu(['texture-compression-etc2'])
    restore = gpu.restore
    const r = await prewarmCompressTexture({ hint: 'color', quality: 'low' })
    expect(r.targets).toEqual([{ backend: 'webgpu', format: TextureFormat.ETC2_RGB8_SRGB }])
    expect(gpu.modules).toEqual(['etc2-encoder'])
  })
})
