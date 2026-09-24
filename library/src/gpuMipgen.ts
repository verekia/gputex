// GPU mip-chain generation.
//
// The CPU path (mipgen.ts) costs real main-thread time at 4K: canvas
// `getImageData` readback (~33 ms) + the JS box filter (~38 ms) + a
// writeTexture upload per level. This module uploads the source bitmap ONCE
// and box-filters the whole chain on the GPU in a single compute pass —
// each dispatch samples mip level i and storage-writes level i+1 (compute
// passes scope usage per dispatch, so same-texture subresource ping-pong is
// valid). The encoder then reads the levels straight from texture views
// (`Encoder.encodeMipChainFromTexture`), so pixels never transit the CPU
// between decode and readback of the compressed bytes.
//
// The filter is INTEGER-EXACT against mipgen.ts's `downsample2x`: loads are
// rounded back to byte values, summed, and `(a+b+c+d+2)>>2` is evaluated in
// exact f32 integer arithmetic (sums ≤ 1022 are exact in f32), so GPU- and
// CPU-generated chains produce byte-identical compressed output. It also
// shares the CPU path's colour-space trade-off: bytes are filtered raw, not
// linearised (see the mipgen.ts header).
//
// The CPU path remains for devices with a broken copyExternalImageToTexture
// (workarounds.ts) and for engine-agnostic callers that already hold CPU
// pixel chains.

/** Matches mipgen's chain length: floor(log2(max(w, h))) + 1 levels. */
export function gpuMipLevelCount(width: number, height: number): number {
  return 32 - Math.clz32(Math.max(width, height))
}

const DOWNSAMPLE_WGSL = /* wgsl */ `
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var dst : texture_storage_2d<rgba8unorm, write>;

// 2x2 box filter, clamp-to-edge fold on odd source dimensions, integer
// round-to-nearest — bit-exact with mipgen.ts downsample2x.
@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dstSize = textureDimensions(dst);
  if (gid.x >= dstSize.x || gid.y >= dstSize.y) { return; }
  let srcMax = textureDimensions(src) - vec2<u32>(1u, 1u);
  let x0 = min(gid.x * 2u, srcMax.x);
  let x1 = min(gid.x * 2u + 1u, srcMax.x);
  let y0 = min(gid.y * 2u, srcMax.y);
  let y1 = min(gid.y * 2u + 1u, srcMax.y);
  // round() recovers the exact byte values (unorm->f32 conversion is only
  // correctly-rounded, so a*255 is n +/- ~1e-5 — floor() would misround
  // sums divisible by 4 without it).
  let s = round(textureLoad(src, vec2<u32>(x0, y0), 0) * 255.0) +
          round(textureLoad(src, vec2<u32>(x1, y0), 0) * 255.0) +
          round(textureLoad(src, vec2<u32>(x0, y1), 0) * 255.0) +
          round(textureLoad(src, vec2<u32>(x1, y1), 0) * 255.0);
  textureStore(dst, gid.xy, floor((s + 2.0) * 0.25) / 255.0);
}
`

// One downsample pipeline per device, compiled on first use. WeakMap so a
// destroyed device's pipeline is collectable.
const pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>()

function getDownsamplePipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  let pipeline = pipelineCache.get(device)
  if (!pipeline) {
    pipeline = device.createComputePipelineAsync({
      label: 'gputex-mipgen-pipeline',
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ label: 'gputex-mipgen', code: DOWNSAMPLE_WGSL }),
        entryPoint: 'downsample',
      },
    })
    pipelineCache.set(device, pipeline)
    pipeline.catch(() => pipelineCache.delete(device))
  }
  return pipeline
}

/** Compile the mip-generation pipeline for `device` ahead of first use. */
export async function warmGpuMipgen(device: GPUDevice): Promise<void> {
  await getDownsamplePipeline(device)
}

/**
 * Upload `source` and generate its full mip chain on the GPU. Returns an
 * `rgba8unorm` texture with `gpuMipLevelCount` levels whose dimensions
 * follow the standard floor-halving chain; feed it to
 * `Encoder.encodeMipChainFromTexture()`. The caller owns the texture —
 * destroy it after encoding.
 *
 * The returned promise resolves once the work is submitted (not completed);
 * queue ordering makes the levels visible to any later submission.
 */
export async function generateGpuMipChain(
  device: GPUDevice,
  source: ImageBitmap | ImageData | HTMLCanvasElement | OffscreenCanvas,
  { flipY = false }: { flipY?: boolean } = {},
): Promise<GPUTexture> {
  const width = source.width
  const height = source.height
  if (!width || !height) {
    throw new Error('generateGpuMipChain: source has no dimensions')
  }
  const mipLevelCount = gpuMipLevelCount(width, height)
  const texture = device.createTexture({
    label: 'gputex-mip-chain',
    size: [width, height, 1],
    format: 'rgba8unorm',
    mipLevelCount,
    // COPY_DST + RENDER_ATTACHMENT for copyExternalImageToTexture (a blit
    // internally), TEXTURE_BINDING for downsample/encoder reads,
    // STORAGE_BINDING for downsample writes (rgba8unorm write-only storage
    // is core WebGPU).
    usage:
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING,
  })
  device.queue.copyExternalImageToTexture({ source: source as ImageBitmap, flipY }, { texture }, [width, height, 1])

  const pipeline = await getDownsamplePipeline(device)
  const enc = device.createCommandEncoder({ label: 'gputex-mipgen' })
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  for (let level = 0; level + 1 < mipLevelCount; level++) {
    const bindGroup = device.createBindGroup({
      label: `gputex-mipgen-bg-${level}`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: texture.createView({ baseMipLevel: level, mipLevelCount: 1 }) },
        { binding: 1, resource: texture.createView({ baseMipLevel: level + 1, mipLevelCount: 1 }) },
      ],
    })
    pass.setBindGroup(0, bindGroup)
    const dstW = Math.max(1, width >> (level + 1))
    const dstH = Math.max(1, height >> (level + 1))
    pass.dispatchWorkgroups(Math.ceil(dstW / 8), Math.ceil(dstH / 8), 1)
  }
  pass.end()
  device.queue.submit([enc.finish()])
  return texture
}
