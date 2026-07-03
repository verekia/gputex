// Device-specific workarounds.
//
// copyExternalImageToTexture produces black textures on Pixel 10
// (Imagination Technologies "img-tec" / "d-series" PowerVR DXT GPU).
// On those GPUs we rasterise to CPU pixels and upload via writeTexture
// instead.

/**
 * Returns true when the adapter's GPU is known to have a broken
 * `copyExternalImageToTexture` implementation.
 *
 * Pixel 10 reports: vendor "img-tec", architecture "d-series".
 */
export function needsWriteTextureWorkaround(adapter: GPUAdapter): boolean {
  const { vendor, architecture } = adapter.info ?? {}
  return vendor === 'img-tec' && architecture === 'd-series'
}

/**
 * Upload an image source to a GPU texture.
 *
 * `ImageData` goes through `writeTexture`: for CPU-resident raw bytes it's
 * the direct path (no colour-space blit) and it sidesteps the broken
 * `copyExternalImageToTexture` devices above. `writeTexture` can't flip, so
 * `ImageData` + `flipY` falls back to the blit path — callers that need the
 * Pixel 10 workaround (see `compressTexture`) bake the flip into the pixels
 * before reaching here.
 */
export function uploadSourceTexture(
  device: GPUDevice,
  srcTex: GPUTexture,
  source: ImageData | ImageBitmap | TexImageSource,
  width: number,
  height: number,
  flipY: boolean,
): void {
  if (source instanceof ImageData && !flipY) {
    device.queue.writeTexture({ texture: srcTex }, source.data, { bytesPerRow: width * 4 }, [width, height, 1])
  } else {
    device.queue.copyExternalImageToTexture({ source: source as ImageBitmap, flipY }, { texture: srcTex }, [
      width,
      height,
      1,
    ])
  }
}
