// Device-specific workarounds.
//
// copyExternalImageToTexture produces black textures on some Mali
// drivers (Pixel 10 / Mali-G925). On those GPUs we rasterise to CPU
// pixels and upload via writeTexture instead.

/**
 * Returns true when the adapter's GPU is known to have a broken
 * `copyExternalImageToTexture` implementation.
 */
export function needsWriteTextureWorkaround(adapter: GPUAdapter): boolean {
  return adapter.info?.architecture?.includes('Mali') ?? false
}

/**
 * Upload an image source to a GPU texture. Uses `writeTexture` with raw
 * pixel bytes when `useWriteTexture` is true (Mali workaround), otherwise
 * falls back to the standard `copyExternalImageToTexture`.
 */
export function uploadSourceTexture(
  device: GPUDevice,
  srcTex: GPUTexture,
  source: ImageData | ImageBitmap | TexImageSource,
  width: number,
  height: number,
  flipY: boolean,
  useWriteTexture: boolean,
): void {
  if (useWriteTexture && source instanceof ImageData) {
    device.queue.writeTexture({ texture: srcTex }, source.data, { bytesPerRow: width * 4 }, [width, height, 1])
  } else {
    device.queue.copyExternalImageToTexture({ source: source as ImageBitmap, flipY }, { texture: srcTex }, [
      width,
      height,
      1,
    ])
  }
}
