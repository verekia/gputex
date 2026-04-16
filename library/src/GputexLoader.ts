// Three.js-compatible loader that wraps `compressTexture()`.
//
// Matches the classic `THREE.Loader` contract — callback-style `load()`
// + Promise-returning `loadAsync()` — so existing application code that
// already uses `TextureLoader` or similar can drop this in with the
// identical call shape:
//
//   const loader = new GputexLoader()
//   loader.hint = 'normal'
//   loader.mipmaps = true
//   const tex = await loader.loadAsync('/normal.png')
//   material.normalMap = tex
//
// Only .load() needs to be overridden; the base class's `loadAsync()`
// calls `load()` with resolve/reject wrapping the onLoad/onError
// callbacks, so we get both APIs from the single implementation.

import { Loader } from 'three'

import { compressTexture, type CompressResult } from './compressTexture.js'

import type { Texture } from 'three'

import type { TextureHint } from './selectFormat.js'

export class GputexLoader extends Loader<Texture> {
  /** Format-selection hint. Default 'color'. */
  hint: TextureHint = 'color'
  /** Pick the sRGB or linear variant of the chosen format. Default 'srgb'. */
  colorSpace: 'srgb' | 'linear' = 'srgb'
  /** Flip the image vertically before encoding. Default true (matches Three.js convention). */
  flipY: boolean = true
  /** Generate + encode a full mip chain. Default false. */
  mipmaps: boolean = false
  /**
   * Optional pre-existing WebGPU device. Reusing the renderer's device
   * avoids spinning up a second WebGPU context for encoding.
   */
  device?: GPUDevice
  adapter?: GPUAdapter

  /**
   * Most recent full encode result. Useful when the caller wants format
   * / mipLevels / astcNormalRemap metadata without threading a separate
   * callback through `load()`. Cleared when a new load starts.
   */
  lastResult: CompressResult | null = null

  /**
   * THREE.Loader contract: returns void, drives callbacks. `loadAsync`
   * (inherited from the base class) wraps this with Promise semantics.
   * Errors routed through `manager.itemError` so the LoadingManager's
   * aggregate state stays accurate.
   */
  override load(
    url: string,
    onLoad?: (texture: Texture) => void,
    _onProgress?: (event: ProgressEvent) => void,
    onError?: (err: unknown) => void,
  ): void {
    this.lastResult = null
    this.manager.itemStart(url)
    compressTexture(url, {
      hint: this.hint,
      colorSpace: this.colorSpace,
      flipY: this.flipY,
      mipmaps: this.mipmaps,
      device: this.device,
      adapter: this.adapter,
    }).then(
      result => {
        this.lastResult = result
        const mip0 = (result.texture as { mipmaps?: { data: Uint8Array }[] }).mipmaps?.[0]
        result.texture.userData.gputex = {
          format: result.format,
          fallbackUncompressed: result.fallbackUncompressed,
          astcNormalRemap: result.astcNormalRemap,
          width: result.width,
          height: result.height,
          mipLevels: result.mipLevels,
          encodeMs: result.encodeMs,
          compressedBytes: result.fallbackUncompressed
            ? result.width * result.height * 4
            : (mip0?.data.byteLength ?? 0),
        }
        onLoad?.(result.texture)
        this.manager.itemEnd(url)
      },
      (err: unknown) => {
        onError?.(err)
        this.manager.itemError(url)
        this.manager.itemEnd(url)
      },
    )
  }
}
