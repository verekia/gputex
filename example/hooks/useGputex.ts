import { useLayoutEffect } from 'react'

import { WebGPUCompressedTextureLoader } from 'gputex'
import type { TextureHint } from 'gputex'

import { useLoader } from '@react-three/fiber'

import type { Texture } from 'three'

export interface EncodeInfo {
  format: string | null
  fallbackUncompressed: boolean
  astcNormalRemap: boolean
  width: number
  height: number
  mipLevels: number
  encodeMs: number
  compressedBytes: number
}

interface UseGputexOptions {
  hint?: TextureHint
  colorSpace?: 'srgb' | 'linear'
  flipY?: boolean
  mipmaps?: boolean
}

export function useGputex(
  url: string | string[],
  options?: UseGputexOptions,
  onLoad?: (texture: Texture | Texture[], result: EncodeInfo | null) => void,
): Texture | Texture[] {
  const textures = useLoader(WebGPUCompressedTextureLoader, url, loader => {
    if (options?.hint !== undefined) loader.hint = options.hint
    if (options?.colorSpace !== undefined) loader.colorSpace = options.colorSpace
    if (options?.flipY !== undefined) loader.flipY = options.flipY
    if (options?.mipmaps !== undefined) loader.mipmaps = options.mipmaps
  })

  useLayoutEffect(() => {
    const tex = Array.isArray(textures) ? textures[0] : textures
    const result = (tex?.userData?.gputex as EncodeInfo) ?? null
    onLoad?.(textures, result)
  }, [onLoad, textures])

  return textures
}

useGputex.preload = (url: string | string[], options?: UseGputexOptions) => {
  useLoader.preload(WebGPUCompressedTextureLoader, url, loader => {
    if (options?.hint !== undefined) loader.hint = options.hint
    if (options?.colorSpace !== undefined) loader.colorSpace = options.colorSpace
    if (options?.flipY !== undefined) loader.flipY = options.flipY
    if (options?.mipmaps !== undefined) loader.mipmaps = options.mipmaps
  })
}

useGputex.clear = (url: string | string[]) => {
  useLoader.clear(WebGPUCompressedTextureLoader, url)
}
