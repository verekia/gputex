import { useLayoutEffect } from 'react'

import type { SvgRasterSize, TextureHint } from 'gputex'

import { useLoader } from '@react-three/fiber'
import { GputexLoader } from 'gputex/three'

import { forceWebGL } from '../lib/forceWebGL'

import type { Texture } from 'three'

export interface EncodeInfo {
  format: string | null
  fallbackUncompressed: boolean
  backend: 'webgpu' | 'webgl' | 'none'
  astcNormalRemap: boolean
  width: number
  height: number
  mipLevels: number
  encodeMs: number
  decodeMs: number
  totalMs: number
  cacheHit: boolean
  compressedBytes: number
}

interface UseGputexOptions {
  hint?: TextureHint
  colorSpace?: 'srgb' | 'linear'
  svgSize?: SvgRasterSize
  flipY?: boolean
  mipmaps?: boolean
  cache?: boolean
}

export function useGputex(
  url: string | string[],
  options?: UseGputexOptions,
  onLoad?: (texture: Texture | Texture[], result: EncodeInfo | null) => void,
): Texture | Texture[] {
  const textures = useLoader(GputexLoader, url, loader => {
    loader.forceWebGL = forceWebGL
    if (options?.hint !== undefined) loader.hint = options.hint
    if (options?.colorSpace !== undefined) loader.colorSpace = options.colorSpace
    if (options?.svgSize !== undefined) loader.svgSize = options.svgSize
    if (options?.flipY !== undefined) loader.flipY = options.flipY
    if (options?.mipmaps !== undefined) loader.mipmaps = options.mipmaps
    if (options?.cache !== undefined) loader.cache = options.cache
  })

  useLayoutEffect(() => {
    const tex = Array.isArray(textures) ? textures[0] : textures
    const result = (tex?.userData?.gputex as EncodeInfo) ?? null
    onLoad?.(textures, result)
  }, [onLoad, textures])

  return textures
}

useGputex.preload = (url: string | string[], options?: UseGputexOptions) => {
  useLoader.preload(GputexLoader, url, loader => {
    loader.forceWebGL = forceWebGL
    if (options?.hint !== undefined) loader.hint = options.hint
    if (options?.colorSpace !== undefined) loader.colorSpace = options.colorSpace
    if (options?.svgSize !== undefined) loader.svgSize = options.svgSize
    if (options?.flipY !== undefined) loader.flipY = options.flipY
    if (options?.mipmaps !== undefined) loader.mipmaps = options.mipmaps
    if (options?.cache !== undefined) loader.cache = options.cache
  })
}

useGputex.clear = (url: string | string[]) => {
  useLoader.clear(GputexLoader, url)
}
