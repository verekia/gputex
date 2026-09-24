import { useEffect, useState } from 'react'

import { compressTexture, prewarmCompressTexture } from 'gputex/three'

import { forceWebGL } from '../lib/forceWebGL'

import type { CompressOptions, CompressResult } from 'gputex/three'
import type { Texture } from 'three'

import type { EncodeInfo } from './useGputex'

// Compress a dropped/picked File straight from its bytes. Deliberately NOT
// routed through useLoader + a blob/data URL: compressTexture() accepts the
// Blob itself, which skips the URL fetch hop entirely (a blob-URL fetch of
// a 20 MB file costs ~200 ms; a data URL >1 s). `cache: true` keeps the
// compressed bytes in the session's in-memory transcode cache keyed by
// content hash, so re-dropping the same file skips decode + encode
// (~30 ms instead of hundreds).

const DEMO_OPTIONS: CompressOptions = { hint: 'color', colorSpace: 'srgb', svgSize: 1024, cache: true, forceWebGL }

// Compile the demo's encoder at page load, while the user picks a file: the
// same selection compressTexture() makes for DEMO_OPTIONS on this client, so
// the first drop doesn't pay a cold shader compile.
if (typeof window !== 'undefined') {
  void prewarmCompressTexture(DEMO_OPTIONS).then(({ targets, ms }) =>
    console.log(`[GPUtex] prewarmed ${targets.map(t => `${t.backend} ${t.format}`).join(', ')} in ${ms.toFixed(1)} ms`),
  )
}

export function useCompressedFile(file: File | null, onResult?: (info: EncodeInfo) => void): Texture | null {
  const [texture, setTexture] = useState<Texture | null>(null)

  useEffect(() => {
    if (!file) return
    let cancelled = false
    let result: CompressResult | null = null
    setTexture(null)

    compressTexture(file, DEMO_OPTIONS)
      .then(r => {
        if (cancelled) {
          r.destroy()
          return
        }
        result = r
        setTexture(r.texture)
        const mip0 = (r.texture as { mipmaps?: { data: Uint8Array }[] }).mipmaps?.[0]
        onResult?.({
          format: r.format,
          fallbackUncompressed: r.fallbackUncompressed,
          backend: r.backend,
          astcNormalRemap: r.astcNormalRemap,
          width: r.width,
          height: r.height,
          mipLevels: r.mipLevels,
          encodeMs: r.encodeMs,
          decodeMs: r.decodeMs,
          totalMs: r.totalMs,
          cacheHit: r.cacheHit,
          compressedBytes: r.fallbackUncompressed ? r.width * r.height * 4 : (mip0?.data.byteLength ?? 0),
        })
      })
      .catch((e: unknown) => console.error('[useCompressedFile]', e))

    return () => {
      cancelled = true
      result?.destroy()
    }
  }, [file, onResult])

  return texture
}
