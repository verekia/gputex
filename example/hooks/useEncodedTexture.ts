import { useEffect, useState } from 'react'

import type { EncoderConstructor, EncodeQuality } from 'gputex'

import { encodeToTexture } from 'gputex/three'

import type { Texture } from 'three'

// Encode a single image with ONE specific encoder class, bypassing
// `compressTexture()`'s capability-based auto-selection. The per-format test
// pages (/bc1, /bc5, /bc7, /astc) use this to force exactly the format they
// want — `compressTexture` would otherwise pick BC7 (desktop) or ASTC (mobile)
// for a colour hint and never BC1 at all.
//
// WebGPU compute path only: these pages exist to eyeball the compute encoders.

export interface EncodedInfo {
  /** Encoder label, e.g. 'bc7'. */
  label: string
  /** Concrete GPUTextureFormat string, e.g. 'bc7-rgba-unorm-srgb'. */
  format: string
  quality: EncodeQuality
  width: number
  height: number
  paddedWidth: number
  paddedHeight: number
  /** Size of the compressed level-0 bytes. */
  compressedBytes: number
  /** What the same image would cost as uncompressed RGBA8. */
  rgba8Bytes: number
  encodeMs: number
}

interface Options {
  colorSpace?: 'srgb' | 'linear'
  quality?: EncodeQuality
  /** Bake a vertical flip into the encoded bytes. Default true (matches the
   *  TextureLoader original, which Three flips on upload). */
  flipY?: boolean
}

export interface EncodedResult {
  texture: Texture | null
  info: EncodedInfo | null
  error: string | null
  loading: boolean
}

export function useEncodedTexture(url: string, EncoderClass: EncoderConstructor, options: Options = {}): EncodedResult {
  const { colorSpace = 'srgb', quality = 'fast', flipY = true } = options
  const [result, setResult] = useState<EncodedResult>({ texture: null, info: null, error: null, loading: true })

  useEffect(() => {
    let cancelled = false
    let encoder: Awaited<ReturnType<EncoderConstructor['create']>> | null = null
    let texture: Texture | null = null

    const run = async () => {
      setResult({ texture: null, info: null, error: null, loading: true })
      try {
        if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser.')
        const adapter = await navigator.gpu.requestAdapter()
        if (!adapter) throw new Error('No WebGPU adapter found.')

        // Pre-flight the feature the output texture needs to be *sampled*. The
        // encoder still writes a storage buffer without it, but the renderer
        // couldn't display the result — so fail with a clear message instead.
        const feat = EncoderClass.requiredFeature
        if (feat && !adapter.features.has(feat)) {
          throw new Error(
            `This GPU doesn't expose "${feat}", so the encoded texture can't be sampled here. ` +
              'BC formats are typically desktop (Windows/Linux/Intel-Mac); ASTC is typically Apple Silicon / mobile.',
          )
        }

        const resp = await fetch(url)
        if (!resp.ok) {
          throw new Error(`Couldn't load "${url}" (HTTP ${resp.status}). Expected the file at example/public${url}.`)
        }
        const blob = await resp.blob()
        const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })

        encoder = await EncoderClass.create()
        const { texture: built, ...bytes } = await encodeToTexture(encoder, bitmap, { flipY, quality, colorSpace })
        bitmap.close()
        texture = built

        if (cancelled) return
        const effSrgb = colorSpace === 'srgb' && encoder.supportsSrgb
        setResult({
          texture: built,
          info: {
            label: encoder.label,
            format: encoder.gpuTextureFormat({ colorSpace: effSrgb ? 'srgb' : 'linear' }),
            quality,
            width: bytes.width,
            height: bytes.height,
            paddedWidth: bytes.paddedWidth,
            paddedHeight: bytes.paddedHeight,
            compressedBytes: bytes.data.byteLength,
            rgba8Bytes: bytes.width * bytes.height * 4,
            encodeMs: bytes.encodeMs,
          },
          error: null,
          loading: false,
        })
      } catch (e) {
        if (!cancelled) {
          setResult({ texture: null, info: null, error: e instanceof Error ? e.message : String(e), loading: false })
        }
      }
    }
    run()

    return () => {
      cancelled = true
      texture?.dispose()
      encoder?.destroy()
    }
  }, [url, EncoderClass, colorSpace, quality, flipY])

  return result
}
