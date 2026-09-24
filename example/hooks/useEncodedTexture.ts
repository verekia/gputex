import { useEffect, useState } from 'react'

import {
  ASTC4x4Encoder,
  ASTC4x4WebGLEncoder,
  BC1Encoder,
  BC1WebGLEncoder,
  BC5Encoder,
  BC5WebGLEncoder,
  BC7Encoder,
  BC7WebGLEncoder,
  detectWebGLCapabilities,
  ETC2Encoder,
  ETC2WebGLEncoder,
  getSharedWebGLContext,
} from 'gputex'
import type { EncoderConstructor, TextureFormat, WebGLCapabilities, WebGLEncoderConstructor } from 'gputex'

import { buildCompressedTexture, encodeToTexture } from 'gputex/three'

import { forceWebGL } from '../lib/forceWebGL'

import type { Texture } from 'three'

// Encode a single image with ONE specific encoder class, bypassing
// `compressTexture()`'s capability-based auto-selection. The per-format test
// pages (/bc1, /bc5, /bc7, /astc) use this to force exactly the format they
// want — `compressTexture` would otherwise pick BC7 (desktop) or ASTC (mobile)
// for a colour hint and never BC1 at all.
//
// WebGPU compute path by default; with `?forcewebgl=1` the same format is
// encoded by its WebGL2 fragment-shader twin instead (see lib/forceWebGL).

// WebGL2 twin of each WebGPU encoder, plus the extension that makes the
// output sampleable there.
const WEBGL_TWINS = new Map<EncoderConstructor, { cls: WebGLEncoderConstructor; cap: keyof WebGLCapabilities }>([
  [BC1Encoder, { cls: BC1WebGLEncoder, cap: 's3tc' }],
  [BC5Encoder, { cls: BC5WebGLEncoder, cap: 'rgtc' }],
  [BC7Encoder, { cls: BC7WebGLEncoder, cap: 'bptc' }],
  [ASTC4x4Encoder, { cls: ASTC4x4WebGLEncoder, cap: 'astc' }],
  [ETC2Encoder, { cls: ETC2WebGLEncoder, cap: 'etc' }],
])

export interface EncodedInfo {
  /** Encoder label, e.g. 'bc7'. */
  label: string
  /** Which encoder backend produced the bytes. */
  backend: 'webgpu' | 'webgl'
  /** Concrete GPUTextureFormat string on WebGPU (e.g. 'bc7-rgba-unorm-srgb'),
   *  the logical TextureFormat on WebGL (e.g. 'BC7_SRGB'). */
  format: string
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
  const { colorSpace = 'srgb', flipY = true } = options
  const [result, setResult] = useState<EncodedResult>({ texture: null, info: null, error: null, loading: true })

  useEffect(() => {
    let cancelled = false
    let encoder: Awaited<ReturnType<EncoderConstructor['create']>> | null = null
    let texture: Texture | null = null

    const run = async () => {
      setResult({ texture: null, info: null, error: null, loading: true })
      try {
        if (forceWebGL) {
          await runWebGL()
          return
        }
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
        const { texture: built, ...bytes } = await encodeToTexture(encoder, bitmap, { flipY, colorSpace })
        bitmap.close()
        texture = built

        if (cancelled) return
        const effSrgb = colorSpace === 'srgb' && encoder.supportsSrgb
        setResult({
          texture: built,
          info: {
            label: encoder.label,
            backend: 'webgpu',
            format: encoder.gpuTextureFormat({ colorSpace: effSrgb ? 'srgb' : 'linear' }),
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
    const loadBitmap = async (): Promise<ImageBitmap> => {
      const resp = await fetch(url)
      if (!resp.ok) {
        throw new Error(`Couldn't load "${url}" (HTTP ${resp.status}). Expected the file at example/public${url}.`)
      }
      return createImageBitmap(await resp.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
    }

    // ?forcewebgl=1: the WebGL2 twin encodes, the same logical format is built.
    const runWebGL = async () => {
      const twin = WEBGL_TWINS.get(EncoderClass)
      if (!twin) throw new Error('This format has no WebGL2 encoder — remove ?forcewebgl=1 to use WebGPU.')
      const gl = getSharedWebGLContext()
      if (!gl) throw new Error('WebGL2 is not available in this browser.')
      if (!detectWebGLCapabilities(gl)[twin.cap]) {
        throw new Error(`This browser's WebGL2 lacks the "${twin.cap}" compressed-texture extension for this format.`)
      }
      const bitmap = await loadBitmap()
      const glEncoder = twin.cls.create(gl)
      try {
        const bytes = glEncoder.encodeToBytes(bitmap, { flipY })
        const formats = (EncoderClass as unknown as { textureFormats: readonly TextureFormat[] }).textureFormats
        const srgb = colorSpace === 'srgb' && glEncoder.supportsSrgb
        const format = formats.find(f => f.endsWith('_SRGB') === srgb) ?? formats[0]!
        const built = buildCompressedTexture([bytes], format)
        texture = built
        if (cancelled) return
        setResult({
          texture: built,
          info: {
            label: glEncoder.label,
            backend: 'webgl',
            format,
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
      } finally {
        bitmap.close()
        glEncoder.destroy()
      }
    }

    run()

    return () => {
      cancelled = true
      texture?.dispose()
      encoder?.destroy()
    }
  }, [url, EncoderClass, colorSpace, flipY])

  return result
}
