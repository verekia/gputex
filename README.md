# GPUtex

Runtime GPU texture compression via WebGPU compute shaders, with a WebGL2 fragment-shader fallback. Feed it a PNG/JPG/WebP/AVIF and get back a GPU-compressed texture (BC7, BC5, ASTC 4x4, or BC1) ready for Three.js or React Three Fiber.

⚠️ 100% vibe-coded. The code is completely unreviewed and under-tested. Do not use for anything important.

## Install

```sh
npm install gputex
# or
pnpm add gputex
# or
bun add gputex
```

### Entry points

`gputex` ships two entry points:

- **`gputex`** — the engine-agnostic core: the `*Encoder` classes, capability / format detection, and mip helpers. Nothing here imports `three`, so it works with Babylon.js, raw WebGPU/WebGL, workers, etc. Encoders return raw compressed block bytes via `encodeToBytes()`.
- **`gputex/three`** — the Three.js layer. Re-exports the entire core **plus** `compressTexture()`, `GputexLoader`, and `buildCompressedTexture()` / `encodeToTexture()`. This is the only entry that imports `three`.

`three` is an **optional** peer dependency (`>=0.170`): install it only if you import `gputex/three`. Pure-core consumers (e.g. Babylon.js) can skip it entirely.

## Formats

| Format       | Bytes / 4x4 block | Use case                                                  |
| ------------ | ----------------- | --------------------------------------------------------- |
| **BC7**      | 16 (8 bpp)        | Color / RGBA on desktop (`texture-compression-bc`)        |
| **BC5**      | 16 (8 bpp)        | Normal maps — RG only (`texture-compression-bc`)          |
| **ASTC 4x4** | 16 (8 bpp)        | Color / RGBA on mobile / iOS (`texture-compression-astc`) |
| **BC1**      | 8 (4 bpp)         | Legacy (never auto-selected)                              |

Format selection is automatic: BC7/BC5 on desktop, ASTC on mobile, uncompressed RGBA8 fallback otherwise.

## WebGL fallback

WebGPU is the primary path. When it's unavailable (older Safari, Firefox without WebGPU, locked-down environments) `compressTexture()` automatically falls back to a **WebGL2** path that runs the same block encoders as fragment shaders — each 4×4 block is computed in one fragment, written to an `RGBA32UI` render target, and read back. The output bytes are identical to the WebGPU encoders, so the resulting `CompressedTexture` looks the same under either renderer.

The fallback chain is **WebGPU → WebGL2 → uncompressed RGBA8**. The `backend` field on the result (`'webgpu' | 'webgl' | 'none'`) tells you which path ran.

Notes on the WebGL path:

- It needs the matching WebGL2 compressed-texture extension to be sampleable: `EXT_texture_compression_bptc` (BC7), `EXT_texture_compression_rgtc` (BC5), `WEBGL_compressed_texture_astc` (ASTC), or `WEBGL_compressed_texture_s3tc` (BC1). Selection mirrors the WebGPU side, with BC1 added as a broadly-available last resort for **opaque** colour when neither BPTC nor ASTC is present.
- It always uses the **fast** encoders — the `quality: 'high'` option and the `device` / `adapter` options apply to the WebGPU path only.
- All encoding happens on one shared, off-screen WebGL2 context; nothing is drawn to a visible canvas.

## Usage

### `compressTexture` — direct API

```ts
import { compressTexture } from 'gputex/three'

const { texture, format } = await compressTexture('/cobblestone.avif', {
  hint: 'color', // 'color' | 'colorWithAlpha' | 'normal'
  colorSpace: 'srgb',
  mipmaps: true,
  quality: 'fast', // 'fast' (default) | 'high'
})

material.map = texture
```

#### Quality

`quality` trades encode speed against compression accuracy:

- **`'fast'` (default)** — a bounding-box endpoint seed plus projection-based
  index assignment (each pixel is projected onto the colinear endpoint line in
  O(1) instead of searching every palette entry) with a single fused
  least-squares refit. On GPUs that report the `shader-f16` feature the whole
  fast path runs in f16 (≈2× on Apple) — the f32 path is the automatic
  fallback. Net vs `'high'` on an Apple GPU: **BC7 ~50×**, **ASTC ~9×**,
  **BC5 ~5×** faster, for a PSNR cost of **≤0.45 dB** (imperceptible). BC1 is
  single-pass and unaffected.
- **`'high'`** — exhaustive endpoint search (farthest-pair seed, full nearest
  search, p-bit search); output is byte-for-byte identical to the CPU reference
  encoders.

### `GputexLoader` — Three.js Loader

```ts
import { GputexLoader } from 'gputex/three'

const loader = new GputexLoader()
loader.hint = 'normal'
loader.mipmaps = true
const normalMap = await loader.loadAsync('/brick_normal.png')
material.normalMap = normalMap
```

### React Three Fiber

The `GputexLoader` works with R3F's `useLoader`:

```tsx
import { useLoader } from '@react-three/fiber'
import { GputexLoader } from 'gputex/three'

function Scene() {
  const texture = useLoader(GputexLoader, '/cobblestone.avif', loader => {
    loader.hint = 'color'
    loader.colorSpace = 'srgb'
    loader.mipmaps = true
  })

  return (
    <mesh>
      <sphereGeometry args={[1, 64, 32]} />
      <meshStandardMaterial map={texture} />
    </mesh>
  )
}
```

For a reusable hook with metadata access:

```tsx
import { useLayoutEffect } from 'react'
import { useLoader } from '@react-three/fiber'
import { GputexLoader } from 'gputex/three'
import type { TextureHint } from 'gputex'

function useGputex(url: string, options?: { hint?: TextureHint; colorSpace?: 'srgb' | 'linear'; mipmaps?: boolean }) {
  const texture = useLoader(GputexLoader, url, loader => {
    if (options?.hint !== undefined) loader.hint = options.hint
    if (options?.colorSpace !== undefined) loader.colorSpace = options.colorSpace
    if (options?.mipmaps !== undefined) loader.mipmaps = options.mipmaps
  })

  return texture
}

// Preload textures outside of components
useGputex.preload = (
  url: string,
  options?: { hint?: TextureHint; colorSpace?: 'srgb' | 'linear'; mipmaps?: boolean },
) => {
  useLoader.preload(GputexLoader, url, loader => {
    if (options?.hint !== undefined) loader.hint = options.hint
    if (options?.colorSpace !== undefined) loader.colorSpace = options.colorSpace
    if (options?.mipmaps !== undefined) loader.mipmaps = options.mipmaps
  })
}
```

Usage:

```tsx
// Preload outside the component tree
useGputex.preload('/cobblestone.avif', { hint: 'color', colorSpace: 'srgb', mipmaps: true })

function Scene() {
  const texture = useGputex('/cobblestone.avif', { hint: 'color', colorSpace: 'srgb', mipmaps: true })

  return (
    <mesh>
      <sphereGeometry args={[1, 64, 32]} />
      <meshStandardMaterial map={texture} />
    </mesh>
  )
}
```

### Low-level encoders (any engine)

The individual encoder classes live in the engine-agnostic core (`gputex`). `encodeToBytes()` returns raw compressed block bytes with no Three.js involvement — feed them into whatever compressed-texture upload your engine exposes (Babylon.js, raw WebGPU/WebGL, …):

```ts
import { BC7Encoder, BC5Encoder, ASTC4x4Encoder, BC1Encoder } from 'gputex'

const encoder = await BC7Encoder.create()
const { data, width, height, paddedWidth, paddedHeight } = await encoder.encodeToBytes(imageBitmap)
// `data` is a Uint8Array of BC7 blocks covering paddedWidth × paddedHeight.
encoder.destroy()
```

To turn an encoder's output into a Three.js `CompressedTexture` directly, use the helpers in `gputex/three`:

```ts
import { BC7Encoder, TextureFormat } from 'gputex'
import { encodeToTexture, buildCompressedTexture } from 'gputex/three'

const encoder = await BC7Encoder.create()

// One-shot: image → CompressedTexture (plus the raw byte metadata)
const { texture } = await encodeToTexture(encoder, imageBitmap, { colorSpace: 'srgb' })

// …or assemble a texture from bytes you already have (e.g. a mip chain):
const bytes = await encoder.encodeToBytes(imageBitmap)
const tex = buildCompressedTexture([bytes], TextureFormat.BC7_SRGB)
```

## Options

### `compressTexture` options

| Option       | Type                 | Default   | Description                                             |
| ------------ | -------------------- | --------- | ------------------------------------------------------- |
| `hint`       | `TextureHint`        | `'color'` | `'color'`, `'colorWithAlpha'`, or `'normal'`            |
| `colorSpace` | `'srgb' \| 'linear'` | `'srgb'`  | Use the sRGB or linear variant of the chosen format     |
| `flipY`      | `boolean`            | `true`    | Flip vertically (matches Three.js convention)           |
| `mipmaps`    | `boolean`            | `false`   | Generate full mip chain down to 1x1                     |
| `device`     | `GPUDevice`          | —         | Reuse an existing WebGPU device instead of creating one |

## Requirements

- WebGPU (primary) **or** WebGL2 (fallback) — almost every current browser has at least one
- A compressed-texture capability for compressed output:
  - WebGPU: `texture-compression-bc` (desktop) or `texture-compression-astc` (mobile)
  - WebGL2: `EXT_texture_compression_bptc` / `_rgtc`, `WEBGL_compressed_texture_astc`, or `WEBGL_compressed_texture_s3tc`
- Falls back to uncompressed RGBA8 when no compressed format is available on either backend

## Device-specific workarounds

- Black texture on Google Pixel 10: `copyExternalImageToTexture` produces black textures on the Pixel 10's PowerVR DXT GPU (vendor `img-tec`, architecture `d-series`). Worked around by uploading via `writeTexture` with rasterised pixel data instead.

## Acknowledgements

The concept of encoding images on the GPU on the fly via compute shaders was first introduced by [spark.js](https://ludicon.com/sparkjs/), which is a much more robust solution for users who can afford its license. GPUtex is not derived from Spark and its encoders have been implemented from scratch using official references, which have been ported to TypeScript, and then converted to WGSL via AI. For any serious production use of GPU-compressed textures, Spark is the recommended choice over GPUtex.
