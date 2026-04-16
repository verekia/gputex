# GPUtex

Runtime GPU texture compression via WebGPU compute shaders. Feed it a PNG/JPG/WebP/AVIF and get back a GPU-compressed texture (BC7, BC5, ASTC 4x4, or BC1) ready for Three.js or React Three Fiber.

⚠️: This is 100% vibe-coded. The code is completely unreviewed, under-tested, it will probably crash on many devices, and this library is very likely to go unmaintained. Do not use for anything important.

## Install

```sh
npm install gputex
# or
pnpm add gputex
# or
bun add gputex
```

`three` is a peer dependency (`>=0.180`).

## Formats

| Format       | Bytes / 4x4 block | Use case                                                  |
| ------------ | ----------------- | --------------------------------------------------------- |
| **BC7**      | 16 (8 bpp)        | Color / RGBA on desktop (`texture-compression-bc`)        |
| **BC5**      | 16 (8 bpp)        | Normal maps — RG only (`texture-compression-bc`)          |
| **ASTC 4x4** | 16 (8 bpp)        | Color / RGBA on mobile / iOS (`texture-compression-astc`) |
| **BC1**      | 8 (4 bpp)         | Legacy (never auto-selected)                              |

Format selection is automatic: BC7/BC5 on desktop, ASTC on mobile, uncompressed RGBA8 fallback otherwise.

## Usage

### `compressTexture` — direct API

```ts
import { compressTexture } from 'gputex'

const { texture, format } = await compressTexture('/cobblestone.avif', {
  hint: 'color', // 'color' | 'colorWithAlpha' | 'normal'
  colorSpace: 'srgb',
  mipmaps: true,
})

material.map = texture
```

### `GputexLoader` — Three.js Loader

```ts
import { GputexLoader } from 'gputex'

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
import { GputexLoader } from 'gputex'

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
import { GputexLoader } from 'gputex'
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

### Low-level encoders

Individual encoder classes are exported for direct control:

```ts
import { BC7Encoder, BC5Encoder, ASTC4x4Encoder, BC1Encoder } from 'gputex'

const encoder = await BC7Encoder.create()
const { data, width, height } = await encoder.encodeToBytes(imageBitmap)
encoder.destroy()
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

- A browser with WebGPU support
- `texture-compression-bc` (desktop) or `texture-compression-astc` (mobile) for compressed output
- Falls back to uncompressed RGBA8 when neither is available

## Known issues

- Black texture on Google Pixel 10: `copyExternalImageToTexture` produces black textures on the Pixel 10's PowerVR DXT GPU (vendor `img-tec`, architecture `d-series`). Worked around by uploading via `writeTexture` with rasterised pixel data instead.

## Acknowledgements

The concept of encoding images on the GPU on the fly via compute shaders was first introduced by [spark.js](https://ludicon.com/sparkjs/), which is a much more robust solution for users who can afford its license. GPUtex is not derived from Spark and its encoders have been implemented from scratch using official references, which have been ported to TypeScript, and then converted to WGSL via AI. For any serious production use of GPU-compressed textures, Spark is the recommended choice over GPUtex.
