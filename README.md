# GPUtex | On-the-fly GPU texture encoding

Runtime GPU texture compression via WebGPU compute shaders, with a WebGL2 fragment-shader fallback. Feed it a PNG/JPG/WebP/AVIF and get back a GPU-compressed texture (BC7, BC5, ASTC 4x4, or BC1) ready for Three.js or React Three Fiber.

⚠️ 100% vibe-coded. The code is completely unreviewed and under-tested. Do not use for anything important.

🚀 Used in production on [Mana Blade](https://manablade.com).

## Install

```sh
npm install gputex
# or
pnpm add gputex
# or
bun add gputex
```

### Entry points

`gputex` ships three entry points:

- **`gputex`** — the engine-agnostic core: the `*Encoder` classes, capability / format detection, and mip helpers. Nothing here imports `three`, so it works with Babylon.js, raw WebGPU/WebGL, workers, etc. Encoders return raw compressed block bytes via `encodeToBytes()`.
- **`gputex/three`** — the Three.js layer. Re-exports the entire core **plus** `compressTexture()`, `GputexLoader`, and `buildCompressedTexture()` / `encodeToTexture()`. This is the only entry that imports `three`.
- **`gputex/testing`** — the CPU reference encoders/decoders the GPU shaders are validated against. Test-suite material, not runtime API (see [Testing](#testing)).

`three` is an **optional** peer dependency (`>=0.170`): install it only if you import `gputex/three`. Pure-core consumers (e.g. Babylon.js) can skip it entirely.

## Formats

| Format       | Bytes / 4x4 block | Use case                                                  |
| ------------ | ----------------- | --------------------------------------------------------- |
| **BC7**      | 16 (8 bpp)        | Color / RGBA on desktop (`texture-compression-bc`)        |
| **BC5**      | 16 (8 bpp)        | Normal maps — RG only (`texture-compression-bc`)          |
| **ASTC 4x4** | 16 (8 bpp)        | Color / RGBA on mobile / iOS (`texture-compression-astc`) |
| **BC1**      | 8 (4 bpp)         | Opaque color at half BC7's size (opt-in)                  |

Format selection is automatic: BC7/BC5 on desktop, ASTC on mobile, uncompressed RGBA8 fallback otherwise.

BC1 is never picked by default — it's half the memory of BC7 but visibly lower
quality, a trade-off only the application can make. Opt in per-texture with
`preferredFormat: 'bc1'`: on BC-capable devices the texture encodes as BC1;
everywhere else (e.g. ASTC-only mobile) selection proceeds as normal. The
preference is only honoured with `hint: 'color'`, since BC1 can't carry real
alpha or a normal map.

## WebGL fallback

WebGPU is the primary path. When it's unavailable (older Safari, Firefox without WebGPU, locked-down environments) `compressTexture()` automatically falls back to a **WebGL2** path that runs the same family of block encoders as fragment shaders — each 4×4 block is computed in one fragment, written to an `RGBA32UI` render target, and read back. The WebGPU fast paths have since been rewritten for speed (projection assignment, f16), so the two backends are no longer byte-identical, but they implement the same algorithms at the same quality level and the resulting `CompressedTexture` looks the same under either renderer.

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
  least-squares refit (accepted per block only when it lowers the error), and
  the block bits packed with straight-line constant shifts. On GPUs that
  report the `shader-f16` feature the whole fast path (all four formats, BC1
  included) runs in f16 — the f32 path is the automatic fallback. Net vs
  `'high'` on an Apple GPU: roughly **10–30× faster** depending on format,
  for a PSNR cost of **≤0.65 dB on smooth/flat content** (imperceptible) and
  up to a few dB on adversarial high-frequency noise, where the bbox seed
  trails `'high'`'s exhaustive search. See the benchmark table below.
- **`'high'`** — exhaustive endpoint search (farthest-pair seed, full nearest
  search, p-bit search); matches the CPU reference encoders block-for-block
  (byte-identical on >96% of blocks; the rest are equal-error FP tie-breaks,
  enforced by the GPU test suite).

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

| Option            | Type                 | Default   | Description                                                                                      |
| ----------------- | -------------------- | --------- | ------------------------------------------------------------------------------------------------ |
| `hint`            | `TextureHint`        | `'color'` | `'color'`, `'colorWithAlpha'`, or `'normal'`                                                     |
| `preferredFormat` | `'bc1'`              | —         | Prefer BC1 (half of BC7's size) when supported; normal selection otherwise. `hint: 'color'` only |
| `colorSpace`      | `'srgb' \| 'linear'` | `'srgb'`  | Use the sRGB or linear variant of the chosen format                                              |
| `flipY`           | `boolean`            | `true`    | Flip vertically (matches Three.js convention)                                                    |
| `mipmaps`         | `boolean`            | `false`   | Generate full mip chain down to 1x1                                                              |
| `device`          | `GPUDevice`          | —         | Reuse an existing WebGPU device instead of creating one                                          |

## Benchmarks

Measured with the repo's GPU test suite (see below) on an Apple Silicon GPU
(`metal-3`) in Chrome, encoding a 2048×2048 image. **GPU pass** is the compute
shader alone (WebGPU timestamp queries, median of 20 runs); end-to-end wall
time adds ~3–4 ms of image upload + result readback regardless of format.
"Before" is the shader generation prior to the 2026-07 optimization pass
(projection-based index assignment everywhere, on-the-fly bit packing, an f16
BC1 fast shader, straight-line block assembly).

| Format   | Quality        | Shader | GPU pass before | GPU pass after | Speedup   |
| -------- | -------------- | ------ | --------------- | -------------- | --------- |
| BC1      | fast (default) | f16    | — (had no f16)  | **0.26 ms**    | **2.8×**¹ |
| BC1      | fast           | f32    | 0.72 ms         | 0.33 ms        | 2.2×      |
| BC5      | fast (default) | f16    | 0.33 ms         | **0.20 ms**    | 1.7×      |
| BC5      | fast           | f32    | 0.79 ms         | 0.39 ms        | 2.0×      |
| BC7      | fast (default) | f16    | 1.38 ms         | **0.52 ms**    | 2.7×      |
| BC7      | fast           | f32    | 2.65 ms         | 1.70 ms        | 1.6×      |
| ASTC 4×4 | fast (default) | f16    | 0.33 ms         | **0.20 ms**    | 1.7×      |
| ASTC 4×4 | fast           | f32    | 0.72 ms         | 0.66 ms        | 1.1×      |
| BC1      | high           | f32    | 2.9 ms          | 3.0 ms         | unchanged |
| BC5      | high           | f32    | 3.4 ms          | 3.4 ms         | unchanged |
| BC7      | high           | f32    | 14.1 ms         | 14.1 ms        | unchanged |
| ASTC 4×4 | high           | f32    | 2.6 ms          | 2.4 ms         | unchanged |

¹ vs the old f32 fast shader, which was the only BC1 fast path before. The
BC1 rows were measured with old and new pipelines interleaved in one session
(the most noise-robust method); the others are cross-run suite medians.

Per-quadrant PSNR on the committed 512² test card is equal to or better than
the previous fast encoders everywhere (flat tiles bit-identical, gradients
+0.01 dB, noise −0.01 dB); the `high` paths still match the CPU reference
encoders. Timestamps are quantised to 100 µs by Chrome, so sub-millisecond
figures are ±0.05–0.1 ms.

## Testing

Unit tests (`bun test`) cover the CPU reference encoders and metadata, but the
WGSL shaders can only be validated on a real GPU. The repo ships a browser
test + benchmark suite at `example/pages/test.tsx` (logic in
`example/lib/gpuTestSuite.ts`):

```sh
bun run --filter gputex build   # build the library the example consumes
cd example && bunx next dev     # then open http://localhost:3000/test
```

The page runs three groups against the live WebGPU device and renders
PASS/FAIL tables (machine-readable copy on `window.__GPUTEX_TESTS__`):

- **Correctness** — `quality: 'high'` output is compared block-by-block
  against the CPU reference encoders (`gputex/testing`), including a
  non-multiple-of-4 image for the clamp-to-edge padding path. Differing blocks
  must have equal decoded error (FP tie-break tolerance) and the aggregate
  PSNR delta must be ≤0.05 dB. Plus determinism checks (same input twice →
  identical bytes).
- **Quality** — `'fast'` and `'high'` output is CPU-decoded and validated on
  the FULL 512² test cards (every quadrant stresses a different failure mode)
  with two gates, for both the f16 and (force-disabled-f16) f32 shaders:
  aggregate PSNR must beat per-format thresholds pinned ~0.15 dB under the
  measured baseline, and — because a handful of catastrophically wrong blocks
  barely moves aggregate PSNR — the worst _easy_ block (one that `'high'`
  encodes near-losslessly) must not exceed `'high'`'s error by more than a
  small per-format limit.
- **Performance** — the benchmark table above: wall + GPU-pass time per
  format × quality × shader variant.

The `gputex/testing` entry point exports the CPU reference
encoders/decoders (`encodeBC7Mode6Block`, `decodeASTC4x4Block`, …) so any
consumer can run the same validation.

## Requirements

- WebGPU (primary) **or** WebGL2 (fallback) — almost every current browser has at least one
- A compressed-texture capability for compressed output:
  - WebGPU: `texture-compression-bc` (desktop) or `texture-compression-astc` (mobile)
  - WebGL2: `EXT_texture_compression_bptc` / `_rgtc`, `WEBGL_compressed_texture_astc`, or `WEBGL_compressed_texture_s3tc`
- Falls back to uncompressed RGBA8 when no compressed format is available on either backend

## Device-specific workarounds

- Black texture on Google Pixel 10: `copyExternalImageToTexture` produces black textures on the Pixel 10's PowerVR DXT GPU (vendor `img-tec`, architecture `d-series`). Worked around by uploading via `writeTexture` with rasterised pixel data instead.

## Acknowledgements

The concept of encoding images on the GPU on the fly via compute shaders was first introduced by [spark.js](https://ludicon.com/sparkjs/). GPUtex is not derived from Spark. Its encoders have been implemented from scratch using official references, which have been ported to TypeScript, and then converted to WGSL and GLSL via AI. Spark was never mentioned or used as reference at any point of the implementation, and multiple reviews have found the implementations to be completely independent. For any serious production use of GPU-compressed textures, Spark is the recommended choice over GPUtex.
