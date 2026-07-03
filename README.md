# GPUtex | On-the-fly GPU texture encoding

Runtime GPU texture compression via WebGPU compute shaders, with a WebGL2 fragment-shader fallback. Feed it a PNG/JPG/WebP/AVIF — or an SVG, rasterised on the fly — and get back a GPU-compressed texture (BC7, BC5, ASTC 4x4, BC1, or ETC2) ready for Three.js or React Three Fiber.

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

| Format        | Bytes / 4x4 block | Use case                                                                        |
| ------------- | ----------------- | ------------------------------------------------------------------------------- |
| **BC7**       | 16 (8 bpp)        | Color / RGBA on desktop (`texture-compression-bc`)                              |
| **BC5**       | 16 (8 bpp)        | Normal maps — RG only (`texture-compression-bc`)                                |
| **ASTC 4x4**  | 16 (8 bpp)        | Color / RGBA on mobile / iOS (`texture-compression-astc`)                       |
| **BC1**       | 8 (4 bpp)         | Opaque color at half BC7's size (`quality: 'low'`)                              |
| **ETC2 RGB8** | 8 (4 bpp)         | Opaque color at half ASTC's size (`texture-compression-etc2`, `quality: 'low'`) |

Format selection is automatic: BC7/BC5 on desktop, ASTC on mobile, ETC2 as the
last-resort compressed format for opaque colour, uncompressed RGBA8 fallback
otherwise.

The 4-bpp formats are never picked by default — half the memory of BC7/ASTC
but visibly lower quality, a trade-off only the application can make. Opt in
with `quality: 'low'`: opaque colour textures then encode as BC1 on BC-capable
devices and as ETC2 RGB8 on ETC2-capable ones (most mobile GPUs), while
`'colorWithAlpha'` and `'normal'` hints keep the high-quality formats (the
4-bpp formats can't carry them). Per-texture, `preferredFormat: 'bc1'` forces
BC1 on BC hardware the same way; both knobs fall back to the normal selection
when unsupported, and both apply to `hint: 'color'` only.

## WebGL fallback

WebGPU is the primary path. When it's unavailable (older Safari, Firefox without WebGPU, locked-down environments) `compressTexture()` automatically falls back to a **WebGL2** path that runs the same family of block encoders as fragment shaders — each 4×4 block is computed in one fragment, written to an `RGBA32UI` render target, and read back. The two backends are not byte-identical (the WebGPU shaders use f16 where available), but they implement the same algorithms at the same quality level and the resulting `CompressedTexture` looks the same under either renderer.

The fallback chain is **WebGPU → WebGL2 → uncompressed RGBA8**. The `backend` field on the result (`'webgpu' | 'webgl' | 'none'`) tells you which path ran.

Notes on the WebGL path:

- It needs the matching WebGL2 compressed-texture extension to be sampleable: `EXT_texture_compression_bptc` (BC7), `EXT_texture_compression_rgtc` (BC5), `WEBGL_compressed_texture_astc` (ASTC), or `WEBGL_compressed_texture_s3tc` (BC1). Selection mirrors the WebGPU side, with BC1 added as a broadly-available last resort for **opaque** colour when neither BPTC nor ASTC is present. ETC2 is WebGPU-only (no WebGL fragment encoder), so `quality: 'low'` on the WebGL tier can only deliver BC1.
- The `device` / `adapter` options apply to the WebGPU path only.
- All encoding happens on one shared, off-screen WebGL2 context; nothing is drawn to a visible canvas.

## Usage

### `compressTexture` — direct API

```ts
import { compressTexture } from 'gputex/three'

const { texture, format } = await compressTexture('/cobblestone.avif', {
  hint: 'color', // 'color' | 'colorWithAlpha' | 'normal'
  colorSpace: 'srgb',
  mipmaps: true,
})

material.map = texture
```

#### The encoding algorithm

There is a single encode mode, built to be both fast and high quality: a
principal-axis endpoint seed (per-block covariance power-iteration — unlike
a bbox diagonal it follows anti-correlated channels, worth **+2–4 dB on
normal-map-like content**) plus projection-based index assignment (each
pixel is projected onto the colinear endpoint line in O(1) instead of
searching every palette entry), and the block bits packed with
straight-line constant shifts. The formats with coarse 4-level palettes
(BC1, ASTC) add up to two least-squares endpoint refit rounds accepted per
block only when they lower the error, and BC5 one; BC7's 16-level mode-6
palette makes the refit redundant on a principal-axis seed (≤0.05 dB), so
it skips it and stays the cheapest per pixel. On GPUs that report the
`shader-f16` feature everything runs in f16 — the f32 shaders are the
automatic fallback.

ETC2 is the exception to the endpoint-line story: its blocks are per-subblock
base colours shifted by scalar modifier tables. The encoder exploits the
algebra of that scalar shift — table and index selection depend only on each
texel's luma-sum difference from the base, exactly (modulo decode clamping) —
so the whole 8-table × 4-modifier search collapses to a handful of scalar
threshold tests against a two-candidate table shortlist, with subblock error
constants and the flip preselect computed O(1) from quadrant sums. A gated
base-colour refit and a closed-form least-squares fit of ETC2's planar mode
(which rescues the smooth gradients ETC1-style blocks band on) complete the
block, all driven by the same estimates. The rewrite took the GPU pass from
6.0 ms to 0.28 ms at 2048² (21×, within ~0.2 dB of the exhaustive search).
It ships as f32 only: the estimates are integer-exact sums that overflow f16.

On the repo's test cards this lands within **≤0.1 dB** of the exhaustive
per-block reference encoders (BC5 matches the reference exactly; ASTC and
BC1-on-normal-maps measure slightly above it), trailing only on adversarial
high-frequency noise, where any single-line seed loses to an exhaustive
search — while encoding an order of magnitude faster. See the benchmark
table below.

#### SVG sources

SVGs work anywhere a raster image does — as a URL, a Blob/File, an inline
markup string (detected by a leading `<`), or an `<img>` element. The vector
is rasterised before encoding, at the SVG's intrinsic size by default
(absolute `width`/`height` attributes, else the `viewBox` dimensions). Use
`svgSize` to pick the raster size — the browser renders the vector directly
at that size, so upscaling stays crisp:

```ts
// Longest side 1024, aspect ratio preserved:
const { texture } = await compressTexture('/logo.svg', { svgSize: 1024 })

// Exact size (aspect mismatches follow the SVG's preserveAspectRatio rules):
await compressTexture('/icon.svg', { svgSize: { width: 512, height: 512 } })

// Inline markup:
await compressTexture('<svg viewBox="0 0 32 32">…</svg>', { svgSize: 256 })
```

An SVG with no `width`/`height` **and** no `viewBox` has no intrinsic size;
`svgSize` is required for those. Rasterisation needs a DOM `Image`, so SVG
sources are main-thread only. Non-Three.js users get the same rasteriser as
a standalone helper: `rasterizeSvg(source, { size })` from the core `gputex`
entry returns an `ImageBitmap` ready for `encodeToBytes()`.

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

For mip chains, `encodeMipChainToBytes()` encodes every level in a **single
GPU submission** — one compute pass and one readback instead of a full
CPU↔GPU round trip per level (an 11-level 1024² chain is one `mapAsync`
wait instead of eleven):

```ts
import { BC7Encoder, generateMipChain } from 'gputex'

const encoder = await BC7Encoder.create()
// level0 = { data: Uint8ClampedArray (RGBA8), width, height }
const { levels, encodeMs } = await encoder.encodeMipChainToBytes(generateMipChain(level0))
// levels[i] = { data, width, height, paddedWidth, paddedHeight }
```

When the source is an image (not raw pixels), skip the CPU entirely:
`generateGpuMipChain()` uploads it once and box-filters the whole chain on
the GPU in one compute pass, and `encodeMipChainFromTexture()` encodes
straight from the texture's mip views — no `getImageData` readback, no JS
filter, no per-level uploads. This is what `compressTexture()` uses for
`mipmaps: true` (mipped BC7: 28 → 7.5 ms at 2048², 110 → 23 ms at 4096²),
and its box filter is integer-exact against the CPU one, so both paths emit
identical bytes:

```ts
import { BC7Encoder, generateGpuMipChain } from 'gputex'

const encoder = await BC7Encoder.create()
const chainTex = await generateGpuMipChain(encoder.device, imageBitmap, { flipY: true })
const { levels, encodeMs } = await encoder.encodeMipChainFromTexture(chainTex)
chainTex.destroy()
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

| Option            | Type                          | Default   | Description                                                                                                              |
| ----------------- | ----------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------ |
| `hint`            | `TextureHint`                 | `'color'` | `'color'`, `'colorWithAlpha'`, or `'normal'`                                                                             |
| `quality`         | `'high' \| 'low'`             | `'high'`  | `'low'` picks the 4-bpp formats (BC1 on desktop, ETC2 RGB8 on mobile) for opaque colour — half the memory, lower quality |
| `preferredFormat` | `'bc1'`                       | —         | Prefer BC1 (half of BC7's size) when supported; normal selection otherwise. `hint: 'color'` only                         |
| `colorSpace`      | `'srgb' \| 'linear'`          | `'srgb'`  | Use the sRGB or linear variant of the chosen format                                                                      |
| `svgSize`         | `number \| { width, height }` | intrinsic | Raster size for SVG sources: longest side (aspect preserved) or exact size                                               |
| `flipY`           | `boolean`                     | `true`    | Flip vertically (matches Three.js convention)                                                                            |
| `mipmaps`         | `boolean`                     | `false`   | Generate full mip chain down to 1x1                                                                                      |
| `cache`           | `boolean`                     | `false`   | Session-scoped in-memory cache; repeat calls skip decode + encode (see below)                                            |
| `cacheKey`        | `string`                      | derived   | Explicit cache identity (skips content hashing; makes pixel sources cacheable)                                           |
| `device`          | `GPUDevice`                   | —         | Reuse an existing WebGPU device instead of creating one                                                                  |

#### In-memory transcode cache

With `cache: true`, the compressed bytes are kept in a session-scoped
in-memory LRU keyed by source identity (URL, or a content hash for
Blobs/Files/data URLs) plus the selected format and encode options. Loading
the same texture again later in the session — say, two worlds sharing an
atlas — skips **both** the image decode and the encode, the two dominant
costs: a 4K PNG that takes ~220 ms to decode + encode comes back in ~30 ms
(content-hashed) or ~2 ms (URL-keyed). Nothing touches persistent storage;
the cache dies with the page. Total compressed payload is capped at 256 MiB
with LRU eviction — `setTranscodeCacheLimit(bytes)` tunes it (0 disables),
`clearTranscodeCache()` empties it (e.g. on world unload). Pixel sources
(ImageBitmap, canvas, ImageData) are only cached when you pass a `cacheKey`.

When neither `device` nor `adapter` is passed, `compressTexture()` shares one
WebGPU device and one encoder per format across calls: the first call pays the
adapter/device request and pipeline compile, subsequent calls skip straight to
the encode and reuse the encoder's cached GPU resources. The result's
`destroy()` only disposes that call's texture; call `releaseSharedGpuResources()`
(also exported from `gputex/three`) to tear down the shared device — the next
`compressTexture()` call transparently recreates it. With `mipmaps: true` the
whole chain is encoded in a single GPU submission (one compute pass, one
readback) rather than a round trip per level.

## Benchmarks

Measured with the repo's GPU test suite (see below) on an Apple Silicon GPU
(`metal-3`) in Chrome, encoding a 2048×2048 image. **GPU pass** is the compute
shader alone (WebGPU timestamp queries, median of 20 runs); end-to-end wall
time adds ~2–4 ms of image upload + result readback regardless of format.
Each encoder caches its GPU resources (source texture, output/staging
buffers, bind group) across encodes, so repeated encodes — including mip
chains — skip per-call allocation: in an interleaved A/B this cuts BC7
end-to-end wall time by ~10% at 512², ~20% at 1024–2048² and ~35% at 4096².

| Format   | Shader        | GPU pass    |
| -------- | ------------- | ----------- |
| BC1      | f16 (default) | **0.26 ms** |
| BC1      | f32           | 0.46 ms     |
| BC5      | f16 (default) | **0.26 ms** |
| BC5      | f32           | 0.26 ms     |
| BC7      | f16 (default) | **0.26 ms** |
| BC7      | f32           | 0.59 ms     |
| ASTC 4×4 | f16 (default) | **0.26 ms** |
| ASTC 4×4 | f32           | 0.56 ms     |
| ETC2     | f32 (only)    | 0.28 ms     |

The ETC2 figure is the interleaved `/ab` harness measurement (batched
dispatches, clock-stable): the scalar-luma selection rewrite took it from
6.0 ms to 0.28 ms in-session — for reference, a 16-loads-only null shader
measures 0.14 ms, so the encode logic itself costs about one load-pass.

Timestamps are quantised to 100 µs by Chrome and Apple GPU clock states swing
timings by ~2×, so sub-millisecond figures are indicative (±0.1 ms); compare
variants only within a single session.

## Testing

Unit tests (`bun test`) cover the CPU reference encoders and metadata, but the
WGSL shaders can only be validated on a real GPU. The repo ships a browser
test + benchmark suite at `example/pages/test.tsx` (logic in
`example/lib/gpuTestSuite.ts`):

```sh
bun run --filter gputex build   # build the library the example consumes
cd example && bunx next dev     # then open http://localhost:3000/test
```

A second page, `/bench`, measures median end-to-end `encodeToBytes()` wall
time per format across image sizes (256²–4096²) — the numbers that matter
for runtime streaming, where host overhead dominates small textures
(results on `window.__GPUTEX_BENCH__`).

The page runs three groups against the live WebGPU device and renders
PASS/FAIL tables (machine-readable copy on `window.__GPUTEX_TESTS__`):

- **Correctness** — determinism (same input twice → identical bytes) and the
  clamp-to-edge padding path: a non-multiple-of-4 image must land within a
  couple of dB of the exhaustive CPU reference encode (`gputex/testing`) — a
  padding bug craters it.
- **Quality** — GPU output is CPU-decoded and validated on the FULL 1024²
  test cards (every tile stresses a different failure mode) with two gates,
  for both the f16 and (force-disabled-f16) f32 shaders: aggregate PSNR must
  beat per-format thresholds pinned ~0.15 dB under the measured baseline,
  and — because a handful of catastrophically wrong blocks barely moves
  aggregate PSNR — the worst _easy_ block (one the exhaustive CPU reference
  encodes near-losslessly) must not exceed the reference's error by more
  than a small per-format limit.
- **Performance** — the benchmark table above: wall + GPU-pass time per
  format × shader variant.

The `gputex/testing` entry point exports the CPU reference
encoders/decoders (`encodeBC7Mode6Block`, `decodeASTC4x4Block`, …) — the
exhaustive per-block yardstick the GPU shaders are gated against — so any
consumer can run the same validation.

## Requirements

- WebGPU (primary) **or** WebGL2 (fallback) — almost every current browser has at least one
- A compressed-texture capability for compressed output:
  - WebGPU: `texture-compression-bc` (desktop), `texture-compression-astc` (mobile), or `texture-compression-etc2` (mobile)
  - WebGL2: `EXT_texture_compression_bptc` / `_rgtc`, `WEBGL_compressed_texture_astc`, or `WEBGL_compressed_texture_s3tc`
- Falls back to uncompressed RGBA8 when no compressed format is available on either backend

## Device-specific workarounds

- Black texture on Google Pixel 10: `copyExternalImageToTexture` produces black textures on the Pixel 10's PowerVR DXT GPU (vendor `img-tec`, architecture `d-series`). Worked around by uploading via `writeTexture` with rasterised pixel data instead.

## Acknowledgements

The concept of encoding images on the GPU on the fly via compute shaders was first introduced by [spark.js](https://ludicon.com/sparkjs/). GPUtex is not derived from Spark. Its encoders have been implemented from scratch using official references, which have been ported to TypeScript, and then converted to WGSL and GLSL via AI. Spark was never mentioned or used as reference at any point of the implementation, and multiple reviews have found the implementations to be completely independent. For any serious production use of GPU-compressed textures, Spark is the recommended choice over GPUtex.
