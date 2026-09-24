# gputex | GPU texture encoding

Runtime GPU texture compression via WebGPU compute shaders, with a WebGL2 fragment-shader fallback. Feed it a PNG/JPG/WebP/AVIF — or an SVG, rasterised on the fly — and get back a GPU-compressed texture (BC7, BC5, ASTC 4x4, BC1, or ETC2) ready for Three.js or React Three Fiber.

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

WebGPU is the primary path. When it's unavailable (older Safari, Firefox without WebGPU, locked-down environments) `compressTexture()` automatically falls back to a **WebGL2** path that runs the same family of block encoders as fragment shaders — each 4×4 block is computed in one fragment, written to an `RGBA32UI` render target, and read back. Each fragment shader is a line-for-line port of the WebGPU f32 shader and produces the same bytes as it (verified on Apple M3); the default WebGPU path runs the f16 variants where `shader-f16` is available, which can differ from f32 on rounding ties only, so the resulting `CompressedTexture` looks the same under either renderer. The WebGL encoders also reuse their textures across encodes and skip re-uploading an `ImageBitmap` they already hold.

The fallback chain is **WebGPU → WebGL2 → uncompressed RGBA8**. The `backend` field on the result (`'webgpu' | 'webgl' | 'none'`) tells you which path ran.

Notes on the WebGL path:

- It needs the matching WebGL2 compressed-texture extension to be sampleable: `EXT_texture_compression_bptc` (BC7), `EXT_texture_compression_rgtc` (BC5), `WEBGL_compressed_texture_astc` (ASTC), `WEBGL_compressed_texture_s3tc` (BC1), or `WEBGL_compressed_texture_etc` (ETC2). Selection mirrors the WebGPU side, with BC1 added as a broadly-available last resort for **opaque** colour when neither BPTC nor ASTC is present; ETC2 RGB8 (`WEBGL_compressed_texture_etc`) is the `quality: 'low'` pick on devices without s3tc and the final opaque-colour fallback.
- The `device` / `adapter` options apply to the WebGPU path only.
- All encoding happens on one shared, off-screen WebGL2 context; nothing is drawn to a visible canvas. `compressTexture()` keeps one compiled encoder per format on it across calls (released by `releaseSharedGpuResources()`).
- `forceWebGL: true` (or `loader.forceWebGL = true`) takes this path on WebGPU-capable browsers too — handy for testing it. In the example app, add `?forcewebgl=1` to any page to encode and render through WebGL2.

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
straight-line constant shifts. BC1's coarse 4-level palette adds up to two
least-squares endpoint refit rounds, solved from per-pass projection
moments and accepted per block only when they lower the error; its
near-flat blocks skip the line fit and take the endpoint pair whose ⅔/⅓
interpolant lands nearest the block colour (direct 565 rounding is up to 4
levels off — worth up to +3.9 dB on maps with flat regions). BC5 refits
once; BC7's 16-level mode-6 palette makes the refit redundant on a
principal-axis seed (≤0.05 dB). ASTC spends every one of its 128 bits: a
wide-span opaque block gets 16 weight levels with 192-level (trit-coded)
endpoints, a small-span one exact 8-bit endpoints with 8 levels,
exactly-grayscale blocks a luminance-only mode with 32 levels. On GPUs that
report the `shader-f16` feature everything runs in f16 — the f32 shaders
are the automatic fallback.

ETC2 is the exception to the endpoint-line story: its blocks are per-subblock
base colours shifted by scalar modifier tables. The encoder exploits the
algebra of that scalar shift — table and index selection depend only on each
texel's luma-sum difference from the base, exactly (modulo decode clamping) —
so the whole 8-table × 4-modifier search collapses to a handful of scalar
threshold tests against a two-candidate table shortlist, with subblock error
constants and the flip preselect computed O(1) from quadrant sums (exactly
gray blocks, which give the preselect nothing to go on, score both flips on
a one-channel path). A closed-form least-squares fit of ETC2's planar mode
(which rescues the smooth gradients ETC1-style blocks band on) completes
the block, driven by the same estimates. There is no base-colour refit
(~0.2 dB on photographic content for ≥13% GPU). The kernel reads the source
through `textureGather` and keeps every per-texel quantity in registers
with constant indexing (numbers below). Its f16 module is EXACT-VALUE: lumas,
D values and thresholds are integers (or halves) f16 represents exactly,
while the sums and estimates stay f32 (they overflow f16), so the two
modules produce byte-identical output wherever the sampler's unorm
conversion is exact (verified on Apple) — f16 buys register space, not
different results.

On the repo's test textures this lands within a few tenths of a dB of the
per-block CPU reference encoders (`gputex/testing`) and above them on
several (BC5 matches exactly; BC1 on flat content, BC7 and ASTC on some
maps measure above), trailing only on adversarial high-frequency noise,
where any single-line seed loses to an exhaustive search — while encoding
an order of magnitude faster. See the benchmark table below.

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
| `forceWebGL`      | `boolean`                     | `false`   | Skip WebGPU and encode on the WebGL2 fallback (testing); pair with `new WebGPURenderer({ forceWebGL: true })`            |

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

#### Prewarming shaders

A cold shader cache (first visit, browser or driver update) costs ~120–160 ms
of pipeline compilation per WebGPU encoder, ~60–90 ms per WebGL2 program,
paid by the first texture that needs it. `prewarmCompressTexture()` compiles
ahead of time exactly what `compressTexture()` will use on this client for
the option sets you pass — the same capability-based selection (BC on
desktop, ASTC/ETC2 on mobile, the WebGL2 tier when WebGPU is missing or
`forceWebGL` is set), so nothing unused compiles:

```ts
import { prewarmCompressTexture } from 'gputex'

// At app boot — pass your texture option presets as-is.
prewarmCompressTexture([{ hint: 'color', quality: 'low', mipmaps: true }, { hint: 'normal' }])
```

It creates the shared device (or WebGL2 context) and the per-format encoders
later `compressTexture()` calls reuse, plus the mip-generation pipeline when
a target sets `mipmaps`, and resolves with the chosen backend/format per
target once everything compiled. Compilation runs off the main thread
(`createComputePipelineAsync`; `KHR_parallel_shader_compile` on WebGL2), and
it never rejects — compile errors surface on the first encode. Calls that
pass their own `device`/`adapter` build their own encoders and aren't
warmed. Independently of prewarming, every `compressTexture()` call starts
its encoder's compile before decoding the image, so compile and decode
overlap.

## Benchmarks

Measured on an Apple Silicon GPU (`metal-3`, M3) in Chrome with the `/eval`
dev page: per-dispatch compute time from timestamp queries over batches of
back-to-back dispatches, all variants interleaved in one session, encoding
the procedural 2048×2048 benchmark image. **GPU pass** is the compute
shader alone.

| Format   | Shader        | GPU pass    |
| -------- | ------------- | ----------- |
| BC1      | f16 (default) | **0.33 ms** |
| BC1      | f32           | 0.50 ms     |
| BC5      | f16 (default) | **0.14 ms** |
| BC5      | f32           | 0.15 ms     |
| BC7      | f16 (default) | **0.19 ms** |
| BC7      | f32           | 0.52 ms     |
| ASTC 4×4 | f16 (default) | **0.17 ms** |
| ASTC 4×4 | f32           | 0.28 ms     |
| ETC2     | f16 (default) | **0.14 ms** |
| ETC2     | f32           | 0.15 ms     |

End-to-end `encodeToBytes()` wall time adds the upload and the readback.
Each encoder caches its GPU resources (source texture, output/staging
buffers, bind groups) across encodes, and outputs above ~3 MB are encoded
in row bands — one submission and staging buffer per ~2 MB of output, so
the readback of one band (a GPU-process copy that dominated large encodes)
overlaps the compute of the next — with the result array pre-faulted while
the GPU works. In an interleaved A/B against the single-submission
readback this cuts wall time by 23–33% at 4096² and 5–25% at 2048²
(bytes identical); a fresh 4096² encode is then dominated by the ~8 ms
`copyExternalImageToTexture` upload.

On a 100 GB/s part just reading the 2048² RGBA8 source costs ~0.14 ms, so
BC5/BC7/ASTC sit within ~1.3× of simply touching the bytes and ETC2 at it;
BC1's refit rounds keep it ALU-bound. On real textures (which Apple's
lossless framebuffer compression makes cheaper to read) and at 1024², where
the source stays cached, the ETC2 kernel is ALU-exposed again: 0.035–0.04 ms
at 1024², 0.13–0.15 ms at 2048² and 0.49–0.57 ms at 4096² across the corpus.
A two-pass 2 B/px prepared-source ETC2 variant lives in git history and was
not shipped: its prep pass is also bandwidth-bound and cannot overlap, so
the per-texture total regressed.

Single-dispatch timestamps are coarse and Apple GPU clock states swing
timings by up to ~2× across page loads, so compare variants only within a
single session, interleaved (as `/eval` and `/ab` do).

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

For shader work, `/eval` compares WGSL variants on speed AND quality in one
session: `example/scripts/ab-sync.sh` snapshots the working-tree shaders
(`<fmt>_work`) and git HEAD's (`<fmt>_head`) into `example/public/ab/`, and
`/eval?shaders=bc7_f16_head,bc7_f16_work` times them interleaved and scores
each image through the GPU's own hardware decoder (the encoded blocks are
copied into a real compressed texture and sampled), reporting PSNR, blocks
that got worse/better, the worst regression on blocks the first variant
encodes near-losslessly, and byte-identical coverage — over the full
1K/2K/4K texture corpus in seconds (`window.__GPUTEX_EVAL__`). For host-side
changes, `example/scripts/ab-lib.sh` builds git HEAD's library and the
working tree side by side (`/ab/gputex_head.js`, `/ab/gputex_work.js`) so
both can be driven in one page with alternating calls.

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
  - WebGL2: `EXT_texture_compression_bptc` / `_rgtc`, `WEBGL_compressed_texture_astc`, `WEBGL_compressed_texture_s3tc`, or `WEBGL_compressed_texture_etc`
- Falls back to uncompressed RGBA8 when no compressed format is available on either backend

## Device-specific workarounds

- Black texture on Google Pixel 10: `copyExternalImageToTexture` produces black textures on the Pixel 10's PowerVR DXT GPU (vendor `img-tec`, architecture `d-series`). Worked around by uploading via `writeTexture` with rasterised pixel data instead.

## Acknowledgements

The concept of encoding images on the GPU on the fly via compute shaders was first introduced by [spark.js](https://ludicon.com/sparkjs/). gputex is not derived from Spark. Its encoders have been implemented from scratch using official references, which have been ported to TypeScript, and then converted to WGSL and GLSL via AI. Spark was never mentioned or used as reference at any point of the implementation, and multiple reviews have found the implementations to be completely independent. For any serious production use of GPU-compressed textures, Spark is the recommended choice over gputex.
