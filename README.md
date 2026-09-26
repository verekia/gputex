# gputex

Compress textures on the GPU at runtime, in the browser.

Give it a regular image (PNG, JPG, WebP, AVIF, SVG) and get back a GPU-compressed texture that uses 4–8× less video memory. gputex picks the right format for the user's hardware (BC7, BC5, BC1, ASTC or ETC2) and encodes it in a few milliseconds using WebGPU, with a WebGL2 fallback. No build step, no format variants to ship.

Works with Three.js and React Three Fiber out of the box.

🚀 Used in production on [Mana Blade](https://manablade.com).

## Install

```sh
npm install gputex
```

## Usage

### Three.js

```ts
import { compressTexture } from 'gputex/three'

const { texture } = await compressTexture('/cobblestone.avif', {
  mipmaps: true,
})
material.map = texture
```

Or as a loader:

```ts
import { GputexLoader } from 'gputex/three'

const loader = new GputexLoader()
loader.hint = 'normal'
loader.mipmaps = true
material.normalMap = await loader.loadAsync('/brick_normal.png')
```

### React Three Fiber

```tsx
import { useLoader } from '@react-three/fiber'
import { GputexLoader } from 'gputex/three'

function Ground() {
  const texture = useLoader(GputexLoader, '/cobblestone.avif', loader => {
    loader.mipmaps = true
  })

  return (
    <mesh>
      <planeGeometry args={[10, 10]} />
      <meshStandardMaterial map={texture} />
    </mesh>
  )
}
```

## Options

| Option       | Default   | Description                                                                                        |
| ------------ | --------- | -------------------------------------------------------------------------------------------------- |
| `hint`       | `'color'` | What the texture is for: `'color'`, `'colorWithAlpha'` or `'normal'`. Decides which format is used |
| `quality`    | `'high'`  | `'low'` halves the memory of opaque color textures at a visible quality cost                       |
| `colorSpace` | `'srgb'`  | `'srgb'` or `'linear'`                                                                             |
| `mipmaps`    | `false`   | Generate mipmaps                                                                                   |
| `flipY`      | `true`    | Flip vertically (Three.js convention)                                                              |
| `svgSize`    | intrinsic | Size to rasterize SVGs at: a number (longest side) or `{ width, height }`                          |
| `cache`      | `false`   | Remember compressed textures for the session so loading the same image again is instant            |
| `device`     | —         | Encode on an existing WebGPU device (e.g. your renderer's) instead of creating one                 |

Sources can be a URL, a `Blob`/`File`, an `ImageBitmap`, an `<img>`, a canvas, `ImageData`, or an inline SVG string.

The result also exposes `format`, `backend` (`'webgpu'`, `'webgl'` or `'none'`), `width`, `height`, `mipLevels`, timings, and `destroy()`.

### Normal maps on mobile

ASTC has no two-channel format, so on ASTC devices a `'normal'` texture stores X in the red channel and Y in the alpha channel. Check `result.astcNormalRemap` and read `.ra` instead of `.rg` in your shader when it's `true`.

## Tips

**Prewarm shaders** at app boot so the first texture doesn't pay ~100 ms of shader compilation:

```ts
import { prewarmCompressTexture } from 'gputex'

prewarmCompressTexture([{ hint: 'color', mipmaps: true }, { hint: 'normal' }])
```

**Reuse your renderer's WebGPU device** with the `device` option to avoid creating a second one.

## Without Three.js

The root `gputex` entry has no Three.js dependency and returns raw compressed bytes you can upload with any engine:

```ts
import { compressTextureToBytes } from 'gputex'

const { levels, format } = await compressTextureToBytes('/cobblestone.avif', {
  mipmaps: true,
})
// levels[i] = { data: Uint8Array, width, height }
```

Individual encoders (`BC7Encoder`, `BC5Encoder`, `BC1Encoder`, `ASTC4x4Encoder`, `ETC2Encoder`) are exported too.

## Browser support

Any browser with WebGPU or WebGL2. Compressed output needs a GPU with at least one compressed texture format, which is essentially all of them; if none is available, gputex returns an uncompressed texture so your app keeps working.

## Acknowledgements

The idea of compressing textures on the GPU on the fly was first introduced by [spark.js](https://ludicon.com/sparkjs/). gputex is not derived from Spark: its encoders were implemented from scratch from the official format specifications. For serious production use of GPU-compressed textures, Spark is the recommended choice.

## License

MIT
