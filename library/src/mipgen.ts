// Mip-chain generation (CPU, box filter).
//
// The plan calls for a compute pass here. We chose CPU because:
//   • `rgba8unorm-storage` is an optional WebGPU feature and not all
//     adapters report it, so a compute-based mipgen would need a
//     render-pass fallback anyway (two code paths to test).
//   • Mip generation runs once at load time on a bitmap that already
//     sits in CPU memory after `createImageBitmap`/`getImageData`.
//     A 1024² texture produces ~1.33× the base pixel count across the
//     mip chain — a JS box filter handles that in well under 50 ms.
//   • The encoder then re-uploads each level to the GPU for block
//     compression. That dominates wall-clock time either way.
// Upgrading to GPU mip-gen is a drop-in replacement behind this module
// if profiling ever justifies it.
//
// Colour-space caveat: this module box-filters the raw byte values. For
// sRGB-encoded inputs that's slightly too dark compared to filtering in
// linear light (the standard trade-off every naïve mip-gen makes,
// including browsers' automatic `generateMipmap` in WebGL). If a caller
// needs perceptually-correct mips they can decode → linearise → filter
// → re-encode upstream of this function; our output format doesn't
// change.

/** One level of a mip chain. 4 bytes per pixel (RGBA8). */
export interface MipLevel {
  data: Uint8ClampedArray
  width: number
  height: number
}

/**
 * Produce the full mip chain from a level-0 image. The chain goes down
 * to a 1×1 level — the standard OpenGL / WebGPU convention — so the
 * caller gets `floor(log2(max(w, h))) + 1` levels total.
 *
 * Levels whose logical dimensions are below the encoder's 4×4 block
 * grid are still produced here at their true logical size; padding up
 * to a single block is the encoder's job, not ours.
 */
export function generateMipChain(level0: MipLevel): MipLevel[] {
  if (level0.width < 1 || level0.height < 1) {
    throw new Error(`generateMipChain: level 0 must be at least 1×1, got ${level0.width}×${level0.height}`)
  }
  if (level0.data.length !== level0.width * level0.height * 4) {
    throw new Error(
      `generateMipChain: level 0 data length ${level0.data.length} does not match ` +
        `${level0.width}×${level0.height}×4 = ${level0.width * level0.height * 4}`,
    )
  }
  const chain: MipLevel[] = [level0]
  let prev = level0
  while (prev.width > 1 || prev.height > 1) {
    prev = downsample2x(prev)
    chain.push(prev)
  }
  return chain
}

/**
 * Halve a mip level in each dimension via 2×2 box filter. Either
 * dimension can be odd; the rightmost / bottom texel "folds onto itself"
 * (clamp-to-edge) so the filter weight stays 1.0.
 *
 * Rounding uses `(a+b+c+d+2) >> 2` — round-to-nearest, matching the
 * conventional integer box filter. Without the +2 bias the filter
 * systematically darkens by ~1 LSB per level over 10+ levels.
 */
function downsample2x(src: MipLevel): MipLevel {
  const dstW = Math.max(1, src.width >> 1)
  const dstH = Math.max(1, src.height >> 1)
  const dst = new Uint8ClampedArray(dstW * dstH * 4)
  const sW = src.width
  const sMaxX = src.width - 1
  const sMaxY = src.height - 1
  for (let y = 0; y < dstH; y++) {
    const sy0 = y * 2
    const sy1 = Math.min(sy0 + 1, sMaxY)
    for (let x = 0; x < dstW; x++) {
      const sx0 = x * 2
      const sx1 = Math.min(sx0 + 1, sMaxX)
      const i00 = (sy0 * sW + sx0) * 4
      const i10 = (sy0 * sW + sx1) * 4
      const i01 = (sy1 * sW + sx0) * 4
      const i11 = (sy1 * sW + sx1) * 4
      const o = (y * dstW + x) * 4
      // Unrolled per channel — predictable, and typed-array indexing
      // is the hot path here.
      dst[o] = (src.data[i00]! + src.data[i10]! + src.data[i01]! + src.data[i11]! + 2) >> 2
      dst[o + 1] = (src.data[i00 + 1]! + src.data[i10 + 1]! + src.data[i01 + 1]! + src.data[i11 + 1]! + 2) >> 2
      dst[o + 2] = (src.data[i00 + 2]! + src.data[i10 + 2]! + src.data[i01 + 2]! + src.data[i11 + 2]! + 2) >> 2
      dst[o + 3] = (src.data[i00 + 3]! + src.data[i10 + 3]! + src.data[i01 + 3]! + src.data[i11 + 3]! + 2) >> 2
    }
  }
  return { data: dst, width: dstW, height: dstH }
}

/**
 * Pad a mip level up to a multiple of 4 in each dimension using clamp-
 * to-edge sampling. Used before handing sub-4×4 levels to a block-
 * compression encoder, which requires at least one full block per level.
 *
 * If the input is already block-aligned this returns the input unchanged.
 */
export function padToBlockMultiple(level: MipLevel): MipLevel {
  const pw = (level.width + 3) & ~3
  const ph = (level.height + 3) & ~3
  if (pw === level.width && ph === level.height) return level
  const out = new Uint8ClampedArray(pw * ph * 4)
  const maxX = level.width - 1
  const maxY = level.height - 1
  for (let y = 0; y < ph; y++) {
    const sy = Math.min(y, maxY)
    for (let x = 0; x < pw; x++) {
      const sx = Math.min(x, maxX)
      const si = (sy * level.width + sx) * 4
      const di = (y * pw + x) * 4
      out[di] = level.data[si]!
      out[di + 1] = level.data[si + 1]!
      out[di + 2] = level.data[si + 2]!
      out[di + 3] = level.data[si + 3]!
    }
  }
  return { data: out, width: pw, height: ph }
}
