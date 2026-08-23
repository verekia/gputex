// Mip-chain generation (CPU, box filter).
//
// The GPU counterpart lives in gpuMipgen.ts and is what `compressTexture()`
// uses on healthy WebGPU devices — profiling showed this CPU path costing
// ~70 ms of main-thread time at 4K (getImageData readback + filter) plus a
// writeTexture upload per level. This module remains for:
//   • devices with a broken `copyExternalImageToTexture` (workarounds.ts),
//     which can't upload the bitmap for GPU filtering in the first place;
//   • the WebGL2 fallback tier (no compute);
//   • engine-agnostic callers that already hold CPU pixel chains.
// The GPU filter is integer-exact against `downsample2x`, so both paths
// produce byte-identical compressed output.
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
  // Hot loop: `s` hoisted (one property load instead of two per texel read)
  // and all indices carried as running byte offsets — no per-texel
  // `(y * W + x) * 4` multiplies.
  const s = src.data
  const rowBytes = src.width * 4
  const lastColByte = (src.width - 1) * 4
  const sMaxY = src.height - 1
  let o = 0
  for (let y = 0; y < dstH; y++) {
    const sy0 = y << 1
    const sy1 = sy0 < sMaxY ? sy0 + 1 : sMaxY
    const r0 = sy0 * rowBytes
    const r1 = sy1 * rowBytes
    for (let x = 0; x < dstW; x++) {
      const sx0 = x << 3 // byte offset of source texel 2x
      const sx1 = sx0 + 4 <= lastColByte ? sx0 + 4 : lastColByte
      const i00 = r0 + sx0
      const i10 = r0 + sx1
      const i01 = r1 + sx0
      const i11 = r1 + sx1
      // Unrolled per channel — predictable, and typed-array indexing
      // is the hot path here.
      dst[o] = (s[i00]! + s[i10]! + s[i01]! + s[i11]! + 2) >> 2
      dst[o + 1] = (s[i00 + 1]! + s[i10 + 1]! + s[i01 + 1]! + s[i11 + 1]! + 2) >> 2
      dst[o + 2] = (s[i00 + 2]! + s[i10 + 2]! + s[i01 + 2]! + s[i11 + 2]! + 2) >> 2
      dst[o + 3] = (s[i00 + 3]! + s[i10 + 3]! + s[i01 + 3]! + s[i11 + 3]! + 2) >> 2
      o += 4
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
  const src = level.data
  const w = level.width
  const h = level.height
  const srcRowBytes = w * 4
  const dstRowBytes = pw * 4
  // Copy each real row wholesale, then replicate its last texel across the
  // ≤ 3 padding columns. `set` on a subarray is a memcpy; the per-pixel
  // index arithmetic this replaces ran over the WHOLE level, so on a
  // non-block-aligned 4K base level it was 16M iterations of scalar work.
  for (let y = 0; y < h; y++) {
    const di = y * dstRowBytes
    out.set(src.subarray(y * srcRowBytes, y * srcRowBytes + srcRowBytes), di)
    const lastTexel = di + srcRowBytes - 4
    for (let x = w; x < pw; x++) {
      const d = di + x * 4
      out[d] = out[lastTexel]!
      out[d + 1] = out[lastTexel + 1]!
      out[d + 2] = out[lastTexel + 2]!
      out[d + 3] = out[lastTexel + 3]!
    }
  }
  // Padding rows repeat the last real row verbatim, padding included.
  const lastRow = (h - 1) * dstRowBytes
  for (let y = h; y < ph; y++) {
    out.copyWithin(y * dstRowBytes, lastRow, lastRow + dstRowBytes)
  }
  return { data: out, width: pw, height: ph }
}
