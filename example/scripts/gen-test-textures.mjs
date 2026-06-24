// Procedural test textures for the per-format encoder pages (/bc1, /bc5, /bc7).
//
// No dependencies — hand-rolls a PNG (zlib is built into Node). Run with:
//   node scripts/gen-test-textures.mjs
//
// Outputs into public/textures/:
//   • color.png  — RGB test card used by the BC1 and BC7 pages. Designed to
//     expose codec weaknesses: smooth gradients (banding), saturated colour
//     edges (block/endpoint artifacts), and a high-frequency region (detail
//     loss). Encoding the SAME image with BC1 vs BC7 makes BC7's quality edge
//     obvious side by side.
//   • normal.png — a tangent-space normal map (bump field) used by the BC5
//     page. Smoothly varying R/G (normal x/y) is exactly what BC5 targets.
//
// These are committed defaults so the pages render out of the box. Drop your
// own color.png / normal.png in public/textures/ to test real assets.

import { writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { deflateSync } from 'node:zlib'

const SIZE = 512
const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'textures')

// ---------------------------------------------------------------- PNG writer

const CRC_TABLE = (() => {
  const t = new Uint32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    t[n] = c >>> 0
  }
  return t
})()

const crc32 = buf => {
  let c = 0xffffffff
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

const chunk = (type, data) => {
  const typeBuf = Buffer.from(type, 'ascii')
  const body = Buffer.concat([typeBuf, data])
  const out = Buffer.alloc(8 + body.length + 4 - 4) // len(4) + type+data + crc(4)
  out.writeUInt32BE(data.length, 0)
  body.copy(out, 4)
  out.writeUInt32BE(crc32(body), 4 + body.length)
  return out
}

// rgba: Uint8Array of length width*height*4
const encodePNG = (width, height, rgba) => {
  const stride = width * 4
  const raw = Buffer.alloc((stride + 1) * height)
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0 // filter: none
    Buffer.from(rgba.buffer, rgba.byteOffset + y * stride, stride).copy(raw, y * (stride + 1) + 1)
  }
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0)
  ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8 // bit depth
  ihdr[9] = 6 // colour type: RGBA
  // 10..12 = compression / filter / interlace = 0
  return Buffer.concat([
    sig,
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ])
}

// ---------------------------------------------------------------- helpers

const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v)
const to255 = v => Math.max(0, Math.min(255, Math.round(v * 255)))

// Deterministic value noise (hash-based) so runs are reproducible.
const hash = (x, y) => {
  let h = (x * 374761393 + y * 668265263) >>> 0
  h = (h ^ (h >>> 13)) >>> 0
  h = Math.imul(h, 1274126177) >>> 0
  return ((h ^ (h >>> 16)) >>> 0) / 0xffffffff
}

const hsv2rgb = (h, s, v) => {
  const i = Math.floor(h * 6)
  const f = h * 6 - i
  const p = v * (1 - s)
  const q = v * (1 - f * s)
  const t = v * (1 - (1 - f) * s)
  switch (i % 6) {
    case 0:
      return [v, t, p]
    case 1:
      return [q, v, p]
    case 2:
      return [p, v, t]
    case 3:
      return [p, q, v]
    case 4:
      return [t, p, v]
    default:
      return [v, p, q]
  }
}

// ---------------------------------------------------------------- color card

const genColor = () => {
  const px = new Uint8Array(SIZE * SIZE * 4)
  const half = SIZE / 2
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      let r, g, b
      const left = x < half
      const top = y < half
      if (left && top) {
        // TL: smooth hue × saturation gradient → banding test.
        const lu = x / (half - 1)
        const lv = y / (half - 1)
        ;[r, g, b] = hsv2rgb(lu, 0.35 + 0.65 * lv, 0.95)
      } else if (!left && top) {
        // TR: hard-edged saturated colour blocks → block/edge test.
        const bx = Math.floor(((x - half) / half) * 4)
        const by = Math.floor((y / half) * 4)
        const palette = [
          [0.9, 0.1, 0.1],
          [0.1, 0.8, 0.2],
          [0.1, 0.3, 0.95],
          [0.95, 0.85, 0.1],
          [0.95, 0.45, 0.1],
          [0.6, 0.1, 0.8],
          [0.1, 0.85, 0.85],
          [0.95, 0.95, 0.95],
        ]
        const c = palette[(bx + by * 4) % palette.length]
        ;[r, g, b] = c
        // thin black grid lines between blocks → high-contrast edges.
        const fx = ((x - half) / half) * 4
        const fy = (y / half) * 4
        if (fx - Math.floor(fx) < 0.04 || fy - Math.floor(fy) < 0.04) {
          r = g = b = 0.03
        }
      } else if (left && !top) {
        // BL: grayscale + per-channel ramps → endpoint/banding test.
        const lv = (y - half) / (half - 1)
        const band = Math.floor(lv * 4)
        const lu = x / (half - 1)
        if (band === 0) {
          r = g = b = lu
        } else if (band === 1) {
          r = lu
          g = 0.1
          b = 0.1
        } else if (band === 2) {
          r = 0.1
          g = lu
          b = 0.1
        } else {
          r = 0.1
          g = 0.1
          b = lu
        }
      } else {
        // BR: high-frequency checker + noise → detail-loss test, with a smooth
        // radial gradient disc overlaid in the centre (banding on curves).
        const cx = (x - half) / half
        const cy = (y - half) / half
        const checker = ((x >> 1) + (y >> 1)) & 1
        const n = hash(x, y) * 0.25
        if (checker) {
          r = 0.85 + n
          g = 0.2 + n
          b = 0.5 + n
        } else {
          r = 0.15 + n
          g = 0.6 + n
          b = 0.85 - n
        }
        const dx = cx - 0.5
        const dy = cy - 0.5
        const d = Math.sqrt(dx * dx + dy * dy)
        if (d < 0.42) {
          const t = clamp01(1 - d / 0.42)
          ;[r, g, b] = [0.95 * t, 0.7 * t + 0.1, 0.2 + 0.7 * (1 - t)]
        }
      }
      const o = (y * SIZE + x) * 4
      px[o] = to255(clamp01(r))
      px[o + 1] = to255(clamp01(g))
      px[o + 2] = to255(clamp01(b))
      px[o + 3] = 255
    }
  }
  return px
}

// ---------------------------------------------------------------- normal map

const genNormal = () => {
  const px = new Uint8Array(SIZE * SIZE * 4)
  // Height field: a grid of rounded bumps + a couple of diagonal ripples.
  const height = (x, y) => {
    const u = x / SIZE
    const v = y / SIZE
    let h = 0
    const cells = 5
    for (let gy = 0; gy < cells; gy++) {
      for (let gx = 0; gx < cells; gx++) {
        const cxp = (gx + 0.5) / cells
        const cyp = (gy + 0.5) / cells
        const dx = u - cxp
        const dy = v - cyp
        const r2 = (dx * dx + dy * dy) * cells * cells
        h += Math.exp(-r2 * 3.0) * (0.6 + 0.4 * hash(gx + 1, gy + 1))
      }
    }
    h += 0.15 * Math.sin((u + v) * Math.PI * 8)
    return h
  }
  const eps = 1 / SIZE
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const hL = height(x - 1, y)
      const hR = height(x + 1, y)
      const hD = height(x, y - 1)
      const hU = height(x, y + 1)
      // Tangent-space normal from the height gradient.
      const scale = 2.0
      let nx = (-(hR - hL) / (2 * eps)) * scale
      let ny = (-(hU - hD) / (2 * eps)) * scale
      let nz = 1
      const inv = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz)
      nx *= inv
      ny *= inv
      nz *= inv
      const o = (y * SIZE + x) * 4
      px[o] = to255(clamp01(nx * 0.5 + 0.5))
      px[o + 1] = to255(clamp01(ny * 0.5 + 0.5))
      px[o + 2] = to255(clamp01(nz * 0.5 + 0.5))
      px[o + 3] = 255
    }
  }
  return px
}

// ---------------------------------------------------------------- main

mkdirSync(OUT_DIR, { recursive: true })
writeFileSync(join(OUT_DIR, 'color.png'), encodePNG(SIZE, SIZE, genColor()))
writeFileSync(join(OUT_DIR, 'normal.png'), encodePNG(SIZE, SIZE, genNormal()))
console.log(`Wrote ${SIZE}×${SIZE} color.png and normal.png to ${OUT_DIR}`)
