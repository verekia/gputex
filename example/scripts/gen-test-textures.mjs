// Procedural test textures for the per-format encoder pages (/bc1, /bc5, /bc7).
//
// No dependencies — hand-rolls a PNG (zlib is built into Node). Run with:
//   node scripts/gen-test-textures.mjs
//
// Outputs into public/textures/:
//   • alpha.png  — 1024² RGBA test card for the alpha-capable formats
//     (BC7, ASTC): a 4×4 grid of 256² tiles, each stressing ONE
//     alpha-specific failure mode. Row 1: smooth alpha gradients (alpha
//     banding — over flat colour, radial falloff, joint colour+alpha ramp,
//     shallow low-contrast alpha). Row 2: hard alpha edges (decal cutout
//     over mismatched RGB, foliage mask, diagonal chevrons, thin strokes).
//     Row 3: high-frequency/mixed alpha (Nyquist checker, alpha noise,
//     UI-like translucent panes at fixed alpha flats, per-cell Voronoi
//     alpha). Row 4: natural-ish + class boundaries (smoke wisps, low-alpha
//     glass, a fully OPAQUE plasma tile and an exactly-GRAY opaque tile so
//     the per-block class selection and its tile-border transitions are
//     exercised inside one image). Transparent regions keep meaningful RGB
//     underneath — encoders must preserve colour under alpha ≈ 0 too.
//   • color.png  — 1024² RGB test card used by the BC1/BC7/ASTC pages: a 4×4
//     grid of 256² tiles, each stressing ONE codec failure mode so artifacts
//     are attributable at a glance. Row 1: smooth gradients (banding). Row 2:
//     hard edges (block/endpoint artifacts — includes the disc-over-checker
//     probe that catches invented-hue fringes on multi-cluster blocks).
//     Row 3: high-frequency detail (detail loss, chroma collapse). Row 4:
//     natural-ish content (what real assets look like). Encoding the SAME
//     card with BC1 vs BC7 makes BC7's quality edge obvious side by side.
//   • normal.png — 1024² tangent-space normal map for the BC5 page, four
//     quadrants: hemisphere domes (smooth full-range normals), pyramids
//     (flat faces + hard creases — flat faces must stay flat), a bevelled
//     brick wall (fine structured detail), and a ripple frequency sweep with
//     a flat strip (banding on a flat normal shows as lighting blotches).
//
// These are committed defaults so the pages render out of the box. Drop your
// own color.png / normal.png in public/textures/ to test real assets.
//
// NOTE: the /test suite's PSNR floors and excess limits (gpuTestSuite.ts) are
// measured against these exact images — regenerate them and the thresholds
// must be re-baselined.

import { writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { deflateSync } from 'node:zlib'

const SIZE = 1024
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
const lerp = (a, b, t) => a + (b - a) * t
const mix3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]

// Deterministic value noise (hash-based) so runs are reproducible.
const hash = (x, y) => {
  let h = (x * 374761393 + y * 668265263) >>> 0
  h = (h ^ (h >>> 13)) >>> 0
  h = Math.imul(h, 1274126177) >>> 0
  return ((h ^ (h >>> 16)) >>> 0) / 0xffffffff
}

// Smooth value noise: bilinear hash-lattice interpolation with smoothstep.
const smoothNoise = (x, y) => {
  const xi = Math.floor(x)
  const yi = Math.floor(y)
  const fx = x - xi
  const fy = y - yi
  const sx = fx * fx * (3 - 2 * fx)
  const sy = fy * fy * (3 - 2 * fy)
  const n00 = hash(xi, yi)
  const n10 = hash(xi + 1, yi)
  const n01 = hash(xi, yi + 1)
  const n11 = hash(xi + 1, yi + 1)
  return lerp(lerp(n00, n10, sx), lerp(n01, n11, sx), sy)
}

// Fractional Brownian motion — 4 octaves of smooth value noise in [0,1).
const fbm = (x, y) => {
  let sum = 0
  let amp = 0.5
  let freq = 1
  for (let o = 0; o < 4; o++) {
    sum += amp * smoothNoise(x * freq, y * freq)
    amp *= 0.5
    freq *= 2
  }
  return sum / 0.9375
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
//
// 4×4 grid of 256² tiles separated by 2px near-black rules (crisp edges are
// themselves a stressor, and the rules make the card read as deliberate).
// Every tile function gets tile-local (u, v) in [0,1) plus the absolute pixel
// (x, y) for pixel-exact patterns, and returns [r, g, b] in [0,1].

const TILE = 256

const TILES = [
  // ---- Row 1: smooth gradients — banding. ----
  // Hue sweep × saturation ramp.
  (u, v) => hsv2rgb(u, 0.35 + 0.65 * v, 0.95),
  // Very shallow grayscale dome — the classic banding killer (span ~0.14).
  (u, v) => {
    const d = Math.hypot(u - 0.5, v - 0.5) / 0.7071
    const g = 0.36 + 0.14 * (1 - clamp01(d))
    return [g, g, g]
  },
  // Deep shadow gradient: black → dark blue, where 565/endpoint quantisation
  // bands are most visible.
  (u, v) => mix3([0.0, 0.01, 0.03], [0.16, 0.2, 0.52], v * (0.7 + 0.3 * u)),
  // Duotone diagonal: orange → teal through neutral (chroma axis banding).
  (u, v) => mix3([0.93, 0.52, 0.13], [0.09, 0.5, 0.58], (u + v) / 2),

  // ---- Row 2: hard edges — block/endpoint artifacts. ----
  // Saturated tile grid with thin dark rules (per-tile flatness + edges).
  (u, v) => {
    const PALETTE = [
      [0.9, 0.1, 0.1],
      [0.1, 0.8, 0.2],
      [0.1, 0.3, 0.95],
      [0.95, 0.85, 0.1],
      [0.95, 0.45, 0.1],
      [0.6, 0.1, 0.8],
      [0.1, 0.85, 0.85],
      [0.95, 0.95, 0.95],
    ]
    const fx = u * 4
    const fy = v * 4
    if (fx - Math.floor(fx) < 0.04 || fy - Math.floor(fy) < 0.04) return [0.03, 0.03, 0.03]
    return PALETTE[(Math.floor(fx) + Math.floor(fy) * 4) % PALETTE.length]
  },
  // Radial-gradient disc over a crisp 2px pink/teal checker — the
  // multi-cluster edge probe (blocks on the rim hold 3 colour clusters; an
  // encoder that extrapolates endpoints paints fringe hues here).
  (u, v, x, y) => {
    const d = Math.hypot(u - 0.5, v - 0.5) / 0.42
    if (d < 1) {
      const t = 1 - d
      return [0.95 * t, 0.7 * t + 0.1, 0.2 + 0.7 * (1 - t)]
    }
    return ((x >> 1) + (y >> 1)) & 1 ? [0.85, 0.2, 0.5] : [0.15, 0.6, 0.85]
  },
  // 45° chevron bars in complementary red/cyan — diagonal edges cut every
  // block, the worst orientation for a 4×4 grid.
  (u, v, x, y) => (Math.floor((x + y) / 8) & 1 ? [0.88, 0.12, 0.14] : [0.1, 0.78, 0.84]),
  // Zone plate: sin(r²) rings sweep every frequency and orientation up to
  // ~Nyquist at the corners — detail loss shows as ring dropout/moiré.
  (u, v) => {
    const dx = (u - 0.5) * TILE
    const dy = (v - 0.5) * TILE
    const g = 0.5 + 0.45 * Math.sin((dx * dx + dy * dy) / 170)
    return [g, g, g]
  },

  // ---- Row 3: highest frequency — detail loss, chroma collapse. ----
  // 1px black/white checker — Nyquist luma.
  (u, v, x, y) => ((x + y) & 1 ? [0.95, 0.95, 0.95] : [0.05, 0.05, 0.05]),
  // 1px red/blue checker — Nyquist chroma (BC1's classic failure: collapses
  // to purple mush).
  (u, v, x, y) => ((x + y) & 1 ? [0.88, 0.1, 0.14] : [0.12, 0.18, 0.88]),
  // Pinstripe frequency sweep: stripe width doubles per column band (1→32px);
  // top half vertical, bottom half horizontal.
  (u, v, x, y) => {
    const w = 1 << Math.min(Math.floor(u * 6), 5)
    const on = (Math.floor((v < 0.5 ? x : y) / w) & 1) === 1
    return on ? [0.92, 0.9, 0.85] : [0.16, 0.14, 0.2]
  },
  // Contained noise patch — worst-case entropy. Left half grayscale, right
  // half independent RGB (chroma entropy).
  (u, v, x, y) => {
    if (u < 0.5) {
      const g = 0.15 + 0.7 * hash(x, y)
      return [g, g, g]
    }
    return [0.15 + 0.7 * hash(x, y), 0.15 + 0.7 * hash(x + 7919, y), 0.15 + 0.7 * hash(x, y + 4483)]
  },

  // ---- Row 4: natural-ish content — what real assets degrade like. ----
  // Marble: fBm-warped sine veins, warm stone tint.
  (u, v, x, y) => {
    const t = fbm(x / 48, y / 48)
    const vein = 0.5 + 0.5 * Math.sin((u * 6 + t * 5) * Math.PI)
    return mix3([0.23, 0.2, 0.28], [0.88, 0.84, 0.78], vein)
  },
  // Plasma: layered sines — smooth, colourful, every hue direction at once.
  (u, v) => {
    const a = Math.sin(u * 5.1 + Math.sin(v * 3.7)) + Math.sin(Math.hypot(u - 0.7, v - 0.3) * 9)
    const b = Math.sin(v * 4.3 + Math.sin(u * 2.9)) + Math.sin(Math.hypot(u - 0.2, v - 0.8) * 7)
    return [
      0.5 + 0.4 * Math.sin(a * 1.7),
      0.5 + 0.4 * Math.sin(b * 1.9 + 2.1),
      0.5 + 0.4 * Math.sin((a + b) * 1.3 + 4.2),
    ]
  },
  // Voronoi cells: pastel flats with darkened borders — organic hard edges.
  (u, v) => {
    let d1 = Infinity
    let d2 = Infinity
    let cell = 0
    for (let i = 0; i < 20; i++) {
      const px = hash(i * 3 + 1, 17)
      const py = hash(31, i * 5 + 2)
      const d = Math.hypot(u - px, v - py)
      if (d < d1) {
        d2 = d1
        d1 = d
        cell = i
      } else if (d < d2) {
        d2 = d
      }
    }
    const base = hsv2rgb(hash(cell, 97), 0.35, 0.6 + 0.35 * hash(cell, 131))
    const edge = clamp01((d2 - d1) * 22) // 0 at borders → 1 inside
    return mix3([0.08, 0.07, 0.1], base, 0.25 + 0.75 * edge)
  },
  // Soft additive blobs on a dark ground — smooth multi-hue gradients.
  (u, v) => {
    let r = 0.06
    let g = 0.06
    let b = 0.09
    for (let i = 0; i < 7; i++) {
      const px = hash(i + 3, 211)
      const py = hash(223, i + 5)
      const w = Math.exp(-(((u - px) ** 2 + (v - py) ** 2) / 0.022))
      const c = hsv2rgb(hash(i, 241), 0.65, 0.8)
      r += c[0] * w * 0.6
      g += c[1] * w * 0.6
      b += c[2] * w * 0.6
    }
    return [clamp01(r), clamp01(g), clamp01(b)]
  },
]

const genColor = () => {
  const px = new Uint8Array(SIZE * SIZE * 4)
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const tx = Math.floor(x / TILE)
      const ty = Math.floor(y / TILE)
      const lx = x - tx * TILE
      const ly = y - ty * TILE
      // 2px separator rules between tiles (not on the outer border).
      const onRule = (lx < 2 && tx > 0) || (ly < 2 && ty > 0)
      const [r, g, b] = onRule ? [0.04, 0.04, 0.05] : TILES[ty * 4 + tx](lx / TILE, ly / TILE, x, y)
      const o = (y * SIZE + x) * 4
      px[o] = to255(clamp01(r))
      px[o + 1] = to255(clamp01(g))
      px[o + 2] = to255(clamp01(b))
      px[o + 3] = 255
    }
  }
  return px
}

// ---------------------------------------------------------------- alpha card
//
// Same 4×4 tile grid as the colour card, but every tile is an ALPHA
// stressor; tiles return [r, g, b, a]. See the header for the row plan.

const ALPHA_TILES = [
  // ---- Row 1: smooth alpha gradients — alpha banding. ----
  // Pure alpha ramp over a flat warm colour (only alpha varies).
  u => [0.9, 0.45, 0.12, u],
  // Radial soft-particle falloff (gaussian-ish glow).
  (u, v) => {
    const d = Math.hypot(u - 0.5, v - 0.5) / 0.5
    return [0.95, 0.8, 0.35, Math.exp(-d * d * 3.2)]
  },
  // Joint colour + alpha ramp — the single-RGBA-line case.
  (u, v) => {
    const [r, g, b] = hsv2rgb(u * 0.8, 0.6, 0.9)
    return [r, g, b, v]
  },
  // Shallow alpha dome (span ~0.14) over a textured ground — the alpha
  // analogue of the colour card's banding-killer tile.
  (u, v, x, y) => {
    const d = Math.hypot(u - 0.5, v - 0.5) / 0.7071
    const t = fbm(x / 40, y / 40)
    const [r, g, b] = mix3([0.2, 0.3, 0.5], [0.4, 0.55, 0.7], t)
    return [r, g, b, 0.36 + 0.14 * (1 - clamp01(d))]
  },

  // ---- Row 2: hard alpha edges — cutout/decal artifacts. ----
  // Opaque patterned disc over a fully transparent ground whose RGB
  // deliberately clashes — blocks on the rim mix a=1 and a=0 clusters.
  (u, v) => {
    const d = Math.hypot(u - 0.5, v - 0.5) / 0.42
    if (d < 1) {
      const ring = 0.5 + 0.5 * Math.sin(d * 18)
      return [0.9 - 0.5 * ring, 0.25 + 0.55 * ring, 0.2, 1]
    }
    return [0.05, 0.9, 0.05, 0]
  },
  // Foliage mask: fBm thresholded into leaf clumps, green shades.
  (u, v, x, y) => {
    const t = fbm(x / 26, y / 26)
    const a = t > 0.52 ? 1 : 0
    const [r, g, b] = mix3([0.1, 0.35, 0.08], [0.35, 0.7, 0.2], fbm(x / 9 + 40, y / 9))
    return [r, g, b, a]
  },
  // 45° alpha chevrons over a colour gradient — diagonal alpha edges cut
  // every 4×4 block.
  (u, v, x, y) => {
    const [r, g, b] = mix3([0.85, 0.3, 0.5], [0.2, 0.4, 0.85], u)
    return [r, g, b, Math.floor((x + y) / 8) & 1 ? 1 : 0]
  },
  // Thin strokes: 2px bars in a grid — text/UI-like alpha features.
  (u, v, x, y) => {
    const on = x % 12 < 2 || y % 10 < 2
    return [0.95, 0.95, 0.9, on ? 1 : 0.08]
  },

  // ---- Row 3: high-frequency / mixed alpha. ----
  // 1px alpha checker — Nyquist alpha over constant colour.
  (u, v, x, y) => [0.8, 0.6, 0.2, (x + y) & 1 ? 1 : 0],
  // Independent alpha noise over a smooth colour gradient.
  (u, v, x, y) => {
    const [r, g, b] = mix3([0.15, 0.5, 0.6], [0.7, 0.25, 0.5], v)
    return [r, g, b, 0.15 + 0.7 * hash(x + 131, y + 57)]
  },
  // UI panes: overlapping rectangles at alpha flats {0.25, 0.5, 0.75, 1} —
  // hard alpha steps between large flat regions.
  (u, v) => {
    let a = 0.0
    let c = [0.12, 0.12, 0.16]
    const PANES = [
      [0.05, 0.08, 0.6, 0.55, 0.25, [0.9, 0.9, 0.95]],
      [0.3, 0.25, 0.92, 0.7, 0.5, [0.25, 0.6, 0.9]],
      [0.15, 0.5, 0.7, 0.93, 0.75, [0.95, 0.6, 0.25]],
      [0.55, 0.05, 0.95, 0.4, 1.0, [0.4, 0.85, 0.45]],
    ]
    for (const [x0, y0, x1, y1, pa, pc] of PANES) {
      if (u >= x0 && u < x1 && v >= y0 && v < y1) {
        a = pa
        c = pc
      }
    }
    return [c[0], c[1], c[2], a]
  },
  // Voronoi with per-cell alpha flats and opaque borders.
  (u, v) => {
    let d1 = Infinity
    let d2 = Infinity
    let cell = 0
    for (let i = 0; i < 20; i++) {
      const px = hash(i * 7 + 3, 53)
      const py = hash(71, i * 3 + 11)
      const d = Math.hypot(u - px, v - py)
      if (d < d1) {
        d2 = d1
        d1 = d
        cell = i
      } else if (d < d2) {
        d2 = d
      }
    }
    const base = hsv2rgb(hash(cell, 177), 0.5, 0.75)
    const border = clamp01((d2 - d1) * 22)
    return [base[0], base[1], base[2], border < 0.5 ? 1 : hash(cell, 191)]
  },

  // ---- Row 4: natural-ish alpha + block-class boundaries. ----
  // Smoke: fBm wisps, alpha 0..0.8 over a cool gray.
  (u, v, x, y) => {
    const t = fbm(x / 56, y / 56 + 90)
    const a = clamp01((t - 0.35) * 2.2) * 0.8 * (1 - v * 0.6)
    const g = 0.55 + 0.25 * fbm(x / 22 + 7, y / 22)
    return [g, g, g * 1.05, a]
  },
  // Low-alpha glass: colour gradient at a = 0.2..0.4 — precision where a
  // handful of alpha codes must carry a smooth ramp.
  (u, v) => {
    const [r, g, b] = mix3([0.3, 0.75, 0.85], [0.7, 0.3, 0.8], u)
    return [r, g, b, 0.2 + 0.2 * v]
  },
  // Fully opaque plasma — inside an alpha card, so class transitions at the
  // tile borders are part of the test.
  (u, v) => {
    const a = Math.sin(u * 5.1 + Math.sin(v * 3.7)) + Math.sin(Math.hypot(u - 0.7, v - 0.3) * 9)
    const b = Math.sin(v * 4.3 + Math.sin(u * 2.9)) + Math.sin(Math.hypot(u - 0.2, v - 0.8) * 7)
    return [
      0.5 + 0.4 * Math.sin(a * 1.7),
      0.5 + 0.4 * Math.sin(b * 1.9 + 2.1),
      0.5 + 0.4 * Math.sin((a + b) * 1.3 + 4.2),
      1,
    ]
  },
  // Exactly-gray opaque dome — the grayscale block class, adjacent to
  // translucent tiles.
  (u, v) => {
    const d = Math.hypot(u - 0.5, v - 0.5) / 0.7071
    const g = 0.3 + 0.4 * (1 - clamp01(d))
    return [g, g, g, 1]
  },
]

const genAlpha = () => {
  const px = new Uint8Array(SIZE * SIZE * 4)
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const tx = Math.floor(x / TILE)
      const ty = Math.floor(y / TILE)
      const lx = x - tx * TILE
      const ly = y - ty * TILE
      // 2px separator rules between tiles (opaque, like the colour card).
      const onRule = (lx < 2 && tx > 0) || (ly < 2 && ty > 0)
      const [r, g, b, a] = onRule ? [0.04, 0.04, 0.05, 1] : ALPHA_TILES[ty * 4 + tx](lx / TILE, ly / TILE, x, y)
      const o = (y * SIZE + x) * 4
      px[o] = to255(clamp01(r))
      px[o + 1] = to255(clamp01(g))
      px[o + 2] = to255(clamp01(b))
      px[o + 3] = to255(clamp01(a))
    }
  }
  return px
}

// ---------------------------------------------------------------- normal map
//
// Height field in pixel units → tangent-space normals via central
// differences. Four 512² quadrants, each a different BC5 stressor:
//   TL domes      — hemispheres, smooth normals through the full x/y range
//   TR pyramids   — flat faces (must stay flat) meeting in hard creases
//   BL bricks     — bevelled grooves, fine structured detail at block scale
//   BR ripples    — frequency sweep (64→6px) + a dead-flat strip at the
//                   bottom (any banding there shows as lighting blotches)

const HALF = SIZE / 2

// Quadrant-local height functions, (lx, ly) in [0, HALF).
const domeHeight = (lx, ly) => {
  const cells = 4
  const cs = HALF / cells
  const cx = Math.floor(lx / cs)
  const cy = Math.floor(ly / cs)
  const R = cs * 0.42
  const dx = lx - (cx + 0.5) * cs
  const dy = ly - (cy + 0.5) * cs
  const d2 = dx * dx + dy * dy
  const r = R * (0.75 + 0.25 * hash(cx + 51, cy + 87))
  return d2 < r * r ? Math.sqrt(r * r - d2) : 0
}

const pyramidHeight = (lx, ly) => {
  const cells = 4
  const cs = HALF / cells
  const cx = Math.floor(lx / cs)
  const cy = Math.floor(ly / cs)
  const dx = Math.abs(lx - (cx + 0.5) * cs)
  const dy = Math.abs(ly - (cy + 0.5) * cs)
  const R = cs * 0.44
  // Alternate square (chessboard-axis) and diamond (45°) pyramids.
  const d = (cx + cy) & 1 ? dx + dy : Math.max(dx, dy)
  return Math.max(0, R - d) * 0.9
}

const brickHeight = (lx, ly) => {
  const bw = 128
  const bh = 64
  const groove = 8 // full groove width
  const bevel = 7 // bevel ramp on each side of the groove
  const row = Math.floor(ly / bh)
  const ox = row & 1 ? bw / 2 : 0
  const bx = (((lx + ox) % bw) + bw) % bw
  const by = ((ly % bh) + bh) % bh
  // Distance to the nearest brick border along each axis.
  const ex = Math.min(bx, bw - bx)
  const ey = Math.min(by, bh - by)
  const e = Math.min(ex, ey)
  const h = clamp01((e - groove / 2) / bevel) * 10
  // Subtle per-brick height variation so faces aren't all identical.
  const col = Math.floor((lx + ox) / bw)
  return h * (0.8 + 0.2 * hash(col + 13, row + 29))
}

const rippleHeight = (lx, ly) => {
  if (ly > HALF * 0.8) return 0 // dead-flat strip
  const band = Math.floor(ly / ((HALF * 0.8) / 5)) // 5 frequency bands
  const wavelength = 64 / 1.6 ** band // 64 → ~9.8px
  const slope = 0.55
  const amp = (wavelength * slope) / (2 * Math.PI)
  return amp * Math.sin((2 * Math.PI * lx) / wavelength)
}

const height = (x, y) => {
  // Clamp into the owning quadrant so central differences never sample across
  // a quadrant seam (which would smear one stressor into another).
  const qx = x < HALF ? 0 : 1
  const qy = y < HALF ? 0 : 1
  const lx = Math.min(Math.max(x - qx * HALF, 0), HALF - 1)
  const ly = Math.min(Math.max(y - qy * HALF, 0), HALF - 1)
  if (qy === 0) return qx === 0 ? domeHeight(lx, ly) : pyramidHeight(lx, ly)
  return qx === 0 ? brickHeight(lx, ly) : rippleHeight(lx, ly)
}

const genNormal = () => {
  const px = new Uint8Array(SIZE * SIZE * 4)
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const hL = height(x - 1, y)
      const hR = height(x + 1, y)
      const hU = height(x, y - 1)
      const hD = height(x, y + 1)
      // Tangent-space normal from the height gradient (heights in px units).
      let nx = -(hR - hL) / 2
      let ny = -(hD - hU) / 2
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
writeFileSync(join(OUT_DIR, 'alpha.png'), encodePNG(SIZE, SIZE, genAlpha()))
console.log(`Wrote ${SIZE}×${SIZE} color.png, normal.png and alpha.png to ${OUT_DIR}`)
