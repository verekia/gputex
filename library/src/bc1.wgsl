// BC1 (DXT1) compute shader encoder.
//
// Each invocation encodes one 4x4 pixel block into an 8-byte BC1 block
// written as 2 x u32 into the destination storage buffer. This is the f32
// fallback; bc1_fast_f16.wgsl is the same algorithm and is preferred when
// the device reports shader-f16.
//
// BC1 block layout (little-endian):
//   u32[0]: color0 (low 16) | color1 (high 16)   both in RGB565
//   u32[1]: 16 x 2-bit indices, pixel 0 = bits 0..1, pixel 15 = bits 30..31
//
// We always force the 4-color mode (color0 > color1, numeric 16-bit):
//   idx 0 -> color0
//   idx 1 -> color1
//   idx 2 -> (2*color0 +   color1) / 3
//   idx 3 -> (  color0 + 2*color1) / 3
//
// ALGORITHM (same as bc1_fast_f16.wgsl, which documents the measurements):
// near-flat blocks take a solid colour — per channel the endpoint pair whose
// ⅔/⅓ interpolant lands nearest the block mean; other blocks get a
// principal-axis endpoint seed (covariance power-iteration; inset bbox on
// degenerate blocks), inset by ~half a 565 cell along the axis. A
// projection pass then assigns every pixel the rounded projection onto the
// decoded-endpoint line (the 4 palette entries are colinear and evenly
// spaced, so that is the nearest entry) while accumulating the block error
// and projection moments, followed by up to TWO least-squares refit rounds
// solved from those moments (re-quantise, reproject, accept only on lower
// block error).

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
  y0:       u32, // first block row of this dispatch (row-band encodes)
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn to565(c: vec3<f32>) -> u32 {
  // Round-to-nearest quantization into 5-6-5.
  let r = u32(clamp(floor(c.r * 31.0 + 0.5), 0.0, 31.0));
  let g = u32(clamp(floor(c.g * 63.0 + 0.5), 0.0, 63.0));
  let b = u32(clamp(floor(c.b * 31.0 + 0.5), 0.0, 31.0));
  return (r << 11u) | (g << 5u) | b;
}

fn from565(c: u32) -> vec3<f32> {
  let r = (c >> 11u) & 31u;
  let g = (c >>  5u) & 63u;
  let b =  c         & 31u;
  // 5/6-bit -> 8-bit: (x*527+23)>>6 (6-bit: 259/33) — round-to-nearest
  // scaling, matching bc1_ref.ts and typical hardware decoders (white ->
  // 255). Integer u32 math is exact. NOTE: this is NOT plain bit-replication
  // ((x<<3)|(x>>2)) — they differ for some codes (e.g. 5-bit 3 -> 25 vs 24).
  // Selecting indices against this palette is what makes the encoder agree
  // with what the GPU will actually sample.
  let r8 = (r * 527u + 23u) >> 6u;
  let g8 = (g * 259u + 33u) >> 6u;
  let b8 = (b * 527u + 23u) >> 6u;
  return vec3<f32>(vec3<u32>(r8, g8, b8)) / 255.0;
}

// Force 4-colour mode: c0 > c1 strictly.
fn order565(a: u32, b: u32) -> vec2<u32> {
  var c0 = a; var c1 = b;
  if (c0 == c1) {
    if (c1 > 0u) { c1 = c1 - 1u; } else { c0 = c0 + 1u; }
  } else if (c0 < c1) {
    let t = c0; c0 = c1; c1 = t;
  }
  return vec2<u32>(c0, c1);
}

// One projection pass against the decoded endpoints of (c0,c1): levels
// L = 0..3 along p0→p1 (palette = p0, p0+⅓d, p0+⅔d, p1 — colinear, evenly
// spaced, so rounding the projection IS the nearest-entry search), the
// packed indices, the block's squared error, and the projection MOMENTS a
// refit needs: ΣL, ΣL², Σu, ΣL·u (u = v − p0). Level → BC1 index: 0→0
// (c0), 1→2 (⅔c0+⅓c1), 2→3, 3→1 (c1); as a packed LUT: (0x78 >> 2L) & 3.
struct Moments { sL: f32, sLL: f32, sU: vec3<f32>, sLu: vec3<f32>, indices: u32, err: f32 };
fn moments(pix: ptr<function, array<vec3<f32>, 16>>, c0: u32, c1: u32) -> Moments {
  let p0 = from565(c0);
  let dir = from565(c1) - p0;
  let inv = 3.0 / dot(dir, dir);
  var out: Moments;
  out.sL = 0.0;
  out.sLL = 0.0;
  out.sU = vec3<f32>(0.0);
  out.sLu = vec3<f32>(0.0);
  out.indices = 0u;
  out.err = 0.0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let u = (*pix)[k] - p0;
    let L = clamp(floor(dot(u, dir) * inv + 0.5), 0.0, 3.0);
    out.sL = out.sL + L;
    out.sLL = out.sLL + L * L;
    out.sU = out.sU + u;
    out.sLu = out.sLu + L * u;
    out.indices = out.indices | (((0x78u >> (u32(L) * 2u)) & 3u) << (k * 2u));
    let e = u - L * (1.0 / 3.0) * dir;
    out.err = out.err + dot(e, e);
  }
  return out;
}

// One least-squares refit from moments (b = L/3, a = 1 − b):
//   sBB = ΣL²/9   sAB = ΣL/3 − ΣL²/9   sAA = 16 − 2ΣL/3 + ΣL²/9
//   Σb·u = ΣL·u/3   Σa·u = Σu − Σb·u
// clamped to [lim_lo, lim_hi], re-quantised and ordered. Returns (c0, c1)
// unchanged when every pixel sits on ONE level (16·ΣL² == (ΣL)², singular).
fn solve(m: Moments, c0: u32, c1: u32, lim_lo: vec3<f32>, lim_hi: vec3<f32>) -> vec2<u32> {
  if (16.0 * m.sLL == m.sL * m.sL) { return vec2<u32>(c0, c1); }
  let sBB = m.sLL * (1.0 / 9.0);
  let sAB = m.sL * (1.0 / 3.0) - sBB;
  let sAA = 16.0 - m.sL * (2.0 / 3.0) + sBB;
  let det = sAA * sBB - sAB * sAB;
  let p0 = from565(c0);
  let sBu = m.sLu * (1.0 / 3.0);
  let sAu = m.sU - sBu;
  let e0 = clamp(p0 + (sBB * sAu - sAB * sBu) / det, lim_lo, lim_hi);
  let e1 = clamp(p0 + (sAA * sBu - sAB * sAu) / det, lim_lo, lim_hi);
  return order565(to565(e0), to565(e1));
}

// Solid-colour channel code: the pair (a, b) of `bits`-bit codes whose ⅔/⅓
// interpolant (2·dec(a) + dec(b))/3 — palette index 2 — lands nearest v
// (8-bit units). See bc1_fast_f16.wgsl.
fn solid_pair(v: f32, bits: u32) -> vec2<u32> {
  let maxc = (1u << bits) - 1u;
  let q = min(u32(v * f32(maxc) / 255.0), maxc - 1u);
  var x: f32; var y: f32;
  if (bits == 5u) {
    x = f32((q * 527u + 23u) >> 6u);
    y = f32(((q + 1u) * 527u + 23u) >> 6u);
  } else {
    x = f32((q * 259u + 33u) >> 6u);
    y = f32(((q + 1u) * 259u + 33u) >> 6u);
  }
  var best = vec2<u32>(q, q);
  var be = abs(x - v);
  let c1 = (2.0 * x + y) / 3.0;
  if (abs(c1 - v) < be) { be = abs(c1 - v); best = vec2<u32>(q, q + 1u); }
  let c2 = (x + 2.0 * y) / 3.0;
  if (abs(c2 - v) < be) { be = abs(c2 - v); best = vec2<u32>(q + 1u, q); }
  if (abs(y - v) < be) { best = vec2<u32>(q + 1u, q + 1u); }
  return best;
}

// Principal colour axis via covariance power-iteration, seeded with the bbox
// diagonal. Returns a unit axis, or vec3(0) for a degenerate (constant)
// block. The bbox diagonal alone is sign-blind and points across
// anti-correlated data (normal maps, hue edges) instead of along it.
fn principal_axis(
  pixels: ptr<function, array<vec3<f32>, 16>>,
  mean: vec3<f32>,
  seed: vec3<f32>,
) -> vec3<f32> {
  // Symmetric 3x3 covariance, stored as its three rows.
  var c0v = vec3<f32>(0.0);
  var c1v = vec3<f32>(0.0);
  var c2v = vec3<f32>(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let d = (*pixels)[k] - mean;
    c0v = c0v + d.x * d;
    c1v = c1v + d.y * d;
    c2v = c2v + d.z * d;
  }
  var v = seed;
  var len = length(v);
  if (len < 1e-9) { return vec3<f32>(0.0); }
  v = v / len;
  for (var iter: u32 = 0u; iter < 8u; iter = iter + 1u) {
    let nv = vec3<f32>(dot(c0v, v), dot(c1v, v), dot(c2v, v));
    len = length(nv);
    if (len < 1e-12) { return vec3<f32>(0.0); }
    v = nv / len;
  }
  return v;
}

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid_raw: vec3<u32>) {
  // Row-band encodes dispatch a slice of the block grid starting at row y0.
  let gid = vec3<u32>(gid_raw.x, gid_raw.y + params.y0, gid_raw.z);
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) {
    return;
  }

  let bx = gid.x;
  let by = gid.y;
  let block_index = by * params.blocks_x + bx;

  let base = vec2<i32>(i32(bx) * 4, i32(by) * 4);
  let max_xy = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  var pixels: array<vec3<f32>, 16>;
  var bb_min = vec3<f32>(1.0, 1.0, 1.0);
  var bb_max = vec3<f32>(0.0, 0.0, 0.0);
  var mean = vec3<f32>(0.0);
  var gd = 0.0;

  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let lx = i32(i & 3u);
    let ly = i32(i >> 2u);
    // Clamp to edge for non-multiple-of-4 textures.
    let p  = clamp(base + vec2<i32>(lx, ly), vec2<i32>(0, 0), max_xy);
    let c  = textureLoad(src_tex, p, 0).rgb;
    pixels[i] = c;
    bb_min = min(bb_min, c);
    bb_max = max(bb_max, c);
    mean = mean + c;
    gd = max(gd, max(abs(c.x - c.y), abs(c.x - c.z)));
  }
  mean = mean * (1.0 / 16.0);
  // Exactly-gray blocks free the refit from the bbox clamp (no hue to
  // protect; smooth gradients want endpoints outside the data range) —
  // see bc1_fast_f16.wgsl.
  let gray = gd == 0.0;
  let lim_lo = select(bb_min, vec3<f32>(0.0), gray);
  let lim_hi = select(bb_max, vec3<f32>(1.0), gray);

  // Near-flat blocks (every channel within 3 levels): solid colour at the
  // block mean, per channel the endpoint pair whose ⅔/⅓ interpolant lands
  // nearest; they share the index pass below and skip the seed + refits.
  let span = bb_max - bb_min;
  let flat = max(max(span.x, span.y), span.z) <= 3.0 / 255.0;
  var c0: u32;
  var c1: u32;
  if (flat) {
    let m8 = mean * 255.0;
    let pr = solid_pair(m8.x, 5u);
    let pg = solid_pair(m8.y, 6u);
    let pb = solid_pair(m8.z, 5u);
    let s0 = (pr.x << 11u) | (pg.x << 5u) | pb.x;
    let s1 = (pr.y << 11u) | (pg.y << 5u) | pb.y;
    c0 = max(s0, s1);
    c1 = min(s0, s1);
  } else {
    // Seed endpoints from the block's principal colour axis at the exact
    // projection extents, inset by ~half a 565 cell along the axis
    // (stb_dxt heuristic). Degenerate blocks keep the inset-bbox seed.
    var seed_hi: vec3<f32>;
    var seed_lo: vec3<f32>;
    let axis = principal_axis(&pixels, mean, bb_max - bb_min);
    if (dot(axis, axis) > 0.0) {
      var t_min: f32 = 1e30;
      var t_max: f32 = -1e30;
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let t = dot(pixels[k] - mean, axis);
        t_min = min(t_min, t);
        t_max = max(t_max, t);
      }
      let pad = (t_max - t_min) / 16.0;
      seed_hi = clamp(mean + (t_max - pad) * axis, vec3<f32>(0.0), vec3<f32>(1.0));
      seed_lo = clamp(mean + (t_min + pad) * axis, vec3<f32>(0.0), vec3<f32>(1.0));
    } else {
      let inset = (bb_max - bb_min) / 16.0;
      seed_hi = clamp(bb_max - inset, vec3<f32>(0.0), vec3<f32>(1.0));
      seed_lo = clamp(bb_min + inset, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let seed = order565(to565(seed_hi), to565(seed_lo));
    c0 = seed.x;
    c1 = seed.y;
  }

  // Projection pass on the seed, then up to two least-squares refit rounds
  // (solve() off the previous pass's moments), each re-projected and
  // accepted only if the block error drops — the refit minimises a
  // continuous objective and can lose after 565 quantisation. Equal flat
  // codes encode the colour itself: index 0 (opaque in either mode).
  var indices = 0u;
  if (c0 != c1) {
    var cur = moments(&pixels, c0, c1);
    for (var it: u32 = 0u; it < select(2u, 0u, flat); it = it + 1u) {
      let cand = solve(cur, c0, c1, lim_lo, lim_hi);
      if (cand.x == c0 && cand.y == c1) { break; }
      let nxt = moments(&pixels, cand.x, cand.y);
      if (nxt.err >= cur.err) { break; }
      c0 = cand.x;
      c1 = cand.y;
      cur = nxt;
    }
    indices = cur.indices;
  }

  let out = block_index * 2u;
  dst[out]      = c0 | (c1 << 16u);
  dst[out + 1u] = indices;
}
