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
// ALGORITHM: principal-axis endpoint seed (covariance power-iteration; inset
// bbox on degenerate blocks), inset by ~half a 565 cell along the axis, then
// a fused pass that projects every pixel onto the decoded-endpoint line (the
// 4 palette entries are colinear and evenly spaced, so the nearest entry is
// the rounded projection — no 4-entry search) while accumulating the
// least-squares refit sums, followed by up to TWO refit rounds (re-quantise,
// reproject with indices packed on the fly, accept only on lower block
// error).

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
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

// One projection pass against the decoded endpoints of (c0,c1): the packed
// 2-bit indices, the block's squared error, and the LSQ normal-equation sums
// of the resulting assignment — so an accepted refit can seed the next
// round. Levels s run 0..3 along p0→p1 (palette = p0, p0+⅓d, p0+⅔d, p1 —
// colinear, evenly spaced, so rounding the projection IS the nearest-entry
// search). Level → BC1 index: 0→0 (c0), 1→2 (⅔c0+⅓c1), 2→3, 3→1 (c1); as a
// packed LUT: (0x78 >> 2L) & 3.
struct ProjStats {
  indices: u32,
  err: f32,
  sAA: f32, sBB: f32, sAB: f32,
  sAV: vec3<f32>, sBV: vec3<f32>,
  s_min: f32, s_max: f32,
};
fn project_stats(pix: ptr<function, array<vec3<f32>, 16>>, c0: u32, c1: u32) -> ProjStats {
  var out: ProjStats;
  out.indices = 0u;
  out.err = 0.0;
  out.sAA = 0.0; out.sBB = 0.0; out.sAB = 0.0;
  out.sAV = vec3<f32>(0.0); out.sBV = vec3<f32>(0.0);
  out.s_min = 3.0; out.s_max = 0.0;
  let p0 = from565(c0);
  let p1 = from565(c1);
  let dir = p1 - p0;
  let dd = dot(dir, dir);
  if (dd == 0.0) {
    // Unreachable for distinct 565 codes (the decode is injective); kept so
    // a degenerate call still returns a consistent error.
    out.s_min = 0.0;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let e = (*pix)[k] - p0;
      out.err = out.err + dot(e, e);
    }
    return out;
  }
  let inv = 3.0 / dd;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let v = (*pix)[k];
    let s = clamp(floor(dot(v - p0, dir) * inv + 0.5), 0.0, 3.0);
    out.s_min = min(out.s_min, s); out.s_max = max(out.s_max, s);
    let b = s * (1.0 / 3.0); let a = 1.0 - b;
    out.sAA = out.sAA + a * a; out.sBB = out.sBB + b * b; out.sAB = out.sAB + a * b;
    out.sAV = out.sAV + a * v; out.sBV = out.sBV + b * v;
    let e = v - (p0 + b * dir);
    out.err = out.err + dot(e, e);
    out.indices = out.indices | (((0x78u >> (u32(s) * 2u)) & 3u) << (k * 2u));
  }
  return out;
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
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
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

  // Seed endpoints from the block's principal colour axis at the exact
  // projection extents, inset by ~half a 565 cell along the axis (stb_dxt
  // heuristic). Degenerate (near-flat) blocks keep the inset-bbox seed.
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
  var c0 = to565(seed_hi);
  var c1 = to565(seed_lo);
  if (c0 == c1) {
    if (c1 > 0u) { c1 = c1 - 1u; } else { c0 = c0 + 1u; }
  } else if (c0 < c1) {
    let t = c0; c0 = c1; c1 = t;
  }

  // Fused seed pass, then up to TWO least-squares refit rounds, each
  // accepted only if the block's squared error actually decreases — the
  // refit minimises a continuous objective and can lose after 565
  // quantisation. Every pass re-accumulates the normal-equation sums, so an
  // accepted round seeds the next.
  var cur = project_stats(&pixels, c0, c1);
  for (var it: u32 = 0u; it < 2u; it = it + 1u) {
    // Refit only on a well-conditioned system: when every pixel lands on
    // ONE level (flat blocks — the 4-colour nudge forces c0 ≠ c1 even
    // then) the system is rank-1 and det/numerators are pure float noise;
    // the solve would return garbage endpoints. With ≥2 levels
    // det = Σ_i<j (b_j − b_i)² ≥ ~1.67, so 1e-3 is a safe guard.
    if (cur.s_min >= cur.s_max) { break; }
    let det = cur.sAA * cur.sBB - cur.sAB * cur.sAB;
    if (abs(det) <= 1e-3) { break; }
    // Clamp the refit to the block bbox (not [0,1]): on multi-cluster
    // blocks the unconstrained solve extrapolates far outside the block's
    // colours and the per-channel clamp then bends the hue — fringe pixels
    // decode to colours that exist nowhere in the block. Constraining to
    // the bbox also measures better in plain SSE (+1.6 dB on the colour
    // test card), so the accept-if-better guard below keeps more refits.
    let e0 = clamp((cur.sBB * cur.sAV - cur.sAB * cur.sBV) / det, lim_lo, lim_hi);
    let e1 = clamp((cur.sAA * cur.sBV - cur.sAB * cur.sAV) / det, lim_lo, lim_hi);
    var nc0 = to565(e0);
    var nc1 = to565(e1);
    if (nc0 == nc1) {
      if (nc1 > 0u) { nc1 = nc1 - 1u; } else { nc0 = nc0 + 1u; }
    } else if (nc0 < nc1) {
      let t = nc0; nc0 = nc1; nc1 = t;
    }
    if (nc0 == c0 && nc1 == c1) { break; }
    let nxt = project_stats(&pixels, nc0, nc1);
    if (nxt.err >= cur.err) { break; }
    c0 = nc0;
    c1 = nc1;
    cur = nxt;
  }

  let out = block_index * 2u;
  dst[out]      = c0 | (c1 << 16u);
  dst[out + 1u] = cur.indices;
}
