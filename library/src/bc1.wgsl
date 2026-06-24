// BC1 (DXT1) compute shader encoder.
//
// Each invocation encodes one 4x4 pixel block into an 8-byte BC1 block
// written as 2 x u32 into the destination storage buffer.
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
// QUALITY LEVELS (pipeline-overridable constant `QUALITY_HIGH`)
//   fast (0, default): bounding-box endpoints, inset by ~half a 565 cell, then
//     a single least-squares endpoint refit (the refit is accepted only if it
//     lowers the block's squared error). This is what the WebGL2 fragment
//     fallback runs too.
//   high (1): endpoints are seeded from the block's principal colour axis
//     (covariance power-iteration) as well as the bbox diagonal, each refined by
//     several least-squares passes; the lower-error family wins. Mirrors
//     bc1_ref.ts. Strictly ≥ fast in quality, at the cost of the eigen-solve.
//
// Algorithm per block:
//  1. Load the 16 pixels; compute the bounding box (and, for high, the mean).
//  2. Seed endpoints (bbox diagonal; high also tries the principal axis).
//  3. Quantize to RGB565, force 4-color mode, assign each pixel its nearest
//     palette entry (full 4-entry L2 search in the decoded colour space).
//  4. Least-squares refit: re-solve the endpoints for the current indices,
//     re-quantize, re-assign; keep the result only when error decreases.

// 0 = fast (default), 1 = high-quality. Set via pipeline constants.
override QUALITY_HIGH: u32 = 0u;

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
  let r = f32((c >> 11u) & 31u);
  let g = f32((c >>  5u) & 63u);
  let b = f32( c         & 31u);
  // 5/6-bit -> 8-bit. floor((x*527+23)/64) == (x<<3)|(x>>2), i.e. the exact
  // bit-replication a BC1 decoder performs (white -> 255). The inputs are
  // small integers and /64 is exact in f32, so this matches the hardware and
  // is portable. Selecting indices against this palette is what makes the
  // encoder agree with what the GPU will actually sample.
  let r8 = floor((r * 527.0 + 23.0) / 64.0);
  let g8 = floor((g * 259.0 + 33.0) / 64.0);
  let b8 = floor((b * 527.0 + 23.0) / 64.0);
  return vec3<f32>(r8, g8, b8) / 255.0;
}

// 4-color-mode interpolation weights: palette[j] = wa(j)*c0 + wb(j)*c1.
fn wa(j: u32) -> f32 {
  switch j {
    case 0u:  { return 1.0; }
    case 1u:  { return 0.0; }
    case 2u:  { return 2.0 / 3.0; }
    default:  { return 1.0 / 3.0; }   // case 3u
  }
}
fn wb(j: u32) -> f32 {
  switch j {
    case 0u:  { return 0.0; }
    case 1u:  { return 1.0; }
    case 2u:  { return 1.0 / 3.0; }
    default:  { return 2.0 / 3.0; }   // case 3u
  }
}

fn build_palette(c0: u32, c1: u32, pal: ptr<function, array<vec3<f32>, 4>>) {
  let p0 = from565(c0);
  let p1 = from565(c1);
  for (var j: u32 = 0u; j < 4u; j = j + 1u) {
    (*pal)[j] = wa(j) * p0 + wb(j) * p1;
  }
}

// Assign each of the 16 pixels its nearest palette entry (full 4-entry L2),
// writing indices into `out_idx` and returning the total squared error.
fn assign_indices(
  pixels:  ptr<function, array<vec3<f32>, 16>>,
  pal:     ptr<function, array<vec3<f32>, 4>>,
  out_idx: ptr<function, array<u32, 16>>,
) -> f32 {
  var err: f32 = 0.0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let c = (*pixels)[k];
    var best_j: u32 = 0u;
    var best_d: f32 = 1e30;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      let d = (*pal)[j] - c;
      let d2 = dot(d, d);
      if (d2 < best_d) {
        best_d = d2;
        best_j = j;
      }
    }
    (*out_idx)[k] = best_j;
    err = err + best_d;
  }
  return err;
}

// One least-squares refit pass: solve the 2x2 normal equations for the endpoint
// colours that minimise Σ‖wa·e0 + wb·e1 − c‖² under the current indices. The
// three channels share the scalar sums, so it's one 2x2 solve with vec3 RHS.
struct RefitResult { e0: vec3<f32>, e1: vec3<f32>, valid: bool };
fn refit(
  pixels:  ptr<function, array<vec3<f32>, 16>>,
  indices: ptr<function, array<u32, 16>>,
) -> RefitResult {
  var sAA: f32 = 0.0; var sBB: f32 = 0.0; var sAB: f32 = 0.0;
  var sAV: vec3<f32> = vec3<f32>(0.0);
  var sBV: vec3<f32> = vec3<f32>(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let a = wa((*indices)[k]);
    let b = wb((*indices)[k]);
    let v = (*pixels)[k];
    sAA = sAA + a * a;
    sBB = sBB + b * b;
    sAB = sAB + a * b;
    sAV = sAV + a * v;
    sBV = sBV + b * v;
  }
  var out: RefitResult;
  let det = sAA * sBB - sAB * sAB;
  if (abs(det) < 1e-9) {
    out.valid = false;
    return out;
  }
  out.e0 = clamp((sBB * sAV - sAB * sBV) / det, vec3<f32>(0.0), vec3<f32>(1.0));
  out.e1 = clamp((sAA * sBV - sAB * sAV) / det, vec3<f32>(0.0), vec3<f32>(1.0));
  out.valid = true;
  return out;
}

// Candidate solution tracked across endpoint seeds / refit passes.
struct Best { c0: u32, c1: u32, indices: array<u32, 16>, err: f32 };

// Quantize (hi, lo) to 565, force 4-color mode, assign indices, then refine with
// up to `max_refits` least-squares passes. Commits to `*best` only on strict
// improvement.
fn fit_from_endpoints(
  pixels: ptr<function, array<vec3<f32>, 16>>,
  hi: vec3<f32>,
  lo: vec3<f32>,
  max_refits: u32,
  best: ptr<function, Best>,
) {
  var c0 = to565(hi);
  var c1 = to565(lo);
  // 4-color mode requires color0 > color1.
  if (c0 == c1) {
    if (c1 > 0u) { c1 = c1 - 1u; } else { c0 = c0 + 1u; }
  } else if (c0 < c1) {
    let t = c0; c0 = c1; c1 = t;
  }

  var pal: array<vec3<f32>, 4>;
  var idx: array<u32, 16>;
  build_palette(c0, c1, &pal);
  var err = assign_indices(pixels, &pal, &idx);
  if (err < (*best).err) {
    (*best).c0 = c0; (*best).c1 = c1; (*best).indices = idx; (*best).err = err;
  }

  for (var rp: u32 = 0u; rp < max_refits; rp = rp + 1u) {
    let r = refit(pixels, &idx);
    if (!r.valid) { break; }
    var nc0 = to565(r.e0);
    var nc1 = to565(r.e1);
    // A refit that flips/equalises the endpoints would change decode mode;
    // keep 4-color mode, and stop once it stops moving.
    if (nc0 < nc1) { let t = nc0; nc0 = nc1; nc1 = t; }
    if (nc0 == nc1) { break; }
    if (nc0 == c0 && nc1 == c1) { break; }
    build_palette(nc0, nc1, &pal);
    let nerr = assign_indices(pixels, &pal, &idx);
    c0 = nc0; c1 = nc1; err = nerr;
    if (nerr < (*best).err) {
      (*best).c0 = nc0; (*best).c1 = nc1; (*best).indices = idx; (*best).err = nerr;
    }
  }
}

// Principal colour axis via covariance power-iteration, seeded with the bbox
// diagonal. Returns a unit axis, or vec3(0) for a degenerate (constant) block.
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
  }
  mean = mean * (1.0 / 16.0);

  // Inset the bounding box by ~half an RGB565 cell (1/16) so the quantized
  // 4-color palette covers the real data range more tightly (stb_dxt heuristic).
  let inset = (bb_max - bb_min) / 16.0;
  let bbox_hi = clamp(bb_max - inset, vec3<f32>(0.0), vec3<f32>(1.0));
  let bbox_lo = clamp(bb_min + inset, vec3<f32>(0.0), vec3<f32>(1.0));

  var best: Best;
  best.err = 1e30;

  if (QUALITY_HIGH != 0u) {
    // Seed from the principal colour axis: project all texels onto it, take the
    // extreme projections as endpoints, inset along the axis. Then also try the
    // bbox seed and keep whichever family yields the lower error.
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
      let pca_hi = clamp(mean + (t_max - pad) * axis, vec3<f32>(0.0), vec3<f32>(1.0));
      let pca_lo = clamp(mean + (t_min + pad) * axis, vec3<f32>(0.0), vec3<f32>(1.0));
      fit_from_endpoints(&pixels, pca_hi, pca_lo, 3u, &best);
    }
    fit_from_endpoints(&pixels, bbox_hi, bbox_lo, 3u, &best);
  } else {
    fit_from_endpoints(&pixels, bbox_hi, bbox_lo, 1u, &best);
  }

  var indices: u32 = 0u;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    indices = indices | ((best.indices[k] & 3u) << (k * 2u));
  }

  let out = block_index * 2u;
  dst[out]      = best.c0 | (best.c1 << 16u);
  dst[out + 1u] = indices;
}
