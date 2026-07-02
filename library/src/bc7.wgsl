// BC7 (BPTC) mode 6 compute shader encoder.
//
// One invocation per 4×4 block. Emits 16 bytes = 4 u32s into the storage
// buffer at `dst[block_index * 4 .. + 3]`.
//
// QUALITY LEVELS (pipeline-overridable constant `QUALITY_HIGH`)
//   fast (0, default): O(N) bounding-box seed → one fused pass that projects
//     each pixel onto the endpoint line (the 16 palette entries are colinear,
//     so the nearest index is the rounded projection — no palette build, no
//     16-entry search) while accumulating the least-squares refit sums, then
//     a reprojection against the quantised refit endpoints for the final
//     indices, packed on the fly into two nibble words.
//   high (1): farthest-pair seed, exhaustive p-bit search over all four
//     (p0,p1) ∈ {0,1}² combos, full 16-entry nearest search, one LSQ refit —
//     matches bc7_ref.ts up to FP tie-breaks.
//
// The fast/high branch is selected at pipeline-compile time, so the driver
// eliminates the unused code entirely.
//
// MODE 6 LAYOUT (LSB-first, bit 0 = byte 0's bit 0)
//   bits 0..6    mode field      (0b0000001 — only bit 6 is 1)
//   bits 7..13   R0 (7-bit)   bits 14..20 R1   bits 21..27 G0   bits 28..34 G1
//   bits 35..41  B0   bits 42..48 B1   bits 49..55 A0   bits 56..62 A1
//   bit  63      P0   bit 64 P1
//   bits 65..67  pixel 0 index (3 bits; anchor, MSB implicit 0)
//   bits 68..71  pixel 1 index (4 bits) ... bits 124..127 pixel 15 index
//
// Effective 8-bit endpoint channel = (7_bit_value << 1) | p_bit.
// Palette[i] = ((64 − W4[i]) × e0_8 + W4[i] × e1_8 + 32) >> 6, integer.
//
// The block is assembled with straight-line constant shifts (see the layout
// summary in bc7_fast_f16.wgsl) — a generic write_bits() helper's dynamic
// word indexing keeps the output array out of registers.

// 0 = fast (default), 1 = exhaustive/high-quality. Set via pipeline constants.
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

// Mode 6 interpolation weights (× 1/64), fixed by the spec (`W4` in bc7_ref.ts).
fn w4(i: u32) -> u32 {
  switch i {
    case 0u:  { return  0u; }
    case 1u:  { return  4u; }
    case 2u:  { return  9u; }
    case 3u:  { return 13u; }
    case 4u:  { return 17u; }
    case 5u:  { return 21u; }
    case 6u:  { return 26u; }
    case 7u:  { return 30u; }
    case 8u:  { return 34u; }
    case 9u:  { return 38u; }
    case 10u: { return 43u; }
    case 11u: { return 47u; }
    case 12u: { return 51u; }
    case 13u: { return 55u; }
    case 14u: { return 60u; }
    default:  { return 64u; }   // case 15u
  }
}

fn interp4(e0: vec4<i32>, e1: vec4<i32>, w: i32) -> vec4<i32> {
  return ((64 - w) * e0 + w * e1 + vec4<i32>(32)) >> vec4<u32>(6u);
}

fn to8(v: vec4<f32>) -> vec4<i32> {
  return vec4<i32>(clamp(floor(v * 255.0 + 0.5), vec4<f32>(0.0), vec4<f32>(255.0)));
}

fn dist2(a: vec4<i32>, b: vec4<i32>) -> i32 {
  let d = a - b;
  let e = d * d;
  return e.x + e.y + e.z + e.w;
}

// Quantize an 8-bit ideal endpoint to (7-bit value, reconstructed 8-bit) under
// a fixed p-bit, all four channels at once. q7 = round((ideal8 − p)/2); used by
// both paths.
struct QuantPair { seven: vec4<i32>, eight: vec4<i32> };
fn quantize_endpoint(ideal8: vec4<i32>, p: u32) -> QuantPair {
  let q = vec4<i32>(clamp(
    floor((vec4<f32>(ideal8) - f32(p)) / 2.0 + 0.5),
    vec4<f32>(0.0), vec4<f32>(127.0),
  ));
  let eff = (q << vec4<u32>(1u)) | vec4<i32>(i32(p));
  return QuantPair(q, eff);
}

// ============================ FAST PATH ================================ //

// Endpoint with its chosen p-bit, picked by minimum quantisation error.
struct Ep { seven: vec4<i32>, eight: vec4<i32>, p: u32 };
fn pick_ep(ideal: vec4<i32>) -> Ep {
  let a = quantize_endpoint(ideal, 0u);
  let b = quantize_endpoint(ideal, 1u);
  if (dist2(b.eight, ideal) < dist2(a.eight, ideal)) { return Ep(b.seven, b.eight, 1u); }
  return Ep(a.seven, a.eight, 0u);
}

// One pass over the block: project every pixel onto the e0→e1 line and
// accumulate the least-squares normal-equation sums; solve for the refit
// endpoints (in 8-bit space). Indices are not produced here — the caller
// reprojects against the quantised refit endpoints anyway.
struct Fit { e0: vec4<i32>, e1: vec4<i32>, valid: bool };
fn proj_fit(pixels: ptr<function, array<vec4<i32>, 16>>, e0: vec4<i32>, e1: vec4<i32>) -> Fit {
  var out: Fit;
  out.valid = false;
  let dir = vec4<f32>(e1 - e0);
  let dd = dot(dir, dir);
  if (dd == 0.0) { return out; }
  let e0f = vec4<f32>(e0);
  let inv = 15.0 / dd;
  var sAA: f32 = 0.0; var sBB: f32 = 0.0; var sAB: f32 = 0.0;
  var sAV: vec4<f32> = vec4<f32>(0.0); var sBV: vec4<f32> = vec4<f32>(0.0);
  var s_min = 15.0; var s_max = 0.0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let v = vec4<f32>((*pixels)[k]);
    let s = clamp(floor(dot(v - e0f, dir) * inv + 0.5), 0.0, 15.0);
    s_min = min(s_min, s); s_max = max(s_max, s);
    let b = s * (1.0 / 15.0); let a = 1.0 - b;
    sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b;
    sAV = sAV + a * v; sBV = sBV + b * v;
  }
  // Rank-1 guard: if every pixel projects to ONE level the system is
  // singular — det and the numerators are pure float rounding noise and the
  // solve returns garbage endpoints. With ≥2 levels det ≥ 15/225 ≈ 0.067.
  if (s_min == s_max) { return out; }
  let det = sAA * sBB - sAB * sAB;
  if (abs(det) < 1e-3) { return out; }
  out.e0 = vec4<i32>(clamp(round((sBB * sAV - sAB * sBV) / det), vec4<f32>(0.0), vec4<f32>(255.0)));
  out.e1 = vec4<i32>(clamp(round((sAA * sBV - sAB * sAV) / det), vec4<f32>(0.0), vec4<f32>(255.0)));
  out.valid = true;
  return out;
}

// ============================ HIGH PATH ================================ //

struct Pair { a: vec4<i32>, b: vec4<i32> };
fn farthest_pair(pixels: ptr<function, array<vec4<i32>, 16>>) -> Pair {
  var best_d: i32 = 0;
  var pa = (*pixels)[0];
  var pb = (*pixels)[1];
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let xi = (*pixels)[i];
    for (var j: u32 = i + 1u; j < 16u; j = j + 1u) {
      let d = dist2(xi, (*pixels)[j]);
      if (d > best_d) { best_d = d; pa = xi; pb = (*pixels)[j]; }
    }
  }
  return Pair(pa, pb);
}

fn build_palette_6(e0: vec4<i32>, e1: vec4<i32>, pal: ptr<function, array<vec4<i32>, 16>>) {
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    (*pal)[i] = interp4(e0, e1, i32(w4(i)));
  }
}

fn assign_all(
  pixels:  ptr<function, array<vec4<i32>, 16>>,
  pal:     ptr<function, array<vec4<i32>, 16>>,
  out_idx: ptr<function, array<u32, 16>>,
) -> i32 {
  var err: i32 = 0;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let px = (*pixels)[k];
    var best_i: u32 = 0u;
    var best_d: i32 = 2147483647;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
      let d = dist2(px, (*pal)[i]);
      if (d < best_d) { best_d = d; best_i = i; }
    }
    (*out_idx)[k] = best_i;
    err = err + best_d;
  }
  return err;
}

struct BestMode6 {
  e0_7: vec4<i32>, e1_7: vec4<i32>,
  p0: u32,         p1: u32,
  indices: array<u32, 16>,
  err: i32,
};

// Exhaustive p-bit search (high path); commits to `*best` only on improvement.
fn try_pbit_combos(
  pixels: ptr<function, array<vec4<i32>, 16>>,
  ideal0: vec4<i32>,
  ideal1: vec4<i32>,
  best:   ptr<function, BestMode6>,
) {
  var local_best = (*best).err;
  var pal: array<vec4<i32>, 16>;
  var tmp: array<u32, 16>;
  for (var p0: u32 = 0u; p0 < 2u; p0 = p0 + 1u) {
    let q0 = quantize_endpoint(ideal0, p0);
    for (var p1: u32 = 0u; p1 < 2u; p1 = p1 + 1u) {
      let q1 = quantize_endpoint(ideal1, p1);
      build_palette_6(q0.eight, q1.eight, &pal);
      let err = assign_all(pixels, &pal, &tmp);
      if (err < local_best) {
        local_best = err;
        (*best).e0_7 = q0.seven;
        (*best).e1_7 = q1.seven;
        (*best).p0 = p0;
        (*best).p1 = p1;
        (*best).indices = tmp;
        (*best).err = err;
      }
    }
  }
}

// Exact-weight LSQ refit (high path); matches bc7_ref.ts `refitEndpointsMode6`.
struct RefitResult { e0: vec4<i32>, e1: vec4<i32>, valid: bool };
fn refit_endpoints(
  pixels: ptr<function, array<vec4<i32>, 16>>,
  indices: ptr<function, array<u32, 16>>,
) -> RefitResult {
  var sAA: f32 = 0.0;
  var sBB: f32 = 0.0;
  var sAB: f32 = 0.0;
  var sAV: vec4<f32> = vec4<f32>(0.0);
  var sBV: vec4<f32> = vec4<f32>(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let i = (*indices)[k];
    let a = f32(64u - w4(i)) / 64.0;
    let b = f32(w4(i)) / 64.0;
    let v = vec4<f32>((*pixels)[k]);
    sAA = sAA + a * a;
    sBB = sBB + b * b;
    sAB = sAB + a * b;
    sAV = sAV + a * v;
    sBV = sBV + b * v;
  }
  let det = sAA * sBB - sAB * sAB;
  var out: RefitResult;
  if (abs(det) < 1e-9) {
    out.valid = false;
    return out;
  }
  let e0f = (sBB * sAV - sAB * sBV) / det;
  let e1f = (sAA * sBV - sAB * sAV) / det;
  out.e0 = vec4<i32>(clamp(round(e0f), vec4<f32>(0.0), vec4<f32>(255.0)));
  out.e1 = vec4<i32>(clamp(round(e1f), vec4<f32>(0.0), vec4<f32>(255.0)));
  out.valid = true;
  return out;
}

// ------------------------------- Entry --------------------------------- //

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) {
    return;
  }

  let bx = gid.x;
  let by = gid.y;
  let block_index = by * params.blocks_x + bx;

  let base   = vec2<i32>(i32(bx) * 4, i32(by) * 4);
  let max_xy = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  // Load 16 RGBA pixels (8-bit integer domain) and the per-channel bbox.
  var pixels: array<vec4<i32>, 16>;
  var lo = vec4<i32>(255);
  var hi = vec4<i32>(0);
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let lx = i32(i & 3u);
    let ly = i32(i >> 2u);
    let p  = clamp(base + vec2<i32>(lx, ly), vec2<i32>(0, 0), max_xy);
    let px = to8(textureLoad(src_tex, p, 0));
    pixels[i] = px;
    lo = min(lo, px);
    hi = max(hi, px);
  }

  // Both branches produce: 7-bit endpoints + p-bits, and the 16 4-bit indices
  // packed LSB-first into two nibble words (pixel k → bits 4k..4k+3).
  var e0_7: vec4<i32>;
  var e1_7: vec4<i32>;
  var p0: u32;
  var p1: u32;
  var ilo: u32 = 0u;
  var ihi: u32 = 0u;

  if (QUALITY_HIGH != 0u) {
    let fp = farthest_pair(&pixels);
    var best: BestMode6;
    best.err = 2147483647;
    try_pbit_combos(&pixels, fp.a, fp.b, &best);
    let refit = refit_endpoints(&pixels, &best.indices);
    if (refit.valid) {
      try_pbit_combos(&pixels, refit.e0, refit.e1, &best);
    }
    e0_7 = best.e0_7; e1_7 = best.e1_7; p0 = best.p0; p1 = best.p1;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      ilo = ilo | (best.indices[k] << (k * 4u));
    }
    for (var k: u32 = 8u; k < 16u; k = k + 1u) {
      ihi = ihi | (best.indices[k] << ((k - 8u) * 4u));
    }
  } else {
    // Seed the fused LSQ fit from the raw bbox, then quantise the refit
    // endpoints and reproject for the final indices.
    let r = proj_fit(&pixels, lo, hi);
    var ep0: Ep;
    var ep1: Ep;
    if (r.valid) { ep0 = pick_ep(r.e0); ep1 = pick_ep(r.e1); }
    else         { ep0 = pick_ep(lo);   ep1 = pick_ep(hi);   }
    let dir = vec4<f32>(ep1.eight - ep0.eight);
    let dd = dot(dir, dir);
    if (dd > 0.0) {
      let e0f = vec4<f32>(ep0.eight);
      let inv = 15.0 / dd;
      for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let s = clamp(floor(dot(vec4<f32>(pixels[k]) - e0f, dir) * inv + 0.5), 0.0, 15.0);
        ilo = ilo | (u32(s) << (k * 4u));
      }
      for (var k: u32 = 8u; k < 16u; k = k + 1u) {
        let s = clamp(floor(dot(vec4<f32>(pixels[k]) - e0f, dir) * inv + 0.5), 0.0, 15.0);
        ihi = ihi | (u32(s) << ((k - 8u) * 4u));
      }
    }
    e0_7 = ep0.seven; e1_7 = ep1.seven; p0 = ep0.p; p1 = ep1.p;
  }

  // Anchor rule — pixel 0's index MSB must be 0. Swapping endpoints reflects
  // every index (i → 15−i), which on packed nibbles is a bitwise NOT.
  if ((ilo & 0x8u) != 0u) {
    let t7 = e0_7; e0_7 = e1_7; e1_7 = t7;
    let tp = p0;   p0   = p1;   p1   = tp;
    ilo = ~ilo; ihi = ~ihi;
  }

  // Straight-line mode-6 packing (see layout at the top of the file).
  let e0 = vec4<u32>(e0_7);
  let e1 = vec4<u32>(e1_7);
  let w0 = 0x40u | (e0.x << 7u) | (e1.x << 14u) | (e0.y << 21u) | (e1.y << 28u);
  let w1 = (e1.y >> 4u) | (e0.z << 3u) | (e1.z << 10u) | (e0.w << 17u) | (e1.w << 24u) | (p0 << 31u);
  let w2 = p1 | ((ilo & 0x7u) << 1u) | (ilo & 0xFFFFFFF0u);
  let w3 = ihi;

  let out = block_index * 4u;
  dst[out + 0u] = w0;
  dst[out + 1u] = w1;
  dst[out + 2u] = w2;
  dst[out + 3u] = w3;
}
