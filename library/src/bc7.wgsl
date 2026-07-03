// BC7 (BPTC) mode 6 compute shader encoder.
//
// One invocation per 4×4 block. Emits 16 bytes = 4 u32s into the storage
// buffer at `dst[block_index * 4 .. + 3]`. This is the f32 fallback;
// bc7_fast_f16.wgsl is the same algorithm and is preferred when the device
// reports shader-f16.
//
// ALGORITHM: principal-axis seed (covariance power-iteration; bbox on
// degenerate blocks) at the exact projection extents, quantised directly —
// no LSQ refit; with the seed on the principal axis, mode 6's 16-level
// palette leaves the refit under 0.15 dB, unlike the 4-level BC1/ASTC
// encoders which keep theirs — then one pass that projects each pixel onto
// the endpoint line (the 16 palette entries are colinear, so the nearest
// index is the rounded projection — no palette build, no 16-entry search),
// packed on the fly into two nibble words.
//
// A MODE 1 (2-subset) candidate was built and evaluated (2026-07) and
// dropped: ~+1.3 dB on multi-modal content but up to ~3× the pass cost on
// exactly that content — see bc7_fast_f16.wgsl. The CPU reference decoder
// keeps mode 1 support (bc7_ref.ts).
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

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn to8(v: vec4<f32>) -> vec4<i32> {
  return vec4<i32>(clamp(floor(v * 255.0 + 0.5), vec4<f32>(0.0), vec4<f32>(255.0)));
}

fn dist2(a: vec4<i32>, b: vec4<i32>) -> i32 {
  let d = a - b;
  let e = d * d;
  return e.x + e.y + e.z + e.w;
}

// Quantize an 8-bit ideal endpoint to (7-bit value, reconstructed 8-bit)
// under a fixed p-bit, all four channels at once. q7 = round((ideal8 − p)/2).
struct QuantPair { seven: vec4<i32>, eight: vec4<i32> };
fn quantize_endpoint(ideal8: vec4<i32>, p: u32) -> QuantPair {
  let q = vec4<i32>(clamp(
    floor((vec4<f32>(ideal8) - f32(p)) / 2.0 + 0.5),
    vec4<f32>(0.0), vec4<f32>(127.0),
  ));
  let eff = (q << vec4<u32>(1u)) | vec4<i32>(i32(p));
  return QuantPair(q, eff);
}

// Endpoint with its chosen p-bit, picked by minimum quantisation error.
struct Ep { seven: vec4<i32>, eight: vec4<i32>, p: u32 };
fn pick_ep(ideal: vec4<i32>) -> Ep {
  let a = quantize_endpoint(ideal, 0u);
  let b = quantize_endpoint(ideal, 1u);
  if (dist2(b.eight, ideal) < dist2(a.eight, ideal)) { return Ep(b.seven, b.eight, 1u); }
  return Ep(a.seven, a.eight, 0u);
}

// Principal colour axis via power-iteration over precomputed, mean-corrected
// covariance rows (the moments are accumulated for free in the pixel-load
// loop), seeded with the bbox diagonal. Returns a unit axis, or vec4(0) for
// a degenerate (constant) block. Same family as bc1.wgsl's principal_axis —
// the bbox diagonal alone is sign-blind and points across anti-correlated
// data (normal maps, hue edges) instead of along it.
fn principal_axis4(
  c0v: vec4<f32>,
  c1v: vec4<f32>,
  c2v: vec4<f32>,
  c3v: vec4<f32>,
  seed: vec4<f32>,
) -> vec4<f32> {
  var v = seed;
  var len = length(v);
  if (len < 1e-9) { return vec4<f32>(0.0); }
  v = v / len;
  for (var iter: u32 = 0u; iter < 8u; iter = iter + 1u) {
    let nv = vec4<f32>(dot(c0v, v), dot(c1v, v), dot(c2v, v), dot(c3v, v));
    len = length(nv);
    if (len < 1e-12) { return vec4<f32>(0.0); }
    v = nv / len;
  }
  return v;
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

  // Load 16 RGBA pixels (8-bit integer domain) and the per-channel bbox,
  // with the covariance moments FUSED in: d = px − pixel0 (first-pixel-
  // relative, so the sums scale with the block's span; d is integer-valued
  // and ≤255, exact in f32).
  var pixels: array<vec4<i32>, 16>;
  var lo = vec4<i32>(255);
  var hi = vec4<i32>(0);
  var gd = 0;
  var p0f = vec4<f32>(0.0);
  var sd = vec4<f32>(0.0);
  var c0v = vec4<f32>(0.0);
  var c1v = vec4<f32>(0.0);
  var c2v = vec4<f32>(0.0);
  var c3v = vec4<f32>(0.0);
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let lx = i32(i & 3u);
    let ly = i32(i >> 2u);
    let p  = clamp(base + vec2<i32>(lx, ly), vec2<i32>(0, 0), max_xy);
    let px = to8(textureLoad(src_tex, p, 0));
    pixels[i] = px;
    lo = min(lo, px);
    hi = max(hi, px);
    gd = max(gd, max(abs(px.x - px.y), abs(px.x - px.z)));
    if (i == 0u) { p0f = vec4<f32>(px); }
    let d = vec4<f32>(px) - p0f;
    sd = sd + d;
    c0v = c0v + d.x * d;
    c1v = c1v + d.y * d;
    c2v = c2v + d.z * d;
    c3v = c3v + d.w * d;
  }
  let mean = p0f + sd * (1.0 / 16.0);

  // Seed endpoints from the block's principal colour axis at the exact
  // projection extents (see header), quantise, and assign indices in one
  // projection pass.
  // Mean-correct the fused moments: C = Σddᵀ − (Σd)(Σd)ᵀ/16.
  let sd16 = sd * (1.0 / 16.0);
  let r0v = c0v - sd.x * sd16;
  let r1v = c1v - sd.y * sd16;
  let r2v = c2v - sd.z * sd16;
  let r3v = c3v - sd.w * sd16;
  var seed0 = lo;
  var seed1 = hi;
  // Gray + opaque blocks: the axis is analytically (1,1,1,0)/√3 with
  // extents at the luma min/max — skip iteration + extents pass entirely
  // (see bc7_fast_f16.wgsl).
  let gray = lo.w == 255 && gd == 0;
  if (gray) {
    seed0 = vec4<i32>(lo.x, lo.x, lo.x, 255);
    seed1 = vec4<i32>(hi.x, hi.x, hi.x, 255);
  } else {
    let axis = principal_axis4(r0v, r1v, r2v, r3v, vec4<f32>(hi - lo));
    if (dot(axis, axis) > 0.0) {
      // Exact projection extents along the axis. (A Rayleigh-quotient span
      // estimate was tried in place of this pass — it saves 16 dots but
      // costs 0.1–0.8 dB and 4–10× on the worst-easy-block gate: σ
      // misjudges two-cluster and outlier blocks. The pass stays.)
      var t_min: f32 = 1e30;
      var t_max: f32 = -1e30;
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let t = dot(vec4<f32>(pixels[k]) - mean, axis);
        t_min = min(t_min, t);
        t_max = max(t_max, t);
      }
      seed0 = vec4<i32>(clamp(round(mean + t_min * axis), vec4<f32>(0.0), vec4<f32>(255.0)));
      seed1 = vec4<i32>(clamp(round(mean + t_max * axis), vec4<f32>(0.0), vec4<f32>(255.0)));
    }
  }

  // The 16 4-bit indices, packed LSB-first into two nibble words
  // (pixel k → bits 4k..4k+3).
  var ilo: u32 = 0u;
  var ihi: u32 = 0u;
  var ep0 = pick_ep(seed0);
  var ep1 = pick_ep(seed1);
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
  var e0_7 = ep0.seven;
  var e1_7 = ep1.seven;
  var p0 = ep0.p;
  var p1 = ep1.p;

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
