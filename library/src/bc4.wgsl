// BC4 (RGTC1) compute shader encoder.
//
// Each invocation encodes one 4x4 pixel block of the source's R channel into
// an 8-byte BC4 block written as 2 x u32 into the destination storage
// buffer. This is the f32 fallback; bc4_fast_f16.wgsl is the same algorithm
// and is preferred when the device reports shader-f16.
//
// BC4 block (8 bytes):
//   byte 0     : red0 (8-bit endpoint)
//   byte 1     : red1 (8-bit endpoint)
//   bytes 2..7 : 16 × 3-bit indices, LSB-first, pixel 0 at bit 0
//
// We always produce the 6-interpolation mode (red0 > red1). The algorithm is
// the single-channel half of bc5.wgsl — a BC4 block is byte-identical to the
// R half of the BC5 block for the same texels (see bc5_fast_f16.wgsl for the
// full notes):
//   • seed at the extremes (spans ≤ 7: a lossless 7-wide window); pass-1
//     levels come from that range inset by ~5.5/256 of the span and
//     accumulate MOMENTS (ΣL, ΣL², Σd, ΣL·d with d = v − r0);
//   • refit = the least-squares line through those levels straight off the
//     moments (den = 16ΣL² − (ΣL)² = 0 keeps the seed);
//   • pass 2 derives the levels once, against the refit endpoints, then an
//     offset round shifts both endpoints by the rounded mean residual;
//   • the 16 texel reads are 4 textureGather fetches through a clamp-to-edge
//     sampler, byte-identical to per-texel loads;
//   • 3-bit levels pack as Σ L·8^k in f32 (exact below 2^24), then one SWAR
//     level → BC4 index map (0→0, 7→1, L→L+1) per 24-bit word.
// Values are kept in the [0,255] f32 domain throughout.

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
@group(0) @binding(3) var smp: sampler;

// Float -> u32 for exact integers in [0, 2^23): x + 2^23 holds x in its
// mantissa, so bitcast(x + MAGIC) ^ MAGIC_BITS == x. WGSL's u32(float) is a
// SATURATING conversion (compares + selects around the convert).
const MAGIC = 8388608.0;
const MAGIC_BITS = 0x4B000000u;
fn fu(x: f32) -> u32 {
  return bitcast<u32>(x + MAGIC) ^ MAGIC_BITS;
}

// Edge blocks: one quad's R from four clamped texel loads, in gather order
// (x=(0,1) y=(1,1) z=(1,0) w=(0,0)).
fn load_quad(qo: vec2<i32>, mx: vec2<i32>) -> vec4<f32> {
  return vec4<f32>(
    textureLoad(src_tex, min(qo + vec2<i32>(0, 1), mx), 0).r,
    textureLoad(src_tex, min(qo + vec2<i32>(1, 1), mx), 0).r,
    textureLoad(src_tex, min(qo + vec2<i32>(1, 0), mx), 0).r,
    textureLoad(src_tex, min(qo, mx), 0).r);
}

// 8 packed 3-bit levels → BC4 indices (0→0, 7→1, L→L+1 otherwise).
fn lvl_to_idx(x: u32) -> u32 {
  let y = ((x & 0x6DB6DBu) + 0x249249u) ^ (x & 0x924924u);
  return y ^ (~((y >> 1u) | (y >> 2u)) & 0x249249u);
}

// Per-quad pixel weights 8^k (gather order x,y,z,w = (0,1),(1,1),(1,0),(0,0)).
const W0 = vec4<f32>(4096.0, 32768.0, 8.0, 1.0);
const W1 = vec4<f32>(262144.0, 2097152.0, 512.0, 64.0);

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid_raw: vec3<u32>) {
  // Row-band encodes dispatch a slice of the block grid starting at row y0.
  let gid = vec3<u32>(gid_raw.x, gid_raw.y + params.y0, gid_raw.z);
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) {
    return;
  }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);

  // Load 4×4 R as quad-major vec4s in gather order (w=(0,0) z=(1,0) x=(0,1)
  // y=(1,1) of quad q = (x ≥ 2) + 2·(y ≥ 2)). Interior blocks read via 4
  // gathers normalised by the PHYSICAL (padded) texture size; blocks
  // straddling the source edge of a non-multiple-of-4 image fall back to
  // per-texel loads clamped to the last real texel.
  var v: array<vec4<f32>, 4>;
  if (u32(base.x) + 4u <= params.width && u32(base.y) + 4u <= params.height) {
    let cc = (vec2<f32>(base) + 1.0) / vec2<f32>(textureDimensions(src_tex));
    v[0] = textureGather(0, src_tex, smp, cc) * 255.0;
    v[1] = textureGather(0, src_tex, smp, cc, vec2<i32>(2, 0)) * 255.0;
    v[2] = textureGather(0, src_tex, smp, cc, vec2<i32>(0, 2)) * 255.0;
    v[3] = textureGather(0, src_tex, smp, cc, vec2<i32>(2, 2)) * 255.0;
  } else {
    let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    v[0] = load_quad(base + vec2<i32>(0, 0), mx) * 255.0;
    v[1] = load_quad(base + vec2<i32>(2, 0), mx) * 255.0;
    v[2] = load_quad(base + vec2<i32>(0, 2), mx) * 255.0;
    v[3] = load_quad(base + vec2<i32>(2, 2), mx) * 255.0;
  }
  let mn = min(min(v[0], v[1]), min(v[2], v[3]));
  let mxv = max(max(v[0], v[1]), max(v[2], v[3]));
  let vmin = min(min(mn.x, mn.y), min(mn.z, mn.w));
  let vmax = max(max(mxv.x, mxv.y), max(mxv.z, mxv.w));

  // Seed endpoints at the extremes (round-to-nearest, the same rule the CPU
  // reference uses); spans ≤ 7 (incl. flat blocks) seed a 7-wide window
  // instead — lossless, and the refit keeps it.
  let vhi = clamp(floor(vmax + 0.5), 0.0, 255.0);
  let vlo = clamp(floor(vmin + 0.5), 0.0, 255.0);
  let small = vhi - vlo <= 7.0;
  let r1f = select(vlo, min(vlo, 248.0), small);
  let r0f = select(vhi, r1f + 7.0, small);
  // Pass-1 levels come from the seed range INSET by ~5.5/256 of the span:
  // scale ×7.3125/7, offset ½ − 0.15625 (exact dyadic constants).
  let scale = 7.3125 / (r1f - r0f);

  // Pass 1 — MOMENTS only. t = d·scale ∈ [0,7.3125] (the seed covers the
  // data), L = floor(t + 11/32) ∈ [0,7], no clamp.
  var sL = 0.0; var sLL = 0.0; var sd = 0.0; var sLd = 0.0;
  for (var q: u32 = 0u; q < 4u; q = q + 1u) {
    let d = v[q] - r0f;
    let L = floor(d * scale + 0.34375);
    sL = sL + dot(L, vec4<f32>(1.0));
    sLL = sLL + dot(L, L);
    sd = sd + dot(d, vec4<f32>(1.0));
    sLd = sLd + dot(L, d);
  }

  // Refit: least-squares line v ≈ r0 + α + β·L through the pass-1 levels;
  // den = 0 ⟺ every pixel on one level — keep the seed then. Endpoints
  // clamp to [0,255], NOT the block's value range.
  let den = 16.0 * sLL - sL * sL;
  let beta = (16.0 * sLd - sL * sd) / den;
  let e0 = r0f + (sd - beta * sL) * (1.0 / 16.0);
  let q0f = floor(clamp(e0, 0.0, 255.0) + 0.5);
  let q1f = floor(clamp(e0 + 7.0 * beta, 0.0, 255.0) + 0.5);
  let acc = (den > 0.0) & (q0f > q1f);
  let n0f = select(r0f, q0f, acc);
  let n1f = select(r1f, q1f, acc);

  // Pass 2 — levels against the refit endpoints, packed as Σ L·8^k
  // (A = pixels 0..7, B = pixels 8..15), plus ΣL for the offset round.
  let sc2 = 7.0 / (n1f - n0f);
  var pk = vec2<f32>(0.0);
  var sL2 = 0.0;
  for (var q: u32 = 0u; q < 4u; q = q + 1u) {
    let L = clamp(floor((v[q] - n0f) * sc2 + 0.5), vec4<f32>(0.0), vec4<f32>(7.0));
    let w = select(W0, W1, (q & 1u) == 1u);
    let hi = q >= 2u;
    let p = dot(L, w);
    pk = pk + vec2<f32>(select(p, 0.0, hi), select(0.0, p, hi));
    sL2 = sL2 + dot(L, vec4<f32>(1.0));
  }

  // Offset round: shift both endpoints by the rounded mean residual of the
  // shipped levels (never worse on those indices, under any decoder).
  let res = sd + 16.0 * (r0f - n0f) - (n1f - n0f) * sL2 * (1.0 / 7.0);
  let sh = clamp(floor(res * (1.0 / 16.0) + 0.5), -n1f, 255.0 - n0f);
  let m0 = fu(n0f + sh);
  let m1 = fu(n1f + sh);
  let iA = lvl_to_idx(u32(pk.x));
  let iB = lvl_to_idx(u32(pk.y));

  // BC4 block = 2 endpoint bytes + 48 index bits = 2 u32s.
  let o = bi * 2u;
  dst[o] = m0 | (m1 << 8u) | (iA << 16u);
  dst[o + 1u] = (iA >> 16u) | (iB << 8u);
}
