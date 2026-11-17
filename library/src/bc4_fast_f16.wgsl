// bc4 "fast" encoder — f16 variant (requires the shader-f16 feature).
// One BC4 (RGTC1) block per 4×4 tile, from the source's R channel: the
// single-channel half of bc5_fast_f16.wgsl — same seed, same refit, same
// offset round, so a BC4 block is byte-identical to the R half of the BC5
// block for the same texels. See that file for the full notes and
// measurements; in short:
//
//   • The 8-entry palette in 6-interpolation mode is colinear and evenly
//     spaced from r0 to r1, so the nearest entry is the rounded projection
//     of v onto the r0→r1 axis.
//   • Math runs in the exact-integer [0,255] f16 domain.
//   • The 16 texel reads are 4 textureGather fetches for interior blocks;
//     blocks straddling the source edge of a non-multiple-of-4 image use
//     clamped per-texel loads (the upload pads the texture with zeros).
//   • SEED: endpoints at the extremes, pass-1 levels assigned against that
//     range inset by ~5.5/256 of the span (scale ×7.3125/7, offset 11/32).
//     Spans ≤ 7 (incl. flat blocks) seed a 7-wide window instead, whose
//     levels land on every integer the block holds: lossless.
//   • Pass 1 accumulates the moments ΣL, ΣL², Σd, ΣL·d (d = v − r0); the
//     refit is the least-squares line through those levels, accepted
//     whenever it stays in 6-interpolation mode.
//   • Pass 2 derives the shipped levels against the refit endpoints, then
//     an offset round shifts both endpoints by the rounded mean residual.
//   • 3-bit indices are packed as float (Σ L·8^k, exact below 2^24) and
//     mapped level → BC4 index with SWAR bit tricks.
//
// The host selects this module only when the device reports shader-f16,
// falling back to bc4.wgsl otherwise.
enable f16;
alias h = f16;
alias h4 = vec4<f16>;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, y0: u32, };
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
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);

  // Load 4×4 R as quad-major vec4s in gather order: component w=(0,0)
  // z=(1,0) x=(0,1) y=(1,1) of quad q = (x ≥ 2) + 2·(y ≥ 2). ×255 then f16
  // lands every value on an exact integer.
  var v: array<h4, 4>;
  if (u32(base.x) + 4u <= params.width && u32(base.y) + 4u <= params.height) {
    let cc = (vec2<f32>(base) + 1.0) / vec2<f32>(textureDimensions(src_tex));
    v[0] = h4(textureGather(0, src_tex, smp, cc) * 255.0);
    v[1] = h4(textureGather(0, src_tex, smp, cc, vec2<i32>(2, 0)) * 255.0);
    v[2] = h4(textureGather(0, src_tex, smp, cc, vec2<i32>(0, 2)) * 255.0);
    v[3] = h4(textureGather(0, src_tex, smp, cc, vec2<i32>(2, 2)) * 255.0);
  } else {
    let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    v[0] = h4(load_quad(base + vec2<i32>(0, 0), mx) * 255.0);
    v[1] = h4(load_quad(base + vec2<i32>(2, 0), mx) * 255.0);
    v[2] = h4(load_quad(base + vec2<i32>(0, 2), mx) * 255.0);
    v[3] = h4(load_quad(base + vec2<i32>(2, 2), mx) * 255.0);
  }
  let mn = min(min(v[0], v[1]), min(v[2], v[3]));
  let mxv = max(max(v[0], v[1]), max(v[2], v[3]));
  let vmin = min(min(mn.x, mn.y), min(mn.z, mn.w));
  let vmax = max(max(mxv.x, mxv.y), max(mxv.z, mxv.w));

  // Seed endpoints at the exact extremes; spans ≤ 7 (incl. flat blocks)
  // seed a 7-wide window instead — lossless, and the refit keeps it.
  let small = vmax - vmin <= h(7.0);
  let r1 = fu(f32(select(vmin, min(vmin, h(248.0)), small)));
  let r0 = select(fu(f32(vmax)), r1 + 7u, small);

  let r0h = h(f32(r0));
  let dirf = f32(r1) - f32(r0);
  // Pass-1 levels come from the seed range INSET by ~5.5/256 of the span.
  let scale = h(7.3125 / dirf);

  // Pass 1 — t = d·scale ∈ [0,7.3125] by construction (seed covers the
  // data), so L = floor(t + 11/32) ∈ [0,7] needs no clamp.
  var sL = h(0.0); var sLL = h(0.0); var sd = 0.0; var sLd = 0.0;
  for (var q: u32 = 0u; q < 4u; q = q + 1u) {
    let d = v[q] - r0h;
    let L = floor(d * scale + h(0.34375));
    sL = sL + dot(L, h4(1.0)); sLL = sLL + dot(L, L);
    sd = sd + f32(dot(d, h4(1.0)));
    sLd = sLd + dot(vec4<f32>(L), vec4<f32>(d));
  }

  // Refit in f32: least-squares line v ≈ r0 + α + β·L through the pass-1
  // levels. den = 0 ⟺ every pixel on one level — keep the seed then.
  let sLf = f32(sL);
  let den = 16.0 * f32(sLL) - sLf * sLf;
  let beta = (16.0 * sLd - sLf * sd) / den;
  let e0 = f32(r0) + (sd - beta * sLf) * (1.0 / 16.0);
  let q0f = floor(clamp(e0, 0.0, 255.0) + 0.5);
  let q1f = floor(clamp(e0 + 7.0 * beta, 0.0, 255.0) + 0.5);
  let acc = (den > 0.0) & (q0f > q1f);
  let n0 = select(r0, fu(q0f), acc);
  let n1 = select(r1, fu(q1f), acc);

  // Pass 2 — levels against the FINAL endpoints, accumulated as 3-bit
  // fields in f32 (Σ L·8^k ≤ 2^24 − 1, exact): A = pixels 0..7, B = 8..15.
  let n0h = h(f32(n0));
  let sc2 = h(7.0 / (f32(n1) - f32(n0)));
  var pk = vec2<f32>(0.0);
  var sLq = h(0.0);
  for (var q: u32 = 0u; q < 4u; q = q + 1u) {
    let L = clamp(floor((v[q] - n0h) * sc2 + h(0.5)), h4(0.0), h4(7.0));
    let w = select(W0, W1, (q & 1u) == 1u);
    let hi = q >= 2u;
    let p = dot(vec4<f32>(L), w);
    pk = pk + vec2<f32>(select(p, 0.0, hi), select(0.0, p, hi));
    sLq = sLq + dot(L, h4(1.0));
  }
  // Offset round: shift both endpoints by the rounded mean residual.
  let n0f = f32(n0);
  let res = sd + 16.0 * (f32(r0) - n0f) - (f32(n1) - n0f) * f32(sLq) * (1.0 / 7.0);
  let sh = clamp(floor(res * (1.0 / 16.0) + 0.5), -f32(n1), 255.0 - n0f);
  let m0 = fu(n0f + sh);
  let m1 = fu(f32(n1) + sh);
  let iA = lvl_to_idx(u32(pk.x));
  let iB = lvl_to_idx(u32(pk.y));

  // BC4 block = 2 endpoint bytes + 48 index bits = 2 u32s.
  let o = bi * 2u;
  dst[o] = m0 | (m1 << 8u) | (iA << 16u);
  dst[o + 1u] = (iA >> 16u) | (iB << 8u);
}
