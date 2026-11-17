// astc4x4 encoder for OPAQUE sources — f16 (requires the shader-f16 feature).
//
// The host selects this module instead of astc4x4_fast_f16.wgsl when the
// caller promises that every texel of the source has alpha 255 (see
// `EncodeOptions.opaque`; `compressTexture` derives the promise from the
// image file itself — JPEG, PNG without an alpha channel or tRNS chunk, WebP
// without alpha). It is the general kernel's algorithm restricted to the
// two opaque block classes, and produces blocks of the same quality (within
// 0.02 dB on real textures, +0.08 dB on the colour test card):
//
//       gray   → CEM 0 (luminance), 5-bit weights, mode 0x253
//       colour → CEM 8 (RGB): QUANT_192 endpoints + 4-bit weights (mode
//                0x242) when the block spans more than 12 levels, exact
//                8-bit endpoints + 3-bit weights (mode 0x053) otherwise
//
// (principal-axis endpoints from a covariance power iteration seeded with
// the largest-variance column, bbox-clamped; rounded-projection weights —
// astc4x4_fast_f16.wgsl documents the design measurements, astc4x4_ref.ts
// the block layouts and ISE.)
//
// What the promise buys is a different SHAPE, which is where the time goes
// on mobile GPUs:
//   • 12 texture fetches per block instead of 16: three channels × four
//     textureGather quads, no alpha. Adreno's cost is close to proportional
//     to the fetch count.
//   • No translucent path. The general kernel's RGBA covariance + refit is
//     the largest part of its code, and mobile shader compilers price the
//     WHOLE shader: with that path present the same colour-path code ran
//     up to 2× slower on Mali and 3× on Adreno (register allocation falls
//     off a cliff), whichever way the path was written.
//   • Quad-major data: the block is twelve vec4s (one per channel per 2×2
//     quad) held in registers, and every per-texel pass is four vec4
//     operations — no function-local texel array (a memory round trip per
//     access on Mali) and no 16-step loops.
// Per-dispatch compute time against the general kernel on the same opaque
// 1024² images (interleaved, same session):
//       Mali-G76 (Galaxy S10)   colour ×0.68–0.73   gray ×0.38–0.39
//       Adreno 620 (Pixel 5)    colour ×0.84–0.92   gray ×0.76
//       Apple M3                colour ×0.92–0.94   gray ×0.88
// The same quad-major rewrite of the general kernel (all three classes) did
// NOT pay: translucent content 1.3–2.2× slower on Mali and everything ~3×
// slower on Adreno — which is why this is a second module and not a
// replacement.
enable f16;
alias h = f16;
alias h3 = vec3<f16>;
alias h4 = vec4<f16>;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, y0: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var smp: sampler;

// Float -> u32 for exact integers in [0, 2^23): x + 2^23 holds x in its
// mantissa, so bitcast(x + MAGIC) ^ MAGIC_BITS == x (or masked, for small
// fields). WGSL's u32(float) is a SATURATING conversion (compares + selects
// around the convert).
const MAGIC = 8388608.0;
const MAGIC_BITS = 0x4B000000u;
fn hbits(x: f16, mask: u32) -> u32 {
  return bitcast<u32>(f32(x) + MAGIC) & mask;
}
fn fu(x: f32) -> u32 {
  return bitcast<u32>(x + MAGIC) ^ MAGIC_BITS;
}
fn fu3(x: vec3<f32>) -> vec3<u32> {
  return bitcast<vec3<u32>>(x + MAGIC) ^ vec3<u32>(MAGIC_BITS);
}
fn hu2(x: vec2<f16>) -> vec2<u32> {
  return bitcast<vec2<u32>>(vec2<f32>(x) + MAGIC) ^ vec2<u32>(MAGIC_BITS);
}
fn hu4(x: h4) -> vec4<u32> {
  return bitcast<vec4<u32>>(vec4<f32>(x) + MAGIC) ^ vec4<u32>(MAGIC_BITS);
}

// Horizontal reductions of one quad.
fn hmin(v: h4) -> h { return min(min(v.x, v.y), min(v.z, v.w)); }
fn hmax(v: h4) -> h { return max(max(v.x, v.y), max(v.z, v.w)); }
fn hsum(v: h4) -> h { return (v.x + v.y) + (v.z + v.w); }

// One 2×2 quad, one vec4 per channel, in textureGather component order:
// x = (0,1), y = (1,1), z = (1,0), w = (0,0).
struct Quad { r: h4, g: h4, b: h4, };
// Edge blocks (sources whose size is not a multiple of 4): four texel loads
// clamped to the last real row / column, arranged like a gather.
fn load_quad(p: vec2<i32>, mx: vec2<i32>) -> Quad {
  let t00 = textureLoad(src_tex, min(p, mx), 0);
  let t10 = textureLoad(src_tex, min(p + vec2<i32>(1, 0), mx), 0);
  let t01 = textureLoad(src_tex, min(p + vec2<i32>(0, 1), mx), 0);
  let t11 = textureLoad(src_tex, min(p + vec2<i32>(1, 1), mx), 0);
  return Quad(
    h4(vec4<f32>(t01.r, t11.r, t10.r, t00.r)),
    h4(vec4<f32>(t01.g, t11.g, t10.g, t00.g)),
    h4(vec4<f32>(t01.b, t11.b, t10.b, t00.b)),
  );
}


// ISE trit-block encoder: 5 trits → the 8-bit T field (the inverse of the
// spec's trit-block decode; all 243 tuples round-trip — see the ref tests).
fn trit_enc(t0: u32, t1: u32, t2: u32, t3: u32, t4: u32) -> u32 {
  let c = select(select((t2 << 4u) | (t1 << 2u) | t0, (t1 << 4u) | (t0 << 2u) | 3u, t2 == 2u), 12u | t0, t2 == 2u && t1 == 2u);
  return select(select((t4 << 7u) | (t3 << 5u) | c, (t3 << 7u) | 96u | c, t4 == 2u), ((c >> 2u) << 5u) | 28u | (c & 3u), t3 == 2u && t4 == 2u);
}

// Nearest QUANT_192 endpoint (one trit + 6 bits) to x ∈ [0,255]. The 192
// unquantised levels are every value ≤ 127 that is not ≡ 3 (mod 4), plus
// their mirror images 255 − u above: lower-half value u = 4q + t (t < 3)
// is ISE value trit t, bits q << 1; the upper half sets bit 0 (the spec's
// A-mask inversion). A value that falls on an excluded slot moves to the
// nearer representable neighbour (a fixed direction instead costs up to
// 0.4 dB on smooth content). Returns (ISE value = trit·64 + bits, level).
fn q192(x: f32) -> vec2<u32> {
  let v = fu(clamp(floor(x + 0.5), 0.0, 255.0));
  let up = v > 127u;
  var u = select(v, 255u - v, up);
  if ((u & 3u) == 3u) {
    let xu = select(x, 255.0 - x, up);
    u = select(u - 1u, u + 1u, xu > f32(u) && u < 127u);
  }
  return vec2<u32>(((u & 3u) << 6u) | ((u >> 2u) << 1u) | u32(up), select(u, 255u - u, up));
}


// Luminance weights of one quad: exact nearest entry of the QUANT_32 grid
// (unq = 2w for w ≤ 15, 2w + 2 for w ≥ 16 — a 4-wide gap at the middle, so
// uniform rounding is wrong there): best candidate of each half, keep the
// closer.
fn wl(x: h4, l0f: h, sc: h) -> vec4<u32> {
  let v = floor(x * h(255.0) + h(0.5)) - l0f; // exact integers
  let u = clamp(v * sc, h4(0.0), h4(64.0));
  let wlo = clamp(floor(u * h(0.5) + h(0.5)), h4(0.0), h4(15.0));
  let whi = clamp(floor((u - h(2.0)) * h(0.5) + h(0.5)), h4(16.0), h4(31.0));
  let pick = abs(u - wlo * h(2.0)) <= abs(u - (whi * h(2.0) + h(2.0)));
  return hu4(select(whi, wlo, pick));
}

// Colour weights of one quad on the d0 → d0 + dir/32 line, packed as two
// bytes of 4-bit weights: x = texels (0,0),(1,0), y = texels (0,1),(1,1).
fn wq(r: h4, g: h4, b: h4, d0: h3, dir: h3, inv: h, lmax: h) -> vec2<u32> {
  let w = clamp(floor(((r - d0.x) * dir.x + (g - d0.y) * dir.y + (b - d0.z) * dir.z) * inv + h(0.5)), h4(0.0), h4(lmax));
  return hu2(vec2<f16>(w.w + w.z * h(16.0), w.x + w.y * h(16.0)));
}

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid_raw: vec3<u32>) {
  // Row-band encodes dispatch a slice of the block grid starting at row y0.
  let gid = vec3<u32>(gid_raw.x, gid_raw.y + params.y0, gid_raw.z);
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);

  // Load pass: the block as four 2×2 quads per channel (cr0 = red of the
  // top-left quad; quad 1 = top-right, 2 = bottom-left, 3 = bottom-right).
  var cr0: h4; var cg0: h4; var cb0: h4;
  var cr1: h4; var cg1: h4; var cb1: h4;
  var cr2: h4; var cg2: h4; var cb2: h4;
  var cr3: h4; var cg3: h4; var cb3: h4;
  if (u32(base.x) + 4u <= params.width && u32(base.y) + 4u <= params.height) {
    let cc = (vec2<f32>(base) + vec2<f32>(1.0)) / vec2<f32>(textureDimensions(src_tex));
    cr0 = h4(textureGather(0, src_tex, smp, cc));
    cg0 = h4(textureGather(1, src_tex, smp, cc));
    cb0 = h4(textureGather(2, src_tex, smp, cc));
    cr1 = h4(textureGather(0, src_tex, smp, cc, vec2<i32>(2, 0)));
    cg1 = h4(textureGather(1, src_tex, smp, cc, vec2<i32>(2, 0)));
    cb1 = h4(textureGather(2, src_tex, smp, cc, vec2<i32>(2, 0)));
    cr2 = h4(textureGather(0, src_tex, smp, cc, vec2<i32>(0, 2)));
    cg2 = h4(textureGather(1, src_tex, smp, cc, vec2<i32>(0, 2)));
    cb2 = h4(textureGather(2, src_tex, smp, cc, vec2<i32>(0, 2)));
    cr3 = h4(textureGather(0, src_tex, smp, cc, vec2<i32>(2, 2)));
    cg3 = h4(textureGather(1, src_tex, smp, cc, vec2<i32>(2, 2)));
    cb3 = h4(textureGather(2, src_tex, smp, cc, vec2<i32>(2, 2)));
  } else {
    let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    let q0 = load_quad(base + vec2<i32>(0, 0), mx); cr0 = q0.r; cg0 = q0.g; cb0 = q0.b;
    let q1 = load_quad(base + vec2<i32>(2, 0), mx); cr1 = q1.r; cg1 = q1.g; cb1 = q1.b;
    let q2 = load_quad(base + vec2<i32>(0, 2), mx); cr2 = q2.r; cg2 = q2.g; cb2 = q2.b;
    let q3 = load_quad(base + vec2<i32>(2, 2), mx); cr3 = q3.r; cg3 = q3.g; cb3 = q3.b;
  }
  let lo3 = h3(hmin(min(min(cr0, cr1), min(cr2, cr3))), hmin(min(min(cg0, cg1), min(cg2, cg3))), hmin(min(min(cb0, cb1), min(cb2, cb3))));
  let hi3 = h3(hmax(max(max(cr0, cr1), max(cr2, cr3))), hmax(max(max(cg0, cg1), max(cg2, cg3))), hmax(max(max(cb0, cb1), max(cb2, cb3))));
  // Exactly grayscale: R == G == B in every texel (equal 8-bit channels
  // convert to identical f16s).
  let gray = all(cr0 == cg0) && all(cr0 == cb0) && all(cr1 == cg1) && all(cr1 == cb1) && all(cr2 == cg2) && all(cr2 == cb2) && all(cr3 == cg3) && all(cr3 == cb3);

  var w0: u32; var w1: u32; var w2: u32; var w3: u32;

  if (gray) {
    // ---------------- Luminance path: CEM 0, 5-bit weights ----------------
    // Scalar [0,255]-integer domain. Endpoints at the exact extremes (the
    // 8-bit values round-trip f16 exactly); 32 palette levels make an LSQ
    // refit unnecessary.
    let L0 = hbits(floor(lo3.x * h(255.0) + h(0.5)), 255u);
    let L1 = hbits(floor(hi3.x * h(255.0) + h(0.5)), 255u);
    var s0 = 0u; var s1 = 0u; var s2 = 0u;
    if (L1 > L0) {
      let l0f = h(f32(L0));
      let sc = h(64.0) / h(f32(L1 - L0)); // span ≥ 1 → sc ≤ 64, no overflow
      let g0 = wl(cr0, l0f, sc); let g1 = wl(cr1, l0f, sc); let g2 = wl(cr2, l0f, sc); let g3 = wl(cr3, l0f, sc);
      // Stream bit q = 5k + j for texel k = 4y + x; quad components are
      // w = (0,0), z = (1,0), x = (0,1), y = (1,1).
      s0 = g0.w | (g0.z << 5u) | (g1.w << 10u) | (g1.z << 15u) | (g0.x << 20u) | (g0.y << 25u) | (g1.x << 30u);
      s1 = (g1.x >> 2u) | (g1.y << 3u) | (g2.w << 8u) | (g2.z << 13u) | (g3.w << 18u) | (g3.z << 23u) | (g2.x << 28u);
      s2 = (g2.x >> 4u) | (g2.y << 1u) | (g3.x << 6u) | (g3.y << 11u);
    }
    // Mode 0x253, partitions−1 = 0, CEM 0, L0 @17, L1 @25 (top bit spills
    // into w1 bit 0); stream word q∈[0,31] → block bits 127…96 via
    // reverseBits, q∈[32,63] → 95…64, q∈[64,79] → 63…48.
    w0 = 0x253u | (L0 << 17u) | (L1 << 25u);
    w1 = (L1 >> 7u) | reverseBits(s2);
    w2 = reverseBits(s1);
    w3 = reverseBits(s0);
  } else {
    // ------------- Colour: CEM 8, two bit budgets (header) -------------
    // Seed endpoints from the block's principal colour axis (covariance
    // power-iteration from the largest-variance column; the bbox diagonal
    // is sign-blind on anti-correlated channels). Deviations are
    // pre-scaled ×16 so covariance entries for shallow blocks stay in
    // f16's normal range (span ~1/255 → d² ≈ 1e-3) while full-range sums
    // stay ≤4096; the iteration renormalises by the max component (a
    // plain length() of the matvec output could overflow f16), so only the
    // direction survives.
    let mean3 = h3(hsum(((cr0 + cr1) + (cr2 + cr3))), hsum(((cg0 + cg1) + (cg2 + cg3))), hsum(((cb0 + cb1) + (cb2 + cb3)))) * h(1.0 / 16.0);
    let dr0 = (cr0 - mean3.x) * h(16.0); let dr1 = (cr1 - mean3.x) * h(16.0); let dr2 = (cr2 - mean3.x) * h(16.0); let dr3 = (cr3 - mean3.x) * h(16.0);
    let dg0 = (cg0 - mean3.y) * h(16.0); let dg1 = (cg1 - mean3.y) * h(16.0); let dg2 = (cg2 - mean3.y) * h(16.0); let dg3 = (cg3 - mean3.y) * h(16.0);
    let db0 = (cb0 - mean3.z) * h(16.0); let db1 = (cb1 - mean3.z) * h(16.0); let db2 = (cb2 - mean3.z) * h(16.0); let db3 = (cb3 - mean3.z) * h(16.0);
    let cxx = ((dot(dr0, dr0) + dot(dr1, dr1)) + (dot(dr2, dr2) + dot(dr3, dr3)));
    let cyy = ((dot(dg0, dg0) + dot(dg1, dg1)) + (dot(dg2, dg2) + dot(dg3, dg3)));
    let czz = ((dot(db0, db0) + dot(db1, db1)) + (dot(db2, db2) + dot(db3, db3)));
    let cxy = ((dot(dr0, dg0) + dot(dr1, dg1)) + (dot(dr2, dg2) + dot(dr3, dg3)));
    let cxz = ((dot(dr0, db0) + dot(dr1, db1)) + (dot(dr2, db2) + dot(dr3, db3)));
    let cyz = ((dot(dg0, db0) + dot(dg1, db1)) + (dot(dg2, db2) + dot(dg3, db3)));
    let c0v = h3(cxx, cxy, cxz);
    let c1v = h3(cxy, cyy, cyz);
    let c2v = h3(cxz, cyz, czz);
    var seed_lo = lo3;
    var seed_hi = hi3;
    // The seed column's largest entry is its diagonal one (|cov_ij| ≤
    // √(var_i·var_j) ≤ the largest variance), so dividing by it normalises
    // the seed the same way the iteration does. A zero variance everywhere
    // is a flat block: keep the bbox endpoints (lo == hi).
    let dmax = max(cxx, max(cyy, czz));
    var axis = c0v;
    if (cyy > cxx) { axis = c1v; }
    if (czz > max(cxx, cyy)) { axis = c2v; }
    axis = axis / max(dmax, h(1e-4));
    var axis_ok = dmax >= h(1e-4);
    // 6 iterations: the axis IS the endpoint quality here (no refit).
    for (var it: u32 = 0u; it < 6u; it = it + 1u) {
      let nv = h3(dot(c0v, axis), dot(c1v, axis), dot(c2v, axis));
      let m = max(max(abs(nv.x), abs(nv.y)), abs(nv.z));
      if (m < h(1e-4)) { axis_ok = false; break; }
      axis = nv / m;
    }
    if (axis_ok) {
      axis = axis / length(axis);
      // Extents along the axis, from the ×16 deviations.
      let t0 = dr0 * axis.x + dg0 * axis.y + db0 * axis.z; let t1 = dr1 * axis.x + dg1 * axis.y + db1 * axis.z; let t2 = dr2 * axis.x + dg2 * axis.y + db2 * axis.z; let t3 = dr3 * axis.x + dg3 * axis.y + db3 * axis.z;
      let t_min = hmin(min(min(t0, t1), min(t2, t3))) * h(1.0 / 16.0);
      let t_max = hmax(max(max(t0, t1), max(t2, t3))) * h(1.0 / 16.0);
      seed_lo = mean3 + t_min * axis;
      seed_hi = mean3 + t_max * axis;
    }
    // Endpoints: the PCA extents, bbox-clamped (on multi-cluster blocks the
    // axis extents overshoot the data per channel and would decode to
    // colours that exist nowhere in the block).
    let x0 = vec3<f32>(clamp(seed_lo, lo3, hi3)) * 255.0;
    let x1 = vec3<f32>(clamp(seed_hi, lo3, hi3)) * 255.0;
    // Bit budget: span ≤ 12 → exact QUANT_256 endpoints + 8 weight levels
    // (mode 0x053), wider → QUANT_192 endpoints + 16 levels (mode 0x242).
    let span3 = hi3 - lo3;
    let small = max(max(span3.x, span3.y), span3.z) <= h(12.0 / 255.0);
    var r0: vec2<u32>; var g0: vec2<u32>; var b0: vec2<u32>;
    var r1: vec2<u32>; var g1: vec2<u32>; var b1: vec2<u32>;
    if (small) {
      let e0 = fu3(clamp(floor(x0 + 0.5), vec3<f32>(0.0), vec3<f32>(255.0)));
      let e1 = fu3(clamp(floor(x1 + 0.5), vec3<f32>(0.0), vec3<f32>(255.0)));
      r0 = vec2<u32>(e0.x); g0 = vec2<u32>(e0.y); b0 = vec2<u32>(e0.z);
      r1 = vec2<u32>(e1.x); g1 = vec2<u32>(e1.y); b1 = vec2<u32>(e1.z);
    } else {
      r0 = q192(x0.x); g0 = q192(x0.y); b0 = q192(x0.z);
      r1 = q192(x1.x); g1 = q192(x1.y); b1 = q192(x1.z);
    }
    // Blue-contraction ordering on the UNQUANTISED levels, before the
    // weight pass so the weights come out oriented (no reflection).
    if (r0.y + g0.y + b0.y > r1.y + g1.y + b1.y) {
      let tr = r0; r0 = r1; r1 = tr;
      let tg = g0; g0 = g1; g1 = tg;
      let tb = b0; b0 = b1; b1 = tb;
    }
    let d0 = h3(vec3<f32>(vec3<u32>(r0.y, g0.y, b0.y))) * h(1.0 / 255.0);
    let d1 = h3(vec3<f32>(vec3<u32>(r1.y, g1.y, b1.y))) * h(1.0 / 255.0);
    // Weight pass — one code path for both budgets (mixed warps would
    // otherwise run two): weights land as 4-bit nibbles, levels = 16 or 8.
    // ×32 pre-scale as in the other formats (distinct levels are ≥1/255
    // apart, so dd₃₂ ≥ 0.0157 and the flat threshold only catches identical
    // ones).
    let lmax = select(h(15.0), h(7.0), small);
    let dir = (d1 - d0) * h(32.0);
    let dd = dot(dir, dir);
    var s0 = 0u; var s1 = 0u;
    if (dd >= h(0.008)) {
      let inv = lmax * h(32.0) / dd;
      let p0 = wq(cr0, cg0, cb0, d0, dir, inv, lmax); let p1 = wq(cr1, cg1, cb1, d0, dir, inv, lmax); let p2 = wq(cr2, cg2, cb2, d0, dir, inv, lmax); let p3 = wq(cr3, cg3, cb3, d0, dir, inv, lmax);
      // Texel k = 4y + x carries stream bits 4k…4k+3.
      s0 = p0.x | (p1.x << 8u) | (p0.y << 16u) | (p1.y << 24u);
      s1 = p2.x | (p3.x << 8u) | (p2.y << 16u) | (p3.y << 24u);
    }
    if (small) {
      // Mode 0x053: endpoints as plain bytes from bit 17; the 3-bit weight
      // stream (48 bits) is the nibbles compacted: 8 nibbles → 24 bits.
      var c0 = (s0 & 0x07070707u) | ((s0 & 0x70707070u) >> 1u);
      c0 = (c0 & 0x003F003Fu) | ((c0 & 0x3F003F00u) >> 2u);
      c0 = (c0 & 0x00000FFFu) | ((c0 & 0x0FFF0000u) >> 4u);
      var c1 = (s1 & 0x07070707u) | ((s1 & 0x70707070u) >> 1u);
      c1 = (c1 & 0x003F003Fu) | ((c1 & 0x3F003F00u) >> 2u);
      c1 = (c1 & 0x00000FFFu) | ((c1 & 0x0FFF0000u) >> 4u);
      w0 = 0x053u | (8u << 13u) | (r0.x << 17u) | (r1.x << 25u);
      w1 = (r1.x >> 7u) | (g0.x << 1u) | (g1.x << 9u) | (b0.x << 17u) | (b1.x << 25u);
      w2 = (b1.x >> 7u) | reverseBits(c1 >> 8u);
      w3 = reverseBits(c0 | (c1 << 24u));
    } else {
      // Mode 0x242, CEM 8 @13. Endpoint ISE (v0..v5 = R0 R1 G0 G1 B0 B1,
      // 46 bits from bit 17): group 1 = v0..v4 with trit field T, group 2 =
      // v5 with its lone trit as 2 bits (T of (t5,0,0,0,0) is t5 itself).
      // Weight stream bit q = 4k + j lives at block bit 127 − q.
      let tg = trit_enc(r0.x >> 6u, r1.x >> 6u, g0.x >> 6u, g1.x >> 6u, b0.x >> 6u);
      w0 = 0x242u | (8u << 13u) | ((r0.x & 63u) << 17u) | ((tg & 3u) << 23u) | ((r1.x & 63u) << 25u) | (((tg >> 2u) & 1u) << 31u);
      w1 = ((tg >> 3u) & 1u) | ((g0.x & 63u) << 1u) | (((tg >> 4u) & 1u) << 7u) | ((g1.x & 63u) << 8u)
        | (((tg >> 5u) & 3u) << 14u) | ((b0.x & 63u) << 16u) | ((tg >> 7u) << 22u) | ((b1.x & 63u) << 23u)
        | ((b1.x >> 6u) << 29u);
      w2 = reverseBits(s1);
      w3 = reverseBits(s0);
    }
  }

  let o = bi * 4u;
  dst[o] = w0; dst[o + 1u] = w1; dst[o + 2u] = w2; dst[o + 3u] = w3;
}
