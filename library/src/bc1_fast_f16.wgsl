// bc1 encoder — f16 variant (requires the shader-f16 feature).
//
// BC1 quantises endpoints to RGB565 anyway, so nothing here needs f32
// precision except the refit sums (see moments()). Same algorithm as the
// f32 fallback in bc1.wgsl:
//
//   1. NEAR-FLAT blocks (every channel within 3 levels) take a solid colour
//      at the block mean: per channel the endpoint pair whose ⅔/⅓
//      interpolant lands nearest (solid_pair — the stb_dxt single-colour
//      idea). Direct 565 quantisation is up to 4 levels off on R/B there,
//      and a line fit has nothing to fit. +0.05..3.9 dB on content with
//      flat regions (displacement/AO maps, UI, the normal card).
//   2. Otherwise a principal-axis endpoint seed (covariance power
//      iteration; inset bbox on degenerate blocks), inset by ~half a 565
//      cell (stb_dxt heuristic), quantised to 565 in 4-colour mode (c0 > c1).
//   3. Projection passes: every pixel's level is the rounded projection
//      onto the decoded-endpoint line (the 4 palette entries are colinear
//      and evenly spaced, so that IS the nearest entry), packed as indices
//      on the fly, with the block error and projection MOMENTS accumulated.
//   4. Up to TWO least-squares refit rounds solved from those moments
//      (solve()), each re-projected and accepted only if the block error
//      drops. The moments replace the full normal-equation sums a refit
//      used to accumulate per pixel (the refit rounds were ~60% of the
//      kernel): −10..15% GPU on non-flat content at equal quality.
//
// The host selects this module only when the device reports shader-f16,
// falling back to bc1.wgsl otherwise.
enable f16;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, y0: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
alias h = f16;
alias h3 = vec3<f16>;
// Float -> u32 for exact integers in [0, 2^23): x + 2^23 holds x in its
// mantissa, so bitcast(x + MAGIC) ^ MAGIC_BITS == x. WGSL's u32(float) is a
// SATURATING conversion (compares + selects around the convert).
const MAGIC = 8388608.0;
const MAGIC_BITS = 0x4B000000u;

// Packed in float (r·2048 + g·32 + b, exact), then ONE conversion (−2% GPU
// over three u32() conversions).
fn to565(c: h3) -> u32 {
  let q = clamp(floor(c * h3(31.0, 63.0, 31.0) + h(0.5)), h3(0.0), h3(31.0, 63.0, 31.0));
  return bitcast<u32>(dot(vec3<f32>(q), vec3<f32>(2048.0, 32.0, 1.0)) + MAGIC) ^ MAGIC_BITS;
}

// Decode a 565 endpoint to [0,1]: (x*527+23)>>6 (6-bit: 259/33) —
// round-to-nearest scaling, matching bc1_ref.ts / bc1.wgsl and typical
// hardware decoders. Exact in u32 integer math (f16 could not evaluate the
// products exactly).
fn from565(c: u32) -> h3 {
  let r = (c >> 11u) & 31u;
  let g = (c >> 5u) & 63u;
  let b = c & 31u;
  let r8 = (r * 527u + 23u) >> 6u;
  let g8 = (g * 259u + 33u) >> 6u;
  let b8 = (b * 527u + 23u) >> 6u;
  return h3(vec3<f32>(vec3<u32>(r8, g8, b8))) * h(1.0 / 255.0);
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

// Projection MOMENTS against the decoded endpoints of (c0, c1): levels
// L = 0..3 along p0→p1 (the rounded projection — the palette is colinear
// and evenly spaced, so that IS the nearest entry), their BC1 indices, and
// ΣL, ΣL² (exact small integers), Σu, ΣL·u (u = v − p0, f32: the f16 block
// mean is off by up to ~½ level — its running sum reaches ~8, where f16's
// ulp is a whole level — which skews the closed-form solve by several
// levels) — plus the block's exact squared error against this palette.
// Level → BC1 index: 0→0 (c0), 1→2, 2→3, 3→1 (c1); packed LUT
// (0x78 >> 2L) & 3, with 2L read out of the float by the MAGIC trick (a
// u32(L) conversion per texel cost ~5% GPU).
struct Moments { sL: f32, sLL: f32, sU: vec3<f32>, sLu: vec3<f32>, indices: u32, err: f32 };
struct MomAcc { sL: f16, sLL: f16, err: f16, sU: vec3<f32>, sLu: vec3<f32>, indices: u32 };
fn project(a: ptr<function, MomAcc>, v: h3, p0: h3, dir: h3, inv: f16, k: u32) {
  let u = v - p0;
  let L = clamp(floor(dot(u, dir) * inv + h(0.5)), h(0.0), h(3.0));
  (*a).sL = (*a).sL + L;
  (*a).sLL = (*a).sLL + L * L;
  let uf = vec3<f32>(u);
  (*a).sU = (*a).sU + uf;
  (*a).sLu = (*a).sLu + f32(L) * uf;
  (*a).indices = (*a).indices | (((0x78u >> (bitcast<u32>(f32(L) * 2.0 + MAGIC) & 7u)) & 3u) << (k * 2u));
  let e = u - L * h(1.0 / 3.0) * dir;
  (*a).err = (*a).err + dot(e, e);
}
// Unrolled over constant texel indices: a pix[k] loop kept the array in
// indexable memory (−3.5% GPU unrolled; the f16/f32 sums then reassociate
// differently, moving ~0.5% of blocks by ±0.001 dB).
fn moments(pix: ptr<function, array<h3, 16>>, c0: u32, c1: u32) -> Moments {
  let p0 = from565(c0);
  let dir = from565(c1) - p0;
  let inv = h(3.0) / dot(dir, dir);
  var a = MomAcc(h(0.0), h(0.0), h(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0u);
  project(&a, (*pix)[0], p0, dir, inv, 0u);
  project(&a, (*pix)[1], p0, dir, inv, 1u);
  project(&a, (*pix)[2], p0, dir, inv, 2u);
  project(&a, (*pix)[3], p0, dir, inv, 3u);
  project(&a, (*pix)[4], p0, dir, inv, 4u);
  project(&a, (*pix)[5], p0, dir, inv, 5u);
  project(&a, (*pix)[6], p0, dir, inv, 6u);
  project(&a, (*pix)[7], p0, dir, inv, 7u);
  project(&a, (*pix)[8], p0, dir, inv, 8u);
  project(&a, (*pix)[9], p0, dir, inv, 9u);
  project(&a, (*pix)[10], p0, dir, inv, 10u);
  project(&a, (*pix)[11], p0, dir, inv, 11u);
  project(&a, (*pix)[12], p0, dir, inv, 12u);
  project(&a, (*pix)[13], p0, dir, inv, 13u);
  project(&a, (*pix)[14], p0, dir, inv, 14u);
  project(&a, (*pix)[15], p0, dir, inv, 15u);
  var out: Moments;
  out.sU = a.sU;
  out.sLu = a.sLu;
  out.indices = a.indices;
  out.err = f32(a.err);
  out.sL = f32(a.sL);
  out.sLL = f32(a.sLL);
  return out;
}

// One least-squares refit from moments: every normal-equation sum is an
// O(1) function of them (b = L/3, a = 1 − b)
//   sBB = ΣL²/9   sAB = ΣL/3 − ΣL²/9   sAA = 16 − 2ΣL/3 + ΣL²/9
//   Σb·u = ΣL·u/3   Σa·u = Σu − Σb·u
// solved in f32 for endpoints relative to p0, clamped to [lim_lo, lim_hi]
// (the block bbox, except exactly-gray blocks — see the load pass: on
// multi-cluster blocks the unconstrained solve extrapolates outside the
// block's colours and the clamp would bend the hue), re-quantised and
// ordered for 4-colour mode. Returns (c0, c1) unchanged when every pixel
// sits on ONE level — then 16·ΣL² == (ΣL)² exactly and the system is
// singular.
fn solve(m: Moments, c0: u32, c1: u32, lim_lo: h3, lim_hi: h3) -> vec2<u32> {
  if (16.0 * m.sLL == m.sL * m.sL) { return vec2<u32>(c0, c1); }
  let sBB = m.sLL * (1.0 / 9.0);
  let sAB = m.sL * (1.0 / 3.0) - sBB;
  let sAA = 16.0 - m.sL * (2.0 / 3.0) + sBB;
  let det = sAA * sBB - sAB * sAB;
  let p0 = vec3<f32>(from565(c0));
  let sBu = m.sLu * (1.0 / 3.0);
  let sAu = m.sU - sBu;
  let e0 = clamp(p0 + (sBB * sAu - sAB * sBu) / det, vec3<f32>(lim_lo), vec3<f32>(lim_hi));
  let e1 = clamp(p0 + (sAA * sBu - sAB * sAu) / det, vec3<f32>(lim_lo), vec3<f32>(lim_hi));
  return order565(to565(h3(e0)), to565(h3(e1)));
}

// Solid-colour channel code: the pair (a, b) of `bits`-bit codes whose ⅔/⅓
// interpolant (2·dec(a) + dec(b))/3 — palette index 2 — lands nearest v
// (8-bit units). With a == b it's a plain endpoint; straddling pairs reach
// the ~2.7-level sub-steps between codes that direct quantisation (steps of
// ~8 levels at 5 bits) cannot.
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

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid_raw: vec3<u32>) {
  // Row-band encodes dispatch a slice of the block grid starting at row y0.
  let gid = vec3<u32>(gid_raw.x, gid_raw.y + params.y0, gid_raw.z);
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);
  let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  var pix: array<h3, 16>;
  var mn = h3(1.0);
  var mxv = h3(0.0);
  var mean = h3(0.0);
  var gd = h(0.0);
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = min(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), mx);
    let px = h3(textureLoad(src_tex, p, 0).rgb);
    pix[i] = px; mn = min(mn, px); mxv = max(mxv, px);
    mean = mean + px;
    gd = max(gd, max(abs(px.x - px.y), abs(px.x - px.z)));
  }
  mean = mean * h(1.0 / 16.0);
  // Exactly-gray blocks free the refit from the bbox clamp below: a gray
  // block has no hue to bend (the clamp's whole purpose), and on smooth
  // gradients the LSQ optimum often lies OUTSIDE the data range — endpoints
  // spread wider than the block so the 1/3-2/3 interpolants land on the
  // values. Same rationale as the BC5 scalar channels (+0.32 dB there).
  let gray = gd == h(0.0);
  let lim_lo = select(mn, h3(0.0), gray);
  let lim_hi = select(mxv, h3(1.0), gray);

  // NEAR-FLAT blocks (every channel within 3 levels) take a solid colour
  // at the block mean: per channel the endpoint pair whose ⅔/⅓ interpolant
  // lands nearest (see solid_pair). A line fit has nothing to fit there,
  // and direct 565 quantisation is up to 4 levels off on R/B. They share
  // the index pass below (pixels ±1 level off the mean may sit closer to
  // an endpoint than to the interpolant) and skip the PCA seed and refits.
  let span = mxv - mn;
  let flat = max(max(span.x, span.y), span.z) <= h(3.0 / 255.0);
  var c0: u32;
  var c1: u32;
  if (flat) {
    let m8 = vec3<f32>(mean) * 255.0;
    let pr = solid_pair(m8.x, 5u);
    let pg = solid_pair(m8.y, 6u);
    let pb = solid_pair(m8.z, 5u);
    let s0 = (pr.x << 11u) | (pg.x << 5u) | pb.x;
    let s1 = (pr.y << 11u) | (pg.y << 5u) | pb.y;
    // c0 > c1 keeps 4-colour mode (index 2 = ⅔·c0 + ⅓·c1; swapped, the
    // same colour is index 3). Equal codes encode the colour itself.
    c0 = max(s0, s1);
    c1 = min(s0, s1);
  } else {
    // Seed endpoints from the block's principal colour axis (covariance
    // power-iteration, seeded with the bbox diagonal). The bbox diagonal is
    // sign-blind: on anti-correlated channels (normal maps, hue edges) it
    // points across the data instead of along it, the projection indices
    // come out garbage, and the LSQ refit — which fits endpoints GIVEN
    // those indices — can't recover. Deviations are pre-scaled ×16 so
    // covariance entries for shallow blocks stay in f16's normal range
    // (span ~1/255 → d² ≈ 1e-3) while full-range sums stay ≤4096; the
    // iteration renormalises by the max component (a plain length() of the
    // matvec output could overflow f16), so only the direction survives.
    var seed_hi: h3;
    var seed_lo: h3;
    var c0v = h3(0.0);
    var c1v = h3(0.0);
    var c2v = h3(0.0);
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let d = (pix[k] - mean) * h(16.0);
      c0v = c0v + d.x * d;
      c1v = c1v + d.y * d;
      c2v = c2v + d.z * d;
    }
    var axis = mxv - mn;
    var axis_ok = true;
    for (var it: u32 = 0u; it < 4u; it = it + 1u) {
      let nv = h3(dot(c0v, axis), dot(c1v, axis), dot(c2v, axis));
      let m = max(max(abs(nv.x), abs(nv.y)), abs(nv.z));
      if (m < h(1e-4)) { axis_ok = false; break; }
      axis = nv / m;
    }
    if (axis_ok) {
      axis = axis / length(axis);
      var t_min = h(4.0);
      var t_max = h(-4.0);
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let t = dot(pix[k] - mean, axis);
        t_min = min(t_min, t);
        t_max = max(t_max, t);
      }
      // Inset along the axis by ~half a 565 cell (stb_dxt heuristic,
      // matching the degenerate-case bbox inset below).
      let pad = (t_max - t_min) * h(1.0 / 16.0);
      seed_hi = clamp(mean + (t_max - pad) * axis, h3(0.0), h3(1.0));
      seed_lo = clamp(mean + (t_min + pad) * axis, h3(0.0), h3(1.0));
    } else {
      // Degenerate block: inset bbox seed.
      let inset = (mxv - mn) * h(1.0 / 16.0);
      seed_hi = clamp(mxv - inset, h3(0.0), h3(1.0));
      seed_lo = clamp(mn + inset, h3(0.0), h3(1.0));
    }
    let seed = order565(to565(seed_hi), to565(seed_lo));
    c0 = seed.x;
    c1 = seed.y;
  }

  // Projection pass on the seed, then up to two refit rounds (solve() off
  // the previous pass's moments), each re-projected and accepted only if
  // the block error drops — the refit minimises a continuous objective and
  // can lose after 565 quantisation. Flat blocks keep their solid pair
  // (equal codes: the colour itself, index 0 — opaque in either mode).
  var indices = 0u;
  if (c0 != c1) {
    var cur = moments(&pix, c0, c1);
    for (var it: u32 = 0u; it < select(2u, 0u, flat); it = it + 1u) {
      let cand = solve(cur, c0, c1, lim_lo, lim_hi);
      if (cand.x == c0 && cand.y == c1) { break; }
      let nxt = moments(&pix, cand.x, cand.y);
      if (nxt.err >= cur.err) { break; }
      c0 = cand.x;
      c1 = cand.y;
      cur = nxt;
    }
    indices = cur.indices;
  }

  let o = bi * 2u;
  dst[o] = c0 | (c1 << 16u);
  dst[o + 1u] = indices;
}
