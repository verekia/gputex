// bc1 "fast" encoder — f16 variant (requires the shader-f16 feature).
//
// BC1 quantises endpoints to RGB565 anyway, so the fast path needs nothing
// f32 can do that f16 can't — all projection / least-squares math runs in
// f16 ([0,1] domain). The algorithm is the same family as the BC7/ASTC fast
// paths rather than a port of bc1.wgsl's fast branch:
//
//   1. principal-axis endpoint seed (covariance power-iteration; inset bbox
//      on degenerate blocks), inset by ~half a 565 cell (stb_dxt heuristic)
//   2. quantise to 565, force 4-colour mode (c0 > c1)
//   3. ONE fused pass: project every pixel onto the decoded-endpoint line
//      (the 4 palette entries are colinear and evenly spaced, so the nearest
//      entry is the rounded projection — no 4-entry search) while
//      accumulating the least-squares refit sums, the packed indices and the
//      squared error
//   4. up to TWO refit rounds (mirroring the high path's iterated refits):
//      re-quantise the refit endpoints, reproject (indices packed on the
//      fly, sums re-accumulated to seed the next round), and accept each
//      round only if the block error decreases — flat/single-level blocks
//      skip these passes entirely
//
// vs the pre-projection fast branch (build palette + full 4-entry search × 3
// passes + refit sums pass) this does roughly half the ALU per block. The
// 565 decode uses exact integer math, so the palette base points are exact.
//
// The host selects this module only when the device reports shader-f16,
// falling back to bc1.wgsl otherwise. "high" never uses this.
enable f16;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
alias h = f16;
alias h3 = vec3<f16>;

fn to565(c: h3) -> u32 {
  let r = u32(clamp(floor(c.r * h(31.0) + h(0.5)), h(0.0), h(31.0)));
  let g = u32(clamp(floor(c.g * h(63.0) + h(0.5)), h(0.0), h(63.0)));
  let b = u32(clamp(floor(c.b * h(31.0) + h(0.5)), h(0.0), h(31.0)));
  return (r << 11u) | (g << 5u) | b;
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

// One projection pass against the decoded endpoints of (c0,c1): the packed
// 2-bit indices, the block's squared error, and the LSQ normal-equation sums
// of the resulting assignment — so an accepted refit can seed the next
// round. Levels s run 0..3 along p0→p1 (palette = p0, p0+⅓d, p0+⅔d, p1 —
// colinear, evenly spaced, so rounding the projection IS the nearest-entry
// search). Level → BC1 index: 0→0 (c0), 1→2 (⅔c0+⅓c1), 2→3, 3→1 (c1); as a
// packed LUT: (0x78 >> 2L) & 3.
struct Proj {
  indices: u32,
  err: h,
  sAA: h, sBB: h, sAB: h,
  sAV: h3, sBV: h3,
  s_min: h, s_max: h,
};
fn project_stats(pix: ptr<function, array<h3, 16>>, c0: u32, c1: u32) -> Proj {
  var out: Proj;
  out.indices = 0u;
  out.err = h(0.0);
  out.sAA = h(0.0); out.sBB = h(0.0); out.sAB = h(0.0);
  out.sAV = h3(0.0); out.sBV = h3(0.0);
  out.s_min = h(3.0); out.s_max = h(0.0);
  let p0 = from565(c0);
  let p1 = from565(c1);
  let dir = p1 - p0;
  let dd = dot(dir, dir);
  if (dd == h(0.0)) {
    // Unreachable for distinct 565 codes (the decode is injective); kept so
    // a degenerate call still returns a consistent error.
    out.s_min = h(0.0);
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let e = (*pix)[k] - p0;
      out.err = out.err + dot(e, e);
    }
    return out;
  }
  let inv = h(3.0) / dd;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let v = (*pix)[k];
    let s = clamp(floor(dot(v - p0, dir) * inv + h(0.5)), h(0.0), h(3.0));
    out.s_min = min(out.s_min, s); out.s_max = max(out.s_max, s);
    let b = s * h(1.0 / 3.0); let a = h(1.0) - b;
    out.sAA = out.sAA + a * a; out.sBB = out.sBB + b * b; out.sAB = out.sAB + a * b;
    out.sAV = out.sAV + a * v; out.sBV = out.sBV + b * v;
    let e = v - (p0 + b * dir);
    out.err = out.err + dot(e, e);
    out.indices = out.indices | (((0x78u >> (u32(s) * 2u)) & 3u) << (k * 2u));
  }
  return out;
}

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0), mx);
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

  // Seed endpoints from the block's principal colour axis (covariance
  // power-iteration, seeded with the bbox diagonal — same family as the
  // 'high' path). The bbox diagonal is sign-blind: on anti-correlated
  // channels (normal maps, hue edges) it points across the data instead of
  // along it, the projection indices come out garbage, and the LSQ refit —
  // which fits endpoints GIVEN those indices — can't recover. Deviations are
  // pre-scaled ×16 so covariance entries for shallow blocks stay in f16's
  // normal range (span ~1/255 → d² ≈ 1e-3) while full-range sums stay ≤4096;
  // the iteration renormalises by the max component (a plain length() of the
  // matvec output could overflow f16), so only the direction survives — the
  // ×256 covariance scale is irrelevant.
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
    // Inset along the axis by ~half a 565 cell (stb_dxt heuristic, matching
    // the degenerate-case bbox inset below).
    let pad = (t_max - t_min) * h(1.0 / 16.0);
    seed_hi = clamp(mean + (t_max - pad) * axis, h3(0.0), h3(1.0));
    seed_lo = clamp(mean + (t_min + pad) * axis, h3(0.0), h3(1.0));
  } else {
    // Degenerate (near-flat) block: inset bbox seed, as before.
    let inset = (mxv - mn) * h(1.0 / 16.0);
    seed_hi = clamp(mxv - inset, h3(0.0), h3(1.0));
    seed_lo = clamp(mn + inset, h3(0.0), h3(1.0));
  }
  let seed = order565(to565(seed_hi), to565(seed_lo));
  var c0 = seed.x;
  var c1 = seed.y;

  // Fused seed pass, then up to TWO least-squares refit rounds (mirroring
  // the high path's iterated refits, at projection cost), each accepted only
  // if the block's squared error actually decreases — the refit minimises a
  // continuous objective and can lose after 565 quantisation. Every pass
  // re-accumulates the normal-equation sums, so an accepted round seeds the
  // next.
  var cur = project_stats(&pix, c0, c1);
  for (var it: u32 = 0u; it < 2u; it = it + 1u) {
    // Refit only on a well-conditioned system. When every pixel lands on ONE
    // level (flat / near-flat blocks — note the 4-colour-mode nudge forces
    // c0 ≠ c1 even for perfectly flat blocks) the system is rank-1: det is 0
    // in exact math and the f16-accumulated det/numerators are pure rounding
    // noise, so the solve returns garbage endpoints. With ≥2 distinct levels
    // det = Σ_i<j (b_j − b_i)² ≥ 15·(1/3)² ≈ 1.67, far above the ~0.05 f16
    // noise floor — 0.5 separates the two regimes cleanly.
    if (cur.s_min >= cur.s_max) { break; }
    let det = cur.sAA * cur.sBB - cur.sAB * cur.sAB;
    if (abs(det) <= h(0.5)) { break; }
    // Clamp the refit to the block bbox (not [0,1]) — except for exactly
    // gray blocks, see the load pass: on multi-cluster blocks the
    // unconstrained solve extrapolates far outside the block's colours
    // and the per-channel clamp then bends the hue — fringe pixels decode to
    // colours that exist nowhere in the block. Constraining to the bbox also
    // measures better in plain SSE (+1.6 dB on the colour test card), so the
    // accept-if-better guard below keeps more refits.
    let e0 = clamp((cur.sBB * cur.sAV - cur.sAB * cur.sBV) / det, lim_lo, lim_hi);
    let e1 = clamp((cur.sAA * cur.sBV - cur.sAB * cur.sAV) / det, lim_lo, lim_hi);
    let rq = order565(to565(e0), to565(e1));
    if (rq.x == c0 && rq.y == c1) { break; }
    let nxt = project_stats(&pix, rq.x, rq.y);
    if (nxt.err >= cur.err) { break; }
    c0 = rq.x;
    c1 = rq.y;
    cur = nxt;
  }

  let o = bi * 2u;
  dst[o] = c0 | (c1 << 16u);
  dst[o + 1u] = cur.indices;
}
