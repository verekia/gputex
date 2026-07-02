// bc1 "fast" encoder — f16 variant (requires the shader-f16 feature).
//
// BC1 quantises endpoints to RGB565 anyway, so the fast path needs nothing
// f32 can do that f16 can't — all projection / least-squares math runs in
// f16 ([0,1] domain). The algorithm is the same family as the BC7/ASTC fast
// paths rather than a port of bc1.wgsl's fast branch:
//
//   1. bbox endpoints, inset by ~half a 565 cell (stb_dxt heuristic)
//   2. quantise to 565, force 4-colour mode (c0 > c1)
//   3. ONE fused pass: project every pixel onto the decoded-endpoint line
//      (the 4 palette entries are colinear and evenly spaced, so the nearest
//      entry is the rounded projection — no 4-entry search) while
//      accumulating the least-squares refit sums, the seed solution's packed
//      indices and its squared error
//   4. re-quantise the refit endpoints, reproject (indices packed on the
//      fly), and accept the refit only if the block error decreases —
//      flat/single-level blocks skip this pass entirely
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

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);
  let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  var pix: array<h3, 16>;
  var mn = h3(1.0);
  var mxv = h3(0.0);
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0), mx);
    let px = h3(textureLoad(src_tex, p, 0).rgb);
    pix[i] = px; mn = min(mn, px); mxv = max(mxv, px);
  }

  // Inset bbox by ~half a 565 cell so the quantised palette hugs the data.
  let inset = (mxv - mn) * h(1.0 / 16.0);
  let seed = order565(to565(clamp(mxv - inset, h3(0.0), h3(1.0))), to565(clamp(mn + inset, h3(0.0), h3(1.0))));
  var c0 = seed.x;
  var c1 = seed.y;
  let p0 = from565(c0);
  let p1 = from565(c1);

  // Fused pass: projection assignment + LSQ normal-equation sums + the seed
  // solution's packed indices and squared error. Levels s run 0..3 along
  // p0→p1 (palette = p0, p0+⅓d, p0+⅔d, p1 — colinear, evenly spaced, so
  // rounding the projection IS the nearest-entry search). Level → BC1 index:
  // 0→0 (c0), 1→2 (⅔c0+⅓c1), 2→3, 3→1 (c1); as a packed LUT: (0x78 >> 2L) & 3.
  var indices: u32 = 0u;
  let dir = p1 - p0;
  let dd = dot(dir, dir);
  if (dd > h(0.0)) {
    let inv = h(3.0) / dd;
    var sAA = h(0.0); var sBB = h(0.0); var sAB = h(0.0);
    var sAV = h3(0.0); var sBV = h3(0.0);
    var s_min = h(3.0); var s_max = h(0.0);
    var seed_err = h(0.0);
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let v = pix[k];
      let s = clamp(floor(dot(v - p0, dir) * inv + h(0.5)), h(0.0), h(3.0));
      s_min = min(s_min, s); s_max = max(s_max, s);
      let b = s * h(1.0 / 3.0); let a = h(1.0) - b;
      sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b;
      sAV = sAV + a * v; sBV = sBV + b * v;
      let e = v - (p0 + b * dir);
      seed_err = seed_err + dot(e, e);
      indices = indices | (((0x78u >> (u32(s) * 2u)) & 3u) << (k * 2u));
    }
    let det = sAA * sBB - sAB * sAB;
    // Refit only on a well-conditioned system. When every pixel lands on ONE
    // level (flat / near-flat blocks — note the 4-colour-mode nudge forces
    // c0 ≠ c1 even for perfectly flat blocks) the system is rank-1: det is 0
    // in exact math and the f16-accumulated det/numerators are pure rounding
    // noise, so the solve returns garbage endpoints. With ≥2 distinct levels
    // det = Σ_i<j (b_j − b_i)² ≥ 15·(1/3)² ≈ 1.67, far above the ~0.05 f16
    // noise floor — 0.5 separates the two regimes cleanly.
    if (s_min < s_max && abs(det) > h(0.5)) {
      let e0 = clamp((sBB * sAV - sAB * sBV) / det, h3(0.0), h3(1.0));
      let e1 = clamp((sAA * sBV - sAB * sAV) / det, h3(0.0), h3(1.0));
      let refit = order565(to565(e0), to565(e1));
      let np0 = from565(refit.x);
      let np1 = from565(refit.y);
      let ndir = np1 - np0;
      let ndd = dot(ndir, ndir);
      if (ndd > h(0.0) && !(refit.x == c0 && refit.y == c1)) {
        // Reproject against the refit endpoints and accept them only if the
        // block's squared error actually decreases (the refit minimises a
        // continuous objective; after 565 quantisation it can lose).
        let ninv = h(3.0) / ndd;
        var refit_err = h(0.0);
        var nindices: u32 = 0u;
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
          let v = pix[k];
          let s = clamp(floor(dot(v - np0, ndir) * ninv + h(0.5)), h(0.0), h(3.0));
          let e = v - (np0 + s * h(1.0 / 3.0) * ndir);
          refit_err = refit_err + dot(e, e);
          nindices = nindices | (((0x78u >> (u32(s) * 2u)) & 3u) << (k * 2u));
        }
        if (refit_err < seed_err) {
          c0 = refit.x; c1 = refit.y;
          indices = nindices;
        }
      }
    }
  }

  let o = bi * 2u;
  dst[o] = c0 | (c1 << 16u);
  dst[o + 1u] = indices;
}
