// astc4x4 "fast" encoder — f16 variant (requires the shader-f16 feature).
// Same algorithm family as the f32 fast path in astc4x4.wgsl (bbox seed →
// projection weight assignment with a fused least-squares refit →
// reproject), tuned for throughput:
//
//   • All projection / refit math in f16 ([0,1] domain). The projection
//     direction is pre-scaled by 32 — a shallow block (endpoints ~1/255
//     apart) has dd ≈ 1.5e-5, where 3/dd ≈ 2e5 overflows f16 (max 65504) to
//     +inf and the projection dots go subnormal, turning weights and the LSQ
//     refit to garbage (banding on smooth gradients). Scaling dir by 32 puts
//     every intermediate in f16's normal range; s = dot·(32·3/dd₃₂) is the
//     same quantity (worst case inv = 96/0.0157 ≈ 6.1e3).
//   • The seed pass only accumulates the LSQ sums (no weight output) — the
//     final weights come from reprojecting against the refit endpoints.
//   • Endpoint ordering (the blue-contraction rule: sum(e0.rgb) must not
//     exceed sum(e1.rgb)) is applied BEFORE the final projection, so no
//     weight-reflection pass is needed.
//   • Weights are packed into the reversed-bit-order field on the fly, and
//     the 128-bit block is assembled with straight-line constant shifts
//     instead of a generic write_bits() helper.
//
// RESTRICTED SUBSET + BLOCK LAYOUT: see astc4x4.wgsl (single partition,
// CEM 12, 8-bit endpoints, 2-bit weights).
//
// The host selects this module only when the device reports shader-f16,
// falling back to astc4x4.wgsl otherwise. "high" never uses this.
enable f16;
alias h = f16;
alias h4 = vec4<f16>;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Fit { e0: h4, e1: h4, valid: bool };
fn proj_fit(pix: ptr<function, array<h4, 16>>, e0: h4, e1: h4) -> Fit {
  var out: Fit;
  out.valid = false;
  // dir pre-scaled by 32 to keep dd and the projection dots in f16's normal
  // range (see header). Spans below ~0.7 of an 8-bit step (dd₃₂ < 0.008,
  // possible only for non-8-bit sources) are treated as flat.
  let dir = (e1 - e0) * h(32.0);
  let dd = dot(dir, dir);
  if (dd < h(0.008)) { return out; }
  let inv = h(96.0) / dd; // 32·3/dd₃₂ ≡ 3/dd
  var sAA = h(0.0); var sBB = h(0.0); var sAB = h(0.0);
  var sAV = h4(0.0); var sBV = h4(0.0);
  var s_min = h(3.0); var s_max = h(0.0);
  // Value sums accumulate v − e0 (basis is affine, a + b = 1, so the fit
  // commutes with the shift): accumulators scale with the block span, keeping
  // f16 rounding a fraction of the span instead of ±1 level at high absolute
  // values.
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let vr = (*pix)[k] - e0;
    let s = clamp(floor(dot(vr, dir) * inv + h(0.5)), h(0.0), h(3.0));
    s_min = min(s_min, s); s_max = max(s_max, s);
    let b = s * h(1.0 / 3.0); let a = h(1.0) - b;
    sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b;
    sAV = sAV + a * vr; sBV = sBV + b * vr;
  }
  // Rank-1 guard: if every pixel projects to ONE level the system is
  // singular — det/numerators are pure f16 rounding noise and the solve
  // returns garbage endpoints. With ≥2 distinct levels
  // det = Σ_i<j (b_j − b_i)² ≥ 15·(1/3)² ≈ 1.67 — 0.5 separates cleanly.
  if (s_min == s_max) { return out; }
  let det = sAA * sBB - sAB * sAB;
  if (abs(det) < h(0.5)) { return out; }
  out.e0 = clamp(e0 + (sBB * sAV - sAB * sBV) / det, h4(0.0), h4(1.0));
  out.e1 = clamp(e0 + (sAA * sBV - sAB * sAV) / det, h4(0.0), h4(1.0));
  out.valid = true;
  return out;
}

fn q8(e: h4) -> vec4<u32> {
  return vec4<u32>(clamp(floor(e * h(255.0) + h(0.5)), h4(0.0), h4(255.0)));
}

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);
  let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  var pix: array<h4, 16>;
  var lo = h4(1.0);
  var hi = h4(0.0);
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0), mx);
    let px = h4(textureLoad(src_tex, p, 0));
    pix[i] = px; lo = min(lo, px); hi = max(hi, px);
  }

  let r = proj_fit(&pix, lo, hi);
  var e0 = lo;
  var e1 = hi;
  if (r.valid) { e0 = r.e0; e1 = r.e1; }
  var E0 = q8(e0);
  var E1 = q8(e1);

  // Blue-contraction ordering, applied before the weight pass so weights are
  // already oriented (no reflection needed).
  if (E0.x + E0.y + E0.z > E1.x + E1.y + E1.z) {
    let t = E0; E0 = E1; E1 = t;
  }
  let d0 = h4(vec4<f32>(E0)) * h(1.0 / 255.0);
  let d1 = h4(vec4<f32>(E1)) * h(1.0 / 255.0);

  // Weight pass, packing on the fly: weight k's lsb at bit 31−2k of the last
  // word, msb at bit 30−2k.
  var w3: u32 = 0u;
  // Same ×32 pre-scale as proj_fit; distinct 8-bit endpoints are ≥1/255
  // apart (dd₃₂ ≥ 0.0157), so the threshold only catches identical ones.
  let dir = (d1 - d0) * h(32.0);
  let dd = dot(dir, dir);
  if (dd >= h(0.008)) {
    let inv = h(96.0) / dd;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let s = u32(clamp(floor(dot(pix[k] - d0, dir) * inv + h(0.5)), h(0.0), h(3.0)));
      w3 = w3 | ((s & 1u) << (31u - 2u * k)) | (((s >> 1u) & 1u) << (30u - 2u * k));
    }
  }

  // Straight-line packing: block mode 0x042 @0, partitions−1=0 @11, CEM 12
  // @13, endpoints R0 R1 G0 G1 B0 B1 A0 A1 (8 bits each) from bit 17.
  let w0 = 0x042u | (12u << 13u) | (E0.x << 17u) | (E1.x << 25u);
  let w1 = (E1.x >> 7u) | (E0.y << 1u) | (E1.y << 9u) | (E0.z << 17u) | (E1.z << 25u);
  let w2 = (E1.z >> 7u) | (E0.w << 1u) | (E1.w << 9u);

  let o = bi * 4u;
  dst[o] = w0; dst[o + 1u] = w1; dst[o + 2u] = w2; dst[o + 3u] = w3;
}
