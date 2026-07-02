// bc7 "fast" encoder — f16 variant (requires the shader-f16 feature).
// Same algorithm family as the f32 fast path in bc7.wgsl (bbox seed →
// projection-based index assignment with a fused least-squares refit →
// reproject), tuned for throughput:
//
//   • All projection / refit math in f16 ([0,1] domain). ~2× ALU throughput
//     on f16-capable GPUs. The projection direction is pre-scaled by 32:
//     a shallow block (endpoints ~1/255 apart) has dd = dot(dir,dir) ≈ 1.5e-5,
//     where 15/dd ≈ 10⁶ overflows f16 (max 65504) to +inf and the products
//     inside the projection dot are subnormal — the indices and the LSQ refit
//     feeding on them turn to garbage (visible as banding on smooth
//     gradients). Scaling dir by 32 multiplies the dots by 32 and dd by 1024;
//     s = dot·(32·15/dd₃₂) is the same quantity with every intermediate in
//     f16's normal range (worst case inv = 480/0.0157 ≈ 3.0e4 < 65504).
//   • The LSQ seed pass projects against the RAW bbox endpoints — quantising
//     the seed first (pick_ep) costs two extra quantisation searches and
//     doesn't measurably change where the refit lands.
//   • Indices are packed into two u32 nibble words ON THE FLY during the
//     final projection pass — no array<u32,16> private array. The BC7 anchor
//     reflection (i → 15−i) is then just a bitwise NOT of both words.
//   • The 128-bit block is assembled with straight-line constant shifts
//     instead of a generic write_bits() helper (whose dynamic word indexing
//     defeats register promotion of the output array).
//
// The host selects this module only when the device reports shader-f16,
// falling back to bc7.wgsl otherwise. "high" never uses this.
//
// MODE 6 BIT LAYOUT (LSB-first): see bc7.wgsl. Summary:
//   w0: mode(7 bits, 0x40) R0 R1 G0 G1[3:0]
//   w1: G1[6:4] B0 B1 A0 A1 P0
//   w2: P1, pixel0 index (3 bits), pixels 1..7 (4 bits each)
//   w3: pixels 8..15 (4 bits each)
enable f16;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
alias h = f16;
alias h4 = vec4<f16>;

// Quantise an ideal endpoint (h4 in [0,1]) to 7-bit + p-bit, choosing the
// p-bit with the lower quantisation error. `eight` is the decoded value the
// hardware will interpolate with, back in [0,1].
struct Ep { seven: vec4<u32>, eight: h4, p: u32 };
fn pick_ep(ideal01: h4) -> Ep {
  let ideal = ideal01 * h(255.0);
  let q0 = clamp(floor(ideal * h(0.5) + h(0.5)), h4(0.0), h4(127.0));        // p=0
  let e0 = q0 * h(2.0);
  let q1 = clamp(floor((ideal - h(1.0)) * h(0.5) + h(0.5)), h4(0.0), h4(127.0)); // p=1
  let e1 = q1 * h(2.0) + h(1.0);
  let d0 = e0 - ideal; let d1 = e1 - ideal;
  if (dot(d1, d1) < dot(d0, d0)) { return Ep(vec4<u32>(q1), e1 * h(1.0 / 255.0), 1u); }
  return Ep(vec4<u32>(q0), e0 * h(1.0 / 255.0), 0u);
}

// One pass over the block: project every pixel onto the e0→e1 line and
// accumulate the least-squares normal-equation sums; solve for the refit
// endpoints. Indices are NOT produced here — the caller reprojects against
// the quantised refit endpoints anyway.
//
// The value sums accumulate v − e0, not v: the basis is affine (a + b = 1),
// so fitting the shifted data and adding e0 back is the same fit, but the
// accumulators scale with the block's span instead of its absolute level —
// on a shallow dark block, f16 rounding of absolute sums (ulp ≈ 0.12 of an
// 8-bit level per add) drifts the refit endpoints by ±1 level.
struct Fit { e0: h4, e1: h4, valid: bool };
fn proj_fit(pix: ptr<function, array<h4, 16>>, e0: h4, e1: h4) -> Fit {
  var out: Fit;
  out.valid = false;
  // dir pre-scaled by 32 to keep dd and the projection dots in f16's normal
  // range (see header). Spans below ~0.7 of an 8-bit step (dd₃₂ < 0.008,
  // possible only for non-8-bit sources) are treated as flat — encoding them
  // flat is under half a level of error, while running the math on them risks
  // inv overflowing to +inf.
  let dir = (e1 - e0) * h(32.0);
  let dd = dot(dir, dir);
  if (dd < h(0.008)) { return out; }
  let inv = h(480.0) / dd; // 32·15/dd₃₂ ≡ 15/dd
  var sAA = h(0.0); var sBB = h(0.0); var sAB = h(0.0);
  var sAV = h4(0.0); var sBV = h4(0.0);
  var s_min = h(15.0); var s_max = h(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let vr = (*pix)[k] - e0;
    let s = clamp(floor(dot(vr, dir) * inv + h(0.5)), h(0.0), h(15.0));
    s_min = min(s_min, s); s_max = max(s_max, s);
    let b = s * h(1.0 / 15.0); let a = h(1.0) - b;
    sAA = sAA + a * a; sBB = sBB + b * b; sAB = sAB + a * b;
    sAV = sAV + a * vr; sBV = sBV + b * vr;
  }
  // Rank-1 guard: if every pixel projects to ONE level the system is
  // singular — det/numerators are pure f16 rounding noise and the solve
  // returns garbage endpoints. With ≥2 distinct levels
  // det = Σ_i<j (b_j − b_i)² ≥ 15/225 ≈ 0.067, so 0.02 is a safe floor.
  if (s_min == s_max) { return out; }
  let det = sAA * sBB - sAB * sAB;
  if (abs(det) < h(0.02)) { return out; }
  out.e0 = clamp(e0 + (sBB * sAV - sAB * sBV) / det, h4(0.0), h4(1.0));
  out.e1 = clamp(e0 + (sAA * sBV - sAB * sAV) / det, h4(0.0), h4(1.0));
  out.valid = true;
  return out;
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

  // Seed fit from the raw bbox, then quantise the refit endpoints.
  let r = proj_fit(&pix, lo, hi);
  var ep0: Ep;
  var ep1: Ep;
  if (r.valid) { ep0 = pick_ep(r.e0); ep1 = pick_ep(r.e1); }
  else         { ep0 = pick_ep(lo);   ep1 = pick_ep(hi);   }

  // Final projection against the decoded endpoints, packing the 4-bit indices
  // into two nibble words as we go (pixel k → bits 4k..4k+3 of ilo/ihi).
  var ilo: u32 = 0u;
  var ihi: u32 = 0u;
  // Same ×32 pre-scale as proj_fit; distinct quantised endpoints are ≥1/255
  // apart, i.e. dd₃₂ ≥ 0.0157, so the flat-block threshold only catches
  // truly identical endpoints.
  let dir = (ep1.eight - ep0.eight) * h(32.0);
  let dd = dot(dir, dir);
  if (dd >= h(0.008)) {
    let inv = h(480.0) / dd;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      let s = clamp(floor(dot(pix[k] - ep0.eight, dir) * inv + h(0.5)), h(0.0), h(15.0));
      ilo = ilo | (u32(s) << (k * 4u));
    }
    for (var k: u32 = 8u; k < 16u; k = k + 1u) {
      let s = clamp(floor(dot(pix[k] - ep0.eight, dir) * inv + h(0.5)), h(0.0), h(15.0));
      ihi = ihi | (u32(s) << ((k - 8u) * 4u));
    }
  }

  // Anchor rule — pixel 0's index MSB must be 0. Swapping endpoints reflects
  // every index (i → 15−i), which on packed nibbles is a bitwise NOT.
  if ((ilo & 0x8u) != 0u) {
    let t = ep0; ep0 = ep1; ep1 = t;
    ilo = ~ilo; ihi = ~ihi;
  }

  // Straight-line mode-6 packing (see layout above).
  let e0 = ep0.seven;
  let e1 = ep1.seven;
  let w0 = 0x40u | (e0.x << 7u) | (e1.x << 14u) | (e0.y << 21u) | (e1.y << 28u);
  let w1 = (e1.y >> 4u) | (e0.z << 3u) | (e1.z << 10u) | (e0.w << 17u) | (e1.w << 24u) | (ep0.p << 31u);
  let w2 = ep1.p | ((ilo & 0x7u) << 1u) | (ilo & 0xFFFFFFF0u);
  let w3 = ihi;

  let o = bi * 4u;
  dst[o] = w0; dst[o + 1u] = w1; dst[o + 2u] = w2; dst[o + 3u] = w3;
}
