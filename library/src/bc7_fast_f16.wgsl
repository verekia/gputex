// bc7 "fast" encoder — f16 variant (requires the shader-f16 feature).
// Same algorithm family as the f32 fast path in bc7.wgsl (principal-axis
// seed at the exact projection extents → quantise → one projection-based
// index-assignment pass), tuned for throughput:
//
//   • All projection math in f16 ([0,1] domain). ~2× ALU throughput on
//     f16-capable GPUs. The projection direction is pre-scaled by 32:
//     a shallow block (endpoints ~1/255 apart) has dd = dot(dir,dir) ≈ 1.5e-5,
//     where 15/dd ≈ 10⁶ overflows f16 (max 65504) to +inf and the products
//     inside the projection dot are subnormal — the indices turn to garbage
//     (visible as banding on smooth gradients). Scaling dir by 32 multiplies
//     the dots by 32 and dd by 1024; s = dot·(32·15/dd₃₂) is the same
//     quantity with every intermediate in f16's normal range (worst case
//     inv = 480/0.0157 ≈ 3.0e4 < 65504).
//   • NO least-squares refit, unlike the BC1/BC5/ASTC fast paths: with the
//     seed already on the principal axis at the exact projection extents,
//     mode 6's fine 16-level palette leaves the refit ≤0.05 dB on the colour
//     card, ≤0.15 dB on the normal card and +0.03 dB on the channel-packed
//     packed-materials atlas — not worth its two extra 16-pixel passes. The
//     coarse 4-level formats DO need it (dropping it there costs 0.5–1.3 dB).
//   • A MODE 1 (2-subset) candidate was built and evaluated (2026-07): it
//     buys ~+1.3 dB on multi-modal content (channel-packed atlases, where
//     35–45% of blocks take it; normal-map facets) but its candidate
//     evaluation costs up to ~3× the mode-6 pass on exactly that content,
//     for a break-even outcome against contemporary encoders — dropped in
//     favour of speed. The CPU reference decoder keeps mode 1 support
//     (bc7_ref.ts) should it return as an opt-in.
//   • Indices are packed into two u32 nibble words ON THE FLY during the
//     projection pass — no array<u32,16> private array. The BC7 anchor
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

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);
  let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  // Load pass, with the covariance moments FUSED in (no separate 16-pixel
  // pass): d = (px − pixel0)·16, relative to the block's first pixel so the
  // accumulators scale with the block's span — raw Σv·vᵀ moments would
  // cancel catastrophically in f16 — and pre-scaled ×16 so shallow blocks
  // (span ~1/255 → d² ≈ 1e-3) clear the subnormal floor while full-range
  // sums stay ≤4096. C = Σddᵀ − (Σd)(Σd)ᵀ/16 is the ×256-scaled covariance.
  var pix: array<h4, 16>;
  var lo = h4(1.0);
  var hi = h4(0.0);
  var gd = h(0.0);
  var p0v = h4(0.0);
  var sd = h4(0.0);
  var c0v = h4(0.0);
  var c1v = h4(0.0);
  var c2v = h4(0.0);
  var c3v = h4(0.0);
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0), mx);
    let px = h4(textureLoad(src_tex, p, 0));
    pix[i] = px; lo = min(lo, px); hi = max(hi, px);
    gd = max(gd, max(abs(px.x - px.y), abs(px.x - px.z)));
    if (i == 0u) { p0v = px; }
    let d = (px - p0v) * h(16.0);
    sd = sd + d;
    c0v = c0v + d.x * d;
    c1v = c1v + d.y * d;
    c2v = c2v + d.z * d;
    c3v = c3v + d.w * d;
  }
  let mean = p0v + sd * h(1.0 / 256.0);
  // Mean-correction via sd4·sd4ᵀ with sd4 = Σd/4: (Σd)(Σd)ᵀ/16 with every
  // product ≤4096 (a direct Σd·Σdᵀ could hit 65536 and overflow f16).
  let sd4 = sd * h(0.25);
  c0v = c0v - sd4.x * sd4;
  c1v = c1v - sd4.y * sd4;
  c2v = c2v - sd4.z * sd4;
  c3v = c3v - sd4.w * sd4;

  // Seed endpoints from the block's principal colour axis (covariance
  // power-iteration, seeded with the bbox diagonal — same family as the BC1
  // 'high' path). The bbox diagonal is sign-blind: on anti-correlated
  // channels (normal maps, hue edges) it points across the data instead of
  // along it, and the LSQ refit — which fits endpoints GIVEN the projection
  // indices — can't recover from a wrong axis. The iteration renormalises by
  // the max component (a plain length() of the matvec output could overflow
  // f16), so only the direction survives.
  var seed_lo = lo;
  var seed_hi = hi;
  // GRAY + opaque blocks (every texel R == G == B, A == 1 — exact in f16
  // for 8-bit sources) have their principal axis analytically: (1,1,1,0)/√3,
  // with projection extents at the luma min/max. Skip the power iteration
  // AND the 16-dot extents pass — the seed is exact, so quality is
  // identical, and the grayscale-heavy formats (roughness/AO/displacement)
  // drop a third of their per-block work.
  if (lo.w == h(1.0) && gd == h(0.0)) {
    seed_lo = h4(lo.x, lo.x, lo.x, h(1.0));
    seed_hi = h4(hi.x, hi.x, hi.x, h(1.0));
  } else {
    var axis = hi - lo;
    var axis_ok = true;
    // 8 iterations: 4 was under-converged on noisy 4-D blocks (heavily
    // downscaled photographic/channel-packed content) — going to 8 measured
    // +0.75 dB on the normal card, +0.12 colour, +0.08 packed-materials, and
    // matches the f32 fallback's iteration count. Four extra 4-dot matvecs
    // per block are noise next to the index pass.
    for (var it: u32 = 0u; it < 8u; it = it + 1u) {
      let nv = h4(dot(c0v, axis), dot(c1v, axis), dot(c2v, axis), dot(c3v, axis));
      let m = max(max(abs(nv.x), abs(nv.y)), max(abs(nv.z), abs(nv.w)));
      if (m < h(1e-4)) { axis_ok = false; break; }
      axis = nv / m;
    }
    if (axis_ok) {
      axis = axis / length(axis);
      // Exact projection extents along the axis. (A Rayleigh-quotient span
      // estimate was tried in place of this pass — it saves 16 dots but costs
      // 0.1–0.8 dB and 4–10× on the worst-easy-block gate: σ misjudges
      // two-cluster and outlier blocks and the quantised weight grid can't
      // recover. The pass stays.)
      var t_min = h(4.0);
      var t_max = h(-4.0);
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let t = dot(pix[k] - mean, axis);
        t_min = min(t_min, t);
        t_max = max(t_max, t);
      }
      seed_lo = clamp(mean + t_min * axis, h4(0.0), h4(1.0));
      seed_hi = clamp(mean + t_max * axis, h4(0.0), h4(1.0));
    }
  }

  // Fit from the principal-axis seed (bbox on degenerate blocks), then
  // quantise the refit endpoints. The refit is clamped to the block bbox: on
  // multi-cluster blocks the unconstrained solve extrapolates far outside
  // the block's colours and the per-channel [0,1] clamp then bends the hue —
  // fringe pixels decode to colours that exist nowhere in the block.
  // Constraining to the bbox also measures better in plain SSE (+1.3 dB on
  // the colour test card).
  // Quantise the PCA-extents seed directly — no LSQ refit (see header).
  var ep0 = pick_ep(seed_lo);
  var ep1 = pick_ep(seed_hi);

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
