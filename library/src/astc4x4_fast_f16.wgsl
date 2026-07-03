// astc4x4 "fast" encoder — f16 variant (requires the shader-f16 feature).
// Same algorithm family as the f32 fallback in astc4x4.wgsl (per-block
// class selection, principal-axis seed → projection weight assignment with
// a fused least-squares refit → reproject), tuned for throughput:
//
//   • THREE block classes, picked per block from the loaded pixels (the
//     ASTC bit budget trades endpoint bits against weight bits, so a block
//     only pays for the channels it uses):
//       gray + opaque → CEM 0  (luminance), 5-bit weights, mode 0x253
//       opaque        → CEM 8  (RGB),       3-bit weights, mode 0x053
//       translucent   → CEM 12 (RGBA),      2-bit weights, mode 0x042
//     "gray" = every texel R == G == B exactly (f16 equality is exact for
//     8-bit sources), "opaque" = every texel A == 1. The gray path is
//     scalar — no covariance, no power iteration, no 4-D fit — and runs
//     [0,255]-integer f16 math like the BC5 kernel (exact endpoints, and
//     64/span ≤ 64 never overflows f16, unlike a [0,1]-domain 1/dd).
//   • All projection / refit math in f16 ([0,1] domain, colour paths).
//     The projection direction is pre-scaled by 32 — a shallow block
//     (endpoints ~1/255 apart) has dd ≈ 1.5e-5, where 3/dd ≈ 2e5 overflows
//     f16 (max 65504) to +inf and the projection dots go subnormal,
//     turning weights and the LSQ refit to garbage (banding on smooth
//     gradients). Scaling dir by 32 puts every intermediate in f16's
//     normal range; s = dot·(32·L/dd₃₂) is the same quantity (worst case
//     for L = 7 levels: inv = 224/0.0157 ≈ 1.4e4).
//   • Opaque blocks ship the quantised PCA extents directly (no LSQ fit
//     pass, 8 power iterations — see the endpoint-selection comment); only
//     the translucent CEM 12 path still refits, and its seed pass only
//     accumulates the LSQ sums — final weights always come from a
//     reprojection against the final endpoints.
//   • Endpoint ordering (the blue-contraction rule: sum(e0.rgb) must not
//     exceed sum(e1.rgb)) is applied BEFORE the final projection, so no
//     weight-reflection pass is needed.
//   • Weight streams are accumulated LSB-first into u32 words and placed
//     into the block's reversed-bit-order field with reverseBits() —
//     stream bit q lives at block bit 127 − q, so a whole stream word
//     maps onto a block word with a single bit reversal.
//   • The covariance moments stay in their OWN pass, deliberately: fusing
//     them into the load loop bc7-style (pixel-0 residuals, with or without
//     hoisting pixel 0 out of the loop) measured +5% GPU time on 4096²
//     (/ab A/B, 2026-07, M3) — the separate loop overlaps the 16 texture
//     loads' latency better than a longer in-loop dependency chain does.
//     Per-pass cost on the same rig, for future tuning: covariance+power-
//     iteration ≈ 18%, LSQ fit pass ≈ 24%, extents pass ≈ 6% of the kernel;
//     @workgroup_size 16×8 measured exactly at par with 8×8.
//
// RESTRICTED SUBSET + BLOCK LAYOUT: see astc4x4_ref.ts (single partition,
// CEM 0/8/12, 8-bit endpoints, plain-bit weight ISE, block modes
// 0x253/0x053/0x042).
//
// The host selects this module only when the device reports shader-f16,
// falling back to astc4x4.wgsl otherwise.
enable f16;
alias h = f16;
alias h4 = vec4<f16>;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// Fused projection + least-squares refit against the 4-level QUANT_4
// palette — only the translucent (CEM 12) path still refits: the 2-bit
// weight grid is coarse enough to need it, while the opaque paths get
// more from spending the same time elsewhere (see header).
struct Fit { e0: h4, e1: h4, valid: bool, wstream: u32 };
fn proj_fit(pix: ptr<function, array<h4, 16>>, e0: h4, e1: h4) -> Fit {
  var out: Fit;
  out.valid = false;
  out.wstream = 0u;
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
    out.wstream = out.wstream | (u32(s) << (2u * k));
    s_min = min(s_min, s); s_max = max(s_max, s);
    let b = s * h(1.0 / 3.0);
    let a = h(1.0) - b;
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

  // Load pass. gd tracks the largest chroma deviation — 0 iff the block is
  // exactly grayscale (equal 8-bit channels convert to identical f16s).
  var pix: array<h4, 16>;
  var lo = h4(1.0);
  var hi = h4(0.0);
  var mean = h4(0.0);
  var gd = h(0.0);
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0), mx);
    let px = h4(textureLoad(src_tex, p, 0));
    pix[i] = px; lo = min(lo, px); hi = max(hi, px);
    mean = mean + px;
    gd = max(gd, max(abs(px.x - px.y), abs(px.x - px.z)));
  }
  mean = mean * h(1.0 / 16.0);
  let opaque = lo.w == h(1.0);

  var w0: u32; var w1: u32; var w2: u32; var w3: u32;

  if (opaque && gd == h(0.0)) {
    // ---------------- Luminance path: CEM 0, 5-bit weights ----------------
    // Scalar [0,255]-integer domain. Endpoints at the exact extremes (the
    // 8-bit values round-trip f16 exactly); 32 palette levels make an LSQ
    // refit unnecessary.
    let L0 = u32(floor(lo.x * h(255.0) + h(0.5)));
    let L1 = u32(floor(hi.x * h(255.0) + h(0.5)));
    var s0 = 0u; var s1 = 0u; var s2 = 0u;
    if (L1 > L0) {
      let l0f = h(f32(L0));
      let sc = h(64.0) / h(f32(L1 - L0)); // span ≥ 1 → sc ≤ 64, no overflow
      // Exact nearest entry of the QUANT_32 grid: unq = 2w for w ≤ 15,
      // 2w + 2 for w ≥ 16 (the grid has a 4-wide gap at the middle, so
      // uniform rounding is wrong there). Evaluate the best candidate of
      // each half and keep the closer.
      for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let v = floor(pix[k].x * h(255.0) + h(0.5)) - l0f; // exact integer
        let u = clamp(v * sc, h(0.0), h(64.0));
        let wlo = clamp(floor(u * h(0.5) + h(0.5)), h(0.0), h(15.0));
        let whi = clamp(floor((u - h(2.0)) * h(0.5) + h(0.5)), h(16.0), h(31.0));
        let pick = abs(u - wlo * h(2.0)) <= abs(u - (whi * h(2.0) + h(2.0)));
        let w = u32(select(whi, wlo, pick));
        // Stream bit q = 5k + j; straddles handled with constant shifts.
        let off = 5u * k;
        if (off < 28u) { s0 = s0 | (w << off); }
        else if (off == 30u) { s0 = s0 | (w << 30u); s1 = s1 | (w >> 2u); }
        else if (off < 60u) { s1 = s1 | (w << (off - 32u)); }
        else if (off == 60u) { s1 = s1 | (w << 28u); s2 = s2 | (w >> 4u); }
        else { s2 = s2 | (w << (off - 64u)); }
      }
    }
    // Mode 0x253, partitions−1 = 0, CEM 0, L0 @17, L1 @25 (top bit spills
    // into w1 bit 0); stream word q∈[0,31] → block bits 127…96 via
    // reverseBits, q∈[32,63] → 95…64, q∈[64,79] → 63…48.
    w0 = 0x253u | (L0 << 17u) | (L1 << 25u);
    w1 = (L1 >> 7u) | reverseBits(s2);
    w2 = reverseBits(s1);
    w3 = reverseBits(s0);
  } else {
    // ------------- Colour paths: shared PCA seed + LSQ refit --------------
    // Seed endpoints from the block's principal colour axis (covariance
    // power-iteration, seeded with the bbox diagonal). The bbox diagonal is
    // sign-blind: on anti-correlated channels (normal maps, hue edges) it
    // points across the data instead of along it, and the LSQ refit — which
    // fits endpoints GIVEN the projection weights — can't recover from a
    // wrong axis. Deviations are pre-scaled ×16 so covariance entries for
    // shallow blocks stay in f16's normal range (span ~1/255 → d² ≈ 1e-3)
    // while full-range sums stay ≤4096; the iteration renormalises by the
    // max component (a plain length() of the matvec output could overflow
    // f16), so only the direction survives.
    var seed_lo = lo;
    var seed_hi = hi;
    var c0v = h4(0.0);
    var c1v = h4(0.0);
    var c2v = h4(0.0);
    var c3v = h4(0.0);
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let d = (pix[k] - mean) * h(16.0);
      c0v = c0v + d.x * d;
      c1v = c1v + d.y * d;
      c2v = c2v + d.z * d;
      c3v = c3v + d.w * d;
    }
    var axis = hi - lo;
    var axis_ok = true;
    // 4 shared iterations, then 4 more for opaque blocks only — two
    // FIXED-bound loops rather than one divergent trip count, so both
    // unroll. Opaque blocks need the converged axis (it IS the endpoint
    // quality there; 4 was under-converged on noisy 4-D content), while
    // translucent blocks' LSQ refit absorbs residual axis error — their
    // extra 4 steps measured exactly 0.000 dB on the alpha card for
    // ~5% GPU.
    for (var it: u32 = 0u; it < 4u; it = it + 1u) {
      let nv = h4(dot(c0v, axis), dot(c1v, axis), dot(c2v, axis), dot(c3v, axis));
      let m = max(max(abs(nv.x), abs(nv.y)), max(abs(nv.z), abs(nv.w)));
      if (m < h(1e-4)) { axis_ok = false; break; }
      axis = nv / m;
    }
    if (axis_ok && opaque) {
      for (var it: u32 = 0u; it < 4u; it = it + 1u) {
        let nv = h4(dot(c0v, axis), dot(c1v, axis), dot(c2v, axis), dot(c3v, axis));
        let m = max(max(abs(nv.x), abs(nv.y)), max(abs(nv.z), abs(nv.w)));
        if (m < h(1e-4)) { axis_ok = false; break; }
        axis = nv / m;
      }
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
      seed_lo = clamp(mean + t_min * axis, h4(0.0), h4(1.0));
      seed_hi = clamp(mean + t_max * axis, h4(0.0), h4(1.0));
    }

    // Endpoint selection. OPAQUE blocks (CEM 8, 8-level weights) quantise
    // the PCA-extent seed directly, BC7-style — the LSQ fit pass measured
    // +26% GPU for −0.06..+0.31 dB against the converged 8-step axis
    // (/ab + PSNR A/B, 2026-07): at 8 weight levels the projection is fine
    // enough that a good axis, not a refit, carries the quality.
    // TRANSLUCENT blocks (CEM 12) keep the fit: 4 levels are coarse enough
    // that dropping it costs 0.5+ dB. Its result is clamped to the block
    // bbox: on multi-cluster blocks the unconstrained solve extrapolates
    // far outside the block's colours and the per-channel [0,1] clamp then
    // bends the hue — fringe pixels decode to colours that exist nowhere
    // in the block (and the bbox constraint also measures better in plain
    // SSE, +1.8 dB on the colour test card).
    var e0 = lo;
    var e1 = hi;
    var fitStream = 0u;
    var haveFitWeights = false;
    if (opaque) {
      // Bbox-clamped like the fit output: on multi-cluster blocks the axis
      // extents overshoot the data per-channel and decode to colours that
      // exist nowhere in the block (the odd-size padding gate caught a
      // −3.9 dB crop without this).
      e0 = clamp(seed_lo, lo, hi);
      e1 = clamp(seed_hi, lo, hi);
    } else {
      let r = proj_fit(&pix, seed_lo, seed_hi);
      if (r.valid) {
        e0 = clamp(r.e0, lo, hi);
        e1 = clamp(r.e1, lo, hi);
        fitStream = r.wstream;
        haveFitWeights = true;
      }
    }
    var E0 = q8(e0);
    var E1 = q8(e1);

    // Blue-contraction ordering, applied before the weight pass so weights
    // are already oriented (no reflection needed).
    var swapped = false;
    if (E0.x + E0.y + E0.z > E1.x + E1.y + E1.z) {
      let t = E0; E0 = E1; E1 = t;
      swapped = true;
    }
    let d0 = h4(vec4<f32>(E0)) * h(1.0 / 255.0);
    let d1 = h4(vec4<f32>(E1)) * h(1.0 / 255.0);

    // Weight pass against the decoded endpoints. Same ×32 pre-scale as
    // proj_fit; distinct 8-bit endpoints are ≥1/255 apart (dd₃₂ ≥ 0.0157),
    // so the flat threshold only catches identical ones.
    let dir = (d1 - d0) * h(32.0);
    let dd = dot(dir, dir);
    if (opaque) {
      // CEM 8: 3-bit weights, stream bit q = 3k.
      var s0 = 0u; var s1 = 0u;
      if (dd >= h(0.008)) {
        let inv = h(224.0) / dd;
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
          let w = u32(clamp(floor(dot(pix[k] - d0, dir) * inv + h(0.5)), h(0.0), h(7.0)));
          let off = 3u * k;
          if (off < 30u) { s0 = s0 | (w << off); }
          else if (off == 30u) { s0 = s0 | (w << 30u); s1 = s1 | (w >> 2u); }
          else { s1 = s1 | (w << (off - 32u)); }
        }
      }
      // Mode 0x053, CEM 8 @13, endpoints R0 R1 G0 G1 B0 B1 from bit 17.
      w0 = 0x053u | (8u << 13u) | (E0.x << 17u) | (E1.x << 25u);
      w1 = (E1.x >> 7u) | (E0.y << 1u) | (E1.y << 9u) | (E0.z << 17u) | (E1.z << 25u);
      w2 = (E1.z >> 7u) | reverseBits(s1);
      w3 = reverseBits(s0);
    } else {
      // CEM 12: 2-bit weights, stream bit q = 2k (single stream word).
      // Valid fits ship the FIT-PASS weights instead of reprojecting —
      // worth a whole 16-pixel pass for −0.09 dB on the alpha card (/ab +
      // PSNR A/B, 2026-07; the pre-adaptive-CEM encoder rejected this same
      // trade when EVERY block was CEM 12 — now only translucent blocks
      // pay it). The blue-contraction swap is a full reflection w → 3−w,
      // i.e. bitwise NOT of the packed stream. Invalid fits (rank-1 /
      // degenerate) fall back to reprojection against the bbox endpoints.
      var s0 = 0u;
      if (haveFitWeights) {
        s0 = select(fitStream, ~fitStream, swapped);
      } else if (dd >= h(0.008)) {
        let inv = h(96.0) / dd;
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
          let w = u32(clamp(floor(dot(pix[k] - d0, dir) * inv + h(0.5)), h(0.0), h(3.0)));
          s0 = s0 | (w << (2u * k));
        }
      }
      // Mode 0x042, CEM 12 @13, endpoints R0 R1 G0 G1 B0 B1 A0 A1 from 17.
      w0 = 0x042u | (12u << 13u) | (E0.x << 17u) | (E1.x << 25u);
      w1 = (E1.x >> 7u) | (E0.y << 1u) | (E1.y << 9u) | (E0.z << 17u) | (E1.z << 25u);
      w2 = (E1.z >> 7u) | (E0.w << 1u) | (E1.w << 9u);
      w3 = reverseBits(s0);
    }
  }

  let o = bi * 4u;
  dst[o] = w0; dst[o + 1u] = w1; dst[o + 2u] = w2; dst[o + 3u] = w3;
}
