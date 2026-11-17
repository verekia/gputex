// EAC R11 (unsigned) compute shader encoder — EXACT-VALUE f16 module
// (requires the shader-f16 feature; eac_r11.wgsl is the f32 fallback).
//
// Same algorithm as eac_r11.wgsl, with the per-texel vectors (targets,
// distances, magnitudes, thresholds, residuals) held in f16. Every one of
// those is an integer or half-integer f16 represents exactly — targets
// relative to the block minimum are ≤ 2047, magnitudes ≤ 1740, thresholds
// are floored to integers whenever a multiplier is in play, and
// half-integers only occur with multiplier 0, whose blocks span a few
// levels — so the output matches the f32 module byte for byte. The residual
// sums (squares up to 2^24) stay f32.
//
// Each invocation encodes one 4x4 block of the source's R channel into an
// 8-byte EAC R11 block (2 x u32 in the destination storage buffer). The
// mobile-side analogue of BC4: 4 bpp, one channel, 11-bit reconstruction.
//
// Block layout (big-endian bit order):
//   byte 0     : base codeword
//   byte 1     : multiplier (high nibble) | modifier table (low nibble)
//   bytes 2..7 : 16 × 3-bit modifier indices, pixel 0 in the TOP three
//                bits; pixels are COLUMN-major (pixel = x·4 + y)
// Decode, in 11-bit units: clamp(base·8 + 4 + modifier·M, 0, 2047) with
// M = multiplier·8, or 1 when the multiplier is 0; the sampler returns
// value/2047. Every table is symmetric about −½: its eight modifiers are
// −½ ± (p + ½) for the four positive entries p, so the palette is a centre
// C = base·8 + 4 − M/2 plus/minus four magnitudes (2p + 1)·M/2.
//
// Algorithm — everything in EXACT 11-bit integer (or half-integer)
// arithmetic, so the f16 module, the WebGL port and the CPU reference
// (eac_r11_ref.ts) agree byte for byte:
//   • TARGETS: each texel's 8-bit value v maps to T = round(v·2047/255)
//     (0 → 0 and 255 → 2047 exactly), held relative to the block minimum.
//   • CANDIDATES: the block's 8-bit span and its shape index a 164-entry
//     table of two (modifier table, multiplier rule) picks. The shape is the
//     mean distance of the texels from the mid-range, as a fraction of the
//     half-span, in four buckets: low for blocks that sit near their middle
//     (tables with tight inner steps win), high for two-level blocks (tables
//     with their steps at the ends win). The multiplier rule is
//     round(half-span / largest magnitude) — the table just covers the
//     block — or that plus one, or multiplier 0 (1/8-step precision, for
//     spans of a few levels). The table was fitted offline, per cell, on
//     roughness / AO / displacement / mask / normal-channel textures (best
//     pair on this encoder's own policy, every texture weighted alike).
//   • SCORE: each candidate is scored with the palette centre on the block's
//     mid-range: per texel, the magnitude whose threshold interval holds its
//     distance from the centre (thresholds are the midpoints of neighbouring
//     magnitudes; the outermost magnitude is clamped to the [0, 2047] range
//     on each side). The pass sums the squared residuals E and the SIGNED
//     residuals S.
//   • BASE: S / 16 is the mean residual, so stepping the base by
//     round(S / 128) zeroes it, and with the texels' magnitudes unchanged
//     that leaves an error of E − 16·step·S + 1024·step². The candidate with
//     the lower predicted error wins, at its stepped base — the phase of the
//     palette against the data is fitted without scoring any other base.
//   • INDICES: against the final palette, with every magnitude clamped to
//     [0, 2047] on the texel's side (so texels at 0 or 255 are matched
//     exactly by an over-reaching palette: masks keep true black and white),
//     a texel's index is the number of thresholds below its distance. The
//     3-bit indices are packed as float fields Σ i·8^k (exact below 2^24),
//     one 24-bit word per pair of columns.
// Two scores + one index pass per block, each four vec4 steps. Measured
// over 17 textures (mean PSNR): a third pick is +0.3 dB for +24% GPU time;
// one pick is −0.8 dB; scoring on half the texels −0.3 dB.

enable f16;
alias h = f16;
alias h4 = vec4<f16>;

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

// 2p + 1 for the four positive modifiers p of each table.
const QV = array<vec4<f32>, 16>(
  vec4<f32>(5.0, 11.0, 17.0, 29.0), vec4<f32>(5.0, 13.0, 19.0, 25.0),
  vec4<f32>(3.0, 9.0, 15.0, 25.0),  vec4<f32>(3.0, 7.0, 11.0, 25.0),
  vec4<f32>(5.0, 11.0, 15.0, 23.0), vec4<f32>(5.0, 13.0, 17.0, 21.0),
  vec4<f32>(7.0, 13.0, 15.0, 21.0), vec4<f32>(5.0, 9.0, 15.0, 21.0),
  vec4<f32>(3.0, 11.0, 15.0, 19.0), vec4<f32>(3.0, 9.0, 15.0, 19.0),
  vec4<f32>(3.0, 7.0, 15.0, 19.0),  vec4<f32>(3.0, 9.0, 13.0, 19.0),
  vec4<f32>(5.0, 7.0, 13.0, 19.0),  vec4<f32>(1.0, 3.0, 5.0, 19.0),
  vec4<f32>(7.0, 11.0, 15.0, 17.0), vec4<f32>(5.0, 9.0, 13.0, 17.0));
// Per table, with q0..q3 the row above: the two inner thresholds
// (q0 + q1) / 2 and (q1 + q2) / 2, then 4·q3 + ½ and 1 / (8·q3) for the
// covering-multiplier rule.
const QT = array<vec4<f32>, 16>(
  vec4<f32>(8.0, 14.0, 116.5, 1.0 / 232.0), vec4<f32>(9.0, 16.0, 100.5, 1.0 / 200.0),
  vec4<f32>(6.0, 12.0, 100.5, 1.0 / 200.0), vec4<f32>(5.0, 9.0, 100.5, 1.0 / 200.0),
  vec4<f32>(8.0, 13.0, 92.5, 1.0 / 184.0), vec4<f32>(9.0, 15.0, 84.5, 1.0 / 168.0),
  vec4<f32>(10.0, 14.0, 84.5, 1.0 / 168.0), vec4<f32>(7.0, 12.0, 84.5, 1.0 / 168.0),
  vec4<f32>(7.0, 13.0, 76.5, 1.0 / 152.0), vec4<f32>(6.0, 12.0, 76.5, 1.0 / 152.0),
  vec4<f32>(5.0, 11.0, 76.5, 1.0 / 152.0), vec4<f32>(6.0, 11.0, 76.5, 1.0 / 152.0),
  vec4<f32>(6.0, 10.0, 76.5, 1.0 / 152.0), vec4<f32>(2.0, 4.0, 76.5, 1.0 / 152.0),
  vec4<f32>(9.0, 13.0, 68.5, 1.0 / 136.0), vec4<f32>(7.0, 11.0, 68.5, 1.0 / 136.0));

// (Span, shape) → two candidates, 6 bits each (low first): table in bits
// 0..3, rule in bits 4..5 (0 = covering multiplier, 1 = that + 1,
// 2 = multiplier 0). Index = span bucket · 4 + shape bucket; span bucket =
// the 8-bit span below 32, then 8 sub-buckets per octave of span, 40 for
// full-range blocks (span 255: both black and white).
const LUT = array<u32, 164>(
  0xb63u, 0xb63u, 0xb63u, 0xb63u, 0xae8u, 0xae8u, 0xae8u, 0xae8u, 0xae4u, 0xae4u,
  0x92bu, 0x969u, 0x360u, 0x360u, 0x84du, 0x360u, 0x80du, 0x360u, 0x360u, 0x360u,
  0x80du, 0x80du, 0x360u, 0x340u, 0x342u, 0x342u, 0x74du, 0x74du, 0x74du, 0x74du,
  0x0cdu, 0x303u, 0x35du, 0x35du, 0x0ddu, 0x74cu, 0x0ddu, 0x0ddu, 0x2ddu, 0x09du,
  0x743u, 0x743u, 0x743u, 0x743u, 0x0ddu, 0x0ddu, 0x0ddu, 0x743u, 0x743u, 0x743u,
  0x743u, 0x0cfu, 0x2c3u, 0x2c3u, 0x0cbu, 0x2cfu, 0x2cau, 0x2cau, 0x2cfu, 0x3ceu,
  0x24au, 0x24au, 0x3c9u, 0x3c6u, 0x34bu, 0x289u, 0x3c9u, 0x3ceu, 0x348u, 0x2cau,
  0x24fu, 0x38fu, 0x34bu, 0x2cau, 0x24fu, 0x388u, 0x344u, 0x28bu, 0x24bu, 0x308u,
  0x34au, 0x2cdu, 0x2cau, 0x202u, 0x340u, 0x28bu, 0x2c7u, 0x1c5u, 0x2c7u, 0x2c7u,
  0x2c7u, 0x141u, 0x1c2u, 0x1c2u, 0x147u, 0x144u, 0x084u, 0x084u, 0x081u, 0x044u,
  0x0c2u, 0x0c2u, 0x081u, 0x7c1u, 0x0c2u, 0x0c2u, 0x042u, 0x081u, 0x0c2u, 0x0c2u,
  0x042u, 0x05bu, 0x080u, 0x080u, 0x081u, 0x04fu, 0x002u, 0x002u, 0x040u, 0x38fu,
  0x080u, 0x080u, 0x040u, 0x00eu, 0x002u, 0x002u, 0x200u, 0x00eu, 0x080u, 0x080u,
  0x3c9u, 0x385u, 0x2c2u, 0x2c2u, 0x089u, 0x20eu, 0x08bu, 0x08bu, 0x1c9u, 0x14eu,
  0x084u, 0x084u, 0x207u, 0x10eu, 0x042u, 0x042u, 0x081u, 0x28eu, 0x042u, 0x042u,
  0x042u, 0x14eu, 0x08bu, 0x08bu, 0x2c9u, 0x389u, 0x009u, 0x009u, 0x009u, 0x14eu,
  0x655u, 0x655u, 0x655u, 0x655u);

// Edge blocks: one quad's R from four clamped texel loads, in gather order
// (x=(0,1) y=(1,1) z=(1,0) w=(0,0)).
fn load_quad(qo: vec2<i32>, mx: vec2<i32>) -> vec4<f32> {
  return vec4<f32>(
    textureLoad(src_tex, min(qo + vec2<i32>(0, 1), mx), 0).r,
    textureLoad(src_tex, min(qo + vec2<i32>(1, 1), mx), 0).r,
    textureLoad(src_tex, min(qo + vec2<i32>(1, 0), mx), 0).r,
    textureLoad(src_tex, min(qo, mx), 0).r);
}

// Index weights 8^k for the big-endian, column-major index field. Quads 0
// and 1 hold rows 0..1 of their two columns, quads 2 and 3 rows 2..3
// (gather order x,y,z,w = (0,1),(1,1),(1,0),(0,0)).
const WA = vec4<f32>(262144.0, 64.0, 512.0, 2097152.0);
const WB = vec4<f32>(4096.0, 1.0, 8.0, 32768.0);

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid_raw: vec3<u32>) {
  // Row-band encodes dispatch a slice of the block grid starting at row y0.
  let gid = vec3<u32>(gid_raw.x, gid_raw.y + params.y0, gid_raw.z);
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) {
    return;
  }
  let bi = gid.y * params.blocks_x + gid.x;
  let org = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);

  // Load 4×4 R as quad-major vec4s in gather order, straight to 11-bit
  // targets: round(v·2047/255) = floor(unorm·2047 + ½) (the nearest a
  // product gets to a rounding boundary is 0.002, far above f32 error).
  var u: array<vec4<f32>, 4>;
  if (u32(org.x) + 4u <= params.width && u32(org.y) + 4u <= params.height) {
    let cc = (vec2<f32>(org) + 1.0) / vec2<f32>(textureDimensions(src_tex));
    u[0] = floor(textureGather(0, src_tex, smp, cc) * 2047.0 + 0.5);
    u[1] = floor(textureGather(0, src_tex, smp, cc, vec2<i32>(2, 0)) * 2047.0 + 0.5);
    u[2] = floor(textureGather(0, src_tex, smp, cc, vec2<i32>(0, 2)) * 2047.0 + 0.5);
    u[3] = floor(textureGather(0, src_tex, smp, cc, vec2<i32>(2, 2)) * 2047.0 + 0.5);
  } else {
    let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    u[0] = floor(load_quad(org + vec2<i32>(0, 0), mx) * 2047.0 + 0.5);
    u[1] = floor(load_quad(org + vec2<i32>(2, 0), mx) * 2047.0 + 0.5);
    u[2] = floor(load_quad(org + vec2<i32>(0, 2), mx) * 2047.0 + 0.5);
    u[3] = floor(load_quad(org + vec2<i32>(2, 2), mx) * 2047.0 + 0.5);
  }
  let mn = min(min(u[0], u[1]), min(u[2], u[3]));
  let mxv = max(max(u[0], u[1]), max(u[2], u[3]));
  let tmin = min(min(mn.x, mn.y), min(mn.z, mn.w));
  let tmax = max(max(mxv.x, mxv.y), max(mxv.z, mxv.w));
  // Targets relative to the block minimum, as exact f16 integers.
  var t: array<h4, 4>;
  t[0] = h4(u[0] - tmin);
  t[1] = h4(u[1] - tmin);
  t[2] = h4(u[2] - tmin);
  t[3] = h4(u[3] - tmin);

  let h2f = tmax - tmin;        // twice the half-span
  let mid2f = tmin + tmax;      // twice the mid-range
  // 8-bit span (T is monotone in v, and v = round(T·255/2047)).
  let s8 = u32(floor(tmax * (255.0 / 2047.0) + 0.5) - floor(tmin * (255.0 / 2047.0) + 0.5));
  let sub = (s8 >> (2u + u32(s8 >= 64u) + u32(s8 >= 128u))) & 7u;
  // Shape bucket: floor(4 · Σ|2T − (min + max)| / (16 · span + 1)), as
  // three exact comparisons.
  let f4 = abs(u[0] + u[0] - mid2f) + abs(u[1] + u[1] - mid2f) + abs(u[2] + u[2] - mid2f) + abs(u[3] + u[3] - mid2f);
  let fx = 4.0 * (f4.x + f4.y + f4.z + f4.w);
  let fd = 16.0 * h2f + 1.0;
  let fb = u32(fx >= fd) + u32(fx >= 2.0 * fd) + u32(fx >= 3.0 * fd);
  let lut = LUT[select(select(32u + sub, s8, s8 < 32u), 40u, s8 == 255u) * 4u + fb];

  var best = 3.0e38;
  var bt = 0u; var bmf = 0.0; var bbf = 0.0;   // best table / multiplier / base
  for (var k: u32 = 0u; k < 2u; k = k + 1u) {
    let c = (lut >> (6u * k)) & 63u;
    let tab = c & 15u;
    let rule = c >> 4u;
    let qv = QV[tab];
    let qt = QT[tab];
    // Covering multiplier floor((h2 + 4·q3) / (8·q3)); the extra ½ keeps the
    // product off integer boundaries, so the floor is exact.
    var mulf = clamp(floor((h2f + qt.z) * qt.w), 1.0, 15.0);
    mulf = select(mulf, min(mulf + 1.0, 15.0), rule == 1u);
    let fine = rule == 2u;
    let half_m = select(4.0 * mulf, 0.5, fine);
    let basf = min(floor((mid2f + 2.0 * half_m) * 0.0625), 255.0);
    let cr = 8.0 * basf + 4.0 - half_m - tmin;   // centre, relative
    let mf = qv * half_m;
    // Outermost magnitude and its threshold, clamped per side (+, −). With a
    // multiplier the distances are integers, so a half-integer threshold is
    // floored: the same comparison, and exact in f16 at any magnitude.
    let m3f = min(vec2<f32>(mf.w), vec2<f32>(2047.0 - tmin - cr, cr + tmin));
    let t2r = (vec2<f32>(mf.z) + m3f) * 0.5;
    let t2f = select(floor(t2r), t2r, vec2<bool>(fine));
    let t01 = qt.xy * half_m;
    let mh = h4(mf);
    let m3h = vec2<f16>(m3f);
    let th = h4(vec4<f32>(t01, t2f));
    let m0 = h4(mh.x);
    let m1 = h4(mh.y);
    let m2 = h4(mh.z);
    let m3p = h4(m3h.x);
    let m3n = h4(m3h.y);
    let t0 = h4(th.x);
    let t1 = h4(th.y);
    let t2p = h4(th.z);
    let t2n = h4(th.w);
    let crh = h(cr);
    var ev = vec4<f32>(0.0);
    var sv = vec4<f32>(0.0);
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
      let d = t[q] - crh;
      let a = abs(d);
      let pos = d >= h4(0.0);
      var ms = select(m0, m1, a > t0);
      ms = select(ms, m2, a > t1);
      ms = select(ms, select(m3n, m3p, pos), a > select(t2n, t2p, pos));
      let rh = a - ms;
      let rf = vec4<f32>(select(-rh, rh, pos));
      ev = ev + rf * rf;
      sv = sv + rf;
    }
    let e = ev.x + ev.y + ev.z + ev.w;
    let s = sv.x + sv.y + sv.z + sv.w;
    // Base step that zeroes the mean residual, and the error it predicts.
    let nb = clamp(basf + floor((s + 64.0) * 0.0078125), 0.0, 255.0);
    let db = nb - basf;
    let ep = e - 16.0 * db * s + 1024.0 * db * db;
    if (ep < best) { best = ep; bt = tab; bmf = select(mulf, 0.0, fine); bbf = nb; }
  }
  let bm = u32(bmf);
  let bb = u32(bbf);

  // Index pass against the final (table, multiplier, base): the number of
  // thresholds below the texel's distance, with all four magnitudes clamped
  // to the [0, 2047] range on its side.
  let fine = bm == 0u;
  let half_m = select(4.0 * bmf, 0.5, fine);
  let cr = 8.0 * bbf + 4.0 - half_m - tmin;
  let mf = QV[bt] * half_m;
  let np = min(mf, vec4<f32>(2047.0 - tmin - cr));
  let nn = min(mf, vec4<f32>(cr + tmin));
  let tpr = (np.xyz + np.yzw) * 0.5;
  let tnr = (nn.xyz + nn.yzw) * 0.5;
  let tp = vec3<f16>(select(floor(tpr), tpr, vec3<bool>(fine)));
  let tn = vec3<f16>(select(floor(tnr), tnr, vec3<bool>(fine)));
  let crh = h(cr);
  var pk = vec2<f32>(0.0);   // (columns 0..1, columns 2..3) index words
  for (var q: u32 = 0u; q < 4u; q = q + 1u) {
    let d = t[q] - crh;
    let a = abs(d);
    let pos = d >= h4(0.0);
    let i = select(h4(0.0), h4(4.0), pos)
      + select(h4(0.0), h4(1.0), a > select(h4(tn.x), h4(tp.x), pos))
      + select(h4(0.0), h4(1.0), a > select(h4(tn.y), h4(tp.y), pos))
      + select(h4(0.0), h4(1.0), a > select(h4(tn.z), h4(tp.z), pos));
    let p = dot(vec4<f32>(i), select(WA, WB, q >= 2u));
    let lo = (q & 1u) == 1u;
    pk = pk + vec2<f32>(select(p, 0.0, lo), select(0.0, p, lo));
  }
  let hi24 = u32(pk.x);
  let lo24 = u32(pk.y);

  // Bytes 0..7 as two little-endian u32s.
  let o = bi * 2u;
  dst[o] = bb | (((bm << 4u) | bt) << 8u) | ((hi24 >> 16u) << 16u) | (((hi24 >> 8u) & 255u) << 24u);
  dst[o + 1u] = (hi24 & 255u) | ((lo24 >> 16u) << 8u) | (((lo24 >> 8u) & 255u) << 16u) | ((lo24 & 255u) << 24u);
}
