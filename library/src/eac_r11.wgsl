// EAC R11 (unsigned) compute shader encoder — f32 module.
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
//   • CANDIDATES: the 8-bit span of the block indexes a 41-entry table of
//     three (modifier table, multiplier rule) picks. The multiplier rule is
//     round(half-span / largest magnitude) — the table just covers the
//     block — or that plus one, or multiplier 0 (1/8-step precision, for
//     spans of a few levels). The table was fitted offline, per span, on
//     roughness / AO / displacement / mask / normal-channel textures (greedy,
//     on this encoder's own policy): which quantiser shapes win is almost
//     entirely a function of the span, so three picks land within ~0.3 dB of
//     scoring all 16 tables.
//   • BASE: the palette centre goes to the block's mid-range.
//   • SCORE: exact squared error of each candidate — per texel, the nearest
//     of the four magnitudes on its side of the centre. The outermost
//     magnitude is clamped to the [0, 2047] range on each side, so texels
//     at 0 or 255 are matched exactly by an over-reaching palette (masks
//     keep true black and white).
//   • REFINE: the winner is re-scored with base − 1 and base + 1. On tables
//     with coarse steps the phase of the palette against the data matters
//     more than the table choice; this buys ~0.5 dB for two more scores.
//   • The 3-bit indices are packed as float fields Σ i·8^k (exact below
//     2^24), one 24-bit word per pair of columns.
// Five scores + one index pass per block, each four vec4 steps. The kernel
// is ALU-bound (one-channel source: almost no texture traffic), so its cost
// is the number of scores, ~13% of the pass each. Measured over 17 textures
// (mean PSNR, M3): a 4th pick +0.22 dB, 6 picks +0.38 dB; 2 picks −0.35 dB;
// dropping the refine −0.48 dB. Scoring on half the texels or by |error|
// loses more per unit of time than dropping a pick.

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
// The largest of those per table, as an integer (the multiplier rule divides
// by it).
const QW = array<u32, 16>(29u, 25u, 25u, 25u, 23u, 21u, 21u, 21u, 19u, 19u, 19u, 19u, 19u, 19u, 17u, 17u);

// Span → three candidates, 6 bits each (low first): table in bits 0..3, rule
// in bits 4..5 (0 = covering multiplier, 1 = that + 1, 2 = multiplier 0).
// Index: the 8-bit span below 32, then 8 sub-buckets per octave of span;
// entry 40 is for full-range blocks (span 255: both black and white).
const LUT = array<u32, 41>(
  0x01026u, 0x24a2du, 0x2b368u, 0x21360u, 0x2380du, 0x2380du, 0x0c74du, 0x1d0cdu,
  0x1141du, 0x0f2ddu, 0x0b0ddu, 0x0b0ddu, 0x0a203u, 0x033cbu, 0x0a3cbu, 0x013cbu,
  0x0e3cbu, 0x0f389u, 0x0d389u, 0x0d2c9u, 0x0b289u, 0x06247u, 0x05247u, 0x07101u,
  0x04081u, 0x020c1u, 0x03081u, 0x03081u, 0x02001u, 0x0f040u, 0x083c0u, 0x0f200u,
  0x0424fu, 0x0f089u, 0x00247u, 0x05244u, 0x0a042u, 0x021c1u, 0x00389u, 0x0e245u,
  0x1269bu);

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
  // Targets relative to the block minimum.
  u[0] = u[0] - tmin;
  u[1] = u[1] - tmin;
  u[2] = u[2] - tmin;
  u[3] = u[3] - tmin;

  let h2 = u32(tmax - tmin);        // twice the half-span
  let mid2 = u32(tmin + tmax);      // twice the mid-range
  // 8-bit span (T is monotone in v, and v = round(T·255/2047)).
  let s8 = u32(floor(tmax * (255.0 / 2047.0) + 0.5) - floor(tmin * (255.0 / 2047.0) + 0.5));
  let sub = (s8 >> (2u + u32(s8 >= 64u) + u32(s8 >= 128u))) & 7u;
  let lut = LUT[select(select(32u + sub, s8, s8 < 32u), 40u, s8 == 255u)];

  var best = 3.0e38;
  var bt = 0u; var bm = 0u; var bb = 0u;   // best table / multiplier / base
  var wt = 0u; var wm = 0u; var wb = 0u;   // winner of the three candidates
  for (var k: u32 = 0u; k < 5u; k = k + 1u) {
    var tab: u32; var mul: u32; var bas: u32;
    if (k < 3u) {
      let c = (lut >> (6u * k)) & 63u;
      tab = c & 15u;
      let rule = c >> 4u;
      let dv = 8u * QW[tab];
      mul = clamp((h2 + dv / 2u) / dv, 1u, 15u);
      mul = select(mul, min(mul + 1u, 15u), rule == 1u);
      mul = select(mul, 0u, rule == 2u);
      bas = min((mid2 + select(8u * mul, 1u, mul == 0u)) >> 4u, 255u);
    } else {
      if (k == 3u) { wt = bt; wm = bm; wb = bb; }
      tab = wt;
      mul = wm;
      bas = select(min(wb + 1u, 255u), max(wb, 1u) - 1u, k == 3u);
    }
    let half_m = select(4.0 * f32(mul), 0.5, mul == 0u);
    let cr = f32(8u * bas + 4u) - half_m - tmin;   // centre, relative
    let m = QV[tab] * half_m;
    let m3p = min(m.w, 2047.0 - tmin - cr);
    let m3n = min(m.w, cr + tmin);
    var e = 0.0;
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
      let d = u[q] - cr;
      let a = abs(d);
      let m3 = select(vec4<f32>(m3n), vec4<f32>(m3p), d >= vec4<f32>(0.0));
      let r = min(min(abs(a - m.x), abs(a - m.y)), min(abs(a - m.z), abs(a - m3)));
      e = e + dot(r, r);
    }
    if (e < best) { best = e; bt = tab; bm = mul; bb = bas; }
  }

  // Index pass against the final (table, multiplier, base).
  let half_m = select(4.0 * f32(bm), 0.5, bm == 0u);
  let cr = f32(8u * bb + 4u) - half_m - tmin;
  let m = QV[bt] * half_m;
  let m3p = min(m.w, 2047.0 - tmin - cr);
  let m3n = min(m.w, cr + tmin);
  var pk = vec2<f32>(0.0);   // (columns 0..1, columns 2..3) index words
  for (var q: u32 = 0u; q < 4u; q = q + 1u) {
    let d = u[q] - cr;
    let a = abs(d);
    let pos = d >= vec4<f32>(0.0);
    let r1 = abs(a - m.y);
    let r2 = abs(a - m.z);
    let r3 = abs(a - select(vec4<f32>(m3n), vec4<f32>(m3p), pos));
    var b = abs(a - m.x);
    var i = select(vec4<f32>(0.0), vec4<f32>(1.0), r1 < b);
    b = min(b, r1);
    i = select(i, vec4<f32>(2.0), r2 < b);
    b = min(b, r2);
    i = select(i, vec4<f32>(3.0), r3 < b);
    let p = dot(i + select(vec4<f32>(0.0), vec4<f32>(4.0), pos), select(WA, WB, q >= 2u));
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
