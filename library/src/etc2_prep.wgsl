// ETC2 source-preparation pass: RGBA8 → packed luma + quadrant averages.
//
// The ETC2 encoder is DRAM-bandwidth-bound, not ALU-bound: on a 100 GB/s
// part, just reading a 2048² RGBA8 source costs ~0.15 ms while the entire
// selection ALU adds ~0.03. But the encoder never needs full-resolution
// chroma — per-texel work (modifier tables, indices) runs on LUMA alone,
// and base colours/estimates need only 2×2 QUADRANT sums. This pass bakes
// the source into exactly that split, halving the encode pass's read
// volume to 2 bytes/pixel with near-zero quality cost (luma stays 8-bit
// exact; only within-quadrant chroma detail is lost, which ETC1-family
// blocks cannot represent anyway):
//
//   • y_out  (r32uint, paddedWidth/4 × paddedHeight): per texel four packed
//     8-bit lumas round((r+g+b)/3), little-endian along x — one fetch per
//     block ROW in the encoder.
//   • q_out  (rgba8unorm, paddedWidth/2 × paddedHeight/2): per texel the
//     rounded average of a 2×2 source quad.
//
// Padding is baked here: out-of-range source reads clamp to the last real
// texel, so the encode pass reads its planes unclamped. Each invocation
// covers one 4×2-pixel tile — two packed-luma texels and two quad cells —
// so every source texel is read exactly once.
//
// Rounding is integer-exact and matches the CPU reference: luma = x/3 with
// x ≤ 765 never lands on .5 (fractions are thirds), and quad averages use
// floor(x·¼ + ½) written as n/255 so the unorm conversion is exact.

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var y_out: texture_storage_2d<r32uint, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var q_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  // One tile per invocation: pixels [4x, 4x+3] × [2y, 2y+1].
  if (gid.x >= params.blocks_x || gid.y * 2u >= params.blocks_y * 4u) {
    return;
  }
  let max_xy = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 2);

  var csum0 = vec3<f32>(0.0); // left 2×2 quad
  var csum1 = vec3<f32>(0.0); // right 2×2 quad
  for (var row: i32 = 0; row < 2; row = row + 1) {
    var packed = 0u;
    for (var col: i32 = 0; col < 4; col = col + 1) {
      let p = clamp(base + vec2<i32>(col, row), vec2<i32>(0, 0), max_xy);
      let c = round(textureLoad(src_tex, p, 0).rgb * 255.0);
      let lum = c.r + c.g + c.b;
      // x/3 for integer x never falls on .5 — the round is unambiguous.
      packed = packed | (u32(floor(lum * (1.0 / 3.0) + 0.5)) << (u32(col) * 8u));
      if (col < 2) {
        csum0 = csum0 + c;
      } else {
        csum1 = csum1 + c;
      }
    }
    textureStore(y_out, vec2<i32>(i32(gid.x), base.y + row), vec4<u32>(packed, 0u, 0u, 0u));
  }
  // Quad averages, rounded to integers so the rgba8unorm store is exact.
  let q0 = floor(csum0 * 0.25 + vec3<f32>(0.5));
  let q1 = floor(csum1 * 0.25 + vec3<f32>(0.5));
  let qy = i32(gid.y);
  textureStore(q_out, vec2<i32>(i32(gid.x) * 2, qy), vec4<f32>(q0 * (1.0 / 255.0), 1.0));
  textureStore(q_out, vec2<i32>(i32(gid.x) * 2 + 1, qy), vec4<f32>(q1 * (1.0 / 255.0), 1.0));
}
