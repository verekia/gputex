// BC1 (DXT1) compute shader encoder.
//
// Each invocation encodes one 4x4 pixel block into an 8-byte BC1 block
// written as 2 x u32 into the destination storage buffer.
//
// BC1 block layout (little-endian):
//   u32[0]: color0 (low 16) | color1 (high 16)   both in RGB565
//   u32[1]: 16 x 2-bit indices, pixel 0 = bits 0..1, pixel 15 = bits 30..31
//
// When color0 > color1 (numeric 16-bit), the 4-color mode is used:
//   idx 0 -> color0
//   idx 1 -> color1
//   idx 2 -> (2*color0 +   color1) / 3
//   idx 3 -> (  color0 + 2*color1) / 3
// We always force the 4-color mode here.
//
// Algorithm:
//  1. Compute the bounding box (min/max RGB) of the block.
//  2. Inset slightly to account for endpoint quantization rounding;
//     this is a well-known heuristic from rygorous/stb_dxt that
//     improves quality cheaply.
//  3. Quantize endpoints to RGB565 and ensure color0 > color1.
//  4. Reconstruct the 4-color palette in floating point and assign
//     the closest palette entry to each pixel (full L2 search).

struct Params {
  blocks_x: u32,
  blocks_y: u32,
  width:    u32,
  height:   u32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn to565(c: vec3<f32>) -> u32 {
  // Round-to-nearest quantization into 5-6-5.
  let r = u32(clamp(floor(c.r * 31.0 + 0.5), 0.0, 31.0));
  let g = u32(clamp(floor(c.g * 63.0 + 0.5), 0.0, 63.0));
  let b = u32(clamp(floor(c.b * 31.0 + 0.5), 0.0, 31.0));
  return (r << 11u) | (g << 5u) | b;
}

fn from565(c: u32) -> vec3<f32> {
  let r = f32((c >> 11u) & 31u);
  let g = f32((c >>  5u) & 63u);
  let b = f32( c         & 31u);
  // Expand to 8-bit then normalize, matching typical BC1 decoder behavior.
  let r8 = (r * 527.0 + 23.0) / 256.0;   // = round(r * 255 / 31)
  let g8 = (g * 259.0 + 33.0) / 256.0;   // = round(g * 255 / 63)
  let b8 = (b * 527.0 + 23.0) / 256.0;
  return vec3<f32>(r8, g8, b8) / 255.0;
}

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) {
    return;
  }

  let bx = gid.x;
  let by = gid.y;
  let block_index = by * params.blocks_x + bx;

  let base = vec2<i32>(i32(bx) * 4, i32(by) * 4);
  let max_xy = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);

  var pixels: array<vec3<f32>, 16>;
  var bb_min = vec3<f32>(1.0, 1.0, 1.0);
  var bb_max = vec3<f32>(0.0, 0.0, 0.0);

  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let lx = i32(i & 3u);
    let ly = i32(i >> 2u);
    // Clamp to edge for non-multiple-of-4 textures.
    let p  = clamp(base + vec2<i32>(lx, ly), vec2<i32>(0, 0), max_xy);
    let c  = textureLoad(src_tex, p, 0).rgb;
    pixels[i] = c;
    bb_min = min(bb_min, c);
    bb_max = max(bb_max, c);
  }

  // Inset the bounding box. The magic constant 1/16 approximates half
  // the width of an RGB565 quantization cell; insetting by that much
  // moves the endpoints toward each other so the quantized 4-color
  // palette covers the real data range more tightly.
  let inset = (bb_max - bb_min) / 16.0;
  var hi = clamp(bb_max - inset, vec3<f32>(0.0), vec3<f32>(1.0));
  var lo = clamp(bb_min + inset, vec3<f32>(0.0), vec3<f32>(1.0));

  var c0 = to565(hi);
  var c1 = to565(lo);

  // 4-color mode requires c0 > c1. If equal (flat block), we still use
  // 4-color mode by nudging c1 down when possible; if c1 == 0 the block
  // is truly black so all indices stay 0 and the decoded value is 0.
  if (c0 == c1) {
    if (c1 > 0u) {
      c1 = c1 - 1u;
    } else {
      c0 = c0 + 1u;
    }
  } else if (c0 < c1) {
    let tmp = c0;
    c0 = c1;
    c1 = tmp;
  }

  // Build the palette in the decoded colour space so index selection
  // matches what the hardware decoder will produce.
  let p0 = from565(c0);
  let p1 = from565(c1);
  let p2 = (2.0 * p0 + p1) * (1.0 / 3.0);
  let p3 = (p0 + 2.0 * p1) * (1.0 / 3.0);

  var indices: u32 = 0u;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let c = pixels[i];
    let d0 = dot(c - p0, c - p0);
    let d1 = dot(c - p1, c - p1);
    let d2 = dot(c - p2, c - p2);
    let d3 = dot(c - p3, c - p3);

    var best_d: f32 = d0;
    var best_i: u32 = 0u;
    if (d1 < best_d) { best_d = d1; best_i = 1u; }
    if (d2 < best_d) { best_d = d2; best_i = 2u; }
    if (d3 < best_d) { best_d = d3; best_i = 3u; }

    indices = indices | (best_i << (i * 2u));
  }

  let out = block_index * 2u;
  dst[out]      = c0 | (c1 << 16u);
  dst[out + 1u] = indices;
}
