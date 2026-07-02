// bc5 "fast" encoder — f16 variant (requires the shader-f16 feature).
// Two BC4 halves (R and G), no refit — same output family as bc5.wgsl's fast
// branch, tuned for throughput:
//
//   • The 8-entry palette in 6-interpolation mode is COLINEAR and EVENLY
//     spaced from r0 to r1 (levels 0..7 in palette order 0,2,3,4,5,6,7,1),
//     so the nearest entry is the rounded projection of v onto the r0→r1
//     axis — O(1) per pixel instead of an 8-entry distance search.
//   • Math runs in the exact-integer [0,255] f16 domain: endpoints and pixel
//     values are whole numbers ≤ 255 (exact in f16), so the only rounding is
//     the single 1/(r1−r0) division.
//   • 3-bit indices are packed into the 48-bit field on the fly — no
//     array<u32,16> private array and no separate packing loop.
//
// Level → BC4 index (0→r0 ... 7→r1): 0,2,3,4,5,6,7,1 — packed 3-bit LUT
// 0x3F58D0 = sum(idx[L] << 3L).
//
// The host selects this module only when the device reports shader-f16,
// falling back to bc5.wgsl otherwise. "high" never uses this.
enable f16;
alias h = f16;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// Encode one channel (16 values in exact-integer [0,255] f16) to a BC4 half.
fn encode_bc4(values: ptr<function, array<h, 16>>) -> vec2<u32> {
  var vmin = h(255.0);
  var vmax = h(0.0);
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    vmin = min(vmin, (*values)[k]);
    vmax = max(vmax, (*values)[k]);
  }
  var r0 = u32(vmax); // values are exact integers — no rounding needed
  var r1 = u32(vmin);
  if (r0 == r1) {
    // Flat block: nudge to keep 6-interp mode (r0 > r1 strictly).
    if (r1 > 0u) { r1 = r1 - 1u; } else { r0 = r0 + 1u; }
  }

  // Projection assignment: level = round(7·(v − r0)/(r1 − r0)), clamped.
  // |v − r0| ≤ r0 − r1 for every in-block value, so the product stays ≤ 7.
  let r0f = h(f32(r0));
  let scale = h(7.0) / (h(f32(r1)) - r0f);
  var lo: u32 = 0u;
  var hi = r0 | (r1 << 8u); // endpoint bytes live in the low 16 bits of u32[0]
  // Pixel k's 3-bit index starts at bit 3k of the 48-bit field, i.e. bit
  // 3k+16 of u32[0] for k ≤ 4, straddling into u32[1] from k = 5 (bit 31).
  var w0 = hi;
  var w1 = 0u;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let L = u32(clamp(floor(((*values)[k] - r0f) * scale + h(0.5)), h(0.0), h(7.0)));
    let idx = (0x3F58D0u >> (L * 3u)) & 7u;
    let bit = 3u * k + 16u;
    if (bit <= 29u) {
      w0 = w0 | (idx << bit);
    } else if (bit >= 32u) {
      w1 = w1 | (idx << (bit - 32u));
    } else {
      // k = 5 straddles the word boundary (bits 31..33).
      w0 = w0 | (idx << bit);
      w1 = w1 | (idx >> (32u - bit));
    }
  }
  return vec2<u32>(w0, w1);
}

@compute @workgroup_size(8, 8, 1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.blocks_x || gid.y >= params.blocks_y) { return; }
  let bi = gid.y * params.blocks_x + gid.x;
  let base = vec2<i32>(i32(gid.x) * 4, i32(gid.y) * 4);
  let mx = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
  var rv: array<h, 16>;
  var gv: array<h, 16>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    let p = clamp(base + vec2<i32>(i32(i & 3u), i32(i >> 2u)), vec2<i32>(0), mx);
    let c = textureLoad(src_tex, p, 0);
    rv[i] = h(c.r * 255.0);
    gv[i] = h(c.g * 255.0);
  }
  let rb = encode_bc4(&rv);
  let gb = encode_bc4(&gv);
  let o = bi * 4u;
  dst[o] = rb.x; dst[o + 1u] = rb.y; dst[o + 2u] = gb.x; dst[o + 3u] = gb.y;
}
