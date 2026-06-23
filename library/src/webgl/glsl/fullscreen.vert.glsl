#version 300 es
// Fullscreen-triangle vertex shader for the WebGL block encoders.
//
// Draws a single oversized triangle covering the viewport from gl_VertexID
// alone — no vertex buffers / attributes needed (drawArrays(TRIANGLES, 0, 3)).
// The encoder sets the viewport to (blocks_x × blocks_y), so each rasterised
// fragment corresponds to exactly one 4×4 output block.
//
//   id 0 -> (-1,-1)   id 1 -> ( 3,-1)   id 2 -> (-1, 3)

void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
