// bc5 "fast" encoder — f16 variant (requires the shader-f16 feature).
// Identical algorithm to the f32 fast path in bc5.wgsl, but the bbox + O(1)
// projection index assignment run in f16 ([0,1] domain). On GPUs with 2x f16
// throughput (e.g. Apple) this is ~2x faster at the same quality; endpoints are
// still quantised to exact 8-bit. The host selects this module only when the
// device reports shader-f16, falling back to bc5.wgsl otherwise. "high" never
// uses this.
//
// BC5 fast path in f16 (two BC4 halves, no refit). f16 halves the per-channel ALU.
enable f16;
alias h = f16;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
fn q8(v: h) -> u32 { return u32(clamp(floor(v*h(255.0)+h(0.5)), h(0.0), h(255.0))); }
// Map projection level (0 = r1/min, 7 = r0/max) to BC4's non-monotonic index order.
fn level_to_index(level: u32) -> u32 { switch level { case 0u:{return 1u;} case 7u:{return 0u;} default:{return 8u-level;} } }
fn encode_bc4(values: ptr<function, array<h,16>>) -> vec2<u32> {
  var vmin=h(1.0); var vmax=h(0.0);
  for(var k:u32=0u;k<16u;k=k+1u){ vmin=min(vmin,(*values)[k]); vmax=max(vmax,(*values)[k]); }
  var r0=q8(vmax); var r1=q8(vmin);
  if(r0==r1){ if(r1>0u){r1=r1-1u;}else{r0=r0+1u;} }
  // O(1) projection: the 8 palette entries are uniformly spaced in [r1f, r0f], so
  // the nearest is round((v-r1f)/(r0f-r1f)*7). r0f>r1f (nudge) ⇒ span ≥ 1/255.
  let r0f=h(f32(r0)/255.0); let r1f=h(f32(r1)/255.0); let inv=h(7.0)/(r0f-r1f);
  var indices: array<u32,16>;
  for(var k:u32=0u;k<16u;k=k+1u){ let level=clamp(floor(((*values)[k]-r1f)*inv+h(0.5)), h(0.0), h(7.0)); indices[k]=level_to_index(u32(level)); }
  var lo=0u; var hi=0u;
  for(var k:u32=0u;k<16u;k=k+1u){ let bit=3u*k; let v=indices[k]&7u;
    if(bit+3u<=32u){ lo=lo|(v<<bit); } else if(bit>=32u){ hi=hi|(v<<(bit-32u)); } else { lo=lo|(v<<bit); hi=hi|(v>>(32u-bit)); } }
  return vec2<u32>(r0 | (r1<<8u) | ((lo&0xFFFFu)<<16u), (lo>>16u) | (hi<<16u));
}
@compute @workgroup_size(8,8,1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if(gid.x>=params.blocks_x||gid.y>=params.blocks_y){return;}
  let bi=gid.y*params.blocks_x+gid.x;
  let base=vec2<i32>(i32(gid.x)*4,i32(gid.y)*4); let mx=vec2<i32>(i32(params.width)-1,i32(params.height)-1);
  var rv: array<h,16>; var gv: array<h,16>;
  for(var i:u32=0u;i<16u;i=i+1u){ let p=clamp(base+vec2<i32>(i32(i&3u),i32(i>>2u)),vec2<i32>(0),mx); let c=textureLoad(src_tex,p,0); rv[i]=h(c.r); gv[i]=h(c.g); }
  let rb=encode_bc4(&rv); let gb=encode_bc4(&gv);
  let o=bi*4u; dst[o]=rb.x; dst[o+1u]=rb.y; dst[o+2u]=gb.x; dst[o+3u]=gb.y;
}
