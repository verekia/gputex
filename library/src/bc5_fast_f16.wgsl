// bc5 "fast" encoder — f16 variant (requires the shader-f16 feature).
// Identical algorithm to the f32 fast path in bc5.wgsl, but the projection +
// least-squares refit run in f16 ([0,1] domain). On GPUs with 2x f16 throughput
// (e.g. Apple) this is ~2x faster at the same quality; endpoints are still
// quantised to exact 8-bit. The host selects this module only when the device
// reports shader-f16, falling back to bc5.wgsl otherwise. "high" never uses this.
//
// BC5 fast path in f16 (two BC4 halves, no refit). f16 halves the per-channel ALU.
enable f16;
alias h = f16;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
fn w0(j: u32) -> h { switch j { case 0u:{return h(1.0);} case 1u:{return h(0.0);} case 2u:{return h(6.0/7.0);} case 3u:{return h(5.0/7.0);} case 4u:{return h(4.0/7.0);} case 5u:{return h(3.0/7.0);} case 6u:{return h(2.0/7.0);} default:{return h(1.0/7.0);} } }
fn w1(j: u32) -> h { switch j { case 0u:{return h(0.0);} case 1u:{return h(1.0);} case 2u:{return h(1.0/7.0);} case 3u:{return h(2.0/7.0);} case 4u:{return h(3.0/7.0);} case 5u:{return h(4.0/7.0);} case 6u:{return h(5.0/7.0);} default:{return h(6.0/7.0);} } }
fn q8(v: h) -> u32 { return u32(clamp(floor(v*h(255.0)+h(0.5)), h(0.0), h(255.0))); }
fn encode_bc4(values: ptr<function, array<h,16>>) -> vec2<u32> {
  var vmin=h(1.0); var vmax=h(0.0);
  for(var k:u32=0u;k<16u;k=k+1u){ vmin=min(vmin,(*values)[k]); vmax=max(vmax,(*values)[k]); }
  var r0=q8(vmax); var r1=q8(vmin);
  if(r0==r1){ if(r1>0u){r1=r1-1u;}else{r0=r0+1u;} }
  var pal: array<h,8>; let r0f=h(f32(r0)/255.0); let r1f=h(f32(r1)/255.0);
  for(var j:u32=0u;j<8u;j=j+1u){ pal[j]=w0(j)*r0f+w1(j)*r1f; }
  var indices: array<u32,16>;
  for(var k:u32=0u;k<16u;k=k+1u){ let v=(*values)[k]; var bj=0u; var bd=h(1e4); for(var j:u32=0u;j<8u;j=j+1u){ let d=pal[j]-v; let d2=d*d; if(d2<bd){bd=d2;bj=j;} } indices[k]=bj; }
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
