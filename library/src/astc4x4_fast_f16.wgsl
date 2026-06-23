// astc4x4 "fast" encoder — f16 variant (requires the shader-f16 feature).
// Identical algorithm to the f32 fast path in astc4x4.wgsl, but the projection +
// least-squares refit run in f16 ([0,1] domain). On GPUs with 2x f16 throughput
// (e.g. Apple) this is ~2x faster at the same quality; endpoints are still
// quantised to exact 8-bit. The host selects this module only when the device
// reports shader-f16, falling back to astc4x4.wgsl otherwise. "high" never uses this.
//
enable f16;
alias h = f16; alias h4 = vec4<f16>;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
struct Fit { e0: h4, e1: h4, valid: bool };
fn proj(pix: ptr<function, array<h4,16>>, e0: h4, e1: h4, out_idx: ptr<function, array<u32,16>>, fit: bool) -> Fit {
  var out: Fit; let dir=e1-e0; let dd=dot(dir,dir);
  if(dd==h(0.0)){ for(var k:u32=0u;k<16u;k=k+1u){(*out_idx)[k]=0u;} out.valid=false; return out; }
  let inv=h(3.0)/dd;
  var sAA=h(0.0); var sBB=h(0.0); var sAB=h(0.0); var sAV=h4(0.0); var sBV=h4(0.0);
  for(var k:u32=0u;k<16u;k=k+1u){ let v=(*pix)[k]; let s=clamp(floor(dot(v-e0,dir)*inv+h(0.5)),h(0.0),h(3.0)); (*out_idx)[k]=u32(s);
    if(fit){ let b=s*h(1.0/3.0); let a=h(1.0)-b; sAA=sAA+a*a; sBB=sBB+b*b; sAB=sAB+a*b; sAV=sAV+a*v; sBV=sBV+b*v; } }
  if(!fit){ out.valid=false; return out; }
  let det=sAA*sBB-sAB*sAB; if(abs(det)<h(0.0001)){out.valid=false;return out;}
  out.e0=clamp((sBB*sAV-sAB*sBV)/det,h4(0.0),h4(1.0)); out.e1=clamp((sAA*sBV-sAB*sAV)/det,h4(0.0),h4(1.0)); out.valid=true; return out;
}
fn write_bits(block: ptr<function, array<u32,4>>, pos: u32, n_bits: u32, value: u32) {
  let v=value&((1u<<n_bits)-1u); let wl=pos/32u; let bl=pos%32u; let il=min(n_bits,32u-bl);
  let ml=((1u<<il)-1u)<<bl; (*block)[wl]=((*block)[wl]&~ml)|((v<<bl)&ml);
  if(il<n_bits){ let ih=n_bits-il; let mh=(1u<<ih)-1u; (*block)[wl+1u]=((*block)[wl+1u]&~mh)|((v>>il)&mh); }
}
fn q8(e: h4) -> vec4<i32> { return vec4<i32>(clamp(floor(e*h(255.0)+h(0.5)), h4(0.0), h4(255.0))); }
@compute @workgroup_size(8,8,1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if(gid.x>=params.blocks_x||gid.y>=params.blocks_y){return;}
  let bi=gid.y*params.blocks_x+gid.x;
  let base=vec2<i32>(i32(gid.x)*4,i32(gid.y)*4); let mx=vec2<i32>(i32(params.width)-1,i32(params.height)-1);
  var pix: array<h4,16>; var lo=h4(1.0); var hi=h4(0.0);
  for(var i:u32=0u;i<16u;i=i+1u){ let p=clamp(base+vec2<i32>(i32(i&3u),i32(i>>2u)),vec2<i32>(0),mx); let px=h4(textureLoad(src_tex,p,0)); pix[i]=px; lo=min(lo,px); hi=max(hi,px); }
  var e0=lo; var e1=hi; var indices: array<u32,16>;
  let r=proj(&pix,e0,e1,&indices,true);
  if(r.valid){ e0=r.e0; e1=r.e1; proj(&pix,e0,e1,&indices,false); }
  var E0=q8(e0); var E1=q8(e1);
  if(E0.x+E0.y+E0.z > E1.x+E1.y+E1.z){ let t=E0; E0=E1; E1=t; for(var k:u32=0u;k<16u;k=k+1u){indices[k]=3u-indices[k];} }
  var block: array<u32,4>; block[0]=0u;block[1]=0u;block[2]=0u;block[3]=0u;
  write_bits(&block,0u,11u,0x042u); write_bits(&block,11u,2u,0u); write_bits(&block,13u,4u,12u);
  write_bits(&block,17u,8u,u32(E0.x)); write_bits(&block,25u,8u,u32(E1.x));
  write_bits(&block,33u,8u,u32(E0.y)); write_bits(&block,41u,8u,u32(E1.y));
  write_bits(&block,49u,8u,u32(E0.z)); write_bits(&block,57u,8u,u32(E1.z));
  write_bits(&block,65u,8u,u32(E0.w)); write_bits(&block,73u,8u,u32(E1.w));
  var w3:u32=0u; for(var k:u32=0u;k<16u;k=k+1u){ let w=indices[k]&3u; w3=w3|((w&1u)<<(31u-2u*k))|(((w>>1u)&1u)<<(30u-2u*k)); } block[3]=w3;
  let o=bi*4u; dst[o]=block[0];dst[o+1u]=block[1];dst[o+2u]=block[2];dst[o+3u]=block[3];
}
