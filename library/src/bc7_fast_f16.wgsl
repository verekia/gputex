// bc7 "fast" encoder — f16 variant (requires the shader-f16 feature).
// Identical algorithm to the f32 fast path in bc7.wgsl, but the projection +
// least-squares refit run in f16 ([0,1] domain). On GPUs with 2x f16 throughput
// (e.g. Apple) this is ~2x faster at the same quality; endpoints are still
// quantised to exact 8-bit. The host selects this module only when the device
// reports shader-f16, falling back to bc7.wgsl otherwise. "high" never uses this.
//
// BC7 mode 6 fast path in f16 (Apple GPUs run f16 at 2x). All math in the [0,1]
// domain so dot products stay well under f16's range; endpoints quantised to
// 8-bit at the end. Same bbox seed + projection + fused LSQ refit + reproject.
enable f16;
struct Params { blocks_x: u32, blocks_y: u32, width: u32, height: u32, };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
alias h = f16;
alias h4 = vec4<f16>;
struct Ep { seven: vec4<i32>, eight: h4, p: u32 };
// quantise an ideal endpoint (h4 in [0,1]) to 7-bit + p-bit; returns 8-bit eff in [0,1].
fn pick_ep(ideal01: h4) -> Ep {
  let ideal = ideal01 * h(255.0);
  let q0 = clamp(floor(ideal * h(0.5) + h(0.5)), h4(0.0), h4(127.0));        // p=0
  let e0 = q0 * h(2.0);
  let q1 = clamp(floor((ideal - h(1.0)) * h(0.5) + h(0.5)), h4(0.0), h4(127.0)); // p=1
  let e1 = q1 * h(2.0) + h(1.0);
  let d0 = e0 - ideal; let d1 = e1 - ideal;
  if (dot(d1,d1) < dot(d0,d0)) { return Ep(vec4<i32>(q1), e1 * h(1.0/255.0), 1u); }
  return Ep(vec4<i32>(q0), e0 * h(1.0/255.0), 0u);
}
struct Fit { e0: h4, e1: h4, valid: bool };
fn proj_assign(pix: ptr<function, array<h4,16>>, e0: h4, e1: h4, out_idx: ptr<function, array<u32,16>>, fit: bool) -> Fit {
  var out: Fit; let dir = e1 - e0; let dd = dot(dir,dir);
  if (dd == h(0.0)) { for(var k:u32=0u;k<16u;k=k+1u){(*out_idx)[k]=0u;} out.valid=false; return out; }
  let inv = h(15.0) / dd;
  var sAA=h(0.0); var sBB=h(0.0); var sAB=h(0.0); var sAV=h4(0.0); var sBV=h4(0.0);
  for(var k:u32=0u;k<16u;k=k+1u){
    let v=(*pix)[k];
    let s = clamp(floor(dot(v - e0, dir) * inv + h(0.5)), h(0.0), h(15.0));
    (*out_idx)[k] = u32(s);
    if(fit){ let b=s*h(1.0/15.0); let a=h(1.0)-b; sAA=sAA+a*a; sBB=sBB+b*b; sAB=sAB+a*b; sAV=sAV+a*v; sBV=sBV+b*v; }
  }
  if(!fit){ out.valid=false; return out; }
  let det = sAA*sBB - sAB*sAB; if (abs(det) < h(0.0001)) { out.valid=false; return out; }
  out.e0 = clamp((sBB*sAV - sAB*sBV)/det, h4(0.0), h4(1.0));
  out.e1 = clamp((sAA*sBV - sAB*sAV)/det, h4(0.0), h4(1.0));
  out.valid=true; return out;
}
fn write_bits(block: ptr<function, array<u32,4>>, pos: u32, n_bits: u32, value: u32) {
  let v=value&((1u<<n_bits)-1u); let wl=pos/32u; let bl=pos%32u; let il=min(n_bits,32u-bl);
  let ml=((1u<<il)-1u)<<bl; (*block)[wl]=((*block)[wl]&~ml)|((v<<bl)&ml);
  if(il<n_bits){ let ih=n_bits-il; let mh=(1u<<ih)-1u; (*block)[wl+1u]=((*block)[wl+1u]&~mh)|((v>>il)&mh); }
}
@compute @workgroup_size(8,8,1)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if(gid.x>=params.blocks_x||gid.y>=params.blocks_y){return;}
  let bi=gid.y*params.blocks_x+gid.x;
  let base=vec2<i32>(i32(gid.x)*4,i32(gid.y)*4); let mx=vec2<i32>(i32(params.width)-1,i32(params.height)-1);
  var pix: array<h4,16>; var lo=h4(1.0); var hi=h4(0.0);
  for(var i:u32=0u;i<16u;i=i+1u){
    let p=clamp(base+vec2<i32>(i32(i&3u),i32(i>>2u)),vec2<i32>(0),mx);
    let px=h4(textureLoad(src_tex,p,0)); pix[i]=px; lo=min(lo,px); hi=max(hi,px);
  }
  var ep0=pick_ep(lo); var ep1=pick_ep(hi); var indices: array<u32,16>;
  let r=proj_assign(&pix,ep0.eight,ep1.eight,&indices,true);
  if(r.valid){ ep0=pick_ep(r.e0); ep1=pick_ep(r.e1); proj_assign(&pix,ep0.eight,ep1.eight,&indices,false); }
  if((indices[0]&0x8u)!=0u){ let t=ep0; ep0=ep1; ep1=t; for(var k:u32=0u;k<16u;k=k+1u){indices[k]=15u-indices[k];} }
  var block: array<u32,4>; block[0]=0u;block[1]=0u;block[2]=0u;block[3]=0u; var pos:u32=0u;
  write_bits(&block,pos,7u,0x40u);pos=pos+7u;
  write_bits(&block,pos,7u,u32(ep0.seven.x));pos=pos+7u; write_bits(&block,pos,7u,u32(ep1.seven.x));pos=pos+7u;
  write_bits(&block,pos,7u,u32(ep0.seven.y));pos=pos+7u; write_bits(&block,pos,7u,u32(ep1.seven.y));pos=pos+7u;
  write_bits(&block,pos,7u,u32(ep0.seven.z));pos=pos+7u; write_bits(&block,pos,7u,u32(ep1.seven.z));pos=pos+7u;
  write_bits(&block,pos,7u,u32(ep0.seven.w));pos=pos+7u; write_bits(&block,pos,7u,u32(ep1.seven.w));pos=pos+7u;
  write_bits(&block,pos,1u,ep0.p);pos=pos+1u; write_bits(&block,pos,1u,ep1.p);pos=pos+1u;
  write_bits(&block,pos,3u,indices[0]&0x7u);pos=pos+3u;
  for(var k:u32=1u;k<16u;k=k+1u){write_bits(&block,pos,4u,indices[k]&0xFu);pos=pos+4u;}
  let o=bi*4u; dst[o]=block[0];dst[o+1u]=block[1];dst[o+2u]=block[2];dst[o+3u]=block[3];
}
