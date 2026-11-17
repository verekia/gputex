#!/usr/bin/env bash
# Snapshot encoder WGSL for the /ab benchmark page: the working tree becomes
# /ab/<fmt>_work.wgsl, git HEAD becomes /ab/<fmt>_head.wgsl. Re-run after
# every shader edit before reloading /ab.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p example/public/ab
for f in bc1 bc5 bc7 astc4x4 etc2; do
  for v in "" "_fast_f16"; do
    src="library/src/${f}${v}.wgsl"
    short="${f/astc4x4/astc}${v/_fast_f16/_f16}"
    cp "$src" "example/public/ab/${short}_work.wgsl"
    git show "HEAD:$src" > "example/public/ab/${short}_head.wgsl" 2>/dev/null || true
  done
done
# ASTC's opaque-source kernel (f16 only).
cp library/src/astc4x4_opaque_f16.wgsl example/public/ab/astc_opq_f16_work.wgsl
git show "HEAD:library/src/astc4x4_opaque_f16.wgsl" > example/public/ab/astc_opq_f16_head.wgsl 2>/dev/null || rm -f example/public/ab/astc_opq_f16_head.wgsl
echo "synced $(ls example/public/ab | wc -l | tr -d ' ') files into example/public/ab/"
