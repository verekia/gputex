#!/usr/bin/env bash
# Build two copies of the core library for in-session host-side A/B
# (encodeToBytes wall time, byte identity) on one device:
#   example/public/ab/gputex_head.js  ← git HEAD (or $1: any commit-ish)
#   example/public/ab/gputex_work.js  ← the working tree
# Load both on any example page with `await import('/ab/gputex_head.js')`.
# Cross-page-load wall timings swing ±30%; only interleaved same-session
# comparisons are meaningful.
set -euo pipefail
cd "$(dirname "$0")/../.."
rev="${1:-HEAD}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p example/public/ab
git archive "$rev" library | tar -x -C "$tmp"
ln -s "$PWD/library/node_modules" "$tmp/library/node_modules"
(cd "$tmp/library" && bunx tsup src/index.ts --format esm --loader .wgsl=text --loader .glsl=text --out-dir out --silent)
cp "$tmp/library/out/index.js" example/public/ab/gputex_head.js
(cd library && bunx tsup src/index.ts --format esm --loader .wgsl=text --loader .glsl=text --out-dir "$tmp/work" --silent)
cp "$tmp/work/index.js" example/public/ab/gputex_work.js
echo "built $rev → /ab/gputex_head.js, working tree → /ab/gputex_work.js"
