# Test textures

Used by the per-format encoder pages (`/bc1`, `/bc7`, `/bc5`, `/astc`).

| File         | Used by               | What it is                                                      |
| ------------ | --------------------- | --------------------------------------------------------------- |
| `color.png`  | `/bc1` `/bc7` `/astc` | 16-tile RGB test card — each tile stresses one failure mode     |
| `normal.png` | `/bc5`                | Tangent-space normal map: domes, pyramids, bricks, ripple sweep |

Both are 1024×1024 (a multiple of 4, so no block padding). The colour card's
rows are: smooth gradients (banding), hard edges (incl. the disc-over-checker
multi-cluster probe), Nyquist-frequency detail (1px checkers, pinstripes,
noise), and natural-ish content (marble, plasma, voronoi, blobs).

Drop in your own files with the same names to test real assets — any web image
format works, any size (non-multiples of 4 get clamp-to-edge padding).

The committed defaults are generated procedurally (no dependencies):

```
node scripts/gen-test-textures.mjs
```

NOTE: the `/test` suite's PSNR floors are pinned to these exact images —
regenerating them requires re-baselining `example/lib/gpuTestSuite.ts`.
