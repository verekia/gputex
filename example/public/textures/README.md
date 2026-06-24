# Test textures

Used by the per-format encoder pages (`/bc1`, `/bc7`, `/bc5`, `/astc`).

| File         | Used by               | What it is                                            |
| ------------ | --------------------- | ----------------------------------------------------- |
| `color.png`  | `/bc1` `/bc7` `/astc` | RGB test card: gradients, sharp colour edges, hi-freq |
| `normal.png` | `/bc5`                | Tangent-space normal map (bump field) — smooth R/G    |

Both are 512×512 (a multiple of 4, so no block padding). Drop in your own
files with the same names to test real assets — any web image format works, any
size (non-multiples of 4 get clamp-to-edge padding).

The committed defaults are generated procedurally (no dependencies):

```
node scripts/gen-test-textures.mjs
```
