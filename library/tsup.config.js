import { defineConfig } from 'tsup'

export default defineConfig({
  // Two entries: `gputex` (engine-agnostic core, no `three` import) and
  // `gputex/three` (the Three.js-coupled layer). The object form keys the
  // output filenames, so `src/three/index.ts` emits `dist/three.js`.
  entry: { index: 'src/index.ts', three: 'src/three/index.ts' },
  clean: true,
  format: ['esm'],
  dts: true,
  splitting: false,
  loader: {
    '.wgsl': 'text',
    '.glsl': 'text',
  },
})
