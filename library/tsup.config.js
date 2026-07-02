import { defineConfig } from 'tsup'

export default defineConfig({
  // Three entries: `gputex` (engine-agnostic core, no `three` import),
  // `gputex/three` (the Three.js-coupled layer) and `gputex/testing` (CPU
  // reference encoders/decoders for test suites). The object form keys the
  // output filenames, so `src/three/index.ts` emits `dist/three.js`.
  entry: { index: 'src/index.ts', three: 'src/three/index.ts', testing: 'src/testing.ts' },
  clean: true,
  format: ['esm'],
  dts: true,
  splitting: false,
  loader: {
    '.wgsl': 'text',
    '.glsl': 'text',
  },
})
