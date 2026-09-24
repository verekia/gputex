import { useSyncExternalStore } from 'react'

// `?forcewebgl=1` on any example page: encode with the WebGL2 fallback
// (`compressTexture({ forceWebGL })` on the demo, the *WebGLEncoder classes
// on the per-format pages) and render through three's WebGL2 backend
// (`new WebGPURenderer({ forceWebGL: true })`) — the path browsers without
// WebGPU take, testable on a WebGPU-capable one.
//
// Read once at module load, in the browser: each Canvas creates its renderer
// once, so the flag must be settled before the first client render. The
// static export prerenders without a query string, so anything rendered into
// the DOM must go through `useForceWebGL()` to hydrate cleanly.
export const forceWebGL: boolean =
  typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('forcewebgl') === '1'

const subscribe = (): (() => void) => () => {}

/** Hydration-safe read of `forceWebGL` for rendered output (false during prerender/hydration). */
export const useForceWebGL = (): boolean =>
  useSyncExternalStore(
    subscribe,
    () => forceWebGL,
    () => false,
  )
