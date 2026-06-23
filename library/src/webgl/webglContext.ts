// WebGL2 rendering context management for the WebGL fallback encoders.
//
// The fallback never draws to a visible canvas — it renders block data into an
// off-screen integer framebuffer and reads it back — so a single 1×1 detached
// canvas is enough. Browsers cap the number of live WebGL contexts (~16), so
// the high-level `compressTexture()` path shares one process-wide context
// across every encoder/mip level rather than spinning up a fresh one each call.

const CONTEXT_ATTRS: WebGLContextAttributes = {
  alpha: false,
  antialias: false,
  depth: false,
  stencil: false,
  premultipliedAlpha: false,
  preserveDrawingBuffer: false,
  // Encoding is GPU-bound; prefer the discrete GPU when the browser exposes a
  // choice. Ignored where unsupported.
  powerPreference: 'high-performance',
}

/**
 * Create a fresh WebGL2 context backed by an off-screen 1×1 canvas. Returns
 * null when neither `OffscreenCanvas` nor `document` is available (e.g. a
 * non-DOM worker without OffscreenCanvas) or when the platform has no WebGL2.
 */
export function createWebGLContext(): WebGL2RenderingContext | null {
  if (typeof OffscreenCanvas !== 'undefined') {
    const gl = new OffscreenCanvas(1, 1).getContext('webgl2', CONTEXT_ATTRS)
    return (gl as WebGL2RenderingContext | null) ?? null
  }
  if (typeof document !== 'undefined') {
    return document.createElement('canvas').getContext('webgl2', CONTEXT_ATTRS)
  }
  return null
}

let sharedContext: WebGL2RenderingContext | null | undefined

/**
 * Lazily-created context shared across the high-level fallback path. Re-created
 * if the previous one was lost (tab backgrounding, GPU reset). Returns null
 * when WebGL2 is unavailable on this platform.
 */
export function getSharedWebGLContext(): WebGL2RenderingContext | null {
  if (sharedContext === undefined || (sharedContext !== null && sharedContext.isContextLost())) {
    sharedContext = createWebGLContext()
  }
  return sharedContext
}

/** True when a WebGL2 context can be created on this platform. */
export function isWebGLAvailable(): boolean {
  return getSharedWebGLContext() !== null
}
