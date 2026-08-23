// Transcode-cache keying and LRU behaviour.
//
// The key is what decides whether a repeat load skips the decode and the
// encode entirely, so these cover what must and must not collide: every
// fingerprint field separates entries, equal Blob bytes share one, pixel
// sources without a `cacheKey` have no identity at all — plus the LRU's
// eviction order, its refusal of oversized entries, and its accounting on
// overwrite.

import { TextureFormat } from '../TextureFormat.js'
import {
  buildTranscodeKey,
  clearTranscodeCache,
  readTranscodeCache,
  setTranscodeCacheLimit,
  writeTranscodeCache,
  type TranscodeFingerprint,
} from '../transcodeCache.js'

const FP: TranscodeFingerprint = {
  format: TextureFormat.BC7_SRGB,
  colorSpace: 'srgb',
  flipY: true,
  mipmaps: false,
}

function entry(bytes: number) {
  return {
    format: TextureFormat.BC7_SRGB,
    width: 4,
    height: 4,
    levels: [{ width: 4, height: 4, paddedWidth: 4, paddedHeight: 4, data: new Uint8Array(bytes) }],
  }
}

afterEach(() => {
  clearTranscodeCache()
  setTranscodeCacheLimit(256 * 1024 * 1024)
})

describe('transcode key composition', () => {
  it('gives two Blobs with equal bytes the same key, and different bytes different keys', async () => {
    const a = await buildTranscodeKey(new Blob([new Uint8Array([9, 9])]), undefined, FP)
    const b = await buildTranscodeKey(new Blob([new Uint8Array([9, 9])]), undefined, FP)
    const c = await buildTranscodeKey(new Blob([new Uint8Array([9, 8])]), undefined, FP)
    expect(a).toBe(b)
    expect(a).not.toBe(c)
  })

  it('has no key for a pixel source without a cacheKey', async () => {
    expect(await buildTranscodeKey({ width: 8, height: 8 }, undefined, FP)).toBeNull()
  })

  it('keys a pixel source once an explicit cacheKey is given', async () => {
    expect(await buildTranscodeKey({ width: 8, height: 8 }, 'atlas/tiles', FP)).not.toBeNull()
  })

  it('separates entries by every fingerprint field', async () => {
    const base = await buildTranscodeKey('/t.png', undefined, FP)
    const keys = await Promise.all([
      buildTranscodeKey('/t.png', undefined, { ...FP, format: TextureFormat.ASTC_4x4_SRGB }),
      buildTranscodeKey('/t.png', undefined, { ...FP, colorSpace: 'linear' }),
      buildTranscodeKey('/t.png', undefined, { ...FP, flipY: false }),
      buildTranscodeKey('/t.png', undefined, { ...FP, mipmaps: true }),
      buildTranscodeKey('/t.png', undefined, { ...FP, svgSize: 512 }),
    ])
    for (const k of keys) expect(k).not.toBe(base)
    expect(new Set([base, ...keys]).size).toBe(6)
  })

  it('returns null for every source once the cache is disabled', async () => {
    setTranscodeCacheLimit(0)
    expect(await buildTranscodeKey('/t.png', undefined, FP)).toBeNull()
  })

  // `compressTexture()` resolves the source identity BEFORE it knows the
  // format, so it has to gate that on the cache being enabled itself — the
  // one-shot builder's own limit check comes too late to save the hash, and
  // hashing a large Blob is not free.
})

describe('transcode cache LRU', () => {
  it('round-trips an entry and misses on an unknown key', () => {
    writeTranscodeCache('a', entry(16))
    expect(readTranscodeCache('a')?.width).toBe(4)
    expect(readTranscodeCache('nope')).toBeNull()
  })

  it('evicts least-recently-used entries past the byte cap', () => {
    setTranscodeCacheLimit(100)
    writeTranscodeCache('a', entry(40))
    writeTranscodeCache('b', entry(40))
    // Touching 'a' makes 'b' the least recently used.
    expect(readTranscodeCache('a')).not.toBeNull()
    writeTranscodeCache('c', entry(40))

    expect(readTranscodeCache('a')).not.toBeNull()
    expect(readTranscodeCache('c')).not.toBeNull()
    expect(readTranscodeCache('b')).toBeNull()
  })

  it('refuses an entry larger than the whole cap rather than wiping the cache', () => {
    setTranscodeCacheLimit(100)
    writeTranscodeCache('small', entry(40))
    writeTranscodeCache('huge', entry(200))
    expect(readTranscodeCache('huge')).toBeNull()
    expect(readTranscodeCache('small')).not.toBeNull()
  })

  it('does not double-count a key that is overwritten', () => {
    setTranscodeCacheLimit(100)
    writeTranscodeCache('a', entry(60))
    writeTranscodeCache('a', entry(60))
    writeTranscodeCache('b', entry(40))
    // If the overwrite had leaked its predecessor's bytes, 'a' would have
    // been evicted to make room for 'b'.
    expect(readTranscodeCache('a')).not.toBeNull()
    expect(readTranscodeCache('b')).not.toBeNull()
  })
})
