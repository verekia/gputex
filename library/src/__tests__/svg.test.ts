import {
  hasSvgExtension,
  isSvgBlob,
  isSvgMarkup,
  parseSvgDimensions,
  resolveSvgRasterSize,
  setSvgRootSize,
} from '../svg.js'

describe('isSvgMarkup', () => {
  it('detects inline markup vs URLs', () => {
    expect(isSvgMarkup('<svg viewBox="0 0 1 1"></svg>')).toBe(true)
    expect(isSvgMarkup('  \n<?xml version="1.0"?><svg/>')).toBe(true)
    expect(isSvgMarkup('/icons/logo.svg')).toBe(false)
    expect(isSvgMarkup('https://example.com/a.svg')).toBe(false)
  })
})

describe('hasSvgExtension', () => {
  it('matches .svg paths, ignoring query and hash', () => {
    expect(hasSvgExtension('/a/b/logo.svg')).toBe(true)
    expect(hasSvgExtension('logo.SVG')).toBe(true)
    expect(hasSvgExtension('/logo.svg?v=2#frag')).toBe(true)
    expect(hasSvgExtension('/logo.png')).toBe(false)
    expect(hasSvgExtension('/convert?file=in.svg')).toBe(false)
  })
})

describe('isSvgBlob', () => {
  it('decides by MIME type when present', () => {
    expect(isSvgBlob(new Blob(['<svg/>'], { type: 'image/svg+xml' }))).toBe(true)
    expect(isSvgBlob(new Blob(['<svg/>'], { type: 'image/svg+xml;charset=utf-8' }))).toBe(true)
    expect(isSvgBlob(new Blob([''], { type: 'image/png' }))).toBe(false)
  })

  it('falls back to the filename for untyped Files', () => {
    expect(isSvgBlob(new File(['<svg/>'], 'icon.svg'))).toBe(true)
    expect(isSvgBlob(new File([''], 'photo.png'))).toBe(false)
  })
})

describe('parseSvgDimensions', () => {
  it('reads absolute width/height attributes', () => {
    const d = parseSvgDimensions('<svg width="128" height="64"></svg>')!
    expect(d.width).toBe(128)
    expect(d.height).toBe(64)
    expect(d.viewBoxWidth).toBeNull()
  })

  it('accepts px units and single quotes', () => {
    const d = parseSvgDimensions("<svg width='24px' height='24px'/>")!
    expect(d.width).toBe(24)
    expect(d.height).toBe(24)
  })

  it('treats relative/other units as absent', () => {
    const d = parseSvgDimensions('<svg width="100%" height="10cm" viewBox="0 0 50 25"/>')!
    expect(d.width).toBeNull()
    expect(d.height).toBeNull()
    expect(d.viewBoxWidth).toBe(50)
    expect(d.viewBoxHeight).toBe(25)
  })

  it('parses comma-separated viewBox values', () => {
    const d = parseSvgDimensions('<svg viewBox="0,0,300.5,150"/>')!
    expect(d.viewBoxWidth).toBe(300.5)
    expect(d.viewBoxHeight).toBe(150)
  })

  it('does not confuse stroke-width with width', () => {
    const d = parseSvgDimensions('<svg stroke-width="3" viewBox="0 0 10 10"/>')!
    expect(d.width).toBeNull()
    expect(d.viewBoxWidth).toBe(10)
  })

  it('skips the XML prolog and doctype', () => {
    const text = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE svg>\n<svg width="8" height="8"/>'
    expect(parseSvgDimensions(text)!.width).toBe(8)
  })

  it('returns null when there is no <svg> root', () => {
    expect(parseSvgDimensions('<html><body/></html>')).toBeNull()
    expect(parseSvgDimensions('not markup at all')).toBeNull()
  })
})

describe('resolveSvgRasterSize', () => {
  const dims = (width: number | null, height: number | null, vbW: number | null = null, vbH: number | null = null) => ({
    width,
    height,
    viewBoxWidth: vbW,
    viewBoxHeight: vbH,
  })

  it('uses the intrinsic attribute size by default', () => {
    expect(resolveSvgRasterSize(dims(128, 64))).toEqual({ width: 128, height: 64 })
  })

  it('falls back to the viewBox when attributes are missing', () => {
    expect(resolveSvgRasterSize(dims(null, null, 300, 150))).toEqual({ width: 300, height: 150 })
  })

  it('derives a missing attribute from the viewBox aspect ratio', () => {
    expect(resolveSvgRasterSize(dims(200, null, 100, 50))).toEqual({ width: 200, height: 100 })
    expect(resolveSvgRasterSize(dims(null, 100, 100, 50))).toEqual({ width: 200, height: 100 })
  })

  it('scales the longest side for a numeric size, preserving aspect', () => {
    expect(resolveSvgRasterSize(dims(100, 50), 512)).toEqual({ width: 512, height: 256 })
    expect(resolveSvgRasterSize(dims(50, 100), 512)).toEqual({ width: 256, height: 512 })
    expect(resolveSvgRasterSize(dims(null, null, 32, 32), 512)).toEqual({ width: 512, height: 512 })
  })

  it('assumes square for a numeric size when no aspect is known', () => {
    expect(resolveSvgRasterSize(dims(null, null), 256)).toEqual({ width: 256, height: 256 })
  })

  it('uses an explicit { width, height } verbatim', () => {
    expect(resolveSvgRasterSize(dims(100, 50), { width: 64, height: 640 })).toEqual({ width: 64, height: 640 })
    // Explicit size needs no intrinsic size at all.
    expect(resolveSvgRasterSize(dims(null, null), { width: 4, height: 4 })).toEqual({ width: 4, height: 4 })
  })

  it('never returns a dimension below 1', () => {
    expect(resolveSvgRasterSize(dims(1000, 1), 512).height).toBe(1)
  })

  it('throws without any usable size', () => {
    expect(() => resolveSvgRasterSize(dims(null, null))).toThrow(/svgSize/)
    expect(() => resolveSvgRasterSize(dims(100, 100), 0)).toThrow(/≥1/)
    expect(() => resolveSvgRasterSize(dims(100, 100), { width: 0, height: 8 })).toThrow(/≥1/)
  })
})

describe('setSvgRootSize', () => {
  it('replaces existing width/height attributes', () => {
    const out = setSvgRootSize('<svg width="10" height="20" viewBox="0 0 10 20"><rect/></svg>', 100, 200)
    expect(out).toBe('<svg width="100" height="200" viewBox="0 0 10 20"><rect/></svg>')
  })

  it('adds width/height when absent', () => {
    const out = setSvgRootSize('<svg viewBox="0 0 8 8"/>', 32, 32)
    expect(out).toBe('<svg width="32" height="32" viewBox="0 0 8 8"/>')
  })

  it('synthesises a viewBox from the original size so resizing scales content', () => {
    const out = setSvgRootSize('<svg width="10" height="20"><rect/></svg>', 100, 200)
    expect(out).toBe('<svg width="100" height="200" viewBox="0 0 10 20"><rect/></svg>')
  })

  it('does not synthesise a viewBox when the original had no absolute size', () => {
    const out = setSvgRootSize('<svg><rect/></svg>', 64, 64)
    expect(out).toBe('<svg width="64" height="64"><rect/></svg>')
  })

  it('preserves surrounding markup and other attributes', () => {
    const text =
      '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" fill="red"><g/></svg>'
    const out = setSvgRootSize(text, 40, 40)
    expect(out).toBe(
      '<?xml version="1.0"?><svg width="40" height="40" viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg" fill="red"><g/></svg>',
    )
  })

  it('handles single-quoted attributes and %/relative sizes', () => {
    const out = setSvgRootSize("<svg width='100%' height='100%' viewBox='0 0 16 16'/>", 64, 64)
    expect(out).toBe(`<svg width="64" height="64" viewBox='0 0 16 16'/>`)
  })

  it('throws when there is no <svg> root', () => {
    expect(() => setSvgRootSize('<html/>', 8, 8)).toThrow(/no <svg> root/)
  })
})
