import { Suspense, useEffect, useMemo, useState } from 'react'

import type { EncoderConstructor, EncodeQuality } from 'gputex'

import { OrbitControls } from '@react-three/drei/webgpu'
import { Canvas, useLoader } from '@react-three/fiber/webgpu'
import { LinearFilter, LinearSRGBColorSpace, MOUSE, NearestFilter, SRGBColorSpace, TextureLoader } from 'three'
import { float, max, sqrt, texture as textureNode, vec4 } from 'three/tsl'
import { MeshBasicNodeMaterial } from 'three/webgpu'

import { useEncodedTexture } from '../hooks/useEncodedTexture'
import TestNav from './TestNav'

import type { Texture } from 'three'

const GAP = 0.06 // world-space gap between the two planes

// Normal-map preview material: reconstruct z = √(1 − x² − y²) from the stored
// (R, G) = (x, y) and show the full normal as colour. Applied to BOTH planes on
// the BC5 page so the comparison is apples-to-apples — without it the compressed
// side has no blue (BC5 stores only R/G), which reads as a fake "artifact".
const NormalMaterial = ({ map }: { map: Texture }) => {
  const material = useMemo(() => new MeshBasicNodeMaterial({ toneMapped: false }), [])
  useEffect(() => {
    const s = textureNode(map)
    const x = s.r.mul(2).sub(1)
    const y = s.g.mul(2).sub(1)
    const z = sqrt(max(float(0), float(1).sub(x.mul(x)).sub(y.mul(y))))
    material.colorNode = vec4(x.mul(0.5).add(0.5), y.mul(0.5).add(0.5), z.mul(0.5).add(0.5), float(1))
    material.needsUpdate = true
  }, [map, material])
  return <primitive object={material} attach="material" />
}

const fmtBytes = (n: number): string => {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

const TexturedPlane = ({
  texture,
  x,
  aspect,
  pixelated,
  colorSpace,
  reconstructNormal,
}: {
  texture: Texture | null
  x: number
  aspect: number
  pixelated: boolean
  colorSpace: 'srgb' | 'linear'
  reconstructNormal: boolean
}) => {
  useEffect(() => {
    if (!texture) return
    texture.colorSpace = colorSpace === 'srgb' ? SRGBColorSpace : LinearSRGBColorSpace
    texture.magFilter = pixelated ? NearestFilter : LinearFilter
    texture.minFilter = pixelated ? NearestFilter : LinearFilter
    texture.needsUpdate = true
  }, [texture, pixelated, colorSpace])

  if (!texture) return null
  return (
    <mesh position={[x, 0, 0]}>
      <planeGeometry args={[aspect, 1]} />
      {reconstructNormal ? <NormalMaterial map={texture} /> : <meshBasicMaterial map={texture} toneMapped={false} />}
    </mesh>
  )
}

const OriginalPlane = (props: {
  url: string
  x: number
  aspect: number
  pixelated: boolean
  colorSpace: 'srgb' | 'linear'
  reconstructNormal: boolean
}) => {
  const texture = useLoader(TextureLoader, props.url)
  return <TexturedPlane texture={texture} {...props} />
}

const Row = ({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) => (
  <div className="flex items-baseline justify-between gap-2 py-0.5">
    <span className="text-gray-400">{label}</span>
    <span className={`font-mono text-xs ${highlight ? 'font-semibold text-green-400' : 'text-gray-200'}`}>{value}</span>
  </div>
)

export interface CompareViewProps {
  /** Page path for nav highlighting, e.g. '/bc7'. */
  current: string
  title: string
  subtitle: string
  description: string
  /** Source image under example/public, e.g. '/textures/color.png'. */
  url: string
  encoder: EncoderConstructor
  colorSpace: 'srgb' | 'linear'
  /** Reconstruct z from the stored (R, G) and show the full normal on both
   *  planes. For BC5, whose 2-channel output otherwise renders without blue. */
  reconstructNormal?: boolean
}

const CompareView = ({
  current,
  title,
  subtitle,
  description,
  url,
  encoder,
  colorSpace,
  reconstructNormal = false,
}: CompareViewProps) => {
  const [quality, setQuality] = useState<EncodeQuality>('fast')
  const [pixelated, setPixelated] = useState(false)

  const { texture, info, error, loading } = useEncodedTexture(url, encoder, { colorSpace, quality })

  const aspect = info ? info.width / info.height : 1
  const halfStep = aspect / 2 + GAP

  const savings = info && info.compressedBytes ? info.rgba8Bytes / info.compressedBytes : 0
  const formatLabel = info?.format ?? (loading ? 'encoding…' : title.toLowerCase())

  return (
    <>
      <Canvas
        camera={{ fov: 40, near: 0.1, far: 100, position: [0, 0, 2.4] }}
        className="fixed inset-0 h-screen w-screen bg-neutral-900"
      >
        <Suspense fallback={null}>
          <OriginalPlane
            url={url}
            x={-halfStep}
            aspect={aspect}
            pixelated={pixelated}
            colorSpace={colorSpace}
            reconstructNormal={reconstructNormal}
          />
        </Suspense>
        <TexturedPlane
          texture={texture}
          x={halfStep}
          aspect={aspect}
          pixelated={pixelated}
          colorSpace={colorSpace}
          reconstructNormal={reconstructNormal}
        />
        <OrbitControls
          makeDefault
          enableRotate={false}
          enablePan
          enableZoom
          minDistance={0.6}
          maxDistance={8}
          mouseButtons={{ LEFT: MOUSE.PAN, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN }}
        />
      </Canvas>

      {/* Plane legends */}
      <div className="pointer-events-none fixed top-4 left-1/2 z-10 flex w-full -translate-x-1/2 justify-center gap-[20vw] text-center">
        <span className="rounded-md bg-black/70 px-2 py-1 font-mono text-xs text-gray-300">
          {reconstructNormal ? 'Original → normal' : 'Original (RGBA8)'}
        </span>
        <span className="rounded-md bg-black/70 px-2 py-1 font-mono text-xs text-blue-300">
          {reconstructNormal ? `${formatLabel} → normal` : formatLabel}
        </span>
      </div>

      <TestNav current={current} />

      <div className="fixed top-4 left-4 z-20 w-80 max-w-[calc(100vw-2rem)] rounded-xl border border-white/10 bg-black/80 p-4 text-sm leading-relaxed shadow-xl backdrop-blur-xl">
        <h1 className="text-[15px] font-semibold text-white">{title}</h1>
        <div className="mb-2 text-[11px] text-gray-400">{subtitle}</div>
        <p className="mb-3 text-[11px] leading-snug text-gray-400">{description}</p>

        {error ? (
          <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-2 text-[11px] leading-snug text-red-300">
            {error}
          </div>
        ) : (
          <>
            <div className="border-t border-white/10 pt-2">
              <Row label="Format" value={info?.format ?? (loading ? 'Encoding…' : '—')} />
              <Row label="Quality" value={quality} />
              <Row label="Resolution" value={info ? `${info.width} × ${info.height} px` : '—'} />
              <Row label="Encode time" value={info ? `${info.encodeMs.toFixed(2)} ms` : '—'} />
            </div>
            <div className="border-t border-white/10 pt-2">
              <Row label="RGBA8 VRAM" value={info ? fmtBytes(info.rgba8Bytes) : '—'} />
              <Row label="Compressed" value={info ? fmtBytes(info.compressedBytes) : '—'} />
              <Row label="Saved" value={savings ? `${savings.toFixed(1)}× smaller` : '—'} highlight />
            </div>
          </>
        )}

        <div className="mt-3 flex flex-col gap-2 border-t border-white/10 pt-3">
          <div className="flex items-center gap-2">
            <span className="w-16 text-xs text-gray-400">Quality</span>
            <div className="flex overflow-hidden rounded-lg border border-white/15">
              {(['fast', 'high'] as const).map(q => (
                <button
                  key={q}
                  type="button"
                  onClick={() => setQuality(q)}
                  className={`px-3 py-1 font-mono text-xs transition-colors ${
                    quality === q ? 'bg-blue-500 text-white' : 'text-gray-300 hover:bg-white/10'
                  }`}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={pixelated}
              onChange={e => setPixelated(e.target.checked)}
              className="accent-blue-500"
            />
            <span className="text-xs text-gray-300">Pixelated (nearest filter — crisp 4×4 blocks)</span>
          </label>
        </div>
      </div>
    </>
  )
}

export default CompareView
