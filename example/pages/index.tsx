import { Suspense, useCallback, useEffect, useState } from 'react'

import { OrbitControls } from '@react-three/drei/webgpu'
import { Canvas, useLoader } from '@react-three/fiber/webgpu'
import { LinearFilter, RepeatWrapping, SRGBColorSpace, TextureLoader } from 'three'

import DropZone from '../components/DropZone'
import InfoPanel from '../components/InfoPanel'
import { useGputex } from '../hooks/useGputex'

import type { Texture, CompressedTexture } from 'three'

import type { EncodeInfo } from '../hooks/useGputex'

const Sphere = ({ texture }: { texture: Texture | CompressedTexture | null }) => (
  <mesh>
    <sphereGeometry args={[1, 128, 64]} />
    <meshStandardMaterial map={texture} roughness={0.55} metalness={0} />
  </mesh>
)

const CompressedSphere = ({ url, onResult }: { url: string; onResult: (result: EncodeInfo) => void }) => {
  const texture = useGputex(url, { hint: 'color', colorSpace: 'srgb' }, (_tex, result) => {
    if (result) onResult(result)
  })

  useEffect(() => {
    return () => {
      const tex = Array.isArray(texture) ? texture[0] : texture
      tex?.dispose()
      useGputex.clear(url)
    }
  }, [texture, url])

  return <Sphere texture={texture as Texture | CompressedTexture} />
}

const OriginalSphere = ({ url }: { url: string }) => {
  const texture = useLoader(TextureLoader, url)
  texture.colorSpace = SRGBColorSpace
  texture.wrapS = texture.wrapT = RepeatWrapping
  texture.minFilter = LinearFilter
  texture.magFilter = LinearFilter
  texture.generateMipmaps = false

  useEffect(() => {
    return () => {
      texture.dispose()
      useLoader.clear(TextureLoader, url)
    }
  }, [texture, url])

  return <Sphere texture={texture} />
}

const IndexPage = () => {
  const [file, setFile] = useState<File | null>(null)
  const [dataUrl, setDataUrl] = useState<string | null>(null)
  const [blobUrl, setBlobUrl] = useState<string | null>(null)
  const [result, setResult] = useState<EncodeInfo | null>(null)
  const [useCompressed, setUseCompressed] = useState(true)
  const [encoding, setEncoding] = useState(false)

  const onFileDrop = useCallback(
    (droppedFile: File) => {
      // Dispose previous textures before loading a new file
      if (dataUrl) useGputex.clear(dataUrl)
      if (blobUrl) {
        useLoader.clear(TextureLoader, blobUrl)
        URL.revokeObjectURL(blobUrl)
      }

      setFile(droppedFile)
      setResult(null)
      setEncoding(true)
      setBlobUrl(URL.createObjectURL(droppedFile))
      const reader = new FileReader()
      reader.onload = () => setDataUrl(reader.result as string)
      reader.readAsDataURL(droppedFile)
    },
    [dataUrl, blobUrl],
  )

  const handleResult = useCallback((r: EncodeInfo) => {
    setResult(r)
    setEncoding(false)
  }, [])

  return (
    <>
      <Canvas
        camera={{ fov: 40, near: 0.1, far: 100, position: [0, 0.2, 3.4] }}
        className="fixed top-0 left-0 h-screen w-screen bg-neutral-800"
      >
        <ambientLight intensity={0.15} />
        <hemisphereLight args={[0xbcd1ff, 0x181a20, 0.55]} />
        <directionalLight position={[3, 3, 4]} intensity={1.4} />
        <directionalLight position={[-3, -1, -2]} intensity={0.6} color={0xa6c8ff} />
        <OrbitControls enableDamping dampingFactor={0.08} enablePan={false} minDistance={1.6} maxDistance={6} />
        {dataUrl ? (
          <Suspense fallback={<Sphere texture={null} />}>
            {useCompressed ? (
              <CompressedSphere url={dataUrl} onResult={handleResult} />
            ) : (
              <OriginalSphere url={blobUrl!} />
            )}
          </Suspense>
        ) : (
          <Sphere texture={null} />
        )}
      </Canvas>
      <DropZone onFileDrop={onFileDrop} hasTexture={!!file} />
      <InfoPanel
        file={file}
        result={result}
        encoding={encoding}
        useCompressed={useCompressed}
        onToggleCompressed={setUseCompressed}
      />
    </>
  )
}

export default IndexPage
