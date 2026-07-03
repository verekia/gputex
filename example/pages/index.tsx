import { Suspense, useCallback, useState } from 'react'

import { OrbitControls } from '@react-three/drei/webgpu'
import { Canvas, useLoader } from '@react-three/fiber/webgpu'
import { LinearFilter, NoToneMapping, SRGBColorSpace, TextureLoader } from 'three'

import DropZone from '../components/DropZone'
import InfoPanel from '../components/InfoPanel'
import { useCompressedFile } from '../hooks/useCompressedFile'

import type { Texture, CompressedTexture } from 'three'

import type { EncodeInfo } from '../hooks/useGputex'

const Sphere = ({ texture }: { texture: Texture | CompressedTexture | null }) => (
  <mesh>
    <sphereGeometry args={[1, 128, 64]} />
    <meshStandardMaterial map={texture} roughness={0.55} metalness={0} />
  </mesh>
)

// The dropped File goes straight to compressTexture() (no URL hop) with the
// persistent transcode cache on — re-dropping a file skips decode + encode.
// Keyed by texture so the material remounts with its map already set: the
// WebGPU node material doesn't rebuild on a null→map swap by itself (the old
// Suspense path never hit that transition).
const CompressedSphere = ({ file, onResult }: { file: File; onResult: (result: EncodeInfo) => void }) => {
  const texture = useCompressedFile(file, onResult)
  return <Sphere key={texture ? texture.uuid : 'pending'} texture={texture} />
}

const OriginalSphere = ({ url }: { url: string }) => {
  const texture = useLoader(TextureLoader, url)
  texture.colorSpace = SRGBColorSpace
  texture.minFilter = LinearFilter
  texture.magFilter = LinearFilter
  texture.generateMipmaps = false
  return <Sphere texture={texture} />
}

const IndexPage = () => {
  const [file, setFile] = useState<File | null>(null)
  const [blobUrl, setBlobUrl] = useState<string | null>(null)
  const [result, setResult] = useState<EncodeInfo | null>(null)
  const [useCompressed, setUseCompressed] = useState(true)
  const [encoding, setEncoding] = useState(false)

  const onFileDrop = useCallback((droppedFile: File) => {
    setFile(droppedFile)
    setResult(null)
    setEncoding(true)
    // Only the "original" sphere needs a URL (TextureLoader takes strings);
    // the compressed path consumes the File directly.
    setBlobUrl(URL.createObjectURL(droppedFile))
  }, [])

  const handleResult = useCallback((r: EncodeInfo) => {
    setResult(r)
    setEncoding(false)
  }, [])

  return (
    <>
      <Canvas
        camera={{ fov: 40, near: 0.1, far: 100, position: [0, 0.2, 3.4] }}
        className="fixed top-0 left-0 h-screen w-screen bg-neutral-800"
        renderer={{ toneMapping: NoToneMapping }}
      >
        <ambientLight intensity={0.15} />
        <hemisphereLight args={[0xbcd1ff, 0x181a20, 0.55]} />
        <directionalLight position={[3, 3, 4]} intensity={1.4} />
        <directionalLight position={[-3, -1, -2]} intensity={0.6} color={0xa6c8ff} />
        <OrbitControls enableDamping dampingFactor={0.08} enablePan={false} minDistance={1.6} maxDistance={6} />
        {file && blobUrl ? (
          <Suspense fallback={<Sphere texture={null} />}>
            {useCompressed ? (
              <CompressedSphere file={file} onResult={handleResult} />
            ) : (
              <OriginalSphere url={blobUrl} />
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
