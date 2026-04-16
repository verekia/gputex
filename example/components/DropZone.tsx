import { useCallback, useEffect, useRef, useState } from 'react'

interface DropZoneProps {
  onFileDrop: (file: File) => void
  hasTexture: boolean
}

const DropZone = ({ onFileDrop, hasTexture }: DropZoneProps) => {
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (file: File) => {
      if (file.type.startsWith('image/')) onFileDrop(file)
    },
    [onFileDrop],
  )

  useEffect(() => {
    const onDragOver = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDragOver(true)
    }
    const onDragLeave = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      if ((e as DragEvent).relatedTarget) return
      setIsDragOver(false)
    }
    const onDrop = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDragOver(false)
      const file = e.dataTransfer?.files?.[0]
      if (file) handleFile(file)
    }
    window.addEventListener('dragover', onDragOver)
    window.addEventListener('dragleave', onDragLeave)
    window.addEventListener('drop', onDrop)
    return () => {
      window.removeEventListener('dragover', onDragOver)
      window.removeEventListener('dragleave', onDragLeave)
      window.removeEventListener('drop', onDrop)
    }
  }, [handleFile])

  return (
    <div
      className={`pointer-events-none fixed inset-0 z-10 flex items-end justify-center pb-6 transition-colors sm:items-center sm:pb-0 ${isDragOver ? 'bg-blue-500/10' : ''}`}
    >
      {!hasTexture && (
        <div className="rounded-xl border border-dashed border-white/20 bg-black/70 px-7 py-5 text-center shadow-lg backdrop-blur-md">
          <div className="mb-1 text-sm font-semibold text-white">Drop a PNG / JPG / WebP / AVIF</div>
          <div className="text-xs text-gray-400">or click to pick one</div>
          <button
            className="pointer-events-auto mt-3 cursor-pointer rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-white transition-colors hover:bg-white/10"
            onClick={() => fileInputRef.current?.click()}
          >
            Open image...
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/avif"
            className="hidden"
            onChange={e => {
              const file = e.target.files?.[0]
              if (file) handleFile(file)
              e.target.value = ''
            }}
          />
        </div>
      )}
    </div>
  )
}

export default DropZone
