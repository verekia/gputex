import { useEffect, useState } from 'react'

import { detectCapabilities } from 'gputex'
import type { Capabilities } from 'gputex'

import type { EncodeInfo } from '../hooks/useGputex'

function fmtBytes(n: number | null | undefined): string {
  if (n == null) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

interface InfoPanelProps {
  file: File | null
  result: EncodeInfo | null
  encoding: boolean
  useCompressed: boolean
  onToggleCompressed: (value: boolean) => void
}

const InfoPanel = ({ file, result, encoding, useCompressed, onToggleCompressed }: InfoPanelProps) => {
  const [caps, setCaps] = useState<Capabilities | null>(null)

  useEffect(() => {
    const probe = async () => {
      if (!('gpu' in navigator)) return
      const adapter = await navigator.gpu.requestAdapter()
      if (!adapter) return
      setCaps(detectCapabilities(adapter))
    }
    probe()
  }, [])

  const origVram = result ? result.width * result.height * 4 : 0
  const compVram = result ? result.compressedBytes : 0
  const savings = origVram && compVram ? origVram / compVram : 0
  const savedPct = origVram && compVram ? 100 * (1 - compVram / origVram) : 0

  return (
    <div className="fixed top-4 left-4 z-20 w-80 max-w-[calc(100vw-2rem)] rounded-xl border border-white/10 bg-black/80 p-4 text-sm leading-relaxed shadow-xl backdrop-blur-xl">
      <h1 className="mb-0.5 text-[15px] font-semibold text-white">GPUtex</h1>
      <div className="mb-1 text-[11px] text-gray-400">WebGPU compute shader &middot; BC7 / BC5 / ASTC &middot; R3F</div>

      {caps && (
        <div className="mb-3 font-mono text-[11px]">
          <span className={caps.bc ? 'text-green-400' : 'text-gray-500'}>{caps.bc ? '✓' : '✗'} BC</span>
          {' · '}
          <span className={caps.astc ? 'text-green-400' : 'text-gray-500'}>{caps.astc ? '✓' : '✗'} ASTC</span>
          {' · '}
          <span className={caps.etc2 ? 'text-green-400' : 'text-gray-500'}>{caps.etc2 ? '✓' : '✗'} ETC2</span>
        </div>
      )}

      <div className="border-t border-white/10 pt-2">
        <Row label="File" value={file?.name ?? '—'} />
        <Row label="Resolution" value={result ? `${result.width} × ${result.height} px` : '—'} />
      </div>

      <div className="border-t border-white/10 pt-2">
        <div className="mb-1 text-[10.5px] tracking-wider text-gray-500 uppercase">Original (decoded image)</div>
        <Row label="Download size" value={file ? fmtBytes(file.size) : '—'} />
        <Row label="VRAM (RGBA8)" value={origVram ? fmtBytes(origVram) : '—'} />
      </div>

      <div className="border-t border-white/10 pt-2">
        <div className="mb-1 text-[10.5px] tracking-wider text-gray-500 uppercase">GPU-compressed</div>
        <Row label="Format" value={result?.format ?? (encoding ? 'Encoding…' : '—')} />
        <Row label="VRAM" value={compVram ? fmtBytes(compVram) : '—'} />
        <Row label="Encode time" value={result ? `${result.encodeMs.toFixed(1)} ms` : '—'} />
      </div>

      <div className="border-t border-white/10 pt-2">
        <Row
          label="VRAM saved"
          value={savings ? `${savings.toFixed(1)}× smaller (−${savedPct.toFixed(1)}%)` : '—'}
          highlight
        />
      </div>

      <div className="flex items-center gap-2 border-t border-white/10 pt-3">
        <label className="flex cursor-pointer items-center gap-2">
          <input
            type="checkbox"
            checked={useCompressed}
            onChange={e => onToggleCompressed(e.target.checked)}
            className="accent-blue-500"
          />
          <span className="text-xs text-gray-300">Use GPU-compressed texture</span>
        </label>
      </div>
    </div>
  )
}

const Row = ({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) => (
  <div className="flex items-baseline justify-between gap-2 py-0.5">
    <span className="text-gray-400">{label}</span>
    <span className={`font-mono text-xs ${highlight ? 'font-semibold text-green-400' : 'text-gray-200'}`}>{value}</span>
  </div>
)

export default InfoPanel
