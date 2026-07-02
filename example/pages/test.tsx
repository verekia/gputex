// GPU encoder test + benchmark suite page (WebGPU only). See
// ../lib/gpuTestSuite.ts for what runs. Automation hooks:
//   window.__GPUTEX_TESTS__ = { status: 'running'|'done'|'error', results?, error? }
// The page renders the same data as human-readable tables.

import { useEffect, useRef, useState } from 'react'

import { runSuite } from '../lib/gpuTestSuite'

import type { SuiteResults } from '../lib/gpuTestSuite'

type Status = 'running' | 'done' | 'error'

declare global {
  interface Window {
    __GPUTEX_TESTS__?: { status: Status; results?: SuiteResults; error?: string }
  }
}

const fmtMs = (ms: number | null): string => (ms === null ? '—' : `${ms.toFixed(2)} ms`)
const fmtDb = (db: number): string => (Number.isFinite(db) ? `${db.toFixed(2)} dB` : '∞')

const Badge = ({ pass }: { pass: boolean }) => (
  <span
    className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${pass ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'}`}
  >
    {pass ? 'PASS' : 'FAIL'}
  </span>
)

const Th = ({ children }: { children: React.ReactNode }) => (
  <th className="border-b border-white/10 px-2 py-1 text-left text-[11px] font-semibold text-gray-300">{children}</th>
)
const Td = ({ children, mono = true }: { children: React.ReactNode; mono?: boolean }) => (
  <td className={`border-b border-white/5 px-2 py-1 text-[11px] text-gray-200 ${mono ? 'font-mono' : ''}`}>
    {children}
  </td>
)

const TestPage = () => {
  const [status, setStatus] = useState<Status>('running')
  const [progress, setProgress] = useState('Starting…')
  const [results, setResults] = useState<SuiteResults | null>(null)
  const [error, setError] = useState<string | null>(null)
  const started = useRef(false)

  useEffect(() => {
    if (started.current) return
    started.current = true
    window.__GPUTEX_TESTS__ = { status: 'running' }
    runSuite(setProgress)
      .then(r => {
        setResults(r)
        setStatus('done')
        window.__GPUTEX_TESTS__ = { status: 'done', results: r }
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? (e.stack ?? e.message) : String(e)
        setError(msg)
        setStatus('error')
        window.__GPUTEX_TESTS__ = { status: 'error', error: msg }
      })
  }, [])

  return (
    <div className="min-h-screen bg-neutral-900 p-6 text-sm text-gray-200">
      <h1 className="text-lg font-semibold text-white">GPUtex — GPU test & benchmark suite</h1>
      <p className="mb-4 text-xs text-gray-400">
        WebGPU compute encoders only (the WebGL2 fallback is not exercised here).
      </p>

      {status === 'running' && (
        <div className="rounded-lg border border-blue-500/30 bg-blue-500/10 p-3 font-mono text-xs text-blue-200">
          Running… {progress}
        </div>
      )}
      {status === 'error' && (
        <pre className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-xs whitespace-pre-wrap text-red-300">
          {error}
        </pre>
      )}

      {results && (
        <div className="mt-4 flex flex-col gap-6">
          <div className="font-mono text-xs text-gray-400">
            {results.env.vendor} / {results.env.architecture} · f16: {results.env.hasF16 ? 'yes' : 'no'} · features:{' '}
            {results.env.features.join(', ')}
          </div>

          <div
            className={`w-fit rounded-lg px-3 py-2 font-mono text-xs ${results.failures === 0 ? 'bg-green-500/15 text-green-300' : 'bg-red-500/15 text-red-300'}`}
          >
            {results.failures === 0 ? 'ALL TESTS PASSED' : `${results.failures} FAILURE(S)`}
          </div>

          <section>
            <h2 className="mb-2 font-semibold text-white">Correctness</h2>
            <table className="border-collapse">
              <thead>
                <tr>
                  <Th>Test</Th>
                  <Th>Result</Th>
                  <Th>Detail</Th>
                </tr>
              </thead>
              <tbody>
                {results.correctness.map(c => (
                  <tr key={c.name}>
                    <Td>{c.name}</Td>
                    <Td mono={false}>
                      <Badge pass={c.pass} />
                    </Td>
                    <Td>{c.detail}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section>
            <h2 className="mb-2 font-semibold text-white">Quality (PSNR vs source, CPU-decoded)</h2>
            <table className="border-collapse">
              <thead>
                <tr>
                  <Th>Format</Th>
                  <Th>Quality</Th>
                  <Th>Variant</Th>
                  <Th>Image</Th>
                  <Th>PSNR</Th>
                  <Th>Threshold</Th>
                  <Th>Worst easy block</Th>
                  <Th>Limit</Th>
                  <Th>Worst any</Th>
                  <Th>Result</Th>
                </tr>
              </thead>
              <tbody>
                {results.quality.map(q => (
                  <tr key={`${q.format}-${q.quality}-${q.variant}-${q.image}`}>
                    <Td>{q.format}</Td>
                    <Td>{q.quality}</Td>
                    <Td>{q.variant}</Td>
                    <Td>{q.image}</Td>
                    <Td>{fmtDb(q.psnrDb)}</Td>
                    <Td>{q.thresholdDb ? fmtDb(q.thresholdDb) : '—'}</Td>
                    <Td>{q.quality === 'fast' ? q.worstEasyBlockExcess.toFixed(4) : '—'}</Td>
                    <Td>{q.excessLimit !== null ? q.excessLimit.toFixed(2) : '—'}</Td>
                    <Td>{q.quality === 'fast' ? q.worstBlockExcess.toFixed(2) : '—'}</Td>
                    <Td mono={false}>
                      <Badge pass={q.pass} />
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section>
            <h2 className="mb-2 font-semibold text-white">Performance (median, {results.perf[0]?.image})</h2>
            <table className="border-collapse">
              <thead>
                <tr>
                  <Th>Format</Th>
                  <Th>Quality</Th>
                  <Th>Variant</Th>
                  <Th>Wall</Th>
                  <Th>GPU pass</Th>
                  <Th>MPix/s</Th>
                </tr>
              </thead>
              <tbody>
                {results.perf.map(p => (
                  <tr key={`${p.format}-${p.quality}-${p.variant}`}>
                    <Td>{p.format}</Td>
                    <Td>{p.quality}</Td>
                    <Td>{p.variant}</Td>
                    <Td>{fmtMs(p.wallMsMedian)}</Td>
                    <Td>{fmtMs(p.gpuMsMedian)}</Td>
                    <Td>{p.mpixPerSec.toFixed(0)}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      )}
    </div>
  )
}

export default TestPage
