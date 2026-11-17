// Idle lab page (dev tool): exposes the library on
// `window.__GPUTEX_ENCODERS__` and runs NOTHING by itself, so scripted
// experiments (example/scripts/device-run.mjs --js '…', or the console) have
// the GPU to themselves. /bench exposes the same object but starts its
// benchmark on load, which competes with whatever the script measures.

import { exposeEncoders } from '../lib/exposeEncoders'

exposeEncoders()

const LabPage = () => (
  <div className="min-h-screen bg-neutral-900 p-6 text-sm text-gray-200">
    <h1 className="text-lg font-semibold text-white">GPUtex — lab</h1>
    <p className="text-xs text-gray-400">
      Idle. The library is on <code>window.__GPUTEX_ENCODERS__</code>; drive it from the console or{' '}
      <code>example/scripts/device-run.mjs --js</code>.
    </p>
  </div>
)

export default LabPage
