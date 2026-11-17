#!/usr/bin/env bun
// Drive an example-app page on a real device over the Chrome DevTools
// Protocol and print what it reports. One driver for every target, so the
// same /eval or /bench URL runs unchanged on phones and on this machine:
//
//   bun example/scripts/device-run.mjs <target> '<path?query>' [options]
//   bun example/scripts/device-run.mjs <target> --js '<expression>' [--on '<path>']   (default page: /lab)
//   bun example/scripts/device-run.mjs list
//
// Targets
//   mac            a private headless Chrome (own profile, own debugging
//                  port) — never touches the Chrome the user is working in.
//   <adb serial>   Chrome on an Android device (also matches a unique
//                  substring of the serial or of the model name, e.g. "pixel").
//
// The page is expected to publish { status: 'running'|'done'|'error', ... }
// on a window global (default: whichever of __GPUTEX_EVAL__ / __GPUTEX_AB__ /
// __GPUTEX_BENCH__ / __GPUTEX_TESTS__ / __GPUTEX_LAB__ appears first).
//
// Options
//   --port 3210        dev-server port on this machine (adb-reversed to phones)
//   --global NAME      window global to poll
//   --timeout 900      seconds before giving up
//   --json             print the whole result object as JSON (default: .text
//                      when the page provides one)
//   --keep             leave the tab open afterwards
//
// Resilience: every run opens its own tab and closes it, phones are woken
// and Chrome foregrounded first (a backgrounded tab is throttled), a dropped
// DevTools socket is re-attached to the same tab instead of restarting the
// run, and battery temperature is logged before/after so thermally
// throttled runs can be told apart.

import { spawn, spawnSync } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const args = process.argv.slice(2)
const flag = (name, def) => {
  const i = args.indexOf(`--${name}`)
  if (i < 0) return def
  const v = args[i + 1]
  args.splice(i, 2)
  return v
}
const bool = name => {
  const i = args.indexOf(`--${name}`)
  if (i < 0) return false
  args.splice(i, 1)
  return true
}

const PORT = Number(flag('port', process.env.GPUTEX_PORT ?? 3210))
const GLOBAL = flag('global', null)
const TIMEOUT_S = Number(flag('timeout', 900))
const JS = flag('js', null)
const ON = flag('on', '/lab')
const AS_JSON = bool('json')
const KEEP = bool('keep')
const [targetArg, pathArg] = args

const sh = (cmd, a, opts = {}) => {
  const r = spawnSync(cmd, a, { encoding: 'utf8', ...opts })
  return (r.stdout ?? '') + (r.stderr ?? '')
}
const sleep = ms => new Promise(r => setTimeout(r, ms))
const log = (...a) => console.error('[device-run]', ...a)

function adbDevices() {
  return sh('adb', ['devices', '-l'])
    .split('\n')
    .slice(1)
    .map(l => /^(\S+)\s+device\b.*?model:(\S+)/.exec(l))
    .filter(Boolean)
    .map(m => ({ serial: m[1], model: m[2] }))
}

if (targetArg === 'list' || !targetArg) {
  console.log('mac')
  for (const d of adbDevices()) console.log(`${d.serial}  ${d.model}`)
  process.exit(targetArg ? 0 : 1)
}

// ---- target setup: returns the local DevTools HTTP port

const MAC_PORT = 9333

async function devtoolsUp(port) {
  try {
    const r = await fetch(`http://127.0.0.1:${port}/json/version`, { signal: AbortSignal.timeout(2000) })
    return r.ok ? await r.json() : null
  } catch {
    return null
  }
}

async function setupMac() {
  if (!(await devtoolsUp(MAC_PORT))) {
    const profile = join(tmpdir(), 'gputex-device-run-chrome')
    mkdirSync(profile, { recursive: true })
    const bin = process.env.CHROME_BIN ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    const child = spawn(
      bin,
      [
        '--headless=new',
        `--remote-debugging-port=${MAC_PORT}`,
        `--user-data-dir=${profile}`,
        '--no-first-run',
        '--no-default-browser-check',
        '--enable-unsafe-webgpu',
        '--enable-webgpu-developer-features',
        '--enable-dawn-features=allow_unsafe_apis',
        '--disable-background-timer-throttling',
        '--disable-backgrounding-occluded-windows',
        '--disable-renderer-backgrounding',
        'about:blank',
      ],
      { detached: true, stdio: 'ignore' },
    )
    child.unref()
    for (let i = 0; i < 50 && !(await devtoolsUp(MAC_PORT)); i++) await sleep(200)
    if (!(await devtoolsUp(MAC_PORT))) throw new Error('headless Chrome did not start')
    log('started private headless Chrome on port', MAC_PORT)
  }
  return { devtools: MAC_PORT, label: 'mac', temp: () => null }
}

async function setupAndroid(dev) {
  const adb = (...a) => sh('adb', ['-s', dev.serial, ...a])
  // Stable per-device local port derived from the serial.
  let h = 0
  for (const c of dev.serial) h = (h * 31 + c.charCodeAt(0)) >>> 0
  const devtools = 9400 + (h % 500)
  adb('reverse', `tcp:${PORT}`, `tcp:${PORT}`)
  adb('shell', 'input', 'keyevent', 'KEYCODE_WAKEUP')
  adb('shell', 'svc', 'power', 'stayon', 'true')
  if (/isKeyguardShowing=true|mDreamingLockscreen=true/.test(adb('shell', 'dumpsys', 'window'))) {
    adb('shell', 'wm', 'dismiss-keyguard')
    await sleep(500)
    if (/isKeyguardShowing=true/.test(adb('shell', 'dumpsys', 'window')))
      throw new Error(`${dev.model}: device is locked — unlock it and retry`)
  }
  if (!/ResumedActivity.*com\.android\.chrome/.test(adb('shell', 'dumpsys', 'activity', 'activities'))) {
    adb('shell', 'am', 'start', '-n', 'com.android.chrome/com.google.android.apps.chrome.Main')
    await sleep(1500)
  }
  adb('forward', `tcp:${devtools}`, 'localabstract:chrome_devtools_remote')
  for (let i = 0; i < 20 && !(await devtoolsUp(devtools)); i++) await sleep(300)
  if (!(await devtoolsUp(devtools))) throw new Error(`${dev.model}: Chrome DevTools socket not reachable`)
  const temp = () => {
    const m = /temperature:\s*(\d+)/.exec(adb('shell', 'dumpsys', 'battery'))
    return m ? Number(m[1]) / 10 : null
  }
  return { devtools, label: dev.model, temp }
}

async function setupTarget(name) {
  if (name === 'mac') return setupMac()
  const devs = adbDevices()
  const n = name.toLowerCase()
  const hits = devs.filter(d => d.serial.toLowerCase().includes(n) || d.model.toLowerCase().includes(n))
  if (hits.length !== 1)
    throw new Error(
      `target '${name}' matches ${hits.length} adb devices (have: ${devs.map(d => `${d.serial} ${d.model}`).join(', ') || 'none'})`,
    )
  return setupAndroid(hits[0])
}

// ---- minimal CDP client over the browser-level socket (flat sessions)

class Cdp {
  constructor(port) {
    this.port = port
    this.id = 0
    this.pending = new Map()
    this.ws = null
  }
  async connect() {
    const v = await devtoolsUp(this.port)
    if (!v) throw new Error('DevTools endpoint went away')
    const url = v.webSocketDebuggerUrl.replace(/ws:\/\/[^/]+/, `ws://127.0.0.1:${this.port}`)
    await new Promise((resolve, reject) => {
      const ws = new WebSocket(url)
      ws.onopen = () => resolve()
      ws.onerror = () => reject(new Error('DevTools websocket error'))
      ws.onclose = () => {
        for (const p of this.pending.values()) p.reject(new Error('DevTools socket closed'))
        this.pending.clear()
        this.ws = null
      }
      ws.onmessage = ev => {
        const msg = JSON.parse(ev.data)
        const p = this.pending.get(msg.id)
        if (!p) return
        this.pending.delete(msg.id)
        if (msg.error) p.reject(new Error(msg.error.message))
        else p.resolve(msg.result)
      }
      this.ws = ws
    })
  }
  async send(method, params = {}, sessionId) {
    if (!this.ws) await this.connect()
    const id = ++this.id
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`${method}: no reply in 60 s`))
      }, 60000)
      this.pending.set(id, {
        resolve: v => (clearTimeout(timer), resolve(v)),
        reject: e => (clearTimeout(timer), reject(e)),
      })
      this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }))
    })
  }
  close() {
    this.ws?.close()
  }
}

async function main() {
  const target = await setupTarget(targetArg)
  const base = `http://localhost:${PORT}`
  const url = JS ? base + ON : base + pathArg
  if (!JS && !pathArg) throw new Error('missing page path (or --js)')

  const cdp = new Cdp(target.devtools)
  const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' })
  let sessionId = null
  const attach = async () => {
    sessionId = (await cdp.send('Target.attachToTarget', { targetId, flatten: true })).sessionId
    await cdp.send('Target.activateTarget', { targetId }).catch(() => {})
  }
  const evaluate = async (expression, awaitPromise = false) => {
    for (let attempt = 0; ; attempt++) {
      try {
        if (!sessionId) await attach()
        const r = await cdp.send('Runtime.evaluate', { expression, awaitPromise, returnByValue: true }, sessionId)
        if (r.exceptionDetails)
          throw Object.assign(
            new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text ?? 'page exception'),
            { page: true },
          )
        return r.result.value
      } catch (e) {
        if (e.page || attempt >= 5) throw e
        log(`${e.message} — re-attaching (${attempt + 1}/5)`)
        sessionId = null
        await sleep(1000 * (attempt + 1))
      }
    }
  }

  const t0 = target.temp()
  const started = Date.now()
  try {
    await attach()
    await cdp.send('Page.enable', {}, sessionId)
    await cdp.send('Page.navigate', { url }, sessionId)
    log(`${target.label}: ${url}${t0 === null ? '' : ` (battery ${t0} °C)`}`)

    let result
    if (JS) {
      // Wait for the page's module graph, then run the expression.
      const deadline = Date.now() + 120000
      while (Date.now() < deadline) {
        if (await evaluate(`document.readyState === 'complete'`).catch(() => false)) break
        await sleep(300)
      }
      await sleep(500)
      // Start the expression detached and poll for its settlement: a long
      // awaitPromise evaluate would die with the socket (or its reply
      // timeout) and a retry would start the work a second time.
      await evaluate(
        `(() => { const g = (window.__GPUTEX_JS__ = { status: 'running' }); ` +
          `(async () => (${JS}))().then(v => { g.value = v; g.status = 'done' }, ` +
          `e => { g.error = String(e && e.stack || e); g.status = 'error' }); return true })()`,
      )
      for (;;) {
        if (Date.now() - started > TIMEOUT_S * 1000) throw new Error(`timed out after ${TIMEOUT_S} s`)
        const st = await evaluate(`window.__GPUTEX_JS__ ? window.__GPUTEX_JS__.status : 'lost'`)
        if (st === 'lost') throw new Error('page reloaded while the expression was running')
        if (st !== 'running') break
        await sleep(500)
      }
      const g = await evaluate(`window.__GPUTEX_JS__`)
      result = g.status === 'error' ? { status: 'error', error: g.error } : g.value
    } else {
      const names = GLOBAL
        ? [GLOBAL]
        : ['__GPUTEX_EVAL__', '__GPUTEX_AB__', '__GPUTEX_BENCH__', '__GPUTEX_TESTS__', '__GPUTEX_LAB__']
      const probe = `(() => { for (const n of ${JSON.stringify(names)}) { const g = window[n]; if (g && g.status) return { n, status: g.status, progress: g.progress ?? null }; } return null })()`
      let last = ''
      for (;;) {
        if (Date.now() - started > TIMEOUT_S * 1000) throw new Error(`timed out after ${TIMEOUT_S} s`)
        const s = await evaluate(probe)
        if (s && s.status !== 'running') {
          result = await evaluate(`window[${JSON.stringify(s.n)}]`)
          break
        }
        if (s?.progress && s.progress !== last) {
          last = s.progress
          log(last)
        }
        await sleep(1000)
      }
    }

    const t1 = target.temp()
    log(`done in ${((Date.now() - started) / 1000).toFixed(1)} s${t1 === null ? '' : ` (battery ${t0} → ${t1} °C)`}`)
    if (!AS_JSON && result && typeof result === 'object' && typeof result.text === 'string') console.log(result.text)
    else if (!AS_JSON && result && typeof result === 'object' && result.status === 'error')
      console.log(`ERROR: ${String(result.error).split('\n').slice(0, 3).join('\n')}`)
    else console.log(typeof result === 'string' ? result : JSON.stringify(result, null, AS_JSON ? 0 : 2))
    if (result && typeof result === 'object' && result.status === 'error') process.exitCode = 2
  } finally {
    if (!KEEP) await cdp.send('Target.closeTarget', { targetId }).catch(() => {})
    cdp.close()
  }
}

main().catch(e => {
  console.error(`[device-run] FAILED: ${e.message}`)
  process.exit(1)
})
