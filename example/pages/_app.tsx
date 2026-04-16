import { useEffect } from 'react'

import '../tailwind.css'

import type { AppProps } from 'next/app'

export default function App({ Component, pageProps }: AppProps) {
  useEffect(() => {
    const init = async () => {
      const { default: eruda } = await import('eruda')
      eruda.init()

      if (!('gpu' in navigator)) {
        console.log('[GPUtex] WebGPU not available')
        return
      }
      const adapter = await navigator.gpu.requestAdapter()
      if (!adapter) {
        console.log('[GPUtex] No GPU adapter found')
        return
      }

      const info = adapter.info
      console.log('[GPUtex] GPU Adapter Info:', {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      })
      console.log('[GPUtex] GPU Features:', [...adapter.features].sort().join(', '))
      console.log('[GPUtex] GPU Limits:', {
        maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      })
      if ('userAgentData' in navigator) {
        const ua = (navigator as any).userAgentData
        console.log('[GPUtex] User Agent Data:', {
          platform: ua?.platform,
          mobile: ua?.mobile,
          brands: ua?.brands,
        })
      }
    }
    init()
  }, [])

  return <Component {...pageProps} />
}
