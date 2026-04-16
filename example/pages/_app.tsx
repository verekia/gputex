import '../tailwind.css'
import Script from 'next/script'

import type { AppProps } from 'next/app'

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Script
        src="https://cdn.jsdelivr.net/npm/eruda"
        strategy="afterInteractive"
        onLoad={() => {
          ;(window as any).eruda?.init()
        }}
      />
      <Component {...pageProps} />
    </>
  )
}
