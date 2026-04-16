import { useEffect } from 'react'

import '../tailwind.css'

import type { AppProps } from 'next/app'

export default function App({ Component, pageProps }: AppProps) {
  useEffect(() => {
    import('eruda').then(({ default: eruda }) => eruda.init())
  }, [])

  return <Component {...pageProps} />
}
