import { BC7Encoder } from 'gputex'

import CompareView from '../components/CompareView'

const Bc7Page = () => (
  <CompareView
    current="/bc7"
    title="BC7"
    subtitle="RGBA · sRGB · 16 bytes/block · 4:1 vs RGBA8"
    description="BPTC mode 6 — 7-bit RGBA endpoints + 4-bit indices. Near-lossless on this test card; flip to /bc1 (same source image) to see how much the older format gives up. The fast path uses the O(1) projection encoder."
    url="/textures/color.png"
    encoder={BC7Encoder}
    colorSpace="srgb"
  />
)

export default Bc7Page
