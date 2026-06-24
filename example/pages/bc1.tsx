import { BC1Encoder } from 'gputex'

import CompareView from '../components/CompareView'

const Bc1Page = () => (
  <CompareView
    current="/bc1"
    title="BC1"
    subtitle="RGB · sRGB · 8 bytes/block · 8:1 vs RGBA8"
    description="DXT1 — the oldest BC format: RGB565 endpoints + 2-bit indices over a 4-colour line. Watch for banding in the gradient quadrant and rough colour edges. Same source image as /bc7, for a direct quality comparison."
    url="/textures/color.png"
    encoder={BC1Encoder}
    colorSpace="srgb"
  />
)

export default Bc1Page
