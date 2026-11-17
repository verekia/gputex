import { BC4Encoder } from 'gputex'

import CompareView from '../components/CompareView'

const Bc4Page = () => (
  <CompareView
    current="/bc4"
    title="BC4"
    subtitle="R (linear) · 8 bytes/block · 8:1 vs RGBA8"
    description="RGTC1 — one channel, two 8-bit endpoints and 16 three-bit indices per block: the `hint: 'singleChannel'` pick on BC hardware, for grayscale data maps (roughness, AO, height, masks). Same kernel as one half of /bc5. Both sides show the R channel as gray."
    url="/textures/Rock064_1K-JPG/Rock064_1K-JPG_Roughness.jpg"
    encoder={BC4Encoder}
    colorSpace="linear"
    grayscale
  />
)

export default Bc4Page
