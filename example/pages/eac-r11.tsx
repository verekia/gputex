import { EACR11Encoder } from 'gputex'

import CompareView from '../components/CompareView'

const EacR11Page = () => (
  <CompareView
    current="/eac-r11"
    title="EAC R11"
    subtitle="R (linear) · 8 bytes/block · 8:1 vs RGBA8"
    description="EAC R11 — the ETC2 family's one-channel format and the `hint: 'singleChannel'` pick on BC-less devices: a base value, a multiplier and one of 16 modifier tables per block, reconstructed at 11 bits. Same source image as /bc4, for a direct comparison. Both sides show the R channel as gray."
    url="/textures/Rock064_1K-JPG/Rock064_1K-JPG_Roughness.jpg"
    encoder={EACR11Encoder}
    colorSpace="linear"
    grayscale
  />
)

export default EacR11Page
