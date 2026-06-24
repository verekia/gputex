import { BC5Encoder } from 'gputex'

import CompareView from '../components/CompareView'

const Bc5Page = () => (
  <CompareView
    current="/bc5"
    title="BC5"
    subtitle="RG (linear) · 16 bytes/block · two BC4 halves"
    description="RGTC2 — two independent BC4 halves storing R and G (a normal map's x and y). Both sides reconstruct z = √(1 − x² − y²), exactly as a shader would, so the comparison is apples-to-apples and block artifacts show up in the bump gradients. The fast path uses the O(1) projection BC4 encoder."
    url="/textures/normal.png"
    encoder={BC5Encoder}
    colorSpace="linear"
    reconstructNormal
  />
)

export default Bc5Page
