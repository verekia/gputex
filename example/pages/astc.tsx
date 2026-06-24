import { ASTC4x4Encoder } from 'gputex'

import CompareView from '../components/CompareView'

const AstcPage = () => (
  <CompareView
    current="/astc"
    title="ASTC 4×4"
    subtitle="RGBA · sRGB · 16 bytes/block · 4:1 vs RGBA8"
    description="ASTC 4×4 LDR — 8-bit RGBA endpoints, single partition. Needs the texture-compression-astc feature (Apple Silicon / mobile); on a BC-only desktop GPU this page will say so. Same source image as /bc7. The fast path uses the O(1) projection encoder."
    url="/textures/color.png"
    encoder={ASTC4x4Encoder}
    colorSpace="srgb"
  />
)

export default AstcPage
