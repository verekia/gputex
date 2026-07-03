import { ETC2Encoder } from 'gputex'

import CompareView from '../components/CompareView'

const Etc2Page = () => (
  <CompareView
    current="/etc2"
    title="ETC2"
    subtitle="RGB · sRGB · 8 bytes/block · 8:1 vs RGBA8"
    description="ETC2 RGB8 — the mobile 4-bpp format and the `quality: 'low'` pick on BC-less devices: per-subblock base colours + luma modifier tables, with the planar mode rescuing smooth gradients (compare the top-row banding against /bc1). Same source image as /bc7, for a direct quality comparison."
    url="/textures/color.png"
    encoder={ETC2Encoder}
    colorSpace="srgb"
  />
)

export default Etc2Page
