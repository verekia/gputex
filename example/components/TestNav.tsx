import Link from 'next/link'

// Cross-links between the per-format encoder test pages (and the main demo).
// `current` is the active page's path so it can be highlighted.

const LINKS: { href: string; label: string }[] = [
  { href: '/bc1', label: 'BC1' },
  { href: '/bc7', label: 'BC7' },
  { href: '/bc5', label: 'BC5' },
  { href: '/astc', label: 'ASTC' },
  { href: '/', label: 'Demo' },
]

const TestNav = ({ current }: { current: string }) => (
  <nav className="fixed top-4 right-4 z-30 flex gap-1 rounded-xl border border-white/10 bg-black/80 p-1 text-sm shadow-xl backdrop-blur-xl">
    {LINKS.map(({ href, label }) => {
      const active = href === current
      return (
        <Link
          key={href}
          href={href}
          className={`rounded-lg px-3 py-1.5 font-mono text-xs transition-colors ${
            active ? 'bg-blue-500 text-white' : 'text-gray-300 hover:bg-white/10'
          }`}
        >
          {label}
        </Link>
      )
    })}
  </nav>
)

export default TestNav
