import { NavLink } from 'react-router-dom'
import type { ReactNode } from 'react'

const links = [
  { to: '/', label: 'Overview', icon: '◈' },
  { to: '/papers', label: 'Paper Labs', icon: '📄' },
] as const

function NavItem({ to, label, icon }: { to: string; label: string; icon: string }) {
  return (
    <NavLink
      to={to}
      end={to === '/'}
      className={({ isActive }) =>
        `flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
          isActive
            ? 'bg-primary-800/30 text-primary-300 border border-primary-700/40'
            : 'text-slate-400 hover:text-slate-200 hover:bg-surface-800/60'
        }`
      }
    >
      <span className="text-base">{icon}</span>
      {label}
    </NavLink>
  )
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 border-r border-surface-700/50 bg-surface-900/50 flex flex-col">
        {/* Title */}
        <div className="px-5 py-6 border-b border-surface-700/50">
          <h1 className="text-lg font-serif font-semibold text-slate-100 tracking-tight">
            MBD Lab
          </h1>
          <p className="text-xs text-slate-500 mt-1 font-sans">
            Memory as Baseline Deviation
          </p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {links.map((l) => (
            <NavItem key={l.to} {...l} />
          ))}
        </nav>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-surface-700/50 text-xs text-slate-600">
          <p>Everett, B. (2025)</p>
          <p className="mt-0.5">Apache 2.0 · v0.1.0</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto px-8 py-8">
          {children}
        </div>
      </main>
    </div>
  )
}
