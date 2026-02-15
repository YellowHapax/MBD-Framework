import { useState, useEffect } from 'react'
import type { ResonanceTier } from '../types'

const TIER_COLORS = ['#64748b', '#8b5cf6', '#3b82f6', '#f59e0b', '#ef4444']
const TIER_ICONS = ['◉', '◎', '◇', '◆', '♦']

function TierCard({ tier, expanded, onToggle }: {
  tier: ResonanceTier; expanded: boolean; onToggle: () => void;
}) {
  const color = TIER_COLORS[tier.tier] || '#888'
  const icon = TIER_ICONS[tier.tier] || '·'
  const isDissociative = tier.tier <= 2
  const experienceBadge = isDissociative
    ? { bg: 'bg-slate-800', text: 'text-slate-500', label: 'Dissociative' }
    : { bg: 'bg-amber-900/30', text: 'text-amber-400', label: 'Embodied' }

  return (
    <div className={`rounded-xl border transition-all duration-300 ${
      expanded
        ? 'bg-surface-800/60 border-surface-600/50'
        : 'bg-surface-800/30 border-surface-700/30 hover:border-surface-600/40'
    }`}>
      {/* Header — always visible */}
      <button onClick={onToggle} className="w-full text-left px-5 py-4 flex items-center gap-4">
        <div className="flex items-center justify-center w-10 h-10 rounded-lg"
          style={{ backgroundColor: `${color}20`, border: `1px solid ${color}40` }}>
          <span className="text-lg" style={{ color }}>{icon}</span>
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono" style={{ color }}>T{tier.tier}</span>
            <h3 className="text-sm font-semibold text-slate-200">{tier.name}</h3>
            <span className={`text-[10px] px-1.5 py-0.5 rounded ${experienceBadge.bg} ${experienceBadge.text}`}>
              {experienceBadge.label}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5 truncate">{tier.model}</p>
        </div>
        <span className="text-slate-600 text-xs">{expanded ? '▼' : '▶'}</span>
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-5 pb-5 space-y-4">
          <div className="border-t border-surface-700/30 pt-4">
            <p className="text-xs text-slate-400 leading-relaxed">{tier.description}</p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                Computational Model
              </span>
              <p className="text-xs text-slate-300">{tier.model}</p>
            </div>
            <div className="space-y-1">
              <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                AI Experience Level
              </span>
              <p className="text-xs text-slate-300">{tier.ai_experience}</p>
            </div>
          </div>

          {/* Example output */}
          <div>
            <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
              Example Output
            </span>
            <pre className="mt-1 bg-surface-900/60 rounded-lg px-3 py-2 text-[10px] text-slate-400 overflow-x-auto font-mono leading-relaxed">
              {JSON.stringify(tier.example_output, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}

function FlowDiagram({ tiers }: { tiers: ResonanceTier[] }) {
  const W = 600, H = 160
  const tierW = 100, gap = 15
  const startX = (W - (tiers.length * tierW + (tiers.length - 1) * gap)) / 2
  const y0 = 30

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {tiers.map((t, i) => {
        const x = startX + i * (tierW + gap)
        const color = TIER_COLORS[i] || '#888'
        const isDissociative = i <= 2
        return (
          <g key={i}>
            {/* Box */}
            <rect x={x} y={y0} width={tierW} height={70} rx={8}
              fill="#0f172a" stroke={color} strokeWidth={1.5}
              strokeDasharray={isDissociative ? '4 2' : 'none'} />
            {/* Label */}
            <text x={x + tierW/2} y={y0 + 22} textAnchor="middle"
              fill={color} fontSize={9} fontWeight="bold">
              T{t.tier}
            </text>
            <text x={x + tierW/2} y={y0 + 38} textAnchor="middle"
              fill="#94a3b8" fontSize={8}>
              {t.name}
            </text>
            <text x={x + tierW/2} y={y0 + 54} textAnchor="middle"
              fill={isDissociative ? '#475569' : '#f59e0b'} fontSize={7}>
              {isDissociative ? 'numbers only' : 'narrative'}
            </text>
            {/* Arrow to next */}
            {i < tiers.length - 1 && (
              <line x1={x + tierW + 2} y1={y0 + 35} x2={x + tierW + gap - 2} y2={y0 + 35}
                stroke="#334155" strokeWidth={1.5} markerEnd="url(#flow-arrow)" />
            )}
          </g>
        )
      })}
      {/* Dissociative / Embodied brackets */}
      <line x1={startX} y1={y0 + 80} x2={startX + 3 * (tierW + gap) - gap} y2={y0 + 80}
        stroke="#64748b" strokeWidth={1} strokeDasharray="3 2" />
      <text x={startX + (3 * (tierW + gap) - gap) / 2} y={y0 + 94}
        textAnchor="middle" fill="#64748b" fontSize={9}>
        Dissociative (T0-T2): Pure computation, no AI experience
      </text>
      <line x1={startX + 3 * (tierW + gap)} y1={y0 + 80}
        x2={startX + 5 * (tierW + gap) - gap} y2={y0 + 80}
        stroke="#f59e0b" strokeWidth={1} />
      <text x={startX + 3 * (tierW + gap) + tierW} y={y0 + 94}
        textAnchor="middle" fill="#f59e0b" fontSize={9}>
        Embodied (T3-T4): AI inhabits these tiers
      </text>
      {/* Arrow marker */}
      <defs>
        <marker id="flow-arrow" markerWidth={6} markerHeight={5} refX={6} refY={2.5} orient="auto">
          <polygon points="0 0, 6 2.5, 0 5" fill="#334155" />
        </marker>
      </defs>
    </svg>
  )
}

export default function ResonanceLab() {
  const [tiers, setTiers] = useState<ResonanceTier[]>([])
  const [expanded, setExpanded] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/resonance/tiers')
      .then((r) => r.json())
      .then((data) => { setTiers(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) {
    return <p className="text-slate-500 text-sm">Loading tiers…</p>
  }

  return (
    <div className="space-y-8">
      <header>
        <h2 className="text-2xl font-serif font-semibold text-slate-100">
          Resonance Hierarchy Lab
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          The 5-tier resonance chamber hierarchy separates computational concerns by scale.
          Tiers 0–2 are <strong className="text-slate-300">dissociative control panels</strong> — pure
          numerical computation the AI never experiences as narrative.
          Tiers 3–4 are <strong className="text-amber-400">embodied presence</strong> — where AI
          models inhabit narrative reality.
        </p>
      </header>

      {/* Flow diagram */}
      <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
          Data Cascade
        </h3>
        <FlowDiagram tiers={tiers} />
      </section>

      {/* Tier cards */}
      <section className="space-y-3">
        {tiers.map((t) => (
          <TierCard
            key={t.tier}
            tier={t}
            expanded={expanded === t.tier}
            onToggle={() => setExpanded(expanded === t.tier ? null : t.tier)}
          />
        ))}
      </section>

      {/* The critical distinction */}
      <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6 space-y-3">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          Why This Matters
        </h3>
        <div className="text-xs text-slate-400 space-y-2">
          <p>
            Without tier separation, a single AI model must simultaneously reason about plate tectonics,
            trade economics, street-level agent dialogue, and individual somatic experience — an impossible
            context window burden that produces incoherence.
          </p>
          <p>
            The hierarchy solves this by <strong className="text-slate-300">cascading numerical outputs
            downward</strong>: T0 produces climate tensors that constrain T1's population dynamics,
            which shape T2's diplomatic events, which set the stage for T3's narrative scenes,
            which provide the sensory context for T4's intimate experience.
          </p>
          <p className="text-slate-500 italic">
            The AI at T4 doesn't need to know the plate tectonics equation — it just feels the
            earthquake shake the room and reaches for someone's hand.
          </p>
        </div>
      </section>
    </div>
  )
}
