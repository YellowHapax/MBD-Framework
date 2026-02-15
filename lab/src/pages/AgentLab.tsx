import { useState, useCallback } from 'react'
import type { AgentStepResult, AgentState } from '../types'

function BlanketDiagram({ state }: { state: AgentState | null }) {
  // Simplified Markov Blanket diagram
  const W = 460, H = 320
  if (!state) {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full bg-surface-900/40 rounded-xl border border-surface-700/30">
        <text x={W/2} y={H/2} textAnchor="middle" fill="#475569" fontSize={12}>
          Run simulation to see the Blanket
        </text>
      </svg>
    )
  }

  const safetyColor = state.needs.safety > 0.5 ? '#10b981' : '#ef4444'
  const hungerColor = state.needs.hunger > 0.5 ? '#ef4444' : '#10b981'

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full bg-surface-900/40 rounded-xl border border-surface-700/30">
      {/* World (external) */}
      <rect x={10} y={10} width={W-20} height={H-20} rx={12}
        fill="none" stroke="#334155" strokeWidth={1.5} strokeDasharray="6 3" />
      <text x={30} y={35} fill="#64748b" fontSize={10} fontWeight="bold">WORLD (W)</text>
      <text x={30} y={50} fill="#475569" fontSize={9}>
        Nearby agents: {state.nearby_agents}
      </text>

      {/* Markov Blanket boundary */}
      <rect x={60} y={70} width={W-120} height={H-100} rx={16}
        fill="#1e293b" fillOpacity={0.6} stroke="#3b82f6" strokeWidth={2} />
      <text x={W/2} y={90} textAnchor="middle" fill="#3b82f6" fontSize={10} fontWeight="bold">
        MARKOV BLANKET
      </text>

      {/* Sensory States (left) */}
      <rect x={80} y={110} width={100} height={120} rx={8}
        fill="#0f172a" stroke="#a78bfa" strokeWidth={1.5} />
      <text x={130} y={130} textAnchor="middle" fill="#a78bfa" fontSize={9} fontWeight="bold">
        SENSORY
      </text>
      <text x={130} y={148} textAnchor="middle" fill="#94a3b8" fontSize={8}>
        Perception
      </text>
      <text x={130} y={165} textAnchor="middle" fill="#64748b" fontSize={8}>
        radius = 5
      </text>
      <text x={130} y={185} textAnchor="middle" fill="#94a3b8" fontSize={8}>
        nearby: {state.nearby_agents}
      </text>
      {/* Arrow: World → Sensory */}
      <line x1={50} y1={170} x2={78} y2={170} stroke="#a78bfa" strokeWidth={1.5} markerEnd="url(#arrow)" />

      {/* Internal States (center) */}
      <rect x={195} y={100} width={120} height={140} rx={8}
        fill="#0f172a" stroke="#f59e0b" strokeWidth={1.5} />
      <text x={255} y={120} textAnchor="middle" fill="#f59e0b" fontSize={9} fontWeight="bold">
        INTERNAL
      </text>
      <text x={255} y={140} textAnchor="middle" fill="#94a3b8" fontSize={8}>Beliefs &amp; Needs</text>
      {/* Needs */}
      <text x={210} y={162} fill="#64748b" fontSize={8}>safety:</text>
      <rect x={250} y={154} width={50} height={6} rx={3} fill="#1e293b" />
      <rect x={250} y={154} width={Math.round(state.needs.safety * 50)} height={6} rx={3} fill={safetyColor} />
      <text x={210} y={180} fill="#64748b" fontSize={8}>hunger:</text>
      <rect x={250} y={172} width={50} height={6} rx={3} fill="#1e293b" />
      <rect x={250} y={172} width={Math.round(state.needs.hunger * 50)} height={6} rx={3} fill={hungerColor} />
      {/* Beliefs */}
      <text x={210} y={200} fill="#64748b" fontSize={8}>location:</text>
      <text x={265} y={200} fill="#94a3b8" fontSize={7}>
        {JSON.stringify(state.beliefs.location?.value)}
      </text>
      <text x={210} y={215} fill="#64748b" fontSize={8}>race:</text>
      <text x={265} y={215} fill="#94a3b8" fontSize={7}>
        {String(state.beliefs.race?.value)}
      </text>
      {/* Arrow: Sensory → Internal */}
      <line x1={182} y1={160} x2={193} y2={160} stroke="#a78bfa" strokeWidth={1.5} markerEnd="url(#arrow)" />

      {/* Active States (right) */}
      <rect x={330} y={110} width={90} height={120} rx={8}
        fill="#0f172a" stroke="#10b981" strokeWidth={1.5} />
      <text x={375} y={130} textAnchor="middle" fill="#10b981" fontSize={9} fontWeight="bold">
        ACTIVE
      </text>
      <text x={375} y={148} textAnchor="middle" fill="#94a3b8" fontSize={8}>
        Action Selection
      </text>
      <text x={375} y={175} textAnchor="middle" fill="#e2e8f0" fontSize={10} fontWeight="bold"
        className="capitalize">
        {state.action.type}
      </text>
      {state.action.direction && (
        <text x={375} y={192} textAnchor="middle" fill="#64748b" fontSize={8}>
          → {state.action.direction}
        </text>
      )}
      {/* Arrow: Internal → Active */}
      <line x1={317} y1={170} x2={328} y2={170} stroke="#10b981" strokeWidth={1.5} markerEnd="url(#arrow)" />
      {/* Arrow: Active → World */}
      <line x1={422} y1={170} x2={440} y2={170} stroke="#10b981" strokeWidth={1.5} markerEnd="url(#arrow)" />

      {/* Arrow marker def */}
      <defs>
        <marker id="arrow" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#94a3b8" />
        </marker>
      </defs>
    </svg>
  )
}

function StepTimeline({ history }: { history: AgentState[] }) {
  return (
    <div className="space-y-2">
      {history.map((s, i) => (
        <div key={i} className="flex items-center gap-3 text-xs">
          <span className="font-mono text-slate-600 w-10">t={i+1}</span>
          <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${
            s.action.type === 'forage'
              ? 'bg-amber-900/30 text-amber-400'
              : s.action.type === 'move'
              ? 'bg-blue-900/30 text-blue-400'
              : 'bg-slate-800 text-slate-500'
          }`}>
            {s.action.type}
            {s.action.direction ? ` → ${s.action.direction}` : ''}
          </span>
          <span className="text-slate-600">
            safety={s.needs.safety.toFixed(2)} hunger={s.needs.hunger.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function AgentLab() {
  const [hunger, setHunger] = useState(0.0)
  const [safety, setSafety] = useState(1.0)
  const [nearby, setNearby] = useState(0)
  const [steps, setSteps] = useState(10)
  const [result, setResult] = useState<AgentStepResult | null>(null)
  const [loading, setLoading] = useState(false)

  const run = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/agent/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent_id: 'Agent-α',
          location: [5.0, 5.0],
          hunger, safety,
          nearby_agent_count: nearby,
          steps,
        }),
      })
      setResult(await res.json())
    } finally { setLoading(false) }
  }, [hunger, safety, nearby, steps])

  const latestState = result && result.history.length > 0
    ? result.history[result.history.length - 1]
    : null

  return (
    <div className="space-y-8">
      <header>
        <h2 className="text-2xl font-serif font-semibold text-slate-100">
          Agent Architecture Lab
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Step through the Markov Blanket perception-action loop.
          An agent perceives the world through <strong className="text-purple-400">sensory states</strong>,
          updates <strong className="text-amber-400">internal beliefs and needs</strong>,
          then selects an <strong className="text-emerald-400">action</strong>.
        </p>
      </header>

      {/* Controls */}
      <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
          Initial Conditions
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="space-y-1">
            <label className="text-xs text-slate-500">Hunger</label>
            <input type="range" min={0} max={1} step={0.05} value={hunger}
              onChange={(e) => setHunger(parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-lg appearance-none cursor-pointer" />
            <span className="text-xs font-mono text-amber-400">{hunger.toFixed(2)}</span>
          </div>
          <div className="space-y-1">
            <label className="text-xs text-slate-500">Safety</label>
            <input type="range" min={0} max={1} step={0.05} value={safety}
              onChange={(e) => setSafety(parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-lg appearance-none cursor-pointer" />
            <span className="text-xs font-mono text-emerald-400">{safety.toFixed(2)}</span>
          </div>
          <div className="space-y-1">
            <label className="text-xs text-slate-500">Nearby Agents</label>
            <input type="number" min={0} max={10} value={nearby}
              onChange={(e) => setNearby(parseInt(e.target.value) || 0)}
              className="w-20 bg-surface-900 border border-surface-700/40 rounded px-2 py-1 text-xs text-slate-200" />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-slate-500">Steps</label>
            <input type="number" min={1} max={50} value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value) || 1)}
              className="w-20 bg-surface-900 border border-surface-700/40 rounded px-2 py-1 text-xs text-slate-200" />
          </div>
        </div>
        <button onClick={run} disabled={loading}
          className="mt-4 px-4 py-2 text-xs font-medium rounded-lg bg-amber-600/30 text-amber-300 border border-amber-500/30 hover:bg-amber-600/50 transition-colors disabled:opacity-50">
          {loading ? 'Running…' : 'Step Agent'}
        </button>
      </section>

      {/* Blanket diagram */}
      <section>
        <BlanketDiagram state={latestState} />
      </section>

      {/* Step history */}
      {result && result.history.length > 0 && (
        <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
            Step History
          </h3>
          <StepTimeline history={result.history} />
        </section>
      )}

      {/* Theory */}
      <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6 space-y-3">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          The Perception-Action Loop
        </h3>
        <div className="text-xs text-slate-400 space-y-2">
          <p>
            Each agent implements a <strong className="text-slate-300">Markov Blanket</strong> — a
            statistical boundary separating internal states from the external world. Information
            flows through three interfaces:
          </p>
          <ol className="list-decimal list-inside space-y-1 text-slate-400">
            <li><strong className="text-purple-400">Sensory States</strong> — perceive the world within a radius, detect nearby agents and structures</li>
            <li><strong className="text-amber-400">Internal States</strong> — update beliefs (with certainty), needs (hunger, safety), and goals</li>
            <li><strong className="text-emerald-400">Active States</strong> — select action based on current needs (forage if hungry, move if unsafe, idle otherwise)</li>
          </ol>
          <p>
            The <strong className="text-slate-300">Hypercube</strong> enables inter-agent belief interpolation:
            agents refine their model of the world by incorporating the models of trusted neighbors,
            weighted by the coupling coefficient κ.
          </p>
          <div className="bg-surface-900/60 rounded-lg px-4 py-2 font-mono text-amber-300 text-center text-[11px]">
            M_obj = f(M_self, M_other, κ_trust)
          </div>
        </div>
      </section>
    </div>
  )
}
