import { useState, useCallback } from 'react'
import type { SocialSimulationResult, SocialSnapshot, SocialAgent } from '../types'

const GROUP_COLORS: Record<string, string> = {
  alpha: '#3b82f6',
  beta: '#ef4444',
  gamma: '#a78bfa',
  delta: '#f59e0b',
}

function AgentDot({ agent, x, y, selected, onClick }: {
  agent: SocialAgent; x: number; y: number; selected: boolean;
  onClick: () => void;
}) {
  const color = GROUP_COLORS[agent.race] || '#888'
  return (
    <g onClick={onClick} className="cursor-pointer">
      <circle cx={x} cy={y} r={selected ? 8 : 5}
        fill={color} fillOpacity={0.8}
        stroke={selected ? '#fff' : 'none'} strokeWidth={2} />
      <title>{agent.name} ({agent.race}, {agent.sex}, age {agent.age})</title>
    </g>
  )
}

function EdgeLine({ x1, y1, x2, y2, weight, type }: {
  x1: number; y1: number; x2: number; y2: number;
  weight: number; type: string;
}) {
  const colors: Record<string, string> = {
    intimacy: '#a78bfa',
    love: '#f472b6',
    conflict: '#ef4444',
    pair_bonding: '#10b981',
  }
  return (
    <line x1={x1} y1={y1} x2={x2} y2={y2}
      stroke={colors[type] || '#555'}
      strokeWidth={Math.max(0.5, weight * 4)}
      strokeOpacity={0.3 + weight * 0.5} />
  )
}

function NetworkGraph({ snapshot, edgeType, selectedAgent, onSelect }: {
  snapshot: SocialSnapshot; edgeType: string;
  selectedAgent: string | null; onSelect: (id: string) => void;
}) {
  const W = 500, H = 400
  // Layout: group agents by race in quadrants
  const raceGroups: Record<string, SocialAgent[]> = {}
  for (const a of snapshot.agents) {
    if (!raceGroups[a.race]) raceGroups[a.race] = []
    raceGroups[a.race].push(a)
  }
  const races = Object.keys(raceGroups)
  const quadrants = [
    { cx: W * 0.25, cy: H * 0.3 },
    { cx: W * 0.75, cy: H * 0.3 },
    { cx: W * 0.25, cy: H * 0.75 },
    { cx: W * 0.75, cy: H * 0.75 },
  ]

  const positions: Record<string, { x: number; y: number }> = {}
  races.forEach((race, ri) => {
    const q = quadrants[ri % 4]
    const members = raceGroups[race]
    const angleStep = (2 * Math.PI) / Math.max(members.length, 1)
    const radius = 40 + members.length * 5
    members.forEach((a, i) => {
      const angle = i * angleStep - Math.PI / 2
      positions[a.id] = {
        x: q.cx + Math.cos(angle) * radius,
        y: q.cy + Math.sin(angle) * radius,
      }
    })
  })

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full bg-surface-900/40 rounded-xl border border-surface-700/30">
      {/* Edges */}
      {snapshot.edges.map((e, i) => {
        const pa = positions[e.a]
        const pb = positions[e.b]
        if (!pa || !pb) return null
        const w = (e as Record<string, number>)[edgeType] || 0
        if (w < 0.01) return null
        return <EdgeLine key={i} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} weight={w} type={edgeType} />
      })}
      {/* Race labels */}
      {races.map((race, ri) => {
        const q = quadrants[ri % 4]
        return (
          <text key={race} x={q.cx} y={q.cy - 60} textAnchor="middle"
            fill={GROUP_COLORS[race] || '#888'} fontSize={11} fontWeight="bold" className="capitalize">
            {race}
          </text>
        )
      })}
      {/* Agents */}
      {snapshot.agents.map((a) => {
        const p = positions[a.id]
        if (!p) return null
        return (
          <AgentDot key={a.id} agent={a} x={p.x} y={p.y}
            selected={selectedAgent === a.id} onClick={() => onSelect(a.id)} />
        )
      })}
    </svg>
  )
}

function AgentDetail({ agent }: { agent: SocialAgent }) {
  const bars = [
    { label: 'Trust', value: agent.trust, color: '#3b82f6' },
    { label: 'Playful', value: agent.playful, color: '#f59e0b' },
    { label: 'Aggression', value: agent.aggression, color: '#ef4444' },
    { label: 'Repro. Drive', value: agent.reproductive_drive, color: '#f472b6' },
    { label: 'Frustration', value: agent.frustration, color: '#a855f7' },
  ]
  return (
    <div className="bg-surface-800/60 rounded-xl border border-surface-700/40 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <span className="w-3 h-3 rounded-full" style={{ backgroundColor: GROUP_COLORS[agent.race] || '#888' }} />
        <h4 className="text-sm font-semibold text-slate-200">{agent.name}</h4>
        <span className="text-xs text-slate-500">{agent.sex}, age {agent.age}</span>
      </div>
      <div className="space-y-1.5">
        {bars.map((b) => (
          <div key={b.label} className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500 w-20">{b.label}</span>
            <div className="flex-1 h-1.5 rounded-full bg-surface-900/80 overflow-hidden">
              <div className="h-full rounded-full transition-all duration-300"
                style={{ width: `${Math.round(Math.max(0, Math.min(1, b.value)) * 100)}%`, backgroundColor: b.color }} />
            </div>
            <span className="text-[10px] font-mono text-slate-500 w-8 text-right">
              {b.value.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function SocialLab() {
  const [result, setResult] = useState<SocialSimulationResult | null>(null)
  const [snapIdx, setSnapIdx] = useState(0)
  const [edgeType, setEdgeType] = useState('intimacy')
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [ticks, setTicks] = useState(48)
  const [perRace, setPerRace] = useState(6)

  const run = useCallback(async () => {
    setLoading(true)
    setSelectedAgent(null)
    try {
      const res = await fetch('/api/social/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          per_group: perRace,
          races: ['Alpha', 'Beta', 'Gamma', 'Delta'],
          ticks,
          seed: Math.floor(Math.random() * 10000),
        }),
      })
      const data: SocialSimulationResult = await res.json()
      setResult(data)
      setSnapIdx(0)
    } finally { setLoading(false) }
  }, [ticks, perRace])

  const snap = result?.snapshots[snapIdx] ?? null
  const agent = snap && selectedAgent ? snap.agents.find(a => a.id === selectedAgent) : null

  return (
    <div className="space-y-8">
      <header>
        <h2 className="text-2xl font-serif font-semibold text-slate-100">
          Social Fabric Lab
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Synthesize a multi-cohort agent population and simulate social dynamics
          derived from the MBD Framework research papers. Agents have psyche vectors
          (trust, playfulness, aggression, reproductive drive) and form relationships
          with four pressure types: intimacy, love, conflict, pair bonding.
        </p>
      </header>

      {/* Controls */}
      <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6">
        <div className="flex flex-wrap items-end gap-6">
          <div className="space-y-1">
            <label className="text-xs text-slate-500">Agents per Race</label>
            <input type="number" min={2} max={12} value={perRace}
              onChange={(e) => setPerRace(parseInt(e.target.value) || 4)}
              className="w-20 bg-surface-900 border border-surface-700/40 rounded px-2 py-1 text-xs text-slate-200" />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-slate-500">Simulation Ticks</label>
            <input type="number" min={6} max={200} value={ticks}
              onChange={(e) => setTicks(parseInt(e.target.value) || 24)}
              className="w-20 bg-surface-900 border border-surface-700/40 rounded px-2 py-1 text-xs text-slate-200" />
          </div>
          <button onClick={run} disabled={loading}
            className="px-4 py-2 text-xs font-medium rounded-lg bg-blue-600/30 text-blue-300 border border-blue-500/30 hover:bg-blue-600/50 transition-colors disabled:opacity-50">
            {loading ? 'Simulating…' : 'Run Simulation'}
          </button>
        </div>
      </section>

      {result && snap && (
        <>
          {/* Snapshot scrubber */}
          <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-4">
            <div className="flex items-center gap-4">
              <span className="text-xs text-slate-500">Tick</span>
              <input type="range" min={0} max={result.snapshots.length - 1} value={snapIdx}
                onChange={(e) => { setSnapIdx(parseInt(e.target.value)); setSelectedAgent(null) }}
                className="flex-1 h-1.5 rounded-lg appearance-none cursor-pointer" />
              <span className="font-mono text-sm text-blue-400 w-12 text-right">
                t={snap.tick}
              </span>
            </div>
            <div className="flex items-center gap-4 mt-3">
              <span className="text-xs text-slate-500">Edge Type</span>
              {['intimacy', 'love', 'conflict', 'pair_bonding'].map((t) => (
                <button key={t} onClick={() => setEdgeType(t)}
                  className={`text-xs px-2 py-1 rounded capitalize transition-colors ${
                    edgeType === t
                      ? 'bg-primary-800/30 text-primary-300 border border-primary-700/40'
                      : 'text-slate-500 hover:text-slate-300'
                  }`}>
                  {t.replace('_', ' ')}
                </button>
              ))}
            </div>
          </section>

          {/* Network + Detail */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2">
              <NetworkGraph snapshot={snap} edgeType={edgeType}
                selectedAgent={selectedAgent} onSelect={setSelectedAgent} />
            </div>
            <div className="space-y-4">
              {agent ? (
                <AgentDetail agent={agent} />
              ) : (
                <div className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-4 text-xs text-slate-500 text-center">
                  Click an agent dot to inspect
                </div>
              )}
              {/* Stats */}
              <div className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-4 space-y-2 text-xs text-slate-400">
                <p>Agents: <span className="text-slate-200 font-mono">{snap.agents.length}</span></p>
                <p>Edges: <span className="text-slate-200 font-mono">{snap.edges.length}</span></p>
                <p>Events so far: <span className="text-slate-200 font-mono">
                  {result.events.filter(e => e.tick <= snap.tick).length}
                </span></p>
              </div>
            </div>
          </div>

          {/* Event log */}
          <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6">
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
              Interaction Events (tick ≤ {snap.tick})
            </h3>
            <div className="max-h-48 overflow-y-auto space-y-1 text-xs font-mono text-slate-400">
              {result.events.filter(e => e.tick <= snap.tick).slice(-30).map((e, i) => (
                <div key={i} className="flex gap-3">
                  <span className="text-slate-600 w-12">t={e.tick}</span>
                  <span className="text-blue-400">{e.a}</span>
                  <span className="text-slate-600">↔</span>
                  <span className="text-blue-400">{e.b}</span>
                  <span className="text-slate-600">p={e.probability.toFixed(4)}</span>
                </div>
              ))}
              {result.events.filter(e => e.tick <= snap.tick).length === 0 && (
                <p className="text-slate-600 italic">No interactions yet</p>
              )}
            </div>
          </section>
        </>
      )}

      {/* Theory */}
      <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6 space-y-3">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          Social Fabric Model
        </h3>
        <div className="text-xs text-slate-400 space-y-2">
          <p>
            Each agent&apos;s <strong className="text-slate-300">psyche vector</strong> (trust, playfulness,
            aggression, reproductive drive) is a baseline deviation from a cohort reference
            (Paper&nbsp;1: MBD). <strong className="text-slate-300">Frustration</strong> emerges when
            drive exceeds available bonding opportunity (Paper&nbsp;5: Emergent Gate).
          </p>
          <p>
            Edges carry four pressure types that evolve via <strong className="text-slate-300">field
            translation</strong> &mdash; trust-aligned dyads accumulate positive pressures while
            aggression-aligned dyads amplify conflict (Paper&nbsp;6: Resonant Gate).
          </p>
          <p>
            <strong className="text-slate-300">Interaction probability</strong> contracts three terms:
            psyche alignment, edge memory, and drive urgency &mdash; a simplified Markov tensor
            contraction (Paper&nbsp;2). Coupling is asymmetric: agent&nbsp;A&apos;s influence on B
            differs from B&apos;s on A (Paper&nbsp;4: Coupling Asymmetry).
          </p>
          <p>
            Cohort profiles use dimensionless epoch fractions for timescale-agnostic dynamics.
            The <strong className="text-slate-300">epoch_scale</strong> multiplier creates inter-group
            temporal asymmetries without assuming fixed calendar durations.
          </p>
        </div>
      </section>
    </div>
  )
}
