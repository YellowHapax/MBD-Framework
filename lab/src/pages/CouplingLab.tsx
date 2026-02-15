import { useState, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import type { CouplingSeriesResult, CouplingGridResult, InteractionEvent } from '../types'

const DEFAULT_INTERACTIONS: InteractionEvent[] = [
  { novelty: 0.9, duration: 1.0, label: 'First meeting — high novelty' },
  { novelty: 0.5, duration: 1.0, label: 'Second encounter — moderate' },
  { novelty: 0.2, duration: 1.0, label: 'Comfortable familiarity' },
  { novelty: 0.1, duration: 1.0, label: 'Deep resonance — very low novelty' },
  { novelty: 0.8, duration: 1.0, label: 'Surprise revelation' },
  { novelty: 0.15, duration: 1.0, label: 'Quiet recovery together' },
]

function ParamSlider({ label, value, onChange, min, max, step, color }: {
  label: string; value: number; onChange: (v: number) => void;
  min: number; max: number; step: number; color: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-slate-300 font-medium">{label}</span>
        <span className="font-mono" style={{ color }}>{value.toFixed(3)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-lg appearance-none cursor-pointer"
        style={{ background: `linear-gradient(to right, ${color}33, ${color})` }} />
    </div>
  )
}

function HeatmapCell({ value, maxVal }: { value: number; maxVal: number }) {
  const t = maxVal > 0 ? value / maxVal : 0
  const r = Math.round(30 + t * 180)
  const g = Math.round(200 - t * 140)
  const b = Math.round(255 - t * 160)
  return (
    <td className="text-center text-[10px] font-mono px-1 py-1 border border-surface-700/20"
      style={{ backgroundColor: `rgba(${r},${g},${b},0.3)`, color: `rgb(${r},${g},${b})` }}>
      {value.toFixed(3)}
    </td>
  )
}

export default function CouplingLab() {
  const [alpha, setAlpha] = useState(0.2)
  const [beta, setBeta] = useState(0.05)
  const [initKappa, setInitKappa] = useState(0.1)
  const [interactions] = useState<InteractionEvent[]>(DEFAULT_INTERACTIONS)
  const [series, setSeries] = useState<CouplingSeriesResult | null>(null)
  const [grid, setGrid] = useState<CouplingGridResult | null>(null)
  const [loading, setLoading] = useState(false)

  const runSeries = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/coupling/series', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          initial_kappa: initKappa, alpha, beta, interactions,
        }),
      })
      setSeries(await res.json())
    } finally { setLoading(false) }
  }, [alpha, beta, initKappa, interactions])

  const runGrid = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/coupling/grid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          initial_kappa: initKappa,
          interactions,
          alpha_range: [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
          beta_range:  [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3],
        }),
      })
      setGrid(await res.json())
    } finally { setLoading(false) }
  }, [initKappa, interactions])

  // Chart data
  const chartData = series ? series.kappa_history.map((k, i) => ({
    step: i,
    kappa: k,
    label: series.labels[i],
  })) : []

  const maxGridVal = grid
    ? Math.max(...grid.final_kappa.flat(), 0.001)
    : 1

  return (
    <div className="space-y-8">
      <header>
        <h2 className="text-2xl font-serif font-semibold text-slate-100">
          Coupling Dynamics Lab
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Explore the relational coupling equation:&nbsp;
          <code className="text-purple-400 text-xs">dκ/dt = α(1 − N) − βκ</code>.
          Coupling (κ) grows when novelty is low (familiarity), decays naturally at rate β.
          α controls the gain from comfort; β controls the natural decay.
        </p>
      </header>

      {/* Parameter controls */}
      <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
          Parameters
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <ParamSlider label="α (coupling gain)" value={alpha} onChange={setAlpha}
            min={0.01} max={1.0} step={0.01} color="#a78bfa" />
          <ParamSlider label="β (coupling decay)" value={beta} onChange={setBeta}
            min={0.001} max={0.5} step={0.001} color="#f472b6" />
          <ParamSlider label="κ₀ (initial coupling)" value={initKappa} onChange={setInitKappa}
            min={0.0} max={1.0} step={0.01} color="#60a5fa" />
        </div>
        <div className="mt-4 flex gap-3">
          <button onClick={runSeries} disabled={loading}
            className="px-4 py-2 text-xs font-medium rounded-lg bg-purple-600/30 text-purple-300 border border-purple-500/30 hover:bg-purple-600/50 transition-colors disabled:opacity-50">
            Run Series
          </button>
          <button onClick={runGrid} disabled={loading}
            className="px-4 py-2 text-xs font-medium rounded-lg bg-pink-600/30 text-pink-300 border border-pink-500/30 hover:bg-pink-600/50 transition-colors disabled:opacity-50">
            Run α×β Grid
          </button>
        </div>
      </section>

      {/* Interaction sequence */}
      <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
          Interaction Sequence
        </h3>
        <div className="space-y-2">
          {interactions.map((ix, i) => (
            <div key={i} className="flex items-center gap-3 text-xs text-slate-400">
              <span className="font-mono text-purple-400 w-8">#{i+1}</span>
              <span className="flex-1">{ix.label}</span>
              <span className="font-mono text-slate-500">N={ix.novelty}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Series chart */}
      {series && (
        <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
            κ Trajectory
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="label" tick={{ fontSize: 10, fill: '#94a3b8' }} angle={-20}
                textAnchor="end" height={60} />
              <YAxis domain={[0, 'auto']} tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', fontSize: 11 }}
                labelStyle={{ color: '#e2e8f0' }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="kappa" name="κ (coupling)"
                stroke="#a78bfa" strokeWidth={2} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-3 text-xs text-slate-500 text-center">
            α = {series.alpha} · β = {series.beta} ·
            Final κ = <span className="text-purple-400 font-mono">{series.kappa_history[series.kappa_history.length - 1].toFixed(4)}</span>
          </div>
        </section>
      )}

      {/* Grid heatmap */}
      {grid && (
        <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
            α × β Parameter Grid (Final κ)
          </h3>
          <div className="overflow-x-auto">
            <table className="text-xs">
              <thead>
                <tr>
                  <th className="px-2 py-1 text-slate-500">α ↓ β →</th>
                  {grid.beta_values.map((b) => (
                    <th key={b} className="px-2 py-1 text-pink-400 font-mono">{b}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {grid.alpha_values.map((a, ai) => (
                  <tr key={a}>
                    <td className="px-2 py-1 text-purple-400 font-mono font-medium">{a}</td>
                    {grid.final_kappa[ai].map((k, bi) => (
                      <HeatmapCell key={bi} value={k} maxVal={maxGridVal} />
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[10px] text-slate-600 text-center">
            Each cell shows final κ after {interactions.length} interactions, starting from κ₀ = {initKappa}
          </p>
        </section>
      )}

      {/* Theory */}
      <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6 space-y-3">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          The Coupling Equation
        </h3>
        <div className="text-sm text-slate-300 space-y-2">
          <p>
            The coupling coefficient κ represents the relational bond strength between two agents.
            It evolves according to:
          </p>
          <div className="bg-surface-900/60 rounded-lg px-4 py-3 font-mono text-purple-300 text-center">
            dκ/dt = α · (1 − N) − β · κ
          </div>
          <ul className="text-xs text-slate-400 space-y-1 list-disc list-inside">
            <li><strong className="text-purple-400">α</strong> — coupling gain. How quickly bonds form in familiar (low-novelty) conditions.</li>
            <li><strong className="text-pink-400">β</strong> — coupling decay. Natural attenuation over time, regardless of interaction.</li>
            <li><strong className="text-slate-300">N</strong> — novelty of each interaction (0 = deeply familiar, 1 = completely novel).</li>
            <li><strong className="text-blue-400">κ</strong> — current coupling strength. Self-limiting: higher κ means faster decay.</li>
          </ul>
          <p className="text-xs text-slate-500">
            When novelty is low, α(1−N) dominates and κ grows — familiarity breeds attachment.
            When novelty is high, the gain term shrinks and decay dominates — shock weakens bonds.
          </p>
        </div>
      </section>
    </div>
  )
}
