/**
 * BaselineLab.tsx — Interactive MBD baseline deviation explorer
 *
 * Two modes:
 *  1. Cube Trajectory — run N steps of the vertex-addressed equation
 *     with adjustable influence sliders
 *  2. Trauma Sequence — classic B(t) = B(t)(1-λ) + I(t)·λ with
 *     discrete events and κ dynamics
 */

import { useState, useCallback } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { runTrajectory, runTraumaSimulation } from '../api'
import {
  BASELINE_LABELS,
  BASELINE_COLORS,
  VERTEX_NAMES,
  type Influences,
  type TrajectoryResult,
  type SimulationResult,
} from '../types'

/* ---------- helpers ---------- */

function Slider({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  color = 'text-slate-300',
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  color?: string
}) {
  return (
    <div className="flex items-center gap-3">
      <span className={`text-xs w-28 ${color}`}>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1"
      />
      <span className="text-xs text-slate-500 font-mono w-10 text-right">
        {value.toFixed(2)}
      </span>
    </div>
  )
}

/* ---------- trajectory chart ---------- */

function TrajectoryChart({ data }: { data: TrajectoryResult }) {
  const chartData = data.history.map((b, i) => ({
    step: i,
    ...Object.fromEntries(BASELINE_LABELS.map((l, j) => [l, b[j]])),
    balance: data.balances[i],
  }))

  return (
    <div className="space-y-4">
      <div className="lab-card" style={{ height: 320 }}>
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Baseline Trajectory — B(t)
        </h4>
        <ResponsiveContainer width="100%" height="85%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="step"
              stroke="#475569"
              tick={{ fontSize: 10, fill: '#64748b' }}
              label={{ value: 'Timestep', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }}
            />
            <YAxis
              stroke="#475569"
              tick={{ fontSize: 10, fill: '#64748b' }}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                background: '#0f172a',
                border: '1px solid #334155',
                borderRadius: 8,
                fontSize: 11,
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {BASELINE_LABELS.map((label, i) => (
              <Line
                key={label}
                type="monotone"
                dataKey={label}
                stroke={BASELINE_COLORS[i]}
                strokeWidth={2}
                dot={false}
                animationDuration={600}
              />
            ))}
            <ReferenceLine y={0} stroke="#475569" strokeDasharray="2 2" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

/* ---------- simulation chart ---------- */

function SimulationCharts({ data }: { data: SimulationResult }) {
  const bData = data.baseline_history.map((b, i) => ({
    event: data.trauma_labels[i] || `${i}`,
    ...Object.fromEntries(['Valence', 'Arousal', 'Dominance'].map((l, j) => [l, b[j]])),
  }))

  const kData = data.kappa_history.map((k, i) => ({
    event: data.interaction_labels[i] || `${i}`,
    κ: k,
  }))

  return (
    <div className="space-y-4">
      <div className="lab-card" style={{ height: 280 }}>
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Personality Baseline B(t) — Trauma Response
        </h4>
        <ResponsiveContainer width="100%" height="85%">
          <LineChart data={bData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="event"
              stroke="#475569"
              tick={{ fontSize: 9, fill: '#64748b' }}
              angle={-15}
              textAnchor="end"
              height={50}
            />
            <YAxis stroke="#475569" tick={{ fontSize: 10, fill: '#64748b' }} />
            <Tooltip
              contentStyle={{
                background: '#0f172a',
                border: '1px solid #334155',
                borderRadius: 8,
                fontSize: 11,
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="Valence" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="Arousal" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="Dominance" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
            <ReferenceLine y={0} stroke="#475569" strokeDasharray="2 2" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="lab-card" style={{ height: 240 }}>
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Relational Coupling κ(t)
        </h4>
        <ResponsiveContainer width="100%" height="85%">
          <LineChart data={kData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="event"
              stroke="#475569"
              tick={{ fontSize: 9, fill: '#64748b' }}
              angle={-15}
              textAnchor="end"
              height={50}
            />
            <YAxis stroke="#475569" tick={{ fontSize: 10, fill: '#64748b' }} />
            <Tooltip
              contentStyle={{
                background: '#0f172a',
                border: '1px solid #334155',
                borderRadius: 8,
                fontSize: 11,
              }}
            />
            <Line type="monotone" dataKey="κ" stroke="#a855f7" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

/* ---------- presets ---------- */

const TRAUMA_PRESET = {
  initial_baseline: [0.5, -0.2, 0.1],
  initial_kappa: 0.1,
  alpha: 0.2,
  beta: 0.05,
  traumas: [
    { input_signal: [0.8, 0.9, 0.2], lambda_rate: 0.5, label: 'Intense positive shock' },
    { input_signal: [-0.9, 0.5, -0.8], lambda_rate: 0.8, label: 'Betrayal (high λ)' },
    { input_signal: [-0.9, 0.5, -0.8], lambda_rate: 0.1, label: 'Echo of betrayal (low λ)' },
    { input_signal: [0.1, -0.7, 0.1], lambda_rate: 0.3, label: 'Numb withdrawal' },
  ],
  interactions: [
    { novelty: 0.9, duration: 1.0, label: 'Awkward encounter' },
    { novelty: 0.2, duration: 1.0, label: 'Comforting chat' },
    { novelty: 0.1, duration: 1.0, label: 'Resonant conversation' },
    { novelty: 0.95, duration: 1.0, label: 'Shocking revelation' },
  ],
}

/* ---------- main page ---------- */

type Mode = 'trajectory' | 'simulation'

export default function BaselineLab() {
  const [mode, setMode] = useState<Mode>('trajectory')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Trajectory state
  const [trajResult, setTrajResult] = useState<TrajectoryResult | null>(null)
  const [influences, setInfluences] = useState<Influences>({
    nature: 0.6, nurture: 0.8, haven: 0.7, home: 0.5,
    displacement: 0.1, fixation: 0.4, erosion: 0.0, capture: 0.0,
  })
  const [steps, setSteps] = useState(60)
  const [noise, setNoise] = useState(0.005)

  // Simulation state
  const [simResult, setSimResult] = useState<SimulationResult | null>(null)

  const runTraj = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await runTrajectory({
        initial_B: [0.5, 0.3, 0.7, 0.4],
        steps,
        influences,
        lambdas: { values: [0.04, 0.06, 0.08, 0.03, 0.02, 0.05, 0.01, 0.01], river: 0.0 },
        noise_scale: noise,
        seed: 42,
      })
      setTrajResult(res)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [influences, steps, noise])

  const runSim = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await runTraumaSimulation(TRAUMA_PRESET)
      setSimResult(res)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-serif text-slate-100">Baseline Lab</h2>
        <p className="text-sm text-slate-400 mt-1">
          Explore baseline deviation dynamics — the core MBD equation in action
        </p>
      </div>

      {/* Mode tabs */}
      <div className="flex gap-2 mb-6">
        {(['trajectory', 'simulation'] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              mode === m
                ? 'bg-primary-800/30 text-primary-300 border border-primary-700/40'
                : 'text-slate-400 hover:text-slate-200 bg-surface-800/40 border border-surface-700/30'
            }`}
          >
            {m === 'trajectory' ? 'Cube Trajectory' : 'Trauma Sequence'}
          </button>
        ))}
      </div>

      {error && (
        <div className="lab-card mb-4 text-red-400 text-sm">
          <p className="font-medium">API error</p>
          <p className="text-red-500/60 text-xs mt-1">{error}</p>
        </div>
      )}

      {/* Trajectory mode */}
      {mode === 'trajectory' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Controls */}
          <div className="space-y-5">
            <div className="lab-card">
              <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                Influence Pressures
              </h4>
              <div className="space-y-2.5">
                {VERTEX_NAMES.map((name, i) => (
                  <Slider
                    key={name}
                    label={name.charAt(0).toUpperCase() + name.slice(1)}
                    value={influences[name as keyof Influences]}
                    onChange={(v) =>
                      setInfluences((prev) => ({ ...prev, [name]: v }))
                    }
                    color={i < 4 ? 'text-blue-400' : 'text-rose-400'}
                  />
                ))}
              </div>
            </div>

            <div className="lab-card">
              <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                Parameters
              </h4>
              <div className="space-y-2.5">
                <Slider
                  label="Steps"
                  value={steps}
                  onChange={(v) => setSteps(Math.round(v))}
                  min={10}
                  max={200}
                  step={1}
                />
                <Slider
                  label="Noise (ε scale)"
                  value={noise}
                  onChange={setNoise}
                  min={0}
                  max={0.05}
                  step={0.001}
                />
              </div>
            </div>

            <button
              onClick={runTraj}
              disabled={loading}
              className="w-full py-2.5 rounded-lg bg-primary-700 hover:bg-primary-600 text-white text-sm font-medium transition-colors disabled:opacity-50"
            >
              {loading ? 'Computing…' : 'Run Trajectory'}
            </button>
          </div>

          {/* Chart */}
          <div className="lg:col-span-2">
            {trajResult ? (
              <TrajectoryChart data={trajResult} />
            ) : (
              <div className="lab-card flex items-center justify-center h-80 text-slate-500 text-sm">
                Adjust influences and click Run to visualize B(t)
              </div>
            )}

            {/* Equation reference */}
            <div className="equation-block mt-4">
              B(t+1) = B(t)(1 − Σλ<sub>v</sub>) + Σ[I<sub>v</sub>(t) · λ<sub>v</sub>] + ε
            </div>
            <p className="text-xs text-slate-600 text-center mt-2">
              Vertex-addressed baseline deviation with 8 influence channels
            </p>
          </div>
        </div>
      )}

      {/* Simulation mode */}
      {mode === 'simulation' && (
        <div>
          <div className="lab-card mb-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-medium text-slate-300">
                  Classic MBD Simulation — Trauma & Coupling
                </h4>
                <p className="text-xs text-slate-500 mt-1">
                  4 trauma events → baseline drift · 4 interactions → κ evolution
                </p>
              </div>
              <button
                onClick={runSim}
                disabled={loading}
                className="px-5 py-2 rounded-lg bg-primary-700 hover:bg-primary-600 text-white text-sm font-medium transition-colors disabled:opacity-50"
              >
                {loading ? 'Computing…' : 'Run Simulation'}
              </button>
            </div>
          </div>

          {simResult ? (
            <SimulationCharts data={simResult} />
          ) : (
            <div className="lab-card flex items-center justify-center h-64 text-slate-500 text-sm">
              Click Run to execute the preset trauma sequence
            </div>
          )}

          {/* Equations */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div className="equation-block">
              B(t+1) = B(t) · (1 − λ) + I(t) · λ
            </div>
            <div className="equation-block">
              dκ/dt = α · (1 − N) − β · κ
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
