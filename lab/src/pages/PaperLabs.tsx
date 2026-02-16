/**
 * PaperLabs.tsx — Interactive paper-focused simulation labs
 *
 * Fetches the lab registry from /api/labs, groups by paper,
 * and renders each lab with describe/run/chart capabilities.
 */

import { useState, useEffect, useCallback } from 'react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

/* ── Types ────────────────────────────────────────────────────────────────── */

interface LabMeta {
  key: string
  paper: number
  paper_title: string
  paper_doi: string
  lab_title: string
  thesis: string
}

interface LabGroup {
  paper: number
  paper_title: string
  labs: LabMeta[]
}

/* ── API helpers ──────────────────────────────────────────────────────────── */

const BASE = '/api'

async function fetchLabs(): Promise<LabMeta[]> {
  const res = await fetch(`${BASE}/labs`)
  if (!res.ok) throw new Error(`API ${res.status}`)
  return res.json()
}

async function runLab(key: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
  const [paper, lab_name] = key.split('/')
  const res = await fetch(`${BASE}/labs/${paper}/${lab_name}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) throw new Error(`API ${res.status}`)
  return res.json()
}

/* ── Color palette ────────────────────────────────────────────────────────── */

const COLORS = [
  '#8884d8', '#82ca9d', '#ff8042', '#ff4d4d', '#ffc658',
  '#00C49F', '#FFBB28', '#FF8042', '#a4de6c', '#d0ed57',
]

/* ── Generic result renderer ──────────────────────────────────────────────── */

function ResultCharts({ data }: { data: Record<string, unknown> }) {
  const charts: JSX.Element[] = []

  // Try to render timeseries (the most common shape)
  const ts = (data.timeseries ?? data.series ?? data.trials ?? data.kappa) as unknown
  if (Array.isArray(ts) && ts.length > 0 && typeof ts[0] === 'object') {
    const keys = Object.keys(ts[0] as Record<string, unknown>).filter(
      (k) => k !== 'time' && k !== 'turn' && typeof (ts[0] as Record<string, unknown>)[k] === 'number'
    )
    charts.push(
      <div key="ts" className="h-72">
        <ResponsiveContainer>
          <LineChart data={ts as Record<string, unknown>[]}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="time" stroke="#666" tick={{ fontSize: 11 }} />
            <YAxis stroke="#666" tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#1e1e2e', border: '1px solid #444', borderRadius: 8 }}
            />
            <Legend />
            {keys.map((k, i) => (
              <Line
                key={k}
                dataKey={k}
                stroke={COLORS[i % COLORS.length]}
                dot={false}
                strokeWidth={1.5}
                name={k}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    )
  }

  // If result is a dict of named series (e.g. dual_resonance trials, zeta kappa)
  if (ts && !Array.isArray(ts) && typeof ts === 'object') {
    const seriesMap = ts as Record<string, unknown[]>
    const seriesKeys = Object.keys(seriesMap)
    if (
      seriesKeys.length > 0 &&
      Array.isArray(seriesMap[seriesKeys[0]]) &&
      seriesMap[seriesKeys[0]].length > 0
    ) {
      seriesKeys.forEach((sk, si) => {
        const sd = seriesMap[sk] as Record<string, unknown>[]
        if (sd.length === 0 || typeof sd[0] !== 'object') return
        const numKeys = Object.keys(sd[0]).filter(
          (k) => k !== 'time' && k !== 'turn' && typeof sd[0][k] === 'number'
        )
        charts.push(
          <div key={`series-${sk}`} className="h-60 mt-4">
            <h4 className="text-xs text-slate-400 mb-1">{sk}</h4>
            <ResponsiveContainer>
              <LineChart data={sd}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey={numKeys.length > 0 ? undefined : 'turn'} stroke="#666" tick={{ fontSize: 10 }} />
                <YAxis stroke="#666" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ background: '#1e1e2e', border: '1px solid #444', borderRadius: 8 }} />
                <Legend />
                {numKeys.map((k, i) => (
                  <Line key={k} dataKey={k} stroke={COLORS[(si + i) % COLORS.length]} dot={false} strokeWidth={1.5} name={k} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )
      })
    }
  }

  // Render comparison/constitution bars if present
  const comp = (data.comparison ?? data.constitution ?? data.cross_state_recall) as unknown
  if (Array.isArray(comp) && comp.length > 0) {
    const first = comp[0] as Record<string, unknown>
    const labelKey = Object.keys(first).find((k) => typeof first[k] === 'string') ?? 'label'
    const numKeys = Object.keys(first).filter((k) => typeof first[k] === 'number')
    charts.push(
      <div key="comp" className="h-56 mt-4">
        <ResponsiveContainer>
          <BarChart data={comp as Record<string, unknown>[]}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey={labelKey} stroke="#666" tick={{ fontSize: 10 }} />
            <YAxis stroke="#666" tick={{ fontSize: 10 }} />
            <Tooltip contentStyle={{ background: '#1e1e2e', border: '1px solid #444', borderRadius: 8 }} />
            <Legend />
            {numKeys.map((k, i) => (
              <Bar key={k} dataKey={k} fill={COLORS[i % COLORS.length]} name={k} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    )
  }

  // Summary panel
  const summary = data.summary as Record<string, unknown> | undefined
  if (summary && typeof summary === 'object') {
    charts.push(
      <div key="summary" className="mt-4 p-4 rounded-lg bg-surface-800/60 border border-surface-700/40">
        <h4 className="text-xs font-semibold text-primary-400 mb-2 uppercase tracking-wider">Summary</h4>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
          {Object.entries(summary).map(([k, v]) => (
            <div key={k} className="text-xs text-slate-300">
              <span className="text-slate-500">{k}: </span>
              <span className="font-mono">{typeof v === 'number' ? (v as number).toFixed(4) : String(v)}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (charts.length === 0) {
    return (
      <pre className="text-xs text-slate-400 mt-4 max-h-96 overflow-auto p-3 bg-surface-800/40 rounded-lg">
        {JSON.stringify(data, null, 2)}
      </pre>
    )
  }

  return <>{charts}</>
}

/* ── Lab Card ─────────────────────────────────────────────────────────────── */

function LabCard({ lab }: { lab: LabMeta }) {
  const [results, setResults] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState(false)

  const handleRun = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const r = await runLab(lab.key)
      setResults(r)
      setOpen(true)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [lab.key])

  return (
    <div className="border border-surface-700/40 rounded-xl bg-surface-900/40 overflow-hidden">
      {/* Header */}
      <button
        className="w-full text-left px-5 py-4 hover:bg-surface-800/40 transition-colors flex items-start justify-between gap-3"
        onClick={() => setOpen(!open)}
      >
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-slate-200">{lab.lab_title}</h3>
          <p className="text-xs text-slate-500 mt-1 line-clamp-2">{lab.thesis}</p>
        </div>
        <span className="text-slate-600 text-xs flex-shrink-0 pt-0.5">
          {open ? '▼' : '▶'}
        </span>
      </button>

      {/* Body */}
      {open && (
        <div className="px-5 pb-5 border-t border-surface-700/30">
          <div className="flex items-center gap-3 mt-3">
            <button
              onClick={handleRun}
              disabled={loading}
              className={`px-4 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                loading
                  ? 'bg-surface-700 text-slate-500 cursor-wait'
                  : 'bg-primary-700/40 text-primary-300 hover:bg-primary-700/60 border border-primary-600/30'
              }`}
            >
              {loading ? 'Running…' : 'Run Simulation'}
            </button>
            <span className="text-[10px] text-slate-600 font-mono">{lab.key}</span>
          </div>

          {error && (
            <p className="text-xs text-red-400 mt-3">{error}</p>
          )}

          {results && <ResultCharts data={results} />}
        </div>
      )}
    </div>
  )
}

/* ── Main Page ────────────────────────────────────────────────────────────── */

export default function PaperLabs() {
  const [groups, setGroups] = useState<LabGroup[]>([])
  const [activePaper, setActivePaper] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchLabs()
      .then((labs) => {
        const map = new Map<number, LabGroup>()
        for (const lab of labs) {
          if (!map.has(lab.paper)) {
            map.set(lab.paper, {
              paper: lab.paper,
              paper_title: lab.paper_title,
              labs: [],
            })
          }
          map.get(lab.paper)!.labs.push(lab)
        }
        const sorted = Array.from(map.values()).sort((a, b) => a.paper - b.paper)
        setGroups(sorted)
        if (sorted.length > 0) setActivePaper(sorted[0].paper)
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return <p className="text-sm text-slate-500">Loading labs…</p>
  }
  if (error) {
    return (
      <div className="text-sm text-red-400">
        <p>Failed to load labs. Is the backend running?</p>
        <p className="text-xs text-slate-600 mt-1">{error}</p>
      </div>
    )
  }

  const active = groups.find((g) => g.paper === activePaper)

  return (
    <div>
      <h2 className="text-2xl font-serif font-semibold text-slate-100 mb-1">Paper Labs</h2>
      <p className="text-sm text-slate-500 mb-6">
        Self-contained simulations for each paper in the MBD series.
        Click a lab to expand, then <strong>Run Simulation</strong> to see results.
      </p>

      {/* Paper tabs */}
      <div className="flex gap-1.5 mb-6 flex-wrap">
        {groups.map((g) => (
          <button
            key={g.paper}
            onClick={() => setActivePaper(g.paper)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
              g.paper === activePaper
                ? 'bg-primary-700/40 text-primary-300 border border-primary-600/30'
                : 'text-slate-500 hover:text-slate-300 hover:bg-surface-800/60 border border-transparent'
            }`}
          >
            Paper {g.paper}
          </button>
        ))}
      </div>

      {/* Active paper */}
      {active && (
        <div>
          <div className="flex items-baseline gap-3 mb-4">
            <h3 className="text-lg font-serif text-slate-200">
              Paper {active.paper}: {active.paper_title}
            </h3>
            <span className="text-[10px] text-slate-600 font-mono">
              {active.labs[0]?.paper_doi}
            </span>
          </div>

          <div className="space-y-3">
            {active.labs.map((lab) => (
              <LabCard key={lab.key} lab={lab} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
