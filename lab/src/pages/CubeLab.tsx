/**
 * CubeLab.tsx — Interactive Influence Cube exploration page
 *
 * Loads vertex data from the API, renders the 3D stella octangula,
 * and shows vertex details + geometry verification in a side panel.
 */

import { useEffect, useState } from 'react'
import CubeScene from '../components/CubeScene'
import { getCubeGeometry } from '../api'
import type { CubeGeometry, Vertex } from '../types'

function VertexDetail({ vertex }: { vertex: Vertex }) {
  return (
    <div className="panel-enter">
      <div className="flex items-center gap-2 mb-3">
        <span
          className={
            vertex.constructive
              ? 'vertex-badge-constructive'
              : 'vertex-badge-destructive'
          }
        >
          {vertex.constructive ? 'Constructive' : 'Destructive'}
        </span>
        <span className="text-xs text-slate-500 font-mono">
          ({vertex.coords.join(', ')})
        </span>
      </div>
      <h3 className="text-xl font-serif text-slate-100 mb-2">{vertex.name}</h3>
      <p className="text-sm text-slate-400 leading-relaxed">
        {vertex.description}
      </p>
      <div className="mt-4 pt-3 border-t border-surface-700/50">
        <p className="text-xs text-slate-500">
          <span className="text-slate-400">Dual:</span>{' '}
          <span className="text-purple-400">{vertex.dual}</span>
          <span className="mx-2">·</span>
          <span className="text-slate-400">Symbol:</span>{' '}
          <span className="font-mono">{vertex.symbol}</span>
        </p>
      </div>
    </div>
  )
}

function AxisKey() {
  const axes = [
    {
      name: 'Locus',
      low: 'Internal (0)',
      high: 'External (1)',
      color: 'text-orange-400',
      desc: 'Where does the influence originate?',
    },
    {
      name: 'Coupling',
      low: 'Low-κ (0)',
      high: 'High-κ (1)',
      color: 'text-purple-400',
      desc: 'Does it require resonant connection?',
    },
    {
      name: 'Temporality',
      low: 'Static (0)',
      high: 'Dynamic (1)',
      color: 'text-teal-400',
      desc: 'Already set, or still in motion?',
    },
  ]

  return (
    <div className="space-y-3">
      {axes.map((a) => (
        <div key={a.name}>
          <p className={`text-xs font-medium ${a.color}`}>{a.name}</p>
          <p className="text-xs text-slate-500 mt-0.5">{a.desc}</p>
          <p className="text-xs text-slate-600 font-mono mt-0.5">
            {a.low} ↔ {a.high}
          </p>
        </div>
      ))}
    </div>
  )
}

function GeometryProof({ geo }: { geo: CubeGeometry }) {
  const check = (b: boolean) => (b ? '✓' : '✗')
  return (
    <div className="space-y-1 text-xs font-mono">
      <p className="text-slate-500">
        Constructive regular tetrahedron:{' '}
        <span className={geo.constructive_regular ? 'text-green-400' : 'text-red-400'}>
          {check(geo.constructive_regular)}
        </span>
      </p>
      <p className="text-slate-500">
        Destructive regular tetrahedron:{' '}
        <span className={geo.destructive_regular ? 'text-green-400' : 'text-red-400'}>
          {check(geo.destructive_regular)}
        </span>
      </p>
      <p className="text-slate-500">
        Stella octangula:{' '}
        <span className={geo.is_stella_octangula ? 'text-green-400' : 'text-red-400'}>
          {check(geo.is_stella_octangula)}
        </span>
      </p>
      <p className="text-slate-500">
        Nature↔Capture diagonal:{' '}
        <span className="text-slate-300">
          {geo.nature_capture_diagonal.toFixed(4)}
        </span>
        <span className="text-slate-600"> (√3 = 1.7321)</span>
      </p>
    </div>
  )
}

export default function CubeLab() {
  const [geo, setGeo] = useState<CubeGeometry | null>(null)
  const [selected, setSelected] = useState<Vertex | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getCubeGeometry()
      .then(setGeo)
      .catch((e) => setError(e.message))
  }, [])

  if (error) {
    return (
      <div className="lab-card text-red-400">
        <p className="font-medium">API unavailable</p>
        <p className="text-sm mt-1 text-slate-500">
          Start the lab server: <code className="text-slate-400">python lab/server.py</code>
        </p>
        <p className="text-xs mt-2 text-red-500/60">{error}</p>
      </div>
    )
  }

  if (!geo) {
    return (
      <div className="flex items-center justify-center h-96 text-slate-500">
        Loading geometry…
      </div>
    )
  }

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-serif text-slate-100">The Influence Cube</h2>
        <p className="text-sm text-slate-400 mt-1">
          Stella octangula of developmental pressure — click a vertex to inspect
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 3D Scene */}
        <div className="lg:col-span-2 lab-card" style={{ height: 520 }}>
          <CubeScene
            vertices={geo.vertices}
            selectedVertex={selected}
            onSelectVertex={setSelected}
          />
        </div>

        {/* Detail panel */}
        <div className="space-y-5">
          {/* Selected vertex or prompt */}
          <div className="lab-card">
            {selected ? (
              <VertexDetail vertex={selected} />
            ) : (
              <div className="text-center py-8">
                <p className="text-slate-500 text-sm">
                  Click a vertex to inspect
                </p>
                <p className="text-slate-600 text-xs mt-1">
                  Blue = constructive · Red = destructive
                </p>
              </div>
            )}
          </div>

          {/* Axes key */}
          <div className="lab-card">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Axes
            </h4>
            <AxisKey />
          </div>

          {/* Geometry proof */}
          <div className="lab-card">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Geometry Verification
            </h4>
            <GeometryProof geo={geo} />
          </div>
        </div>
      </div>

      {/* Equation */}
      <div className="equation-block mt-6">
        B(t+1) = B(t)(1 − Σλ<sub>v</sub>) + Σ[I<sub>v</sub>(t) · λ<sub>v</sub>] + ε
      </div>

      {/* Dual pairs table */}
      <div className="lab-card mt-6">
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
          Dual Pairs
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {geo.vertices
            .filter((v) => v.constructive)
            .map((v) => {
              const d = geo.vertices.find((u) => u.name === v.dual)!
              return (
                <div
                  key={v.name}
                  className="flex items-center justify-between bg-surface-800/40 rounded-lg px-4 py-2.5"
                >
                  <button
                    onClick={() => setSelected(v)}
                    className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
                  >
                    {v.name}
                  </button>
                  <span className="text-xs text-purple-500 font-mono">↔</span>
                  <button
                    onClick={() => setSelected(d)}
                    className="text-sm text-rose-400 hover:text-rose-300 transition-colors"
                  >
                    {d.name}
                  </button>
                </div>
              )
            })}
        </div>
      </div>
    </div>
  )
}
