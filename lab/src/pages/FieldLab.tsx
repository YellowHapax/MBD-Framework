import { useState, useCallback } from 'react'
import type { FieldTranslationResult } from '../types'

const POLES = ['trust', 'curiosity', 'playfulness', 'boldness'] as const
const POLE_META: Record<string, { color: string; field: string; icon: string }> = {
  trust:       { color: '#3b82f6', field: 'Buoyancy',           icon: '◇' },
  curiosity:   { color: '#10b981', field: 'Luminosity',         icon: '✧' },
  playfulness: { color: '#f59e0b', field: 'Tactile Response',   icon: '∿' },
  boldness:    { color: '#ef4444', field: 'Resonant Harmonics', icon: '♦' },
}

function Slider({ pole, value, onChange }: { pole: string; value: number; onChange: (v: number) => void }) {
  const meta = POLE_META[pole]
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-slate-300 font-medium capitalize">
          {meta.icon} {pole}
        </span>
        <span className="font-mono" style={{ color: meta.color }}>
          {value > 0 ? '+' : ''}{value.toFixed(1)}
        </span>
      </div>
      <input
        type="range"
        min={-5}
        max={5}
        step={0.1}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-lg appearance-none cursor-pointer"
        style={{
          background: `linear-gradient(to right, ${meta.color}33, ${meta.color})`,
        }}
      />
      <div className="flex justify-between text-[10px] text-slate-600">
        <span>−5</span>
        <span>0</span>
        <span>+5</span>
      </div>
    </div>
  )
}

function FieldCard({ pole, field, value, magnitude, field_effect, somatic, agency }: {
  pole: string; field: string; value: number; magnitude: number;
  field_effect: string; somatic: string; agency: string;
}) {
  const meta = POLE_META[pole] || { color: '#888', icon: '·' }
  const barW = Math.round(magnitude * 100)
  const sign = value >= 0 ? 'positive' : 'negative'
  return (
    <div className="bg-surface-800/60 rounded-xl border border-surface-700/40 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold capitalize" style={{ color: meta.color }}>
          {meta.icon} {pole} → {field}
        </h4>
        <span className={`text-xs font-mono px-2 py-0.5 rounded-full ${
          sign === 'positive'
            ? 'bg-emerald-900/40 text-emerald-400'
            : 'bg-red-900/40 text-red-400'
        }`}>
          {sign}
        </span>
      </div>
      {/* Magnitude bar */}
      <div className="h-2 rounded-full bg-surface-900/80 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${barW}%`, backgroundColor: meta.color }}
        />
      </div>
      <div className="space-y-1.5 text-xs text-slate-400">
        <p><span className="text-slate-500 font-medium">Field:</span> {field_effect}</p>
        <p><span className="text-slate-500 font-medium">Somatic:</span> {somatic}</p>
        <p><span className="text-slate-500 font-medium">Agency:</span> {agency}</p>
      </div>
    </div>
  )
}

export default function FieldLab() {
  const [values, setValues] = useState<Record<string, number>>({
    trust: 0, curiosity: 0, playfulness: 0, boldness: 0,
  })
  const [result, setResult] = useState<FieldTranslationResult | null>(null)
  const [loading, setLoading] = useState(false)

  const translate = useCallback(async (v: Record<string, number>) => {
    setLoading(true)
    try {
      const res = await fetch('/api/fields/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(v),
      })
      const data: FieldTranslationResult = await res.json()
      setResult(data)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleChange = (pole: string, v: number) => {
    const next = { ...values, [pole]: v }
    setValues(next)
    translate(next)
  }

  // Auto-translate on mount
  if (!result && !loading) {
    translate(values)
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <header>
        <h2 className="text-2xl font-serif font-semibold text-slate-100">
          Field Translation Lab
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Map affective state deltas (TCPB: Trust, Curiosity, Playfulness, Boldness) onto
          resonance field descriptors. Each pole translates into a distinct field modality with
          somatic, agency, and narrative dimensions.
        </p>
      </header>

      {/* Controls */}
      <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
          Affective Deltas (TCPB)
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {POLES.map((p) => (
            <Slider key={p} pole={p} value={values[p]} onChange={(v) => handleChange(p, v)} />
          ))}
        </div>
      </section>

      {/* Field cards */}
      {result && (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {result.poles.map((p) => (
            <FieldCard key={p.pole} {...p} />
          ))}
        </section>
      )}

      {/* Narrative prompt */}
      {result && (
        <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
            Composite Narrative Prompt
          </h3>
          <p className="text-slate-200 italic leading-relaxed">
            "{result.narrative_prompt}"
          </p>
        </section>
      )}

      {/* Theory */}
      <section className="bg-surface-800/40 rounded-xl border border-surface-700/30 p-6">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
          Translation Mapping
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="text-slate-500 border-b border-surface-700/40">
                <th className="py-2 pr-4">Affective Pole</th>
                <th className="py-2 pr-4">Field Modality</th>
                <th className="py-2 pr-4">Positive Effect</th>
                <th className="py-2">Negative Effect</th>
              </tr>
            </thead>
            <tbody className="text-slate-400">
              <tr className="border-b border-surface-800/60">
                <td className="py-2 pr-4 text-blue-400">Trust</td>
                <td className="py-2 pr-4">Buoyancy (gravitational)</td>
                <td className="py-2 pr-4">Weight dissolves; floating expansion</td>
                <td className="py-2">Oppressive heaviness</td>
              </tr>
              <tr className="border-b border-surface-800/60">
                <td className="py-2 pr-4 text-emerald-400">Curiosity</td>
                <td className="py-2 pr-4">Luminosity (photonic)</td>
                <td className="py-2 pr-4">Radiant clarity; sharp perception</td>
                <td className="py-2">Dimming fog; muted senses</td>
              </tr>
              <tr className="border-b border-surface-800/60">
                <td className="py-2 pr-4 text-amber-400">Playfulness</td>
                <td className="py-2 pr-4">Tactile Response (haptic)</td>
                <td className="py-2 pr-4">Rich texture and sensation</td>
                <td className="py-2">Numb, muted surfaces</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 text-red-400">Boldness</td>
                <td className="py-2 pr-4">Resonant Harmonics (acoustic)</td>
                <td className="py-2 pr-4">Amplified, ringing space</td>
                <td className="py-2">Dampened, constricted corridor</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
