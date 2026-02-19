/**
 * Overview.tsx — Landing page for MBD Lab
 *
 * Framework summary, paper links, lab directory.
 */

const papers = [
  {
    num: 1,
    title: 'Memory as Baseline Deviation',
    doi: '10.5281/zenodo.17381536',
    desc: 'Personality as a drifting baseline vector. Trauma as the force that moves it.',
    labs: 6,
  },
  {
    num: 2,
    title: 'In Pursuit of the Markov Tensor',
    doi: '10.5281/zenodo.17537185',
    desc: 'Group coupling convergence, echo chambers, and catastrophic fragmentation.',
    labs: 1,
  },
  {
    num: 3,
    title: 'Episodic Recall as Resonant Re-instantiation',
    doi: '10.5281/zenodo.17374270',
    desc: 'Novelty-gated encoding, context-scaffolded recall, and the retrieval threshold.',
    labs: 1,
  },
  {
    num: 4,
    title: 'The Coupling Asymmetry',
    doi: '10.5281/zenodo.18519187',
    desc: 'ASPD, BPD, parasitic coupling, internalised Others, and dissociative fragmentation.',
    labs: 5,
  },
  {
    num: 5,
    title: 'The Emergent Gate',
    doi: '10.5281/zenodo.17344091',
    desc: 'Mood-incongruent recall, dual resonance — amplification vs overshadowing.',
    labs: 2,
  },
  {
    num: 6,
    title: 'The Resonant Gate',
    doi: '10.5281/zenodo.17352481',
    desc: 'Phase-locked coupling, zeta-modulated integration, deontological pressure tests.',
    labs: 3,
  },
]

export default function Overview() {
  return (
    <div>
      {/* Header */}
      <div className="mb-10">
        <h2 className="text-3xl font-serif text-slate-100 tracking-tight">
          Memory as Baseline Deviation
        </h2>
        <p className="text-base text-slate-400 mt-2 max-w-2xl leading-relaxed">
          A computational framework for modeling personality, cognition, and social
          dynamics as state-space systems. This lab provides 18 runnable simulations
          that demonstrate the key phenomena across six published papers.
        </p>
      </div>

      {/* Core equation */}
      <div className="equation-block mb-10">
        <div className="text-lg">
          B(t+1) = B(t) · (1 − λ) + I(t) · λ
        </div>
        <p className="text-xs text-slate-500 mt-3">
          The baseline deviation equation: personality state B is pulled toward input signal I
          at learning rate λ. Every lab below explores a consequence of this single equation.
        </p>
      </div>

      {/* Variable glossary */}
      <div className="mb-10">
        <h3 className="text-lg font-serif text-slate-200 mb-4">Variables &amp; Notation</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {[
            { sym: 'B(t)',  name: 'Baseline state',       def: 'The agent\'s current personality / cognitive resting point — a vector in state space. Drifts over time under experience.' },
            { sym: 'I(t)',  name: 'Input signal',          def: 'The sensory or social experience arriving at this timestep. Pulls B toward it at rate λ.' },
            { sym: 'λ',     name: 'Plasticity (lambda)',   def: 'How deeply an experience rewrites the baseline. High λ = highly malleable; low λ → ossification, nothing lands.' },
            { sym: 'κ',     name: 'Coupling (kappa)',      def: 'Relational bond strength between two agents (0–1). Grows with familiarity and low novelty; decays when unused.' },
            { sym: 'α',     name: 'Learning rate (alpha)', def: 'How quickly κ grows in response to low-novelty interaction.' },
            { sym: 'β',     name: 'Decay rate (beta)',     def: 'How quickly unused κ fades — the relationship\'s forgetting rate.' },
            { sym: 'ζ',     name: 'Zeta',                  def: 'Deontological attention filter. Low ζ: agent attends to world input (I). High ζ: agent tunes out world and locks onto the relational Other (D).' },
            { sym: 'N',     name: 'Novelty',               def: '|I − B| — the prediction error. High novelty opens the encoding gate; low novelty closes it.' },
          ].map(({ sym, name, def }) => (
            <div key={sym} className="flex gap-3 p-3 rounded-lg bg-surface-800/40 border border-surface-700/30">
              <span className="font-serif text-primary-300 text-base w-6 flex-shrink-0">{sym}</span>
              <div>
                <span className="text-xs font-semibold text-slate-200">{name}</span>
                <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{def}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Papers → Labs */}
      <div>
        <h3 className="text-lg font-serif text-slate-200 mb-4">Papers & Simulation Labs</h3>
        <div className="space-y-3">
          {papers.map((p) => (
            <a
              key={p.doi}
              href={`https://doi.org/${p.doi}`}
              target="_blank"
              rel="noopener noreferrer"
              className="block lab-card hover:border-primary-700/50 transition-colors group"
            >
              <div className="flex items-baseline gap-3">
                <span className="text-xs text-slate-600 font-mono">P{p.num}</span>
                <h4 className="text-sm font-medium text-slate-200 group-hover:text-primary-300 transition-colors">
                  {p.title}
                </h4>
                <span className="ml-auto text-xs text-primary-400 font-mono">
                  {p.labs} lab{p.labs > 1 ? 's' : ''}
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-1 ml-8">{p.desc}</p>
              <p className="text-xs text-slate-600 font-mono mt-2 ml-8">
                doi:{p.doi}
              </p>
            </a>
          ))}
        </div>

        <div className="mt-6 text-center">
          <a
            href="/papers"
            className="inline-flex items-center gap-2 px-4 py-2 text-sm text-primary-300 border border-primary-700/40 rounded-lg hover:bg-primary-800/20 transition-colors"
          >
            📄 Run all 18 labs →
          </a>
        </div>
      </div>

      {/* Quick Start */}
      <div className="mt-10 lab-card">
        <h3 className="text-sm font-medium text-slate-200 mb-2">Quick Start</h3>
        <pre className="text-xs text-slate-400 font-mono leading-relaxed">
{`pip install -r requirements.txt

# Run any lab from the command line
python -m labs.paper1_baseline.eq_lab
python -m labs.paper4_coupling.phenomena_bpd

# Or start the interactive web UI
python lab/server.py        # API at :8050
cd lab && npm run dev       # UI at :5173`}
        </pre>
      </div>

      {/* Footer attribution */}
      <div className="mt-12 pt-6 border-t border-surface-700/50 text-center">
        <p className="text-xs text-slate-600">
          Everett, B. (2025). Memory as Baseline Deviation: A Computational Framework.
        </p>
        <p className="text-xs text-slate-700 mt-1">
          Apache 2.0 · github.com/YellowHapax/MBD-Framework
        </p>
      </div>
    </div>
  )
}
