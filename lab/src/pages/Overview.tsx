/**
 * Overview.tsx — Landing page for MBD Lab
 *
 * Framework summary, paper links, quick architecture diagram.
 */

const papers = [
  {
    title: 'Memory as Baseline Deviation',
    doi: '10.5281/zenodo.17381536',
    desc: 'The foundational paper: personality as a drifting baseline vector, trauma as the force that moves it.',
  },
  {
    title: 'In Pursuit of the Markov Tensor',
    doi: '10.5281/zenodo.17537185',
    desc: 'Collective dynamics beyond the individual blanket — how populations form higher-order mathematical structures.',
  },
  {
    title: 'Episodic Recall via Active Inference',
    doi: '10.5281/zenodo.17374270',
    desc: 'Memory retrieval as free-energy minimization, not random-access lookup.',
  },
  {
    title: 'The Coupling Asymmetry',
    doi: '10.5281/zenodo.18519187',
    desc: 'Why κ grows slowly and decays fast — the thermodynamic arrow of bonding.',
  },
  {
    title: 'The Emergent Gate',
    doi: '10.5281/zenodo.17344091',
    desc: 'Phase transitions in social coupling — how populations crystallize.',
  },
  {
    title: 'The Resonant Gate',
    doi: '10.5281/zenodo.17352481',
    desc: 'Synchronization thresholds and the Kuramoto model applied to cognitive agents.',
  },
]

const modules = [
  {
    name: 'mbd/',
    desc: 'Core agent architecture — Markov Blanket with internal, sensory, and active states',
    files: ['agent.py', 'internal_states.py', 'sensory_states.py', 'active_states.py', 'hypercube.py'],
  },
  {
    name: 'markov/',
    desc: 'Tensor geometry — MarkovTensor, Cube, Hypercube, and the Tensorium operations',
    files: ['tensor_library.py', 'engine.py'],
  },
  {
    name: 'dynamics/',
    desc: 'Simulation engines — Influence Cube pressure geometry, paper-derived social interaction model, Quadrafoil environmental fields',
    files: ['influence_cube.py', 'social_fabric.py', 'world_evolution.py'],
  },
  {
    name: 'analysis/',
    desc: 'Analytics and visualization — trauma model, graphing suite',
    files: ['trauma_model.py', 'graphing_suite.py'],
  },
  {
    name: 'fields/',
    desc: 'TCPB ↔ field translation layer',
    files: ['translation.py'],
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
          A computational framework for modeling personality, trauma, and social
          dynamics as deviations from a drifting baseline vector, coupled through
          Kuramoto oscillators and embedded in Markov Tensor geometry.
        </p>
      </div>

      {/* Core equation */}
      <div className="equation-block mb-10">
        <div className="text-lg">
          B(t+1) = B(t) · (1 − λ) + I(t) · λ + ε
        </div>
        <p className="text-xs text-slate-500 mt-3">
          The baseline deviation equation: personality state B is pulled toward input signal I
          at learning rate λ, with residual gradient ε.
        </p>
      </div>

      {/* Two-column: papers + modules */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Papers */}
        <div>
          <h3 className="text-lg font-serif text-slate-200 mb-4">Published Papers</h3>
          <div className="space-y-3">
            {papers.map((p) => (
              <a
                key={p.doi}
                href={`https://doi.org/${p.doi}`}
                target="_blank"
                rel="noopener noreferrer"
                className="block lab-card hover:border-primary-700/50 transition-colors group"
              >
                <h4 className="text-sm font-medium text-slate-200 group-hover:text-primary-300 transition-colors">
                  {p.title}
                </h4>
                <p className="text-xs text-slate-500 mt-1">{p.desc}</p>
                <p className="text-xs text-slate-600 font-mono mt-2">
                  doi:{p.doi}
                </p>
              </a>
            ))}
          </div>
        </div>

        {/* Module map */}
        <div>
          <h3 className="text-lg font-serif text-slate-200 mb-4">Framework Modules</h3>
          <div className="space-y-3">
            {modules.map((m) => (
              <div key={m.name} className="lab-card">
                <h4 className="text-sm font-medium text-primary-400 font-mono">
                  {m.name}
                </h4>
                <p className="text-xs text-slate-400 mt-1">{m.desc}</p>
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {m.files.map((f) => (
                    <span
                      key={f}
                      className="text-xs font-mono text-slate-500 bg-surface-800 px-2 py-0.5 rounded"
                    >
                      {f}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Influence Cube teaser */}
          <div className="lab-card mt-4 border-primary-800/40">
            <h4 className="text-sm font-medium text-slate-200">
              The Influence Cube
            </h4>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Developmental pressure decomposes along three binary axes —
              Locus, Coupling, Temporality — producing a stella octangula:
              two interlocking tetrahedra of constructive and destructive
              influence. Nature ↔ Capture. Haven ↔ Displacement.
              Nurture ↔ Erosion. Home ↔ Fixation.
            </p>
            <p className="text-xs text-slate-600 mt-2 italic">
              Explore it interactively in the Influence Cube tab →
            </p>
          </div>


        </div>
      </div>

      {/* Interactive Labs index */}
      <div className="mt-10">
        <h3 className="text-lg font-serif text-slate-200 mb-4">Interactive Labs</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {[
            { icon: '⬡', name: 'Influence Cube', path: '/cube', desc: '3D stella octangula with dual-pair geometry proof' },
            { icon: '⟿', name: 'Baseline Lab', path: '/baseline', desc: 'Baseline trajectory simulation & trauma presets' },
            { icon: '◇', name: 'Field Translation', path: '/fields', desc: 'TCPB → field modality mapping with narrative prompts' },
            { icon: 'κ', name: 'Coupling Dynamics', path: '/coupling', desc: 'κ equation explorer with α×β parameter grid' },
            { icon: '⊛', name: 'Social Fabric', path: '/social', desc: 'Multi-species agent network simulation' },
            { icon: '⊡', name: 'Agent Architecture', path: '/agent', desc: 'Markov Blanket perception-action loop step-through' },
            { icon: '⊘', name: 'Resonance Tiers', path: '/resonance', desc: '5-tier computational hierarchy: dissociative → embodied' },
          ].map((lab) => (
            <a key={lab.path} href={lab.path}
              className="lab-card hover:border-primary-700/50 transition-colors group flex items-start gap-3">
              <span className="text-lg text-primary-400 mt-0.5">{lab.icon}</span>
              <div>
                <h4 className="text-sm font-medium text-slate-200 group-hover:text-primary-300 transition-colors">
                  {lab.name}
                </h4>
                <p className="text-xs text-slate-500 mt-0.5">{lab.desc}</p>
              </div>
            </a>
          ))}
        </div>
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
