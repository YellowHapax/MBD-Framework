/**
 * Overview.tsx — Landing page for MBD Lab
 *
 * Framework summary, paper links, quick architecture diagram.
 */

const papers = [
  {
    title: 'Memory as Baseline Deviation',
    doi: '10.5281/zenodo.14538419',
    desc: 'The foundational paper: personality as a drifting baseline vector, trauma as the force that moves it.',
  },
  {
    title: 'In Pursuit of the Markov Tensor',
    doi: '10.5281/zenodo.14611281',
    desc: 'Collective dynamics beyond the individual blanket — how populations form higher-order mathematical structures.',
  },
  {
    title: 'Episodic Recall via Active Inference',
    doi: '10.5281/zenodo.14611303',
    desc: 'Memory retrieval as free-energy minimization, not random-access lookup.',
  },
  {
    title: 'The Coupling Asymmetry',
    doi: '10.5281/zenodo.14611399',
    desc: 'Why κ grows slowly and decays fast — the thermodynamic arrow of bonding.',
  },
  {
    title: 'The Emergent Gate',
    doi: '10.5281/zenodo.14611353',
    desc: 'Phase transitions in social coupling — how populations crystallize.',
  },
  {
    title: 'The Resonant Gate',
    doi: '10.5281/zenodo.14611383',
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
    desc: 'Simulation engines — Influence Cube, social fabric, world evolution',
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
          at learning rate λ, with residual gradient ε — the river the cube cannot name.
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
              influence. Nature ↔ Capture. Heaven ↔ Displacement.
              Nurture ↔ Degeneration. Home ↔ Fixation.
            </p>
            <p className="text-xs text-slate-600 mt-2 italic">
              Explore it interactively in the Influence Cube tab →
            </p>
          </div>

          {/* The River */}
          <div className="lab-card mt-4 border-purple-800/30">
            <h4 className="text-sm font-medium text-purple-400 italic font-serif">
              The River
            </h4>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              ε — the gradient the cube cannot name. Not noise: the medium
              through which all eight vertex pressures propagate.
              Every honest model must leave room for it.
            </p>
          </div>
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
