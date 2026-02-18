# MBD-Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18652919.svg)](https://doi.org/10.5281/zenodo.18652919)

**Memory as Baseline Deviation — Computational Labs**

18 standalone simulation labs that demonstrate the key phenomena across a six-paper series on personality, cognition, and social dynamics as state-space systems.

## The Core Equation

$$B(t+1) = B(t) \cdot (1 - \lambda) + I(t) \cdot \lambda$$

An agent's personality is not a label — it is a **vector** in state space. Every significant experience shifts that vector. **λ** is how deeply the event rewrites who you are. Every lab below explores a consequence of this single equation.

## Papers

| # | Paper | Labs | DOI |
|---|-------|------|-----|
| 1 | **Memory as Baseline Deviation** | 6 | [10.5281/zenodo.17381536](https://doi.org/10.5281/zenodo.17381536) |
| 2 | **In Pursuit of the Markov Tensor** | 1 | [10.5281/zenodo.17537185](https://doi.org/10.5281/zenodo.17537185) |
| 3 | **Episodic Recall as Resonant Re-instantiation** | 1 | [10.5281/zenodo.17374270](https://doi.org/10.5281/zenodo.17374270) |
| 4 | **The Coupling Asymmetry** | 5 | [10.5281/zenodo.18519187](https://doi.org/10.5281/zenodo.18519187) |
| 5 | **The Emergent Gate** | 2 | [10.5281/zenodo.17344091](https://doi.org/10.5281/zenodo.17344091) |
| 6 | **The Resonant Gate** | 3 | [10.5281/zenodo.17352481](https://doi.org/10.5281/zenodo.17352481) |

## The 18 Labs

### Paper 1: Baseline Deviation

| Lab | What it shows |
|-----|--------------|
| `eq_lab` | Two-agent convergence under the core B(t+1) equation |
| `phenomena_adhd` | Typical vs ADHD plasticity (λ = 0.1 vs 0.8 with ×0.85 decay) |
| `phenomena_ossification` | Plasticity λ(t) decaying exponentially — why change gets harder |
| `phenomena_sbs` | Chronic error → λ increase → plasticity collapse (Shaken Baby) |
| `phenomena_phantom` | Constant baseline vs null input — phantom limb as MBD prediction |
| `phenomena_bipolar` | Piecewise manic-euthymic-depressive baseline with asymmetric λ |

### Paper 2: Markov Tensor

| Lab | What it shows |
|-----|--------------|
| `echo_chamber` | Group κ convergence → external shock → catastrophic fragmentation |

### Paper 3: Episodic Recall

| Lab | What it shows |
|-----|--------------|
| `reinstantiation` | Novelty-gated encoding (sigmoid at θ_h = 0.35), context-scaffolded recall |

### Paper 4: Coupling Asymmetry

| Lab | What it shows |
|-----|--------------|
| `phenomena_aspd` | κ locked near 0 — immune to relational constitution (ASPD) |
| `phenomena_bpd` | κ oscillates wildly: idealisation ↔ devaluation (BPD) |
| `phenomena_asymmetry` | One-way coupling: Host κ = 0.9, Parasite κ = 0.1 |
| `phenomena_echo` | Internalised past Other (D_self = 0.8, κ_self = 0.7) |
| `phenomena_fragmentation` | Multiple baselines, state-dependent encoding/recall (DID) |

### Paper 5: Emergent Gate

| Lab | What it shows |
|-----|--------------|
| `mood_incongruent` | Sad baseline + happy event = huge novelty (P6 prediction) |
| `dual_resonance` | Resonance amplification vs overshadowing at multiple κ values |

### Paper 6: Resonant Gate

| Lab | What it shows |
|-----|--------------|
| `resonant_gate` | Two agents, κ grows, identical insight at low-κ vs high-κ |
| `zeta_lab` | Comparative trials at ζ = 0.1/0.5/0.9 — self-preservation floor |
| `deontological_tests` | P8 (blindness), P9 (error dissociation), P10 (immunity) |

## Repository Structure

```
MBD-Framework/
├── labs/                    18 standalone paper simulations
│   ├── paper1_baseline/     6 labs: eq_lab, phenomena_*
│   ├── paper2_markov/       1 lab:  echo_chamber
│   ├── paper3_episodic/     1 lab:  reinstantiation
│   ├── paper4_coupling/     5 labs: phenomena_*
│   ├── paper5_emergent_gate/2 labs: mood_incongruent, dual_resonance
│   └── paper6_resonant_gate/3 labs: resonant_gate, zeta_lab, deontological_tests
│
├── analysis/                Core MBD equations
│   └── trauma_model.py     B(t+1) = B(t)(1-λ) + I(t)λ, κ dynamics
│
├── dynamics/                Live MBD field engine
│   ├── influence_cube.py   Stella octangula vertex geometry (InfluenceState, CubeLambdas)
│   ├── field_agent.py      FieldAgent: attractor basins, behavioral exemplars, novelty field
│   └── social_fabric.py    Group interaction probability under coupling pressure
│
├── notebooks/               Jupyter walkthroughs
│   ├── 01_baseline_deviation.ipynb  Paper 1 walkthrough
│   └── 04_executive_load.ipynb      Paper 4 walkthrough
│
├── lab/                     Interactive web UI (optional)
│   ├── server.py            FastAPI backend (paper labs endpoints)
│   └── src/                 React + Recharts frontend
│
├── visualize_stella.py      Interactive 3D Influence Cube visualizer
├── CITATION.cff             Machine-readable citation metadata
├── LICENSE                  Apache-2.0
└── requirements.txt         numpy, pydantic, matplotlib
```

## Quick Start

```bash
pip install -r requirements.txt

# Run any lab from the command line
python -m labs.paper1_baseline.eq_lab
python -m labs.paper4_coupling.phenomena_bpd
python -m labs.paper6_resonant_gate.zeta_lab

# Or start the interactive web UI
python lab/server.py                # API at localhost:8050
cd lab && npm install && npm run dev  # UI at localhost:5173

# Jupyter notebooks
jupyter notebook notebooks/
```

## Lab Module Interface

Every lab module exposes three functions:

```python
describe() -> dict    # Metadata: title, paper, description, parameters
run(**kwargs) -> dict  # Simulation: timeseries, summary, params
plot(results) -> fig   # Matplotlib figure (optional)
```

## Citation

If you use this code in your research, please cite the relevant paper(s). See `CITATION.cff` for machine-readable metadata, or use:

> Everett, B. (2025). *Memory as Baseline Deviation: A Formal Framework for Personality as State-Space Dynamics*. Zenodo. https://doi.org/10.5281/zenodo.17381536

## License

Apache 2.0 — see [LICENSE](LICENSE).
