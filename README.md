# MBD-Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18652919.svg)](https://doi.org/10.5281/zenodo.18652919)

**Memory as Baseline Deviation — Reference Implementations**

This repository contains the computational implementations accompanying a series of papers on the MBD (Memory as Baseline Deviation) framework for modeling personality, cognition, and social dynamics as state-space systems.

## Papers

| Paper | Topic | DOI |
|-------|-------|-----|
| **Memory as Baseline Deviation** | Personality as state-space dynamics; baseline drift under trauma | [10.5281/zenodo.17381536](https://doi.org/10.5281/zenodo.17381536) |
| **In Pursuit of the Markov Tensor** | Geometric framework for social cognition via tensor lattices | [10.5281/zenodo.17537185](https://doi.org/10.5281/zenodo.17537185) |
| **Episodic Recall as Resonant Re-instantiation** | Fokker–Planck account of memory retrieval dynamics | [10.5281/zenodo.17374270](https://doi.org/10.5281/zenodo.17374270) |
| **The Coupling Asymmetry** | Executive dysfunction as an eigenstate of the memory–baseline system | [10.5281/zenodo.18519187](https://doi.org/10.5281/zenodo.18519187) |
| **The Emergent Gate** | Memory encoding as threshold-dependent consolidation | [10.5281/zenodo.17344091](https://doi.org/10.5281/zenodo.17344091) |
| **The Resonant Gate** | Conversational insight as phase-locked coupling | [10.5281/zenodo.17352481](https://doi.org/10.5281/zenodo.17352481) |

## Repository Structure

```
MBD-Framework/
├── mbd/                  Core agent architecture (Markov Blanket agents)
│   ├── agent.py          Agent with internal/sensory/active states
│   ├── hypercube.py      N-dimensional social lattice (κ-coupling)
│   ├── internal_states.py  Beliefs, needs, goals — hidden by the blanket
│   ├── sensory_states.py   Perception interface to the world
│   └── active_states.py    Action selection from internal states
│
├── markov/               Markov Tensor geometry
│   ├── tensor_library.py MarkovTensor → MarkovCube → MarkovHypercube → Tensorium
│   └── engine.py         Levels of Lucidity, blanket-driven simulation scaling
│
├── dynamics/             Simulation engines
│   ├── influence_cube.py   Stella octangula pressure geometry (3 binary axes → 8 poles)
│   ├── social_fabric.py    Paper-derived agent interaction model: MBD baseline
│   │                       deviation, coupling asymmetry, Markov tensor probability,
│   │                       resonant field translation, emergent gating
│   └── world_evolution.py  Quadrafoil environmental field model: sanctuary/arena/
│                           market/sink pressure signatures with 1/r² falloff
│
├── fields/               Affective ↔ physical field translation
│   └── translation.py    TCPB deltas → buoyancy, luminosity, tactile, harmonics
│
├── analysis/             Visualization and analytics
│   ├── graphing_suite.py Trauma tensor energy, collision 4D scatter, ecology heatmap
│   └── trauma_model.py   Core MBD equations: B(t+1) = B(t)(1-λ) + I(t)λ
│
├── notebooks/            Interactive paper walkthroughs (Jupyter)
│   ├── 01_baseline_deviation.ipynb  Paper 1: baseline drift, plasticity, coupling
│   └── 04_executive_load.ipynb      Paper 4: executive load phase transitions
│
├── demos/                Runnable demonstrations
│   └── resonance_hierarchy.py  5-tier ontological stratification demo
│
├── CITATION.cff          Machine-readable citation metadata
├── LICENSE               Apache-2.0
└── requirements.txt      numpy, pydantic, matplotlib
```

## Core Concepts

### Baseline Drift (MBD)

Every agent maintains a personality baseline vector **B**. Traumatic or significant events shift this baseline:

$$B(t+1) = B(t) \cdot (1 - \lambda) + I(t) \cdot \lambda$$

where **I(t)** is the input signal and **λ** is the learning rate (plasticity).

### Coupling Dynamics (κ)

Relational coupling between agents evolves according to:

$$\frac{d\kappa}{dt} = \alpha(1 - N) - \beta\kappa$$

where **N** is novelty (surprise) and **α**, **β** are gain/decay parameters.

### The Quadrafoil of Influence

Deontological structures emerge in the simulation world as expressions of collective mind:

| Pole | Type | Effect on Agent Baselines |
|------|------|--------------------------|
| **Sanctuaries** | Trust/Order | ↑ trust, ↓ aggression |
| **Arenas** | Boldness/Challenge | Structured aggression outlet |
| **Markets** | Playfulness/Curiosity | ↑ novelty, social interaction |
| **Sinks** | Collapse/Dissipation | Amplify negative pressures |

### Markov Tensor Geometry

Social relationships are modeled as probabilistic manifolds. The **Markov Hypercube** provides N-dimensional lattice interpolation of agent beliefs weighted by trust (κ):

$$M_{obj} = f(M_A, M_B, \kappa_{AB})$$

### Levels of Lucidity

The simulation scales computational detail dynamically: areas near observers run at high lucidity (detailed agent simulation), while distant regions collapse to abstract tensor transitions. This is managed by the **Markov Blanket** construct.

### Affective Field Translation

Internal affect (Trust, Curiosity, Playfulness, Boldness) maps onto environmental physics:

| Affect Pole | Physical Field | Positive | Negative |
|-------------|---------------|----------|----------|
| Trust | Buoyancy | Lighter gravity | Crushing pressure |
| Curiosity | Luminosity | Sharp, clear light | Fog and blur |
| Playfulness | Tactile Response | Warm, yielding surfaces | Cold, rigid surfaces |
| Boldness | Resonant Harmonics | Open, resonant space | Dampened, constricted |

## Quick Start

```bash
pip install -r requirements.txt

# Run the trauma mechanics simulation
python -m analysis.trauma_model

# Run the Markov Tensor demonstration
python -m markov.tensor_library

# Run the Markov Engine demonstration
python -m markov.engine

# Run the resonance hierarchy demonstration
python demos/resonance_hierarchy.py

# Interactive paper walkthroughs (Jupyter)
jupyter notebook notebooks/
```

## Citation

If you use this code in your research, please cite the relevant paper(s). See `CITATION.cff` for machine-readable metadata, or use:

> Everett, B. (2025). *Memory as Baseline Deviation: A Formal Framework for Personality as State-Space Dynamics*. Zenodo. https://doi.org/10.5281/zenodo.17381536

## License

Apache 2.0 — see [LICENSE](LICENSE).
