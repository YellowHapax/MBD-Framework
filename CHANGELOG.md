# Changelog

All notable changes to the MBD-Framework will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] — 2026-02-15

Stable patch release.

### Added
- **Jupyter notebooks** (`notebooks/`): literate walkthroughs of Paper 1 (Baseline Deviation) and Paper 4 (Executive Load) with interactive visualisations
- `.zenodo.json` — rich Zenodo landing-page metadata (HTML description, keywords, all six paper DOIs as related identifiers)
- `README.md` now lists notebooks in the directory tree and quick-start section

### Changed
- Internal terminology alignment with published paper series

## [0.1.0] — 2026-02-15

First public release. DOI: [10.5281/zenodo.18652919](https://doi.org/10.5281/zenodo.18652919)

### Added
- **Core agent architecture** (`mbd/`): Markov Blanket agents with internal, sensory, and active states; hypercube κ-coupling lattice
- **Markov Tensor geometry** (`markov/`): tensor library (MarkovTensor → Cube → Hypercube → Tensorium), Levels of Lucidity engine
- **Influence Cube** (`dynamics/influence_cube.py`): stella octangula formalization — 3 binary axes (Locus, Coupling, Temporality), 8 vertices as dual tetrahedra, ε river term
- **Social fabric model** (`dynamics/social_fabric.py`): paper-derived agent interaction — MBD baseline deviation, coupling asymmetry, Markov tensor probability, resonant field translation, emergent gating
- **Quadrafoil environmental fields** (`dynamics/world_evolution.py`): sanctuary/arena/market/sink pressure signatures with 1/r² falloff
- **Affective field translation** (`fields/translation.py`): TCPB deltas → buoyancy, luminosity, tactile response, resonant harmonics
- **Analysis tools** (`analysis/`): trauma model (core MBD equations), graphing suite (tensor energy, 4D scatter, ecology heatmap)
- **Resonance hierarchy demo** (`demos/resonance_hierarchy.py`): 5-tier ontological stratification
- **MBD Lab** (`lab/`): interactive research frontend — FastAPI backend + React/Three.js/Recharts UI with 8 lab pages (Overview, Influence Cube, Baseline, Field Translation, Coupling Dynamics, Social Fabric, Agent Architecture, Resonance Tiers)
- `CITATION.cff` with all six paper DOIs
- Apache-2.0 license

### Changed
- Cohort profiles use dimensionless, timescale-agnostic parametrisation (no hard-coded lifespans)
- Population archetypes are abstract (Alpha/Beta/Gamma/Delta) with no fantasy race references

[Unreleased]: https://github.com/YellowHapax/MBD-Framework/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/YellowHapax/MBD-Framework/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/YellowHapax/MBD-Framework/releases/tag/v0.1.0
