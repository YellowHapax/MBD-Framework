# Changelog

All notable changes to the MBD-Framework will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-10-02

### Added
- `proofs/BaselineTracking.lean`: convolution form of the baseline, partition of unity of its weights (non-negative exactly when 0 ≤ λ ≤ 1), tracking bound for inputs within ε of a level, and mean lag (1 − λ)/λ
- `proofs/README.md`: plain-English section for the tracking results

## [0.2.0] — 2026-10-02

### Added
- `proofs/`: machine-checked Lean 4 proofs of the core baseline update (`BaselineConvergence.lean`): exact error contraction, closed form, convergence for 0 < λ < 2, and the invariant interval for 0 ≤ λ ≤ 1. Includes a plain-English report (`proofs/README.md`).
- `.github/workflows/lean-proofs.yml`: builds the proofs against pinned Mathlib and fails if any theorem depends on `sorry`

### Fixed
- `.gitignore`: exclude `runtime/`, personal API-council scripts, and `.openrouter.key` from all future commits
- `launch_stella_field.bat`, `view_stella_octangula.bat`: replace hardcoded local venv path with `python` (PATH) default; local `venv/` takes precedence if present
- `launch_stella_field.bat`: replace `exec(open(...))` hack with a clean `python visualize_stella.py` invocation
- `visualize_stella.py`: correct author attribution
- `README.md`: document `dynamics/` module in repository structure

## [0.1.0] — 2026-02-16

First public release — paper-focused simulation labs.

### Added
- **18 standalone simulation labs** (`labs/`): each module exposes `describe()`, `run()`, and `plot()` for reproducible paper demonstrations
  - Paper 1 (Baseline Deviation): `eq_lab`, `phenomena_adhd`, `phenomena_ossification`, `phenomena_sbs`, `phenomena_phantom`, `phenomena_bipolar`
  - Paper 2 (Markov Tensor): `echo_chamber`
  - Paper 3 (Episodic Recall): `reinstantiation`
  - Paper 4 (Coupling Asymmetry): `phenomena_aspd`, `phenomena_bpd`, `phenomena_asymmetry`, `phenomena_echo`, `phenomena_fragmentation`
  - Paper 5 (Emergent Gate): `mood_incongruent`, `dual_resonance`
  - Paper 6 (Resonant Gate): `resonant_gate`, `zeta_lab`, `deontological_tests`
- **Core MBD equations** (`analysis/trauma_model.py`): `Baseline`, `TraumaForm`, `update_baseline`, `update_kappa`
- **Interactive web UI** (`lab/`): FastAPI backend + React/Recharts frontend with Paper Labs page
- **Jupyter notebooks** (`notebooks/`): Paper 1 (Baseline Deviation) and Paper 4 (Executive Load) walkthroughs
- `.zenodo.json` — Zenodo landing-page metadata with all six paper DOI cross-references
- `CITATION.cff` with all six paper DOIs
- Apache-2.0 license

[Unreleased]: https://github.com/YellowHapax/MBD-Framework/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/YellowHapax/MBD-Framework/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/YellowHapax/MBD-Framework/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/YellowHapax/MBD-Framework/releases/tag/v0.1.0
