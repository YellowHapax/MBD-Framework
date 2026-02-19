"""MBD Lab — Paper Labs API Server

Serves the 18 standalone paper simulation modules and the core
MBD equations from analysis/trauma_model.

    Start:  python lab/server.py
    Docs:   http://localhost:8050/docs
"""

import sys
from pathlib import Path

# Ensure framework root is importable
FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict
import importlib
import numpy as np

# ---------------------------------------------------------------------------
# Framework imports — paper-derived equations only
# ---------------------------------------------------------------------------
from analysis.trauma_model import (
    Baseline,
    TraumaForm,
    update_baseline,
    update_kappa,
    Interaction,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MBD Lab",
    description="Academic research API for the Memory as Baseline Deviation framework — Paper Labs",
    version="0.1.2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8050",
        "http://127.0.0.1:8050",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas — core MBD equations
# ---------------------------------------------------------------------------

class TraumaEventIn(BaseModel):
    input_signal: List[float]
    lambda_rate: float = 0.5
    label: str = ""

class InteractionEventIn(BaseModel):
    novelty: float = 0.5
    duration: float = 1.0
    label: str = ""

class SimulationIn(BaseModel):
    initial_baseline: List[float] = Field(default_factory=lambda: [0.5, -0.2, 0.1])
    initial_kappa: float = 0.1
    alpha: float = 0.2
    beta: float = 0.05
    traumas: List[TraumaEventIn] = []
    interactions: List[InteractionEventIn] = []

class SimulationOut(BaseModel):
    baseline_history: List[List[float]]
    kappa_history: List[float]
    trauma_labels: List[str]
    interaction_labels: List[str]

class CouplingSeriesIn(BaseModel):
    initial_kappa: float = 0.1
    alpha: float = 0.2
    beta: float = 0.05
    interactions: List[InteractionEventIn] = []

class CouplingSeriesOut(BaseModel):
    kappa_history: List[float]
    labels: List[str]
    alpha: float
    beta: float

class CouplingGridIn(BaseModel):
    interactions: List[InteractionEventIn] = []
    initial_kappa: float = 0.1
    alpha_range: List[float] = Field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5])
    beta_range: List[float] = Field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3])

class CouplingGridOut(BaseModel):
    alpha_values: List[float]
    beta_values: List[float]
    final_kappa: List[List[float]]

# ---------------------------------------------------------------------------
# Routes — core MBD equations (Paper 1)
# ---------------------------------------------------------------------------

@app.post("/api/trauma/simulate", response_model=SimulationOut)
def trauma_simulate(req: SimulationIn):
    """Run the baseline deviation + coupling simulation from Papers 1 & 4."""
    current = Baseline(req.initial_baseline)
    b_hist = [current.vector.tolist()]
    t_labels = ["Initial"]
    for t in req.traumas:
        trauma = TraumaForm(
            input_signal=np.array(t.input_signal),
            lambda_learning_rate=t.lambda_rate,
            description=t.label,
        )
        current = update_baseline(current, trauma)
        b_hist.append(current.vector.tolist())
        t_labels.append(t.label or f"Event {len(t_labels)}")

    kappa = req.initial_kappa
    k_hist = [kappa]
    i_labels = ["Initial"]
    for ix in req.interactions:
        interaction = Interaction(
            novelty=ix.novelty,
            duration=ix.duration,
            description=ix.label,
        )
        kappa = update_kappa(kappa, interaction, req.alpha, req.beta)
        k_hist.append(kappa)
        i_labels.append(ix.label or f"Interaction {len(i_labels)}")

    return SimulationOut(
        baseline_history=b_hist,
        kappa_history=k_hist,
        trauma_labels=t_labels,
        interaction_labels=i_labels,
    )


@app.post("/api/coupling/series", response_model=CouplingSeriesOut)
def coupling_series(req: CouplingSeriesIn):
    """Run coupling dynamics over an interaction sequence."""
    kappa = req.initial_kappa
    k_hist = [kappa]
    labels = ["Initial"]
    for ix in req.interactions:
        interaction = Interaction(
            novelty=ix.novelty,
            duration=ix.duration,
            description=ix.label,
        )
        kappa = update_kappa(kappa, interaction, req.alpha, req.beta)
        k_hist.append(kappa)
        labels.append(ix.label or f"Event {len(labels)}")
    return CouplingSeriesOut(
        kappa_history=k_hist, labels=labels, alpha=req.alpha, beta=req.beta,
    )


@app.post("/api/coupling/grid", response_model=CouplingGridOut)
def coupling_grid(req: CouplingGridIn):
    """Compute final kappa for each (alpha, beta) pair on a fixed interaction sequence."""
    results: List[List[float]] = []
    for alpha in req.alpha_range:
        row: List[float] = []
        for beta in req.beta_range:
            kappa = req.initial_kappa
            for ix in req.interactions:
                interaction = Interaction(
                    novelty=ix.novelty, duration=ix.duration, description=ix.label,
                )
                kappa = update_kappa(kappa, interaction, alpha, beta)
            row.append(round(kappa, 4))
        results.append(row)
    return CouplingGridOut(
        alpha_values=req.alpha_range, beta_values=req.beta_range, final_kappa=results,
    )


# ---------------------------------------------------------------------------
# Paper Labs — 18 standalone simulation modules
# ---------------------------------------------------------------------------

_LAB_REGISTRY: Dict[str, str] = {
    # Paper 1: Baseline
    "paper1/eq_lab":                 "labs.paper1_baseline.eq_lab",
    "paper1/phenomena_adhd":         "labs.paper1_baseline.phenomena_adhd",
    "paper1/phenomena_ossification": "labs.paper1_baseline.phenomena_ossification",
    "paper1/phenomena_sbs":          "labs.paper1_baseline.phenomena_sbs",
    "paper1/phenomena_phantom":      "labs.paper1_baseline.phenomena_phantom",
    "paper1/phenomena_bipolar":      "labs.paper1_baseline.phenomena_bipolar",
    # Paper 2: Markov
    "paper2/echo_chamber":           "labs.paper2_markov.echo_chamber",
    # Paper 3: Episodic
    "paper3/reinstantiation":        "labs.paper3_episodic.reinstantiation",
    # Paper 4: Coupling
    "paper4/phenomena_aspd":         "labs.paper4_coupling.phenomena_aspd",
    "paper4/phenomena_bpd":          "labs.paper4_coupling.phenomena_bpd",
    "paper4/phenomena_asymmetry":    "labs.paper4_coupling.phenomena_asymmetry",
    "paper4/phenomena_echo":         "labs.paper4_coupling.phenomena_echo",
    "paper4/phenomena_fragmentation":"labs.paper4_coupling.phenomena_fragmentation",
    # Paper 5: Emergent Gate
    "paper5/mood_incongruent":       "labs.paper5_emergent_gate.mood_incongruent",
    "paper5/dual_resonance":         "labs.paper5_emergent_gate.dual_resonance",
    # Paper 6: Resonant Gate
    "paper6/resonant_gate":          "labs.paper6_resonant_gate.resonant_gate",
    "paper6/zeta_lab":               "labs.paper6_resonant_gate.zeta_lab",
    "paper6/deontological_tests":    "labs.paper6_resonant_gate.deontological_tests",
}

_lab_cache: Dict[str, object] = {}


def _get_lab(key: str):
    if key not in _LAB_REGISTRY:
        return None
    if key not in _lab_cache:
        _lab_cache[key] = importlib.import_module(_LAB_REGISTRY[key])
    return _lab_cache[key]


@app.get("/api/labs")
def list_labs():
    """Return metadata for all 18 paper labs."""
    out = []
    for key in _LAB_REGISTRY:
        mod = _get_lab(key)
        if mod and hasattr(mod, "describe"):
            info = mod.describe()
            info["key"] = key
            out.append(info)
    return out


@app.get("/api/labs/{paper}/{lab_name}/describe")
def lab_describe(paper: str, lab_name: str):
    """Return metadata for a single lab."""
    key = f"{paper}/{lab_name}"
    mod = _get_lab(key)
    if mod is None:
        return {"error": f"Lab '{key}' not found"}
    return mod.describe()


# ---------------------------------------------------------------------------
# Input hardening — cap steps, validate types, catch crashes
# ---------------------------------------------------------------------------
_MAX_STEPS = 50_000          # absolute ceiling for any 'steps' parameter
_RUN_TIMEOUT_S = 15.0        # per-run wall-clock timeout
_MAX_BODY_BYTES = 64_000     # reject absurdly large param dicts

_NUMERIC_PARAMS = {
    "lambda_a", "lambda_b", "alpha_a", "alpha_b", "alpha", "beta",
    "b0_a", "b0_b", "beta_decay", "kappa_0", "kappa", "gamma",
    "lambda_rate", "plasticity", "novelty", "duration",
    "zeta", "threshold", "decay", "lr",
}

def _sanitise_params(raw: dict | None) -> dict:
    """Validate and clamp user-supplied lab parameters."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Parameters must be a JSON object")
    out: Dict = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        # Step cap
        if k in ("steps", "n_steps", "rounds", "epochs", "n_turns", "turns"):
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise ValueError(f"'{k}' must be an integer")
            if v < 0:
                v = 0
            v = min(v, _MAX_STEPS)
        # Numeric validation
        elif k in _NUMERIC_PARAMS:
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"'{k}' must be a number")
        out[k] = v
    return out


@app.post("/api/labs/{paper}/{lab_name}/run")
def lab_run(paper: str, lab_name: str, params: Dict = None):
    """Run a lab simulation with optional parameters."""
    key = f"{paper}/{lab_name}"
    mod = _get_lab(key)
    if mod is None:
        return {"error": f"Lab '{key}' not found"}
    try:
        kwargs = _sanitise_params(params)
    except ValueError as e:
        return {"error": f"Invalid parameters: {e}"}
    try:
        result = mod.run(**kwargs)
    except (TypeError, ValueError) as e:
        return {"error": f"Lab rejected parameters: {e}"}
    except Exception as e:
        return {"error": f"Lab execution error: {type(e).__name__}: {e}"}
    return result


# ---------------------------------------------------------------------------
# Static file serving — built React SPA
# ---------------------------------------------------------------------------
_DIST = Path(__file__).parent / "dist"

if not _DIST.exists():
    import textwrap
    print(textwrap.dedent("""
        ⚠  lab/dist/ not found.

        The pre-built frontend is missing.  Build it once with:

            cd lab
            npm install
            npm run build
            cd ..

        Then re-run:  python lab/server.py
        API docs are still available at http://localhost:8050/docs
    """))
else:
    # Serve hashed asset bundles from /assets (immutable, cache-friendly)
    _assets = _DIST / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        """Serve the React SPA for any path not matched by an API route."""
        target = _DIST / full_path
        if target.exists() and target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(_DIST / "index.html"))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("MBD Lab API — http://localhost:8050")
    print("MBD Lab Docs — http://localhost:8050/docs")
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="info")
