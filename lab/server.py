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
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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


@app.post("/api/labs/{paper}/{lab_name}/run")
def lab_run(paper: str, lab_name: str, params: Dict = None):
    """Run a lab simulation with optional parameters."""
    key = f"{paper}/{lab_name}"
    mod = _get_lab(key)
    if mod is None:
        return {"error": f"Lab '{key}' not found"}
    kwargs = params or {}
    return mod.run(**kwargs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("MBD Lab API — http://localhost:8050/docs")
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="info")
