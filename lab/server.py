"""MBD Lab — Academic Research API Server

Wraps the MBD-Framework Python modules in a clean REST API
for the interactive research frontend.

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
from typing import List, Optional, Dict
import numpy as np

# ---------------------------------------------------------------------------
# Framework imports
# ---------------------------------------------------------------------------
from dynamics.influence_cube import (
    ALL_VERTICES,
    CONSTRUCTIVE_TETRAHEDRON,
    DESTRUCTIVE_TETRAHEDRON,
    InfluenceState,
    CubeLambdas,
    baseline_step,
    verify_stella_octangula,
    dual_pair,
)
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
    description="Academic research API for the Memory as Baseline Deviation framework",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class VertexOut(BaseModel):
    name: str
    symbol: str
    coords: List[int]
    constructive: bool
    description: str
    dual: str

class CubeGeometryOut(BaseModel):
    vertices: List[VertexOut]
    constructive_regular: bool
    destructive_regular: bool
    is_stella_octangula: bool
    nature_capture_diagonal: float

class InfluenceIn(BaseModel):
    nature: float = 0.5
    nurture: float = 0.5
    heaven: float = 0.5
    home: float = 0.5
    displacement: float = 0.0
    fixation: float = 0.0
    degeneration: float = 0.0
    capture: float = 0.0

class LambdasIn(BaseModel):
    values: List[float] = Field(default_factory=lambda: [0.05]*8, min_length=8, max_length=8)
    river: float = 0.0

class BaselineStepIn(BaseModel):
    B: List[float] = Field(default_factory=lambda: [0.5, 0.3, 0.7, 0.4])
    influences: InfluenceIn = InfluenceIn()
    lambdas: LambdasIn = LambdasIn()

class BaselineStepOut(BaseModel):
    B_prev: List[float]
    B_next: List[float]
    delta: List[float]
    constructive_sum: float
    destructive_sum: float
    balance: float
    centroid: List[float]

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

class TrajectoryIn(BaseModel):
    """Multi-step cube-based simulation."""
    initial_B: List[float] = Field(default_factory=lambda: [0.5, 0.3, 0.7, 0.4])
    steps: int = 50
    influences: InfluenceIn = InfluenceIn()
    lambdas: LambdasIn = LambdasIn()
    noise_scale: float = 0.01
    seed: int = 42

class TrajectoryOut(BaseModel):
    history: List[List[float]]
    centroids: List[List[float]]
    balances: List[float]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/cube", response_model=CubeGeometryOut)
def get_cube_geometry():
    """Return the full Influence Cube geometry with verification."""
    verts = []
    for v in ALL_VERTICES:
        d = dual_pair(v)
        verts.append(VertexOut(
            name=v.name,
            symbol=v.symbol,
            coords=list(v.coords),
            constructive=v.constructive,
            description=v.description,
            dual=d.name,
        ))
    proof = verify_stella_octangula()
    return CubeGeometryOut(
        vertices=verts,
        constructive_regular=proof["constructive_regular"],
        destructive_regular=proof["destructive_regular"],
        is_stella_octangula=proof["is_stella_octangula"],
        nature_capture_diagonal=proof["nature_capture_diagonal"],
    )


@app.post("/api/cube/step", response_model=BaselineStepOut)
def cube_baseline_step(req: BaselineStepIn):
    """Execute one baseline deviation step using the Influence Cube."""
    B = np.array(req.B, dtype=np.float64)
    inf = InfluenceState(
        nature=req.influences.nature,
        nurture=req.influences.nurture,
        heaven=req.influences.heaven,
        home=req.influences.home,
        displacement=req.influences.displacement,
        fixation=req.influences.fixation,
        degeneration=req.influences.degeneration,
        capture=req.influences.capture,
    )
    lam = CubeLambdas(
        lambdas=np.array(req.lambdas.values, dtype=np.float64),
        river=req.lambdas.river,
    )
    B_next = baseline_step(B, inf, lam)
    return BaselineStepOut(
        B_prev=B.tolist(),
        B_next=B_next.tolist(),
        delta=(B_next - B).tolist(),
        constructive_sum=inf.constructive_sum(),
        destructive_sum=inf.destructive_sum(),
        balance=inf.balance(),
        centroid=inf.centroid().tolist(),
    )


@app.post("/api/cube/trajectory", response_model=TrajectoryOut)
def cube_trajectory(req: TrajectoryIn):
    """Run multiple baseline steps and return the full trajectory."""
    B = np.array(req.initial_B, dtype=np.float64)
    inf = InfluenceState(
        nature=req.influences.nature,
        nurture=req.influences.nurture,
        heaven=req.influences.heaven,
        home=req.influences.home,
        displacement=req.influences.displacement,
        fixation=req.influences.fixation,
        degeneration=req.influences.degeneration,
        capture=req.influences.capture,
    )
    lam = CubeLambdas(
        lambdas=np.array(req.lambdas.values, dtype=np.float64),
        river=req.lambdas.river,
    )
    rng = np.random.default_rng(req.seed)
    history = [B.tolist()]
    centroids = [inf.centroid().tolist()]
    balances = [inf.balance()]
    for _ in range(req.steps):
        B = baseline_step(B, inf, lam, noise_scale=req.noise_scale, rng=rng)
        history.append(B.tolist())
        centroids.append(inf.centroid().tolist())
        balances.append(inf.balance())
    return TrajectoryOut(history=history, centroids=centroids, balances=balances)


@app.post("/api/trauma/simulate", response_model=SimulationOut)
def trauma_simulate(req: SimulationIn):
    """Run the classic trauma + coupling simulation from the MBD paper."""
    # Baseline trajectory
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

    # Kappa trajectory
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("MBD Lab API — http://localhost:8050/docs")
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="info")
