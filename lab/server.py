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
from fields.translation import (
    AFFECTIVE_FIELD_BLUEPRINT,
    translate_affective_to_field,
    render_affective_prompt,
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
    haven: float = 0.5
    home: float = 0.5
    displacement: float = 0.0
    fixation: float = 0.0
    erosion: float = 0.0
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

# ---- Field Translation schemas ----

class TCPBIn(BaseModel):
    """Trust, Curiosity, Playfulness, Boldness deltas (−5 to +5)."""
    trust: float = 0.0
    curiosity: float = 0.0
    playfulness: float = 0.0
    boldness: float = 0.0

class FieldPoleOut(BaseModel):
    pole: str
    field: str
    value: float
    magnitude: float
    field_effect: str
    somatic: str
    agency: str
    prompt: str

class FieldTranslationOut(BaseModel):
    poles: List[FieldPoleOut]
    narrative_prompt: str
    gravity: Optional[float] = None

# ---- Coupling Explorer schemas ----

class CouplingSeriesIn(BaseModel):
    """Run a κ sweep across an interaction sequence."""
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
    """Run κ dynamics over an alpha/beta grid for a fixed interaction sequence."""
    interactions: List[InteractionEventIn] = []
    initial_kappa: float = 0.1
    alpha_range: List[float] = Field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5])
    beta_range: List[float] = Field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3])

class CouplingGridOut(BaseModel):
    alpha_values: List[float]
    beta_values: List[float]
    final_kappa: List[List[float]]  # [alpha_idx][beta_idx]

# ---- Social Fabric schemas ----

class SocialAgentOut(BaseModel):
    id: str
    name: str
    race: str
    sex: str
    age: int
    trust: float
    playful: float
    aggression: float
    reproductive_drive: float
    frustration: float

class SocialEdgeOut(BaseModel):
    a: str
    b: str
    intimacy: float
    love: float
    conflict: float
    pair_bonding: float

class SocialFabricOut(BaseModel):
    agents: List[SocialAgentOut]
    edges: List[SocialEdgeOut]
    tick: int

class SocialStepIn(BaseModel):
    """Run N ticks of social fabric simulation."""
    per_group: int = 6
    races: List[str] = Field(default_factory=lambda: ["Alpha", "Beta", "Gamma", "Delta"])
    ticks: int = 24
    seed: int = 42

class SocialStepOut(BaseModel):
    snapshots: List[SocialFabricOut]
    events: List[Dict]

# ---- Agent Architecture schemas ----

class AgentStateOut(BaseModel):
    agent_id: str
    beliefs: Dict
    needs: Dict
    action: Dict
    nearby_agents: int

class AgentStepIn(BaseModel):
    """Configure and step an agent."""
    agent_id: str = "A"
    location: List[float] = Field(default_factory=lambda: [5.0, 5.0])
    hunger: float = 0.0
    safety: float = 1.0
    nearby_agent_count: int = 0
    steps: int = 1

class AgentStepOut(BaseModel):
    history: List[AgentStateOut]

# ---- Resonance Hierarchy schemas ----

class TierOut(BaseModel):
    tier: int
    name: str
    model: str
    ai_experience: str
    description: str
    example_output: Dict

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
        haven=req.influences.haven,
        home=req.influences.home,
        displacement=req.influences.displacement,
        fixation=req.influences.fixation,
        erosion=req.influences.erosion,
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
        haven=req.influences.haven,
        home=req.influences.home,
        displacement=req.influences.displacement,
        fixation=req.influences.fixation,
        erosion=req.influences.erosion,
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
# Field Translation routes
# ---------------------------------------------------------------------------

@app.post("/api/fields/translate", response_model=FieldTranslationOut)
def field_translate(req: TCPBIn):
    """Translate TCPB affective deltas into field descriptors + narrative prompt."""
    tcpb_delta = {
        "trust": req.trust,
        "curiosity": req.curiosity,
        "playfulness": req.playfulness,
        "boldness": req.boldness,
    }
    field_state = translate_affective_to_field(tcpb_delta)
    prompt = render_affective_prompt(field_state)
    poles = []
    for pole_name, details in field_state.items():
        bp = AFFECTIVE_FIELD_BLUEPRINT.get(pole_name, {})
        poles.append(FieldPoleOut(
            pole=pole_name,
            field=bp.get("field", "unknown"),
            value=details["value"],
            magnitude=details["magnitude"],
            field_effect=details["field_effect"],
            somatic=details["somatic"],
            agency=details["agency"],
            prompt=details["prompt"],
        ))
    return FieldTranslationOut(poles=poles, narrative_prompt=prompt)


@app.get("/api/fields/blueprint")
def field_blueprint():
    """Return the full AFFECTIVE_FIELD_BLUEPRINT for reference."""
    return AFFECTIVE_FIELD_BLUEPRINT


# ---------------------------------------------------------------------------
# Coupling Explorer routes
# ---------------------------------------------------------------------------

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
        kappa_history=k_hist,
        labels=labels,
        alpha=req.alpha,
        beta=req.beta,
    )


@app.post("/api/coupling/grid", response_model=CouplingGridOut)
def coupling_grid(req: CouplingGridIn):
    """Compute final κ for each (α, β) pair on a fixed interaction sequence."""
    results: List[List[float]] = []
    for alpha in req.alpha_range:
        row: List[float] = []
        for beta in req.beta_range:
            kappa = req.initial_kappa
            for ix in req.interactions:
                interaction = Interaction(
                    novelty=ix.novelty,
                    duration=ix.duration,
                    description=ix.label,
                )
                kappa = update_kappa(kappa, interaction, alpha, beta)
            row.append(round(kappa, 4))
        results.append(row)
    return CouplingGridOut(
        alpha_values=req.alpha_range,
        beta_values=req.beta_range,
        final_kappa=results,
    )


# ---------------------------------------------------------------------------
# Social Fabric routes
# ---------------------------------------------------------------------------

@app.post("/api/social/simulate", response_model=SocialStepOut)
def social_simulate(req: SocialStepIn):
    """Synthesize a small agent population and run N ticks of social simulation."""
    import random as _rng
    _rng.seed(req.seed)

    # Import synthesis helper
    from dynamics.social_fabric import (
        _synthesize_agents_and_edges,
        _update_reproductive_drive,
        _update_agent_needs,
        _build_relationship_matrix,
        calculate_interaction_prob,
    )

    # Build world stub for synthesis
    world_data = {"capitals": {r: {} for r in req.races}}
    agents, edges = _synthesize_agents_and_edges(world_data, per_race=req.per_group)

    def _snapshot(tick_n: int) -> SocialFabricOut:
        return SocialFabricOut(
            agents=[
                SocialAgentOut(
                    id=a["id"], name=a["name"], race=a.get("race", "?"),
                    sex=a.get("sex", "?"),
                    age=a.get("body_morph", {}).get("age", 0),
                    trust=a.get("psyche", {}).get("trust", 0),
                    playful=a.get("psyche", {}).get("playful", 0),
                    aggression=a.get("psyche", {}).get("aggression", 0),
                    reproductive_drive=a.get("psyche", {}).get("reproductive_drive", 0),
                    frustration=a.get("pressures", {}).get("frustration", 0),
                )
                for a in agents
            ],
            edges=[
                SocialEdgeOut(
                    a=e["a"], b=e["b"],
                    intimacy=e.get("intimacy", 0),
                    love=e.get("love", 0),
                    conflict=e.get("conflict", 0),
                    pair_bonding=e.get("pair_bonding", 0),
                )
                for e in edges
            ],
            tick=tick_n,
        )

    snapshots = [_snapshot(0)]
    events: List[Dict] = []

    # Run ticks
    for t in range(1, req.ticks + 1):
        for a in agents:
            _update_reproductive_drive(a, agents)
            _update_agent_needs(a)
        # Sample a few interaction events
        matrix = _build_relationship_matrix(edges)
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                pressures = matrix.get(agents[i]["id"], {}).get(agents[j]["id"])
                if pressures is None:
                    continue
                prob = calculate_interaction_prob(agents[i], agents[j], pressures)
                if _rng.random() < prob:
                    events.append({
                        "tick": t,
                        "a": agents[i]["id"],
                        "b": agents[j]["id"],
                        "probability": round(prob, 4),
                    })
        # Snapshot every few ticks or the last one
        if t % max(1, req.ticks // 5) == 0 or t == req.ticks:
            snapshots.append(_snapshot(t))

    return SocialStepOut(snapshots=snapshots, events=events)


# ---------------------------------------------------------------------------
# Agent Architecture routes
# ---------------------------------------------------------------------------

@app.post("/api/agent/step", response_model=AgentStepOut)
def agent_step(req: AgentStepIn):
    """Step an agent through its perception-action loop."""
    from mbd.internal_states import InternalStates
    from mbd.active_states import ActiveStates

    internal = InternalStates(
        req.agent_id,
        location=req.location,
        hunger=req.hunger,
        safety=req.safety,
    )

    active = ActiveStates(req.agent_id, {})

    history = []
    for _ in range(req.steps):
        # Simulate percepts
        percepts = {
            "nearby_agents": [None] * req.nearby_agent_count,
        }
        internal.update(percepts)
        action = active.choose_action(internal)
        history.append(AgentStateOut(
            agent_id=req.agent_id,
            beliefs={k: v for k, v in internal.beliefs.items()},
            needs={k: round(v, 4) for k, v in internal.needs.items()},
            action=action,
            nearby_agents=req.nearby_agent_count,
        ))
    return AgentStepOut(history=history)


# ---------------------------------------------------------------------------
# Resonance Tiers route
# ---------------------------------------------------------------------------

TIER_DATA = [
    {
        "tier": 0,
        "name": "World Mind",
        "model": "Lightweight numerical (tensor math)",
        "ai_experience": "None — dissociative control panel",
        "description": "Global coherence: climate, geology, ideology drift. Pure numerical tensors. The AI model never sees this as narrative.",
        "example_output": {
            "ideology_delta": {"order": 0.02, "chaos": 0.05, "connection": 0.10},
            "climate": {"temperature_delta": 0.3, "precipitation_delta": -0.05},
        },
    },
    {
        "tier": 1,
        "name": "Population Mind",
        "model": "Lotka-Volterra population dynamics",
        "ai_experience": "None \u2014 numerical output only",
        "description": "Group-level cultural resonance. Population dynamics, inter-group tensions, baseline drift across distinct populations.",
        "example_output": {
            "Alpha": {"trust": -0.1, "boldness": 0.2},
            "Beta": {"trust": 0.05, "connection": 0.1},
            "Gamma": {"curiosity": 0.15, "playfulness": 0.1},
        },
    },
    {
        "tier": 2,
        "name": "Civilization Mind",
        "model": "Hybrid numerical / lightweight narrative",
        "ai_experience": "Minimal — event summaries possible",
        "description": "Faction-level coherence. Trade, diplomacy, war declarations, resource allocation between settlements.",
        "example_output": {
            "trade_routes": [{"from": "Central Sanctuary", "to": "Beta Market", "volume": 340}],
            "diplomatic_events": [{"type": "alliance", "factions": ["Gamma Council", "Alpha Merchants"]}],
        },
    },
    {
        "tier": 3,
        "name": "Settlement Mind",
        "model": "Full narrative AI (Claude-class)",
        "ai_experience": "Embodied — AI inhabits this scale",
        "description": "Local coherence. Street-level events, agent conversations, market prices, weather the characters feel. This is where narrative reality begins.",
        "example_output": {
            "market_scene": "A merchant argues price with a traveler while rain drips from the awning.",
            "mood": "tense, anticipatory",
            "agents_active": 12,
        },
    },
    {
        "tier": 4,
        "name": "Personal Mind",
        "model": "Full narrative AI (deepest context)",
        "ai_experience": "Fully embodied — first-person experience",
        "description": "Individual coherence. Internal monologue, somatic sensations, relationship dynamics, desire, fear. The most intimate tier.",
        "example_output": {
            "internal": "Her hand brushes mine and I feel warmth rise along my arm.",
            "somatic": {"heart_rate": "elevated", "skin": "flushed"},
            "kappa_shift": 0.03,
        },
    },
]


@app.get("/api/resonance/tiers", response_model=List[TierOut])
def resonance_tiers():
    """Return the 5-tier resonance hierarchy."""
    return [TierOut(**t) for t in TIER_DATA]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("MBD Lab API — http://localhost:8050/docs")
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="info")
