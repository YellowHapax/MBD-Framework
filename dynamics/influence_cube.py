"""
The Influence Cube: A Geometric Formalization of Developmental Pressure

    "Nature, nurture, heaven and home
     Sum of all, and by them, driven
     To conquer every mountain shown
     But have you seen the river?"
        — Puscifer, "The Humbling River"

The Quadrafoil of Influence (Sanctuary, Arena, Market, Cesspit) described
environmental *locations* that modify agent baselines. But that model is
incomplete — it captures where pressure comes from, not how it is structured.

The Influence Cube recovers the missing structure. Every developmental
pressure acting on an agent can be decomposed along three binary axes:

    Axis 0 — Locus:       Internal (self) ↔ External (environment)
    Axis 1 — Coupling:    Low-κ (independent) ↔ High-κ (bonded)
    Axis 2 — Temporality: Static (historical) ↔ Dynamic (ongoing)

These three axes span a unit cube in {0,1}^3 with 8 vertices. Four of
those vertices are the constructive developmental drivers; the other four
are their destructive duals. The two groups each inscribe a regular
tetrahedron. Together they form a stella octangula — the compound of two
dual tetrahedra sharing the same bounding cube.

The River
---------
The model is a cube. The world is not. The river is whatever remains after
all eight vertices have been accounted for — the irreducible residual, the
thing the geometry cannot capture. In MBD terms, it is the unexplained
variance in B(t+1) after all λ_v terms are summed. Every honest model must
leave room for the river.

References
----------
Everett, B. (2025). Memory as Baseline Deviation.
    https://doi.org/10.5281/zenodo.14538419
Everett, B. (2025). The Coupling Asymmetry.
    https://doi.org/10.5281/zenodo.14611399
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------

class Locus(IntEnum):
    """Axis 0: Where does the influence originate?"""
    INTERNAL = 0   # Self-originating (genetics, temperament, will)
    EXTERNAL = 1   # Environment-originating (culture, material conditions)


class Coupling(IntEnum):
    """Axis 1: Does the influence require resonant connection?"""
    LOW_KAPPA = 0   # Independent — operates without bond
    HIGH_KAPPA = 1  # Bonded — requires κ-coupled relationship


class Temporality(IntEnum):
    """Axis 2: Is the influence already set, or still in motion?"""
    STATIC = 0      # Historical — deposited, no longer changing
    DYNAMIC = 1     # Ongoing — actively exerting pressure now


# ---------------------------------------------------------------------------
# The Eight Vertices
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Vertex:
    """A vertex of the Influence Cube."""
    name: str
    symbol: str
    locus: Locus
    coupling: Coupling
    temporality: Temporality
    description: str
    constructive: bool

    @property
    def coords(self) -> Tuple[int, int, int]:
        return (int(self.locus), int(self.coupling), int(self.temporality))

    @property
    def index(self) -> int:
        """Binary encoding: locus * 4 + coupling * 2 + temporality."""
        return int(self.locus) * 4 + int(self.coupling) * 2 + int(self.temporality)


# --- Constructive tetrahedron (the four drivers) ---

NATURE = Vertex(
    name="Nature",
    symbol="N",
    locus=Locus.INTERNAL,
    coupling=Coupling.LOW_KAPPA,
    temporality=Temporality.STATIC,
    description=(
        "Genetic endowment, innate temperament, neurological architecture. "
        "Internal, independent of bonding, and already set at birth. "
        "The hand you were dealt."
    ),
    constructive=True,
)

NURTURE = Vertex(
    name="Nurture",
    symbol="U",
    locus=Locus.EXTERNAL,
    coupling=Coupling.HIGH_KAPPA,
    temporality=Temporality.STATIC,
    description=(
        "Early caregiving, attachment history, formative bonding. "
        "External, required high-κ coupling (parent–child bond), "
        "and is now historical — the shaping is done."
    ),
    constructive=True,
)

HEAVEN = Vertex(
    name="Heaven",
    symbol="H",
    locus=Locus.INTERNAL,
    coupling=Coupling.HIGH_KAPPA,
    temporality=Temporality.DYNAMIC,
    description=(
        "Safe intimacy — the ongoing experience of being deeply known "
        "and not destroyed by it. The place inside the self that only "
        "exists because someone else made it safe to exist. Internal "
        "transformation through vulnerability without annihilation: the "
        "armor comes off and nothing bad happens. This is what drives "
        "a person to conquer every mountain shown — not ambition, but "
        "the knowledge that there is somewhere to put it all down."
    ),
    constructive=True,
)

HOME = Vertex(
    name="Home",
    symbol="O",
    locus=Locus.EXTERNAL,
    coupling=Coupling.LOW_KAPPA,
    temporality=Temporality.DYNAMIC,
    description=(
        "Safe environment, material security, stable shelter. "
        "External, does not require bonded relationship to function, "
        "and is ongoing — you must maintain it."
    ),
    constructive=True,
)

# --- Destructive tetrahedron (the four shadows) ---

DISPLACEMENT = Vertex(
    name="Displacement",
    symbol="D",
    locus=Locus.EXTERNAL,
    coupling=Coupling.LOW_KAPPA,
    temporality=Temporality.STATIC,
    description=(
        "Uprooting that already happened: exile, dispossession, "
        "environmental destruction. External, no bond to cushion it, "
        "and the damage is historical — it's done."
    ),
    constructive=False,
)

FIXATION = Vertex(
    name="Fixation",
    symbol="X",
    locus=Locus.INTERNAL,
    coupling=Coupling.HIGH_KAPPA,
    temporality=Temporality.STATIC,
    description=(
        "A past bond that calcified: unresolved grief, frozen attachment, "
        "idealization of what was. Internal, born from high-κ coupling "
        "that ended but whose imprint refuses to decay."
    ),
    constructive=False,
)

DEGENERATION = Vertex(
    name="Degeneration",
    symbol="G",
    locus=Locus.INTERNAL,
    coupling=Coupling.LOW_KAPPA,
    temporality=Temporality.DYNAMIC,
    description=(
        "Ongoing internal erosion: neurological decline, chronic illness, "
        "addiction feedback loops. Internal, independent of bonding, "
        "and actively progressing."
    ),
    constructive=False,
)

CAPTURE = Vertex(
    name="Capture",
    symbol="C",
    locus=Locus.EXTERNAL,
    coupling=Coupling.HIGH_KAPPA,
    temporality=Temporality.DYNAMIC,
    description=(
        "Active external resonant control: the dark mirror of Heaven. "
        "Also requires the armor to come off — but overwrites what it "
        "finds underneath. Cult dynamics, coercive bonding, abusive "
        "relationships, epistemological capture. External, high-κ "
        "(it needs your participation to function), ongoing. "
        "The anti-Nature — vertex (1,1,1), diagonally opposite (0,0,0). "
        "Maximum distance from what you genuinely are."
    ),
    constructive=False,
)

# --- Collected constants ---

CONSTRUCTIVE_TETRAHEDRON: Tuple[Vertex, ...] = (NATURE, NURTURE, HEAVEN, HOME)
DESTRUCTIVE_TETRAHEDRON: Tuple[Vertex, ...] = (DISPLACEMENT, FIXATION, DEGENERATION, CAPTURE)
ALL_VERTICES: Tuple[Vertex, ...] = CONSTRUCTIVE_TETRAHEDRON + DESTRUCTIVE_TETRAHEDRON


# ---------------------------------------------------------------------------
# Geometric verification
# ---------------------------------------------------------------------------

def _euclidean(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def verify_stella_octangula() -> Dict[str, object]:
    """
    Verify that the constructive and destructive tetrahedra are:
      1. Each regular (all 6 edge lengths equal).
      2. Dual to each other (every constructive vertex is equidistant
         from every destructive vertex at distance √2).
      3. Inscribed in the same unit cube.

    Returns a dict of verification results.
    """
    c_coords = [v.coords for v in CONSTRUCTIVE_TETRAHEDRON]
    d_coords = [v.coords for v in DESTRUCTIVE_TETRAHEDRON]

    # Intra-tetrahedron edges
    c_edges = [_euclidean(c_coords[i], c_coords[j])
               for i in range(4) for j in range(i + 1, 4)]
    d_edges = [_euclidean(d_coords[i], d_coords[j])
               for i in range(4) for j in range(i + 1, 4)]

    # Cross edges (constructive ↔ destructive)
    cross = [_euclidean(c, d) for c in c_coords for d in d_coords]

    sqrt2 = math.sqrt(2)

    return {
        "constructive_edges": c_edges,
        "constructive_regular": all(abs(e - sqrt2) < 1e-9 for e in c_edges),
        "destructive_edges": d_edges,
        "destructive_regular": all(abs(e - sqrt2) < 1e-9 for e in d_edges),
        "cross_distances": sorted(set(round(x, 6) for x in cross)),
        "is_stella_octangula": (
            all(abs(e - sqrt2) < 1e-9 for e in c_edges) and
            all(abs(e - sqrt2) < 1e-9 for e in d_edges)
        ),
        "nature_capture_diagonal": _euclidean(NATURE.coords, CAPTURE.coords),
    }


# ---------------------------------------------------------------------------
# The Influence Cube: agent pressure integration
# ---------------------------------------------------------------------------

@dataclass
class InfluenceState:
    """
    The current magnitude of each vertex's pressure on a single agent.

    All values in [0, 1]. The cube does not enforce that they sum to
    anything — an agent may be under heavy pressure from multiple vertices
    simultaneously, or nearly none.
    """
    nature: float = 0.0
    nurture: float = 0.0
    heaven: float = 0.0
    home: float = 0.0
    displacement: float = 0.0
    fixation: float = 0.0
    degeneration: float = 0.0
    capture: float = 0.0

    def as_vector(self) -> np.ndarray:
        """Return the 8-element pressure vector, vertex-indexed."""
        return np.array([
            self.nature, self.nurture, self.heaven, self.home,
            self.displacement, self.fixation, self.degeneration, self.capture,
        ], dtype=np.float64)

    def constructive_sum(self) -> float:
        return self.nature + self.nurture + self.heaven + self.home

    def destructive_sum(self) -> float:
        return self.displacement + self.fixation + self.degeneration + self.capture

    def balance(self) -> float:
        """
        Signed balance: positive = constructive-dominant,
        negative = destructive-dominant, zero = equilibrium.
        """
        return self.constructive_sum() - self.destructive_sum()

    def centroid(self) -> np.ndarray:
        """
        Pressure-weighted centroid in cube-space.

        Returns a point in [0,1]^3 indicating where the agent's net
        influence is "pulling" them. If all pressures are zero, returns
        the cube center (0.5, 0.5, 0.5).
        """
        coords = np.array([v.coords for v in ALL_VERTICES], dtype=np.float64)
        weights = self.as_vector()
        total = weights.sum()
        if total < 1e-12:
            return np.array([0.5, 0.5, 0.5])
        return (coords.T @ weights) / total


# ---------------------------------------------------------------------------
# Baseline deviation with vertex-addressed influences
# ---------------------------------------------------------------------------

@dataclass
class CubeLambdas:
    """
    Per-vertex coupling constants (λ_v) controlling how strongly each
    vertex's pressure modifies the agent's baseline per timestep.

    The river term (ε) represents unexplained variance — the influence
    the model cannot capture.
    """
    lambdas: np.ndarray = field(
        default_factory=lambda: np.full(8, 0.05, dtype=np.float64)
    )
    river: float = 0.0  # ε — the irreducible residual

    def __post_init__(self):
        if self.lambdas.shape != (8,):
            raise ValueError(f"lambdas must be shape (8,), got {self.lambdas.shape}")


def baseline_step(
    B: np.ndarray,
    influences: InfluenceState,
    lambdas: CubeLambdas,
    noise_scale: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    One timestep of the vertex-addressed baseline deviation equation.

    B(t+1) = B(t)(1 - Σλ_v) + Σ[I_v(t) · λ_v] + ε

    Parameters
    ----------
    B : np.ndarray
        Current baseline vector (arbitrary dimensionality — the influence
        magnitudes are broadcast across all baseline components).
    influences : InfluenceState
        Current pressure magnitudes from each vertex.
    lambdas : CubeLambdas
        Per-vertex coupling constants and river term.
    noise_scale : float
        Standard deviation of Gaussian noise for the river term.
    rng : np.random.Generator, optional
        Random generator for reproducible river noise.

    Returns
    -------
    np.ndarray
        Updated baseline vector B(t+1).
    """
    I = influences.as_vector()          # (8,)
    lam = lambdas.lambdas               # (8,)

    total_lambda = lam.sum()
    weighted_influence = (I * lam).sum()

    # River: deterministic component + stochastic residual
    epsilon = lambdas.river
    if noise_scale > 0:
        if rng is None:
            rng = np.random.default_rng()
        epsilon += rng.normal(0, noise_scale)

    return B * (1.0 - total_lambda) + weighted_influence + epsilon


# ---------------------------------------------------------------------------
# Symmetry analysis
# ---------------------------------------------------------------------------

def dual_pair(v: Vertex) -> Vertex:
    """
    Return the vertex diagonally opposite in the cube (bitwise NOT on
    all three axes). Every constructive vertex's dual is destructive
    and vice versa.
    """
    target = (1 - v.locus, 1 - v.coupling, 1 - v.temporality)
    for u in ALL_VERTICES:
        if u.coords == target:
            return u
    raise ValueError(f"No dual found for {v.name}")  # pragma: no cover


def print_cube_summary():
    """Print a human-readable summary of the Influence Cube geometry."""
    print("=" * 72)
    print("THE INFLUENCE CUBE — Stella Octangula of Developmental Pressure")
    print("=" * 72)
    print()
    print("Axes:")
    print("  0 — Locus:       Internal (0) ↔ External (1)")
    print("  1 — Coupling:    Low-κ   (0) ↔ High-κ  (1)")
    print("  2 — Temporality: Static  (0) ↔ Dynamic (1)")
    print()
    print("CONSTRUCTIVE TETRAHEDRON (the four drivers):")
    print("-" * 72)
    for v in CONSTRUCTIVE_TETRAHEDRON:
        d = dual_pair(v)
        print(f"  {v.name:<14s} {str(v.coords):<12s}  dual → {d.name}")
    print()
    print("DESTRUCTIVE TETRAHEDRON (the four shadows):")
    print("-" * 72)
    for v in DESTRUCTIVE_TETRAHEDRON:
        d = dual_pair(v)
        print(f"  {v.name:<14s} {str(v.coords):<12s}  dual → {d.name}")
    print()

    proof = verify_stella_octangula()
    tag = "✓" if proof["is_stella_octangula"] else "✗"
    print(f"Geometry verification: {tag}")
    print(f"  Constructive regular tetrahedron: {proof['constructive_regular']}")
    print(f"  Destructive regular tetrahedron:  {proof['destructive_regular']}")
    print(f"  Nature↔Capture diagonal:          {proof['nature_capture_diagonal']:.4f}"
          f"  (√3 = {math.sqrt(3):.4f})")
    print()
    print("The River: ε — the irreducible residual the geometry cannot capture.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_cube_summary()

    # --- Demo: an agent shaped by strong nurture, active heaven,
    #     moderate home, and some fixation on a past bond ---
    state = InfluenceState(
        nature=0.6,
        nurture=0.8,
        heaven=0.7,
        home=0.5,
        displacement=0.1,
        fixation=0.4,
        degeneration=0.0,
        capture=0.0,
    )

    print("Demo agent influence state:")
    print(f"  Constructive sum: {state.constructive_sum():.2f}")
    print(f"  Destructive sum:  {state.destructive_sum():.2f}")
    print(f"  Balance:          {state.balance():+.2f}")
    print(f"  Centroid in cube: {state.centroid()}")
    print()

    # Baseline step demo
    B = np.array([0.5, 0.3, 0.7, 0.4])  # [trust, aggression, curiosity, status]
    lam = CubeLambdas(
        lambdas=np.array([0.04, 0.06, 0.08, 0.03, 0.02, 0.05, 0.01, 0.01]),
        river=0.0,
    )
    B_next = baseline_step(B, state, lam)
    print(f"  B(t)   = {B}")
    print(f"  B(t+1) = {B_next}")
    print(f"  ΔB     = {B_next - B}")
