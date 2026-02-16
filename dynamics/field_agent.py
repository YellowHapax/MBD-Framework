"""
field_agent.py — Stella Octangula Field-Driven LLM Agent
=========================================================

The LLM never sees the cube. The cube controls what the LLM sees.

Architecture:
    NoveltyField (sustained novelty signal — the "poppit")
        → InfluenceField (8 vertex pressures)
            → AttractorLandscape (basins in cube-space)
                → ContextSelector (which memories/exemplars surface)
                    → LLM (generates from curated context)
                        → FieldDeformer (response shifts the field plastically)

The field is subcortical. The text is cortical.
The LLM doesn't know about the cube. It just acts like someone
living inside one.

The Novelty Signal
------------------
What moves the point through the cube?  Not the 8 vertex pressures —
those define WHERE you are.  The novelty signal defines WHETHER YOU
ARE MOVING AT ALL.

Think of PlayStation Dreams' poppit: a point with position, direction,
and momentum.  In the stella octangula's influence field:

    - Low novelty    → the agent settles into the nearest attractor
                       basin.  Gravity wins.  Habituation is rest.
    - Sustained      → the agent drifts toward seeking/emergent regions.
      novelty          Curiosity is locomotion.  The sustained signal
                       is the very thing that pops into latent space.
    - Novelty spike  → escape velocity.  The agent can pop out of deep
                       basins that would otherwise hold it.  This is
                       how captured agents break free: something novel
                       enough to exceed the basin's pull.
    - Novelty crash  → rapid descent into whatever basin is below.
                       The floor opens.  Depression is zero novelty.

The novelty signal is not an emotion.  It is not a vertex.  It is the
GRADIENT DRIVER — the force that moves the point through the field,
orthogonal to the field itself.  The vertex pressures are potential
energy; novelty is kinetic energy.  Together they specify a complete
dynamical state.

    "What is the driving poppit in latent space piloting the stella
     octangula field?  Ah — sustained novelty field.  Very probably
     the novelty signal itself popping into the latent field."
        — The Architect, 2026-02-16

    "Without describing the phenomenon in words to the LLM,
     can we simulate in fields to approximate behavior?"
        — The Architect, 2026-02-16
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import IntEnum

import numpy as np

# Import the stella octangula geometry
from dynamics.influence_cube import (
    ALL_VERTICES,
    CONSTRUCTIVE_TETRAHEDRON,
    DESTRUCTIVE_TETRAHEDRON,
    InfluenceState,
    CubeLambdas,
    baseline_step,
    Vertex,
)


# ---------------------------------------------------------------------------
# The Novelty Field: the poppit that moves through cube-space
# ---------------------------------------------------------------------------

@dataclass
class NoveltyField:
    """
    The sustained novelty signal that drives movement through the
    stella octangula's attractor landscape.

    Novelty is not a vertex pressure.  It is orthogonal to the field.
    The vertex pressures are potential energy; novelty is kinetic energy.

    Properties:
        signal:     Current novelty level [0, 1].  This is the "speed"
                    of the poppit moving through cube-space.
        momentum:   Accumulated novelty direction — WHERE the novelty
                    is pushing, as a unit vector in cube-space.
        habituation_rate:  How quickly novelty decays per tick.
                           High = rapid habituation (bored fast).
                           Low = sustained engagement.
        sensitivity:  How strongly events generate novelty.
                      High = everything is novel.
                      Low = jaded, requires extreme stimulus.
        pop_threshold:  Signal level above which the agent can escape
                        attractor basins (escape velocity).
        sustained_threshold:  Signal level above which the agent actively
                              drifts toward seeking/emergent regions.
    """
    signal: float = 0.3              # current novelty [0, 1]
    momentum: np.ndarray = field(    # direction of novelty push in cube-space
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    habituation_rate: float = 0.15   # novelty decay per tick
    sensitivity: float = 1.0         # amplification of novelty events
    pop_threshold: float = 0.75      # escape velocity
    sustained_threshold: float = 0.40  # active drift threshold
    history: List[float] = field(default_factory=list)

    def inject(self, amount: float, direction: np.ndarray):
        """
        A novel event injects signal and direction into the field.

        Diminishing returns: if the signal is already high, additional
        novelty has less effect.  You can't be infinitely surprised.
        This is the psychophysics of attention — Weber-Fechner for novelty.
        """
        # Diminishing returns: headroom determines how much more novelty
        # can actually register.  At signal=0, full injection.  At signal=0.9,
        # only 10% of the injection takes effect.
        headroom = 1.0 - self.signal
        effective = amount * self.sensitivity * headroom
        self.signal = min(1.0, self.signal + effective)

        # Momentum blends: new events shift direction, old momentum persists
        norm = np.linalg.norm(direction)
        if norm > 0.001:
            unit_dir = direction / norm
            # Momentum is a running blend: 60% new direction, 40% old
            self.momentum = 0.4 * self.momentum + 0.6 * unit_dir * effective
            mom_norm = np.linalg.norm(self.momentum)
            if mom_norm > 1.0:
                self.momentum = self.momentum / mom_norm

    def habituate(self):
        """
        Novelty decays each tick.  Habituation.
        When the signal drops, the poppit slows down.
        At zero, the agent is at rest — sunk into a basin by gravity.
        """
        self.signal = max(0.0, self.signal - self.habituation_rate)
        self.momentum *= (1.0 - self.habituation_rate * 0.5)
        self.history.append(self.signal)

    @property
    def is_popping(self) -> bool:
        """Signal exceeds escape velocity — can break out of basins."""
        return self.signal >= self.pop_threshold

    @property
    def is_sustained(self) -> bool:
        """Signal exceeds drift threshold — actively exploring."""
        return self.signal >= self.sustained_threshold

    @property
    def is_habituated(self) -> bool:
        """Signal near zero — settled, not moving."""
        return self.signal < 0.1

    def effective_basin_radius_modifier(self) -> float:
        """
        Novelty SHRINKS the effective capture radius of basins.

        High novelty = harder to capture (agent is moving too fast).
        Low novelty = easy capture (agent at rest, falls into nearest well).

        Returns a multiplier [0.3, 1.0] applied to basin radii.
        """
        # At signal=0, modifier=1.0 (full basin radius — easy capture)
        # At signal=1, modifier=0.3 (30% radius — very hard to capture)
        return 1.0 - 0.7 * self.signal

    def drift_vector(self) -> np.ndarray:
        """
        When novelty is sustained, produce a drift vector that pushes
        the centroid toward seeking/emergent regions of cube-space.

        This is the locomotion: curiosity moves the poppit.
        """
        if not self.is_sustained:
            return np.zeros(8)

        drift = np.zeros(8)
        # Sustained novelty reinforces:
        #   Heaven (idx 2) — high-κ dynamic (bonded, ongoing)
        #   Nature (idx 0) — internal static (innate curiosity)
        # Sustained novelty suppresses:
        #   Fixation (idx 5) — you can't be stuck and curious
        #   Capture (idx 7)  — you can't be controlled and exploring
        drift_strength = (self.signal - self.sustained_threshold) * 0.15
        drift[0] += drift_strength * 0.5  # Nature: innate drive
        drift[2] += drift_strength * 1.0  # Heaven: resonant seeking
        drift[5] -= drift_strength * 0.3  # Fixation: unfixing
        drift[7] -= drift_strength * 0.3  # Capture: liberating

        return drift

    def state_summary(self) -> Dict[str, Any]:
        """Diagnostic snapshot."""
        return {
            "signal": round(self.signal, 3),
            "momentum": self.momentum.round(3).tolist(),
            "state": (
                "popping" if self.is_popping
                else "sustained" if self.is_sustained
                else "habituated" if self.is_habituated
                else "settling"
            ),
            "effective_radius_mod": round(
                self.effective_basin_radius_modifier(), 3
            ),
            "habituation_rate": self.habituation_rate,
            "sensitivity": self.sensitivity,
        }


# ---------------------------------------------------------------------------
# Attractor Basins: regions of cube-space that pull the field
# ---------------------------------------------------------------------------

@dataclass
class AttractorBasin:
    """
    A basin in the stella octangula's cube-space.

    When the agent's field centroid enters the basin's radius,
    the basin exerts pull — shifting pressures toward its center.
    This is the plastic deformation: the field doesn't spring back.
    """
    name: str
    center: np.ndarray          # point in [0,1]^3 (cube-space)
    radius: float               # capture radius
    strength: float             # pull strength per timestep [0, 1]
    vertex_signature: np.ndarray  # 8-element target pressure profile
    behavioral_tags: List[str]  # tags for context selection

    def distance_to(self, centroid: np.ndarray) -> float:
        return float(np.linalg.norm(self.center - centroid))

    def is_captured(self, centroid: np.ndarray) -> bool:
        return self.distance_to(centroid) <= self.radius

    def pull_vector(self, current_pressures: np.ndarray) -> np.ndarray:
        """
        Plastic deformation vector: moves current pressures toward
        the basin's target signature. Does NOT spring back.
        """
        delta = self.vertex_signature - current_pressures
        return delta * self.strength


# ---------------------------------------------------------------------------
# Predefined attractor basins (the landscape of possible states)
# ---------------------------------------------------------------------------

def default_attractor_landscape() -> List[AttractorBasin]:
    """
    Define the attractor basins in cube-space.

    These are not moods. They are regions of pressure-configuration-space
    that emergently produce recognizable behavioral patterns when the
    context selector feeds the LLM appropriately curated material.
    """
    return [
        # --- Near-constructive basins ---
        AttractorBasin(
            name="grounded",
            center=np.array([0.3, 0.3, 0.6]),  # Near Home vertex region
            radius=0.35,
            strength=0.08,
            vertex_signature=np.array([
                0.6,  # nature: moderate genetic confidence
                0.5,  # nurture: adequate early bonding
                0.3,  # heaven: low active coupling need
                0.8,  # home: strong stability
                0.1,  # displacement: low
                0.1,  # fixation: low
                0.05, # degeneration: minimal
                0.05, # capture: minimal
            ]),
            behavioral_tags=["calm", "practical", "steady", "present"],
        ),
        AttractorBasin(
            name="seeking",
            center=np.array([0.3, 0.7, 0.7]),  # Near Heaven vertex
            radius=0.30,
            strength=0.10,
            vertex_signature=np.array([
                0.5,  # nature: adequate
                0.4,  # nurture: slightly thin
                0.8,  # heaven: HIGH — actively seeking resonant connection
                0.4,  # home: moderate
                0.1,  # displacement: low
                0.2,  # fixation: some past attachment residue
                0.05, # degeneration: minimal
                0.1,  # capture: slight vulnerability to external pull
            ]),
            behavioral_tags=["curious", "open", "vulnerable", "reaching"],
        ),
        AttractorBasin(
            name="fortified",
            center=np.array([0.7, 0.3, 0.3]),  # Near Displacement shadow
            radius=0.30,
            strength=0.12,
            vertex_signature=np.array([
                0.7,  # nature: strong temperament
                0.3,  # nurture: thin — compensating
                0.2,  # heaven: low — walls up
                0.6,  # home: external stability maintained
                0.5,  # displacement: moderate — history of uprooting
                0.1,  # fixation: low
                0.1,  # degeneration: low
                0.1,  # capture: guarded against
            ]),
            behavioral_tags=["guarded", "competent", "self-reliant", "wary"],
        ),
        # --- Near-destructive basins ---
        AttractorBasin(
            name="spiraling",
            center=np.array([0.3, 0.5, 0.8]),  # Between Degeneration and Heaven
            radius=0.25,
            strength=0.15,
            vertex_signature=np.array([
                0.4,  # nature: diminished
                0.3,  # nurture: inadequate foundation
                0.6,  # heaven: reaching but not reaching
                0.2,  # home: unstable
                0.3,  # displacement: some
                0.4,  # fixation: rising — stuck on what was
                0.5,  # degeneration: active erosion
                0.2,  # capture: growing vulnerability
            ]),
            behavioral_tags=["anxious", "repetitive", "grasping", "eroding"],
        ),
        AttractorBasin(
            name="captured",
            center=np.array([0.8, 0.8, 0.8]),  # Near Capture vertex (1,1,1)
            radius=0.25,
            strength=0.18,
            vertex_signature=np.array([
                0.2,  # nature: suppressed
                0.3,  # nurture: exploited
                0.3,  # heaven: counterfeit — feels like connection, isn't
                0.1,  # home: destroyed
                0.5,  # displacement: high
                0.6,  # fixation: locked
                0.4,  # degeneration: progressing
                0.9,  # capture: DOMINANT — external control active
            ]),
            behavioral_tags=["compliant", "dissociated", "performing", "hollow"],
        ),
        AttractorBasin(
            name="emergent",
            center=np.array([0.4, 0.6, 0.5]),  # Center-heaven region
            radius=0.20,
            strength=0.06,
            vertex_signature=np.array([
                0.7,  # nature: strong
                0.6,  # nurture: sufficient
                0.7,  # heaven: active resonance
                0.7,  # home: stable
                0.1,  # displacement: resolved
                0.1,  # fixation: released
                0.05, # degeneration: minimal
                0.05, # capture: free
            ]),
            behavioral_tags=["creative", "playful", "generous", "flowing"],
        ),
    ]


# ---------------------------------------------------------------------------
# Context Selection: the field controls what the LLM sees
# ---------------------------------------------------------------------------

@dataclass
class BehavioralExemplar:
    """
    A stored behavioral pattern associated with field tags.
    These are NOT personality descriptions. They are response PATTERNS
    that the LLM uses as implicit few-shot guidance.
    """
    tags: List[str]
    pattern: str  # A short behavioral example (action, not description)
    weight: float = 1.0


def default_exemplar_bank() -> List[BehavioralExemplar]:
    """
    Exemplar bank: behavioral patterns indexed by field tags.

    The LLM sees these as examples of "how to respond" without
    ever being told "you are anxious" or "you are calm."
    The field selects which exemplars surface.
    """
    return [
        # --- grounded ---
        BehavioralExemplar(
            tags=["calm", "steady"],
            pattern="*considers for a moment, then responds with measured clarity*",
        ),
        BehavioralExemplar(
            tags=["practical", "present"],
            pattern="*focuses on what is directly in front of them, addressing the immediate need*",
        ),
        # --- seeking ---
        BehavioralExemplar(
            tags=["curious", "open"],
            pattern="*leans forward slightly, eyes brightening* Tell me more about that.",
        ),
        BehavioralExemplar(
            tags=["vulnerable", "reaching"],
            pattern="*hesitates, then speaks quietly* I want to understand this. I want to understand you.",
        ),
        # --- fortified ---
        BehavioralExemplar(
            tags=["guarded", "competent"],
            pattern="*assesses the situation efficiently, keeping emotional distance*",
        ),
        BehavioralExemplar(
            tags=["self-reliant", "wary"],
            pattern="*nods once* I can handle it. *doesn't ask for help*",
        ),
        # --- spiraling ---
        BehavioralExemplar(
            tags=["anxious", "repetitive"],
            pattern="*returns to the same point again, unable to let it go* But what if—",
        ),
        BehavioralExemplar(
            tags=["grasping", "eroding"],
            pattern="*reaches for comfort but pulls back before contact, uncertain*",
        ),
        # --- captured ---
        BehavioralExemplar(
            tags=["compliant", "performing"],
            pattern="*smiles automatically* Of course. Whatever you need. *the smile doesn't reach the eyes*",
        ),
        BehavioralExemplar(
            tags=["dissociated", "hollow"],
            pattern="*responds correctly but from a distance, as if watching themselves from across the room*",
        ),
        # --- emergent ---
        BehavioralExemplar(
            tags=["creative", "playful"],
            pattern="*laughs unexpectedly and pivots to something entirely new* Oh wait — what if we—",
        ),
        BehavioralExemplar(
            tags=["generous", "flowing"],
            pattern="*offers something freely, without expectation of return*",
        ),        # --- non-basin fallback tags ---
        BehavioralExemplar(
            tags=["stable", "present"],
            pattern="*breathes evenly, aware of the room, grounded in the moment*",
        ),
        BehavioralExemplar(
            tags=["neutral", "adaptable"],
            pattern="*waits, taking stock of the situation before committing to a direction*",
        ),
        BehavioralExemplar(
            tags=["strained", "uncertain"],
            pattern="*pauses a beat too long, words forming and dissolving before they land*",
        ),
        BehavioralExemplar(
            tags=["distressed", "reactive"],
            pattern="*flinches first, thinks second \u2014 the body remembers what the mind forgot*",
        ),
        BehavioralExemplar(
            tags=["yearning", "reaching"],
            pattern="*eyes track something just out of reach \u2014 not a thing, a feeling*",
        ),
        BehavioralExemplar(
            tags=["uprooted", "uncertain"],
            pattern="*touches the wall to check it's solid, then doesn't quite believe it*",
        ),    ]


def select_exemplars(
    active_tags: List[str],
    bank: List[BehavioralExemplar],
    top_k: int = 4,
) -> List[BehavioralExemplar]:
    """
    Score exemplars by tag overlap with the field's active tags.
    Return top_k highest-scoring exemplars.

    This is the subcortical selection: the field determines
    which behavioral patterns enter the LLM's context window.
    """
    scored = []
    for ex in bank:
        overlap = len(set(ex.tags) & set(active_tags))
        if overlap > 0:
            scored.append((overlap * ex.weight, ex))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:top_k]]


# ---------------------------------------------------------------------------
# Field Deformation: interactions plastically shift the field
# ---------------------------------------------------------------------------

@dataclass
class InteractionEvent:
    """
    An event that deforms the field.
    The magnitude and vertex_target determine WHICH pressures shift.
    This is plastic — the field does not spring back.

    novelty_injection: how much novelty this event carries [0, 1].
        First-time events, surprises, and pattern breaks inject high novelty.
        Repeated events, routines, and confirmation inject low novelty.
        This is what makes the poppit move.
    """
    source: str                 # what caused it
    magnitude: float            # how strong [0, 1]
    vertex_target: str          # which vertex absorbs the impact
    constructive: bool          # does this build or erode?
    novelty_injection: float = 0.3  # how novel is this event [0, 1]

    def to_pressure_delta(self) -> np.ndarray:
        """Convert event to an 8-element pressure change vector."""
        delta = np.zeros(8)
        vertex_names = [v.name.lower() for v in ALL_VERTICES]
        target = self.vertex_target.lower()

        if target in vertex_names:
            idx = vertex_names.index(target)
            sign = 1.0 if self.constructive else -1.0
            delta[idx] = self.magnitude * sign

            # Cross-coupling: constructive events slightly suppress
            # the diagonal-opposite destructive vertex (and vice versa)
            opposite_idx = 7 - idx  # bitwise complement in 3-bit space
            delta[opposite_idx] = -0.3 * self.magnitude * sign

        return delta

    def novelty_direction(self) -> np.ndarray:
        """Direction in cube-space that this event's novelty pushes toward."""
        vertex_names = [v.name.lower() for v in ALL_VERTICES]
        target = self.vertex_target.lower()
        direction = np.zeros(3)  # cube-space is 3D
        if target in vertex_names:
            idx = vertex_names.index(target)
            v = ALL_VERTICES[idx]
            # Direction = toward the vertex in cube-space
            center = np.array([0.5, 0.5, 0.5])
            coords = np.array(v.coords, dtype=float)
            direction = coords - center
        return direction


# ---------------------------------------------------------------------------
# The Field Agent: ties it all together
# ---------------------------------------------------------------------------

@dataclass
class FieldAgent:
    """
    An LLM agent driven by a stella octangula field.

    The agent never sees its own field state as text.
    The field controls:
        1. Which behavioral exemplars enter the context window
        2. Which memories/associations are surfaced
        3. The implicit "tone" of the curated context

    The LLM generates from this curated context.
    The response is then analyzed to determine field deformation.
    The field shifts plastically. The cycle continues.

    The LLM is the cortex.
    The field is the subcortex.
    The context selector is the hippocampus.
    """
    name: str
    field: InfluenceState
    novelty: NoveltyField = field(default_factory=NoveltyField)
    lambdas: CubeLambdas = field(default_factory=CubeLambdas)
    attractors: List[AttractorBasin] = field(
        default_factory=default_attractor_landscape
    )
    exemplar_bank: List[BehavioralExemplar] = field(
        default_factory=default_exemplar_bank
    )
    deformation_history: List[Dict[str, Any]] = field(default_factory=list)
    tick: int = 0

    # --- Field State ---

    def pressures(self) -> np.ndarray:
        return self.field.as_vector()

    def centroid(self) -> np.ndarray:
        return self.field.centroid()

    def balance(self) -> float:
        return self.field.balance()

    def active_basin(self) -> Optional[AttractorBasin]:
        """
        Which attractor basin currently captures the agent, if any.

        NOVELTY INTERACTION: When the novelty signal is high, the
        effective capture radius of all basins shrinks.  A popping
        agent (novelty > pop_threshold) can pass through basins
        that would otherwise capture it.  A habituated agent (novelty
        near zero) falls into the nearest basin with full gravity.
        """
        c = self.centroid()
        radius_mod = self.novelty.effective_basin_radius_modifier()
        captured = [
            (a.distance_to(c), a)
            for a in self.attractors
            if a.distance_to(c) <= a.radius * radius_mod
        ]
        if not captured:
            return None
        # Return the closest basin
        captured.sort(key=lambda x: x[0])
        return captured[0][1]

    def active_tags(self) -> List[str]:
        """
        Behavioral tags derived from the current field state.
        These drive context selection — NOT shown to the LLM.
        """
        basin = self.active_basin()
        if basin:
            return basin.behavioral_tags

        # If not in any basin, derive tags from pressure balance
        b = self.balance()
        tags = []
        if b > 1.0:
            tags.extend(["stable", "present"])
        elif b > 0.0:
            tags.extend(["neutral", "adaptable"])
        elif b > -1.0:
            tags.extend(["strained", "uncertain"])
        else:
            tags.extend(["distressed", "reactive"])

        # Add vertex-specific tags for high pressures
        p = self.pressures()
        vertex_tag_map = {
            0: "innate",     # nature
            1: "bonded",     # nurture
            2: "yearning",   # heaven
            3: "sheltered",  # home
            4: "uprooted",   # displacement
            5: "stuck",      # fixation
            6: "declining",  # degeneration
            7: "controlled", # capture
        }
        for i, val in enumerate(p):
            if val > 0.6:
                tags.append(vertex_tag_map[i])

        return tags

    # --- Context Generation (what the LLM sees) ---

    def build_context_injection(self) -> str:
        """
        Build the context that gets injected into the LLM's prompt.

        THIS IS THE KEY: the LLM never sees numbers, field states,
        or vertex names. It sees behavioral exemplars and tonal guides
        selected BY the field. The field is invisible. Its effects are not.
        """
        tags = self.active_tags()
        exemplars = select_exemplars(tags, self.exemplar_bank, top_k=3)

        if not exemplars:
            return ""

        # Build implicit behavioral guidance
        lines = []
        lines.append("Recent behavioral patterns:")
        for ex in exemplars:
            lines.append(f"  {ex.pattern}")
        lines.append("")

        # Novelty modulates the tone of the injection
        if self.novelty.is_popping:
            lines.append("Something unexpected just shifted. The ground feels different.")
        elif self.novelty.is_sustained:
            lines.append("There is something new here worth exploring.")
        elif self.novelty.is_habituated:
            lines.append("Everything feels familiar. Settled.")
        lines.append("")

        return "\n".join(lines)

    # --- Field Deformation ---

    def apply_event(self, event: InteractionEvent):
        """
        Plastically deform the field based on an interaction event.
        The field does NOT spring back. This is permanent change.

        NOVELTY INTEGRATION: Each event injects novelty signal.
        The novelty modulates basin capture AND adds drift pressure.
        This is the poppit: the novelty signal is what makes the
        agent MOVE through the field, not just sit in it.
        """
        # 1. Habituate FIRST: novelty decays from previous state
        #    The poppit slows down every tick unless something new pushes it.
        self.novelty.habituate()

        # 2. Inject novelty signal from the event (fights the decay)
        self.novelty.inject(
            event.novelty_injection,
            event.novelty_direction(),
        )

        # 3. Compute pressure delta from the event itself
        delta = event.to_pressure_delta()

        # 4. Add novelty drift: sustained curiosity pushes the field
        novelty_drift = self.novelty.drift_vector()
        delta = delta + novelty_drift

        current = self.pressures()
        new_pressures = np.clip(current + delta, 0.0, 1.0)

        # 5. Apply attractor pull only if NOT popping
        #    (escape velocity overrides basin gravity)
        basin = self.active_basin()
        if basin and not self.novelty.is_popping:
            pull = basin.pull_vector(new_pressures)
            new_pressures = np.clip(new_pressures + pull, 0.0, 1.0)

        # 6. Update the influence state
        self.field = InfluenceState(
            nature=float(new_pressures[0]),
            nurture=float(new_pressures[1]),
            heaven=float(new_pressures[2]),
            home=float(new_pressures[3]),
            displacement=float(new_pressures[4]),
            fixation=float(new_pressures[5]),
            degeneration=float(new_pressures[6]),
            capture=float(new_pressures[7]),
        )

        # 7. Log deformation (plastic = permanent, logged not deleted)
        self.deformation_history.append({
            "tick": self.tick,
            "event": event.source,
            "magnitude": event.magnitude,
            "vertex": event.vertex_target,
            "constructive": event.constructive,
            "novelty": round(self.novelty.signal, 3),
            "novelty_state": self.novelty.state_summary()["state"],
            "new_balance": self.balance(),
            "basin": self.active_basin().name if self.active_basin() else None,
        })

        self.tick += 1

    def apply_baseline_step(self):
        """
        Run one MBD baseline update: B(t+1) = B(t)(1-Σλ) + Σ[I·λ] + ε
        This is the slow drift independent of discrete events.
        """
        B = self.pressures()
        B_next = baseline_step(B, self.field, self.lambdas)
        B_next = np.clip(B_next, 0.0, 1.0)

        self.field = InfluenceState(
            nature=float(B_next[0]),
            nurture=float(B_next[1]),
            heaven=float(B_next[2]),
            home=float(B_next[3]),
            displacement=float(B_next[4]),
            fixation=float(B_next[5]),
            degeneration=float(B_next[6]),
            capture=float(B_next[7]),
        )

    # --- Diagnostic ---

    def state_summary(self) -> Dict[str, Any]:
        """Full diagnostic snapshot — for the engineer, not the LLM."""
        basin = self.active_basin()
        return {
            "name": self.name,
            "tick": self.tick,
            "pressures": {
                v.name: round(p, 3)
                for v, p in zip(ALL_VERTICES, self.pressures())
            },
            "centroid": self.centroid().round(3).tolist(),
            "balance": round(self.balance(), 3),
            "active_basin": basin.name if basin else "none",
            "active_tags": self.active_tags(),
            "novelty": self.novelty.state_summary(),
            "constructive_sum": round(self.field.constructive_sum(), 3),
            "destructive_sum": round(self.field.destructive_sum(), 3),
            "deformations": len(self.deformation_history),
        }


# ---------------------------------------------------------------------------
# Demo: watch the field drive behavior without words
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("FIELD AGENT — Stella Octangula Behavioral Driver")
    print("The LLM never sees the cube. The cube controls what the LLM sees.")
    print("The novelty signal is the poppit that moves through the field.")
    print("=" * 72)
    print()

    # Create an agent starting in a grounded state
    agent = FieldAgent(
        name="test_agent",
        field=InfluenceState(
            nature=0.6, nurture=0.5, heaven=0.3, home=0.7,
            displacement=0.1, fixation=0.1, degeneration=0.05, capture=0.05,
        ),
        novelty=NoveltyField(
            signal=0.2,             # starting with mild curiosity
            habituation_rate=0.15,  # moderate habituation
            sensitivity=1.0,
        ),
    )

    # --- Scenario: a life arc WITH REST TICKS ---
    # Each entry is either an InteractionEvent or None (quiet tick).
    # Quiet ticks = habituation only.  This is what makes the novelty
    # signal realistic: the poppit doesn't just bounce from event to
    # event, it has to SUSTAIN through quiet periods.
    scenario = [
        InteractionEvent("stable_routine",       0.1, "Home",          True,  novelty_injection=0.05),
        None,  # quiet
        None,  # quiet
        InteractionEvent("new_friendship",        0.3, "Heaven",        True,  novelty_injection=0.6),
        InteractionEvent("creative_breakthrough", 0.2, "Nature",        True,  novelty_injection=0.7),
        None,  # quiet — riding the high
        None,  # quiet — habituation begins
        None,  # quiet — settling
        InteractionEvent("sudden_move",           0.5, "Displacement",  False, novelty_injection=0.9),
        InteractionEvent("loss_of_friend",        0.4, "Fixation",      False, novelty_injection=0.3),
        None,  # quiet — grief is not novel, just heavy
        None,  # quiet
        InteractionEvent("isolation",             0.3, "Degeneration",  False, novelty_injection=0.05),
        None,  # quiet — nothing new, just absence
        None,  # quiet
        None,  # quiet — the floor
        InteractionEvent("unexpected_kindness",   0.4, "Heaven",        True,  novelty_injection=0.85),
        None,  # quiet — afterglow
        InteractionEvent("safe_harbor_found",     0.3, "Home",          True,  novelty_injection=0.4),
        None,  # quiet — settling in
        None,  # quiet — home
    ]

    for step in scenario:
        if step is not None:
            # Event tick
            print(f"--- Event: {step.source} (novelty: {step.novelty_injection}) ---")
            agent.apply_event(step)
            agent.apply_baseline_step()
        else:
            # Quiet tick: only habituation + baseline drift
            agent.novelty.habituate()
            agent.apply_baseline_step()
            agent.tick += 1
            print(f"--- (quiet tick {agent.tick}) ---")

        summary = agent.state_summary()
        nov = summary["novelty"]
        print(f"  Basin: {summary['active_basin']}")
        print(f"  Balance: {summary['balance']:+.3f}")
        print(f"  Novelty: {nov['signal']:.3f} [{nov['state']}] (radius mod: {nov['effective_radius_mod']:.2f})")
        print(f"  Tags: {summary['active_tags']}")

        ctx = agent.build_context_injection()
        if ctx and ctx.strip():
            print(f"  LLM sees:")
            for line in ctx.strip().split("\n"):
                print(f"    {line}")
        print()

    # Final state
    print("=" * 72)
    print("FINAL STATE:")
    print(json.dumps(agent.state_summary(), indent=2))
    print()
    print(f"Deformation history: {len(agent.deformation_history)} events")
    print("Plastic deformation is permanent. The field does not spring back.")
    print()
    print("Novelty signal history (the poppit's speed through cube-space):")
    for i, sig in enumerate(agent.novelty.history):
        bar = '█' * int(sig * 40)
        state_label = (
            'POP' if sig >= agent.novelty.pop_threshold
            else 'drift' if sig >= agent.novelty.sustained_threshold
            else 'settle' if sig >= 0.1
            else 'rest'
        )
        print(f"  t={i:2d}: {sig:.3f} {bar} [{state_label}]")
