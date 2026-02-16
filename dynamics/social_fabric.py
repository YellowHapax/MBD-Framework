"""
social_fabric.py  --  Paper-derived agent interaction model
===========================================================

Demonstrates core concepts from the MBD Framework research papers:

- **Baseline Deviation** (Paper 1): Each agent maintains a psyche vector
  that deviates from a cohort reference baseline under environmental
  pressure.
- **Coupling Asymmetry** (Paper 4): Influence between agents is
  directional -- kappa_ij != kappa_ji -- modelled as asymmetric coupling
  coefficients on edge pressures.
- **Markov Interaction Probability** (Paper 2): The probability of a
  social event is derived from psyche-vector compatibility and
  edge-pressure history, formulated as a tensor contraction.
- **Resonant Field Translation** (Paper 6): Edge pressures evolve each
  tick via field translation -- trust-aligned dyads accumulate positive
  pressures while aggression-aligned dyads amplify conflict.
- **Emergent Gating** (Paper 5): Frustration is an emergent signal that
  arises when drive exceeds available bonding opportunity, gating
  interaction probability upward (urgency) or downward (withdrawal).

All temporal parameters use dimensionless epoch fractions.
No game mechanics (food, inventory, conception, theft) are present.
"""

import math
import random
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Cohort profiles  --  dimensionless temporal parameters
# ---------------------------------------------------------------------------
# All values are fractions of a normalised agent epoch [0, 1].
# ``epoch_scale`` lets cohorts age at different rates relative to each other.

COHORT_PROFILES = {
    "fast":     {"epoch_scale": 0.8, "maturation": 0.22, "bond_onset": 0.20, "bond_offset": 0.56, "senescence": 0.70},
    "default":  {"epoch_scale": 1.0, "maturation": 0.20, "bond_onset": 0.20, "bond_offset": 0.56, "senescence": 0.75},
    "moderate": {"epoch_scale": 1.5, "maturation": 0.16, "bond_onset": 0.12, "bond_offset": 0.60, "senescence": 0.78},
    "slow":     {"epoch_scale": 3.0, "maturation": 0.10, "bond_onset": 0.10, "bond_offset": 0.60, "senescence": 0.85},
}

GROUP_PROFILE = {
    "alpha": "fast", "beta": "default", "gamma": "moderate", "delta": "slow",
}

DEFAULT_EPOCH_TICKS: int = 100  # configurable per experiment

BASE_INTERACTION_PROB = 0.005


def _profile(group: str) -> dict:
    """Resolve a group label to tick-space cohort parameters."""
    key = GROUP_PROFILE.get(group.lower(), "default")
    p = COHORT_PROFILES.get(key, COHORT_PROFILES["default"])
    epoch = DEFAULT_EPOCH_TICKS * p["epoch_scale"]
    return {
        "epoch":       epoch,
        "maturation":  p["maturation"]  * epoch,
        "bond_onset":  p["bond_onset"]  * epoch,
        "bond_offset": p["bond_offset"] * epoch,
        "senescence":  p["senescence"]  * epoch,
    }


# ---------------------------------------------------------------------------
# Agent & edge synthesis
# ---------------------------------------------------------------------------

# Reference baseline -- the "zero-deviation" psyche (Paper 1)
_BASELINE = {"trust": 0.50, "playful": 0.50, "aggression": 0.25, "reproductive_drive": 0.15}


def _synthesize_agents_and_edges(
    world_data: Dict[str, Any], per_race: int = 6,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create a synthetic population with MBD baseline-deviation psyche vectors."""
    groups = list(world_data.get("capitals", {}).keys()) or [
        "Alpha", "Beta", "Gamma", "Delta",
    ]

    agents: List[Dict[str, Any]] = []
    edges:  List[Dict[str, Any]] = []
    group_ids: Dict[str, List[str]] = {}

    for group in groups:
        gkey = group.lower()
        prof = _profile(gkey)
        members: List[str] = []
        for i in range(per_race):
            age = random.uniform(prof["maturation"], prof["bond_offset"])
            agent = {
                "id":   f"{group[:3].upper()}-{i}",
                "name": f"{group} {i}",
                "race": gkey,
                "sex":  random.choice(["female", "male"]),
                "body_morph": {"age": round(age, 1)},
                "psyche": {
                    k: max(0.0, min(1.0, v + random.gauss(0, 0.12 if k != "aggression" else 0.08)))
                    for k, v in _BASELINE.items()
                },
                "pressures": {"frustration": random.uniform(0.0, 0.10)},
            }
            agents.append(agent)
            members.append(agent["id"])
        group_ids[group] = members

    # Sparse intra-group edges (cohort familiarity)
    for members in group_ids.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if random.random() < 0.20:
                    edges.append(_init_edge(members[i], members[j]))

    # Cross-group edges (inter-cohort encounters)
    all_ids = [a["id"] for a in agents]
    for _ in range(max(3, len(all_ids) // 6)):
        a, b = random.sample(all_ids, 2)
        edges.append(_init_edge(a, b))

    return agents, edges


def _init_edge(a_id: str, b_id: str) -> Dict[str, Any]:
    """Create a new edge with low random pressure seeds."""
    return {
        "a": a_id, "b": b_id,
        "intimacy":     random.uniform(0.02, 0.15),
        "love":         random.uniform(0.00, 0.10),
        "conflict":     random.uniform(0.00, 0.12),
        "pair_bonding": random.uniform(0.00, 0.08),
    }


# ---------------------------------------------------------------------------
# Per-tick agent updates
# ---------------------------------------------------------------------------

def _update_reproductive_drive(agent: Dict, all_agents: List[Dict]):
    """
    Paper 4 (Coupling Asymmetry) + Paper 1 (MBD):
    Drive follows a parabolic bonding window modulated by demographic
    pressure.  Frustration emerges when drive exceeds opportunity
    (Paper 5: Emergent Gate).
    """
    prof = _profile(agent.get("race", "default"))
    age  = agent.get("body_morph", {}).get("age", prof["maturation"])
    psy  = agent["psyche"]

    # Pre-maturation gating
    if age < prof["maturation"] * 0.6:
        psy["reproductive_drive"] = 0.0
        return

    # Parabolic bonding-window drive
    if prof["bond_onset"] <= age <= prof["bond_offset"]:
        window = prof["bond_offset"] - prof["bond_onset"]
        mid    = prof["bond_onset"] + window / 2
        urgency = 1.0 - ((age - mid) / (window / 2)) ** 2
        psy["reproductive_drive"] += urgency * 0.01

    # Demographic pressure (more elders -> higher cohort drive)
    same = [a for a in all_agents if a.get("race") == agent.get("race")]
    if same:
        elder_ratio = sum(
            1 for a in same
            if a["body_morph"].get("age", 0) > prof["senescence"]
        ) / len(same)
        if elder_ratio > 0.3:
            psy["reproductive_drive"] += elder_ratio * 0.015

    # Emergent frustration (Paper 5)
    if psy["reproductive_drive"] > 0.6:
        agent["pressures"]["frustration"] = min(
            agent["pressures"].get("frustration", 0) + 0.04, 1.0,
        )
    else:
        agent["pressures"]["frustration"] = max(
            agent["pressures"].get("frustration", 0) * 0.93, 0.0,
        )

    psy["reproductive_drive"] = max(0.0, min(1.0, psy["reproductive_drive"]))


def _update_agent_needs(agent: Dict):
    """
    Per-tick baseline micro-drift (Paper 1).

    Trust and playfulness undergo small random walks representing
    environmental micro-pressures.  Aggression decays toward the cohort
    mean (homeostatic pull).
    """
    psy = agent["psyche"]
    psy["trust"]    = max(0.0, min(1.0, psy["trust"]    + random.gauss(0, 0.005)))
    psy["playful"]  = max(0.0, min(1.0, psy["playful"]  + random.gauss(0, 0.005)))
    # Aggression drifts toward baseline attractor (0.25)
    psy["aggression"] += (0.25 - psy["aggression"]) * 0.02
    psy["aggression"]  = max(0.0, min(1.0, psy["aggression"]))


# ---------------------------------------------------------------------------
# Edge evolution  --  Resonant Field Translation (Paper 6)
# ---------------------------------------------------------------------------

def _evolve_edges(
    edges: List[Dict], agents: List[Dict], matrix: Dict,
):
    """
    Edge pressures shift each tick based on the psyche alignment of the
    connected agents.  Trust-aligned dyads accumulate intimacy/love;
    aggression-aligned dyads amplify conflict.  All pressures undergo
    slight homeostatic decay.  This is the field-translation loop from
    Paper 6 (Resonant Gate).
    """
    agent_map = {a["id"]: a for a in agents}
    for e in edges:
        a1 = agent_map.get(e["a"], {})
        a2 = agent_map.get(e["b"], {})
        p1, p2 = a1.get("psyche", {}), a2.get("psyche", {})

        trust_avg = (p1.get("trust", 0.5) + p2.get("trust", 0.5)) / 2
        agg_avg   = (p1.get("aggression", 0.25) + p2.get("aggression", 0.25)) / 2
        play_avg  = (p1.get("playful", 0.5) + p2.get("playful", 0.5)) / 2

        # Field translation: psyche compatibility -> edge pressure deltas
        e["intimacy"]     += (trust_avg - 0.40) * 0.005
        e["love"]         += (trust_avg * play_avg - 0.20) * 0.003
        e["conflict"]     += (agg_avg - 0.30) * 0.004
        e["pair_bonding"] += max(0, e["intimacy"] + e["love"] - e["conflict"]) * 0.002

        # Homeostatic decay
        for k in ("intimacy", "love", "conflict", "pair_bonding"):
            e[k] = max(0.0, min(1.0, e[k] * 0.998))


# ---------------------------------------------------------------------------
# Relationship matrix
# ---------------------------------------------------------------------------

def _build_relationship_matrix(
    edges: List[Dict],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Bidirectional O(1) lookup table from edge list."""
    matrix: Dict[str, Dict[str, Dict[str, float]]] = {}
    for e in edges:
        p = {k: e.get(k, 0.0) for k in ("intimacy", "love", "conflict", "pair_bonding")}
        matrix.setdefault(e["a"], {})[e["b"]] = p
        matrix.setdefault(e["b"], {})[e["a"]] = p
    return matrix


# ---------------------------------------------------------------------------
# Interaction probability  --  Markov Tensor contraction (Paper 2)
# ---------------------------------------------------------------------------

def calculate_interaction_prob(
    agent1: Dict, agent2: Dict, pressures: Dict[str, float],
) -> float:
    """
    Interaction probability via psyche-vector compatibility and edge memory.

    Three contracted terms:
      1. **Alignment** -- trust / playfulness compatibility (dot-product).
      2. **Edge memory** -- tanh-scaled positive-pressure accumulation.
      3. **Drive urgency** -- combined reproductive drive + frustration
         signal (Paper 5: Emergent Gate).

    This mirrors the tensor-product formulation from Paper 2 where the
    probability of transition between social states is the inner product
    of agent state vectors with the Markov coupling tensor.
    """
    p1, p2 = agent1.get("psyche", {}), agent2.get("psyche", {})

    # Term 1: psyche alignment
    trust_sim = 1.0 - abs(p1.get("trust", 0.5) - p2.get("trust", 0.5))
    play_sim  = 1.0 - abs(p1.get("playful", 0.5) - p2.get("playful", 0.5))
    alignment = 0.3 + 0.7 * (0.6 * trust_sim + 0.4 * play_sim)

    # Term 2: edge memory (positive pressures attract)
    memory = 1.0 + math.tanh(
        pressures.get("intimacy", 0) + pressures.get("love", 0)
    )

    # Term 3: drive urgency (Paper 5)
    drive = 1.0 + 0.5 * (
        p1.get("reproductive_drive", 0) + p2.get("reproductive_drive", 0)
    )
    frust = 1.0 + 0.3 * (
        agent1.get("pressures", {}).get("frustration", 0)
        + agent2.get("pressures", {}).get("frustration", 0)
    )

    return min(BASE_INTERACTION_PROB * alignment * memory * drive * frust, 1.0)
