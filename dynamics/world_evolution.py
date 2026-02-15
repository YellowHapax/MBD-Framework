"""
world_evolution.py  --  Environmental field dynamics (Quadrafoil model)
======================================================================

Demonstrates the MBD Framework's environmental influence model.

Four archetypal site types -- the **Quadrafoil** -- produce deontological
pressure fields that shift agent baselines over time:

+-------------+---------------------------------------------------+
| Pole        | Effect on agents                                  |
+=============+===================================================+
| Sanctuary   | +trust, -aggression  -- fosters cohesion           |
| Arena       | +aggression (structured), -frustration  -- outlet  |
| Market      | +playful, +trust  -- novelty and exchange          |
| Cesspit     | +frustration, -trust  -- consequence of collapse   |
+-------------+---------------------------------------------------+

Each site exerts influence proportional to 1/r^2 (inverse-square falloff)
on agents within range.  The field strength of each site evolves based on
nearby interaction density (reinforcement loop from Paper 6: Resonant Gate).

This module is a reference scaffold.  Full spatial simulation (terrain
generation, pathfinding, biome maps) belongs in application-layer code,
not in the research framework.
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Quadrafoil site archetypes and their pressure signatures
# ---------------------------------------------------------------------------

QUADRAFOIL = {
    "sanctuary": {"trust": +0.04, "aggression": -0.02, "frustration": -0.03},
    "arena":     {"trust": -0.01, "aggression": +0.03, "frustration": -0.01},
    "market":    {"trust": +0.01, "playful":    +0.03, "frustration": -0.01},
    "cesspit":   {"trust": -0.03, "aggression": +0.02, "frustration": +0.04},
}


def field_influence(site_type: str, distance: float) -> Dict[str, float]:
    """
    Compute the pressure delta a Quadrafoil site exerts at a given distance.

    Uses inverse-square falloff with a minimum-distance clamp to prevent
    singularities.  Returns a dict of psyche-key -> delta.
    """
    sig = QUADRAFOIL.get(site_type, {})
    r2 = max(distance * distance, 0.25)  # clamp at 0.5-unit minimum
    return {k: v / r2 for k, v in sig.items()}


def apply_field_to_agent(agent: Dict[str, Any], delta: Dict[str, float]):
    """Apply a pressure delta to an agent's psyche and pressure vectors."""
    psyche    = agent.get("psyche", {})
    pressures = agent.get("pressures", {})
    for key, val in delta.items():
        if key in psyche:
            psyche[key] = max(0.0, min(1.0, psyche[key] + val))
        elif key in pressures:
            pressures[key] = max(0.0, min(1.0, pressures[key] + val))


def evolve_sites(
    sites: List[Dict[str, Any]],
    agents: List[Dict[str, Any]],
    interaction_count: int,
) -> None:
    """
    Reinforce site field strength based on nearby interaction density.

    Sites that facilitate more interactions grow stronger (positive
    feedback loop described in Paper 6: Resonant Gate field translation).
    """
    for site in sites:
        density = interaction_count / max(len(agents), 1)
        site["strength"] = min(
            site.get("strength", 1.0) + density * 0.01, 3.0,
        )
