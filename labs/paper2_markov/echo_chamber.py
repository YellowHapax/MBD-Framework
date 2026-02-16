"""Echo Chamber Collapse — Resonant Cavity Shattering

Paper 2: Pursuit of the Markov Tensor (10.5281/zenodo.17537185)

A group with high average κ and low baseline divergence forms a
resonant cavity.  An undeniable external truth contradicting the
shared baseline causes a catastrophic novelty spike that shatters κ,
producing relational fragmentation.

Ported from MemoryLab-OS/phenomena.tsx — EchoChamberCollapseModel.
"""

from __future__ import annotations
import math, random
from typing import Any, Dict, List, Optional

import numpy as np

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 2
PAPER_TITLE = "Pursuit of the Markov Tensor"
PAPER_DOI = "10.5281/zenodo.17537185"
LAB_TITLE = "Phenomenon: Echo Chamber Collapse"

THESIS = (
    "A group with high average coupling (κ) and low internal baseline "
    "divergence forms a 'resonant cavity'. When faced with a strong, "
    "undeniable external truth contradicting the shared baseline, the "
    "system suffers a catastrophic cascade failure: a massive novelty "
    "spike shatters the resonant state, causing κ to plummet and "
    "relational fragmentation."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    n_agents: int = 4,
    t_shock: int = 60,
    steps: int = 100,
    convergence_rate: float = 1.05,
    novelty_floor: float = 0.05,
    shock_novelty: float = 5.0,
    post_shock_base: float = 1.5,
    kappa_shock_factor: float = 0.20,
    kappa_post_decay: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate echo chamber formation and collapse.

    Parameters
    ----------
    n_agents : int       Number of agents in the group.
    t_shock : int        Timestep of the external shock event.
    steps : int          Total simulation length.
    shock_novelty : float  Novelty magnitude at the shock event.
    """
    rng = np.random.RandomState(seed)
    avg_coupling = 0.1
    history: List[Dict[str, float]] = []

    # Initial baselines
    baselines = [0.1 + rng.uniform(-0.05, 0.15) for _ in range(n_agents)]

    for t in range(steps):
        if t < t_shock:
            avg_coupling = min(0.95, avg_coupling * convergence_rate)
            total_novelty = 0.5 * math.exp(-t * 0.1) + novelty_floor
            # Baselines converge
            mean_b = np.mean(baselines)
            baselines = [b + 0.02 * (mean_b - b) for b in baselines]
        elif t == t_shock:
            total_novelty = shock_novelty
            avg_coupling *= kappa_shock_factor
            # Shock diverges baselines
            baselines = [b + rng.uniform(-0.5, 0.5) for b in baselines]
        else:
            total_novelty = post_shock_base + rng.uniform(0, 1)
            avg_coupling *= kappa_post_decay

        avg_coupling = max(0.0, min(0.95, avg_coupling))
        divergence = float(np.std(baselines))

        history.append(dict(
            time=t,
            avgCoupling=round(avg_coupling, 6),
            totalNovelty=round(total_novelty, 6),
            baselineDivergence=round(divergence, 6),
            **{f"b{i}": round(b, 4) for i, b in enumerate(baselines)},
        ))

    # Snapshots
    pre_shock = history[t_shock - 1]
    post_shock = history[min(t_shock + 5, steps - 1)]
    final = history[-1]

    return dict(
        timeseries=history,
        summary=dict(
            kappa_pre_shock=pre_shock["avgCoupling"],
            kappa_post_shock=post_shock["avgCoupling"],
            kappa_final=final["avgCoupling"],
            divergence_pre=pre_shock["baselineDivergence"],
            divergence_post=post_shock["baselineDivergence"],
            divergence_final=final["baselineDivergence"],
        ),
        params=dict(n_agents=n_agents, t_shock=t_shock, steps=steps,
                    shock_novelty=shock_novelty),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt

    if results is None:
        results = run(**kw)
    ts = results["timeseries"]
    s = results["summary"]
    t_shock = results["params"]["t_shock"]
    n = results["params"]["n_agents"]

    t = [d["time"] for d in ts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    ax1b = ax1.twinx()
    ax1.plot(t, [d["avgCoupling"] for d in ts], label="Avg κ", color="#8884d8")
    ax1b.plot(t, [d["totalNovelty"] for d in ts], label="Total Novelty",
              color="#ff8042", alpha=0.7)
    ax1.axvline(t_shock, color="red", linestyle="--", alpha=0.6, label="External Shock")
    ax1.set(xlabel="Time", ylabel="Avg κ", title="System State Collapse", ylim=(0, 1))
    ax1b.set_ylabel("Total Novelty")
    ax1.legend(loc="upper left")
    ax1b.legend(loc="upper right")

    # Baseline divergence over time
    colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff8042"]
    for i in range(n):
        key = f"b{i}"
        ax2.plot(t, [d[key] for d in ts], color=colors[i % len(colors)],
                 label=f"Agent {i}", alpha=0.8)
    ax2.axvline(t_shock, color="red", linestyle="--", alpha=0.6)
    ax2.set(xlabel="Time", ylabel="Baseline", title="Baseline Divergence & Re-fragmentation")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
