"""Dual Resonance — Amplification & Overshadowing

Paper 5: The Emergent Gate (10.5281/zenodo.17344091)

Phenomenon P7: Two coupled agents can *amplify* each other's novelty
(resonant boost) or *overshadow* it (dominant signal drowns weaker).
Whether amplification or overshadowing wins depends on the coupling
geometry — specifically the ratio of κ to baseline divergence.

Extracted from MemoryLab-OS simulation contexts & the Emergent Gate paper.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List

import numpy as np

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 5
PAPER_TITLE = "The Emergent Gate"
PAPER_DOI = "10.5281/zenodo.17344091"
LAB_TITLE = "Phenomenon: Dual Resonance"

THESIS = (
    "When two agents are coupled via κ, the novelty of an event "
    "experienced by one reverberates through the other. If baselines are "
    "sufficiently divergent and coupling is moderate, the resonance *amplifies* "
    "the original novelty signal. If coupling overwhelms baseline identity "
    "(κ → 1), the agents' baselines collapse and novelty is extinguished. "
    "The sweet spot — moderate κ, distinct baselines — is where memory is "
    "most powerfully formed."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    kappa_values: List[float] = None,
    b0_a: float = 0.30,
    b0_b: float = -0.20,
    event_valence: float = 0.80,
    event_time: int = 40,
    lambda_val: float = 0.10,
    steps: int = 80,
) -> Dict[str, Any]:
    """Compare resonance vs overshadowing at different κ levels.

    At t=event_time, Agent A receives a large input.  Agent B does not.
    We measure how Agent B's novelty changes as a function of κ.
    """
    if kappa_values is None:
        kappa_values = [0.0, 0.2, 0.5, 0.8, 1.0]

    trials: Dict[str, List[Dict]] = {}
    peak_novelties: List[Dict] = []

    for kappa in kappa_values:
        bA, bB = b0_a, b0_b
        series: List[Dict[str, float]] = []

        for t in range(steps):
            I_a = event_valence if t == event_time else 0.0
            I_b = 0.0

            n_a = abs(I_a - bA) + kappa * abs(bB - bA) * 0.5
            n_b = kappa * abs(bA - bB) * 0.5

            bA = bA * (1.0 - lambda_val) + (I_a + kappa * bB) / (1.0 + kappa) * lambda_val
            bB = bB * (1.0 - lambda_val) + (I_b + kappa * bA) / (1.0 + kappa) * lambda_val

            series.append(dict(
                time=t, bA=round(bA, 5), bB=round(bB, 5),
                novelty_a=round(n_a, 5), novelty_b=round(n_b, 5),
            ))

        key = f"κ={kappa:.1f}"
        trials[key] = series
        peak_b = max(series, key=lambda d: d["novelty_b"])
        peak_novelties.append(dict(
            kappa=kappa, peak_novelty_b=round(peak_b["novelty_b"], 4),
            peak_novelty_a=round(max(d["novelty_a"] for d in series), 4),
            baseline_convergence=round(abs(series[-1]["bA"] - series[-1]["bB"]), 4),
        ))

    # Find optimal kappa (max secondary novelty)
    best = max(peak_novelties, key=lambda d: d["peak_novelty_b"])

    return dict(
        trials=trials,
        peak_novelties=peak_novelties,
        summary=dict(
            optimal_kappa=best["kappa"],
            peak_secondary_novelty=best["peak_novelty_b"],
            overshadow_kappa=1.0,
            resonant_range=[0.2, 0.5],
        ),
        params=dict(
            kappa_values=kappa_values, b0_a=b0_a, b0_b=b0_b,
            event_valence=event_valence, event_time=event_time,
            lambda_val=lambda_val, steps=steps,
        ),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt

    if results is None:
        results = run(**kw)
    pn = results["peak_novelties"]
    trials = results["trials"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    # Secondary novelty vs kappa
    ax = axes[0]
    kappas = [d["kappa"] for d in pn]
    sec_n = [d["peak_novelty_b"] for d in pn]
    ax.plot(kappas, sec_n, "o-", color="#8884d8", markersize=8)
    best_k = results["summary"]["optimal_kappa"]
    ax.axvline(best_k, color="#ff8042", linestyle="--", alpha=0.6,
               label=f"Optimal κ = {best_k}")
    ax.set(xlabel="κ", ylabel="Peak Novelty (Agent B)",
           title="Resonance vs. Overshadowing")
    ax.legend()

    # Baseline convergence vs kappa
    ax = axes[1]
    conv = [d["baseline_convergence"] for d in pn]
    ax.plot(kappas, conv, "o-", color="#82ca9d", markersize=8)
    ax.set(xlabel="κ", ylabel="Final |B_A - B_B|",
           title="Baseline Convergence (Identity Loss)")

    # Time series at optimal kappa
    ax = axes[2]
    key = f"κ={best_k:.1f}"
    if key in trials:
        s = trials[key]
        t = [d["time"] for d in s]
        ax.plot(t, [d["novelty_a"] for d in s], label="Agent A Novelty",
                color="#ff8042", alpha=0.8)
        ax.plot(t, [d["novelty_b"] for d in s], label="Agent B Novelty",
                color="#8884d8", alpha=0.8)
        ax.set(xlabel="Time", ylabel="Novelty",
               title=f"Novelty at Optimal κ = {best_k}")
        ax.legend()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
