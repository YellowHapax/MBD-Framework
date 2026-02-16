"""EQ Lab — Relational Learning Rate

Paper 1: Memory as Baseline Deviation (10.5281/zenodo.17381536)

Two-agent baseline convergence simulation.  Demonstrates how
baseline plasticity (λ), coupling learning rate (α), and
initial baseline separation (B₀) interact to produce relational
convergence or divergence.

Ported from MemoryLab-OS/eqLab.tsx (361 lines).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

import numpy as np

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 1
PAPER_TITLE = "Memory as Baseline Deviation"
PAPER_DOI = "10.5281/zenodo.17381536"
LAB_TITLE = "EQ Lab: Relational Learning Rate"

THESIS = (
    "Two agents with distinct baseline plasticity (λ), coupling learning "
    "rate (α), and initial baselines (B₀) interact over time.  As mutual "
    "prediction error decreases, coupling (κ) grows, pulling the baselines "
    "toward convergence.  The lab reveals how λ and α govern the speed and "
    "stability of relational learning."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER,
        paper_title=PAPER_TITLE,
        paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE,
        thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    lambda_a: float = 0.10,
    lambda_b: float = 0.10,
    alpha_a: float = 0.10,
    alpha_b: float = 0.10,
    b0_a: float = 0.80,
    b0_b: float = -0.80,
    beta_decay: float = 0.05,
    steps: int = 150,
) -> Dict[str, Any]:
    """Run the two-agent EQ simulation.

    Parameters
    ----------
    lambda_a, lambda_b : float
        Baseline plasticity for each agent (0.01–0.5).
    alpha_a, alpha_b : float
        Coupling learning rate (0.01–0.5).
    b0_a, b0_b : float
        Initial baselines (-1 to 1).
    beta_decay : float
        Coupling decay rate.
    steps : int
        Number of simulation turns.

    Returns
    -------
    dict with keys:
        timeseries : list[dict]   per-turn {turn, kappa, mutualError, baselineA, baselineB}
        summary    : dict         final state + turns to converge
        params     : dict         echo of input parameters
    """
    b_a = float(b0_a)
    b_b = float(b0_b)
    kappa = 0.0
    turns_to_converge = -1

    history: List[Dict[str, float]] = [
        dict(turn=0, kappa=kappa, mutualError=abs(b_a - b_b),
             baselineA=b_a, baselineB=b_b),
    ]

    for i in range(1, steps + 1):
        mutual_error = abs(b_a - b_b) / 2.0

        if turns_to_converge == -1 and mutual_error < 0.05:
            turns_to_converge = i

        avg_alpha = (alpha_a + alpha_b) / 2.0
        d_kappa = avg_alpha * (1.0 - mutual_error ** 2) - beta_decay * kappa
        kappa = max(0.0, min(1.0, kappa + d_kappa))

        prev_a, prev_b = b_a, b_b
        b_a = prev_a * (1.0 - lambda_a) + (prev_b * kappa) * lambda_a
        b_b = prev_b * (1.0 - lambda_b) + (prev_a * kappa) * lambda_b

        history.append(dict(
            turn=i,
            kappa=round(kappa, 6),
            mutualError=round(abs(b_a - b_b), 6),
            baselineA=round(b_a, 6),
            baselineB=round(b_b, 6),
        ))

    final = history[-1]
    summary = dict(
        finalKappa=final["kappa"],
        finalBaselineA=final["baselineA"],
        finalBaselineB=final["baselineB"],
        finalError=final["mutualError"],
        turnsToConverge=turns_to_converge if turns_to_converge != -1 else steps,
    )

    return dict(
        timeseries=history,
        summary=summary,
        params=dict(
            lambda_a=lambda_a, lambda_b=lambda_b,
            alpha_a=alpha_a, alpha_b=alpha_b,
            b0_a=b0_a, b0_b=b0_b,
            beta_decay=beta_decay, steps=steps,
        ),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results: Optional[Dict[str, Any]] = None, **run_kwargs):
    """Generate a 2×2 matplotlib figure: state-space, baselines, κ+error, summary."""
    import matplotlib.pyplot as plt

    if results is None:
        results = run(**run_kwargs)
    ts = results["timeseries"]
    s = results["summary"]

    turns = [d["turn"] for d in ts]
    ba = [d["baselineA"] for d in ts]
    bb = [d["baselineB"] for d in ts]
    kappas = [d["kappa"] for d in ts]
    errors = [d["mutualError"] for d in ts]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    # State-space scatter
    ax = axes[0, 0]
    ax.plot(ba, bb, "o-", markersize=2, alpha=0.6, color="#daa520")
    ax.plot([-1, 1], [-1, 1], "--", color="grey", alpha=0.4)
    ax.set(xlabel="Baseline A", ylabel="Baseline B",
           title="State-Space Convergence", xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    ax.set_aspect("equal")

    # Baselines over time
    ax = axes[0, 1]
    ax.plot(turns, ba, label="Agent A", color="#4a90e2")
    ax.plot(turns, bb, label="Agent B", color="#ff8042")
    ax.set(xlabel="Turn", ylabel="Baseline", title="Baseline Convergence",
           ylim=(-1.1, 1.1))
    ax.legend()

    # Kappa and error
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(turns, kappas, label="κ", color="#8884d8")
    ax2.plot(turns, errors, label="Error", color="#ff4d4d", alpha=0.7)
    ax.set(xlabel="Turn", ylabel="κ (Coupling)", title="Coupling & Error")
    ax2.set_ylabel("Mutual Error")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Summary text
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = (
        f"Final κ:  {s['finalKappa']:.4f}\n"
        f"Final B_A:  {s['finalBaselineA']:.4f}\n"
        f"Final B_B:  {s['finalBaselineB']:.4f}\n"
        f"Final Error:  {s['finalError']:.4f}\n"
        f"Turns to Converge:  {s['turnsToConverge']}"
    )
    ax.text(0.1, 0.5, summary_text, fontsize=13, family="monospace",
            verticalalignment="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Summary")

    fig.tight_layout()
    return fig


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig = plot()
    plt.show()
