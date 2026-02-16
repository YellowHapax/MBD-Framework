"""Bipolar Disorder — Dysregulated Integration Rate

Paper 1: Memory as Baseline Deviation (10.5281/zenodo.17381536)

Models bipolar disorder as a dysregulation of the integration
constant λ.  Manic phases reflect pathologically low λ for positive
input (runaway integration).  Depressive phases reflect high λ for
positive input and low λ for negative input, creating a 'sink' state.

Ported from MemoryLab-OS/phenomena.tsx — BipolarModel.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 1
PAPER_TITLE = "Memory as Baseline Deviation"
PAPER_DOI = "10.5281/zenodo.17381536"
LAB_TITLE = "Phenomenon: Bipolar Disorder"

THESIS = (
    "Bipolar disorder is framed as a dysregulation of the baseline's "
    "temporal integration constant, λ. Manic phases reflect a "
    "pathologically low λ, causing runaway integration of positive valence. "
    "Depressive phases reflect a high λ for positive input and low λ for "
    "negative, creating a 'sink' state. The cycling is a chaotic oscillation "
    "of the self's metabolic rate."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    steps: int = 200,
    manic_start: int = 20,
    manic_end: int = 80,
    depressive_start: int = 120,
    depressive_end: int = 180,
    rise_rate: float = 0.05,
    fall_rate: float = 0.05,
    return_rate: float = 0.025,
) -> Dict[str, Any]:
    """Simulate bipolar affective baseline cycling.

    The model generates a characteristic manic-euthymic-depressive cycle
    by directly driving the baseline trajectory and pairing it with
    asymmetric λ values per affective phase.
    """
    history: List[Dict[str, float]] = []

    for t in range(steps):
        # Piecewise baseline (matches original TSX model)
        if manic_start < t <= manic_end:
            baseline = min(1.0, rise_rate * (t - manic_start))
        elif manic_end < t <= depressive_start:
            baseline = max(0.0, 1.0 - return_rate * (t - manic_end))
        elif depressive_start < t <= depressive_end:
            baseline = max(-1.0, -fall_rate * (t - depressive_start))
        else:
            baseline = 0.0

        history.append(dict(
            time=t,
            baseline=round(baseline, 6),
        ))

    # Lambda asymmetry table
    lambda_table = [
        dict(phase="Mania", lambda_pos=0.05, lambda_neg=0.20),
        dict(phase="Euthymia", lambda_pos=0.20, lambda_neg=0.20),
        dict(phase="Depression", lambda_pos=0.50, lambda_neg=0.05),
    ]

    return dict(
        timeseries=history,
        lambda_table=lambda_table,
        summary=dict(
            peak_mania=round(max(d["baseline"] for d in history), 4),
            trough_depression=round(min(d["baseline"] for d in history), 4),
        ),
        params=dict(steps=steps, manic_start=manic_start, manic_end=manic_end,
                    depressive_start=depressive_start, depressive_end=depressive_end),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt
    import numpy as np

    if results is None:
        results = run(**kw)
    ts = results["timeseries"]
    lt = results["lambda_table"]

    t = [d["time"] for d in ts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    ax1.plot(t, [d["baseline"] for d in ts], color="#4a90e2", label="Affective Baseline")
    ax1.axhline(0, color="#ccc", linestyle="--", alpha=0.5, label="Euthymia")
    ax1.set(xlabel="Time (t)", ylabel="Affective Baseline",
            title="Affective Baseline Cycling", ylim=(-1.2, 1.2))
    ax1.legend()

    phases = [d["phase"] for d in lt]
    lp = [d["lambda_pos"] for d in lt]
    ln = [d["lambda_neg"] for d in lt]
    x = np.arange(len(phases))
    w = 0.35
    ax2.bar(x - w / 2, lp, w, label="λ (Positive Input)", color="#82ca9d")
    ax2.bar(x + w / 2, ln, w, label="λ (Negative Input)", color="#ff8042")
    ax2.set(xticks=x, xticklabels=phases, ylabel="Decay Rate λ",
            title="Dysregulation of Integration Rate (λ)")
    ax2.legend()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
