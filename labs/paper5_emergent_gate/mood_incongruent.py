"""Mood-Incongruent Memory — Emergent Novelty Amplification

Paper 5: The Emergent Gate (10.5281/zenodo.17344091)

Phenomenon P6: A sad baseline + happy event → unexpectedly high
novelty → strong encoding.  The gate opens widest when the event
deviates from the current affective baseline, not from objective
neutrality.  This reversal of naive novelty prediction demonstrates
that MBD's novelty is *relative to the agent's state*.

Extracted from MemoryLab-OS/phenomena.tsx & simulationLab.tsx contexts.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 5
PAPER_TITLE = "The Emergent Gate"
PAPER_DOI = "10.5281/zenodo.17344091"
LAB_TITLE = "Phenomenon: Mood-Incongruent Memory"

THESIS = (
    "Mood-Incongruent Memory formation. In standard encoding models, "
    "a happy event during a happy mood is 'expected' and barely encoded. "
    "MBD predicts the reverse: a happy event experienced from a depressed "
    "baseline produces enormous novelty, because the deviation from *what "
    "the agent is* is maximal. The gate opens widest at maximum incongruence."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    baseline_sad: float = -0.70,
    baseline_neutral: float = 0.00,
    baseline_happy: float = 0.70,
    event_valence: float = 0.80,
    theta_h: float = 0.35,
    steps: int = 100,
    event_time: int = 50,
    lambda_val: float = 0.08,
) -> Dict[str, Any]:
    """Compare encoding of the same event across three mood baselines.

    Parameters
    ----------
    baseline_sad, baseline_neutral, baseline_happy : float
        Starting baselines for three parallel agents.
    event_valence : float
        Value of the mood-incongruent event.
    theta_h : float
        Novelty threshold for P(WRITE) sigmoid.
    """
    agents = [
        dict(name="Sad Agent", b0=baseline_sad),
        dict(name="Neutral Agent", b0=baseline_neutral),
        dict(name="Happy Agent", b0=baseline_happy),
    ]
    all_series: Dict[str, List[Dict]] = {}
    comparison: List[Dict] = []

    for ag in agents:
        b = ag["b0"]
        series: List[Dict[str, float]] = []
        for t in range(steps):
            I = event_valence if t == event_time else 0.0
            novelty = abs(I - b)
            p_write = 1.0 / (1.0 + math.exp(-10.0 * (novelty - theta_h)))
            prev_b = b
            b = b * (1.0 - lambda_val) + I * lambda_val
            series.append(dict(
                time=t, baseline=round(b, 5), input=round(I, 4),
                novelty=round(novelty, 5), p_write=round(p_write, 5),
            ))

        peak = max(series, key=lambda d: d["novelty"])
        comparison.append(dict(
            agent=ag["name"], b0=ag["b0"],
            peak_novelty=peak["novelty"], peak_p_write=peak["p_write"],
        ))
        all_series[ag["name"]] = series

    return dict(
        series=all_series,
        comparison=comparison,
        summary=dict(
            sad_novelty=comparison[0]["peak_novelty"],
            neutral_novelty=comparison[1]["peak_novelty"],
            happy_novelty=comparison[2]["peak_novelty"],
            incongruence_boost=round(
                comparison[0]["peak_novelty"] - comparison[2]["peak_novelty"], 4),
        ),
        params=dict(
            baselines=[baseline_sad, baseline_neutral, baseline_happy],
            event_valence=event_valence, theta_h=theta_h,
            event_time=event_time, lambda_val=lambda_val, steps=steps,
        ),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt
    import numpy as np

    if results is None:
        results = run(**kw)
    comp = results["comparison"]
    series = results["series"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    # Bar comparison
    ax = axes[0]
    names = [c["agent"] for c in comp]
    novelties = [c["peak_novelty"] for c in comp]
    colors = ["#ff8042", "#ccc", "#82ca9d"]
    ax.bar(names, novelties, color=colors)
    ax.set(ylabel="Peak Novelty", title="Novelty at Event")
    ax.axhline(results["params"]["theta_h"], color="red", linestyle="--",
               alpha=0.5, label=f"θ_h = {results['params']['theta_h']}")
    ax.legend()

    # P(WRITE) comparison
    ax = axes[1]
    pw = [c["peak_p_write"] for c in comp]
    ax.bar(names, pw, color=colors)
    ax.set(ylabel="P(WRITE)", title="Encoding Probability", ylim=(0, 1.1))

    # Time-series: novelty around event
    ax = axes[2]
    et = results["params"]["event_time"]
    window = range(max(0, et - 5), min(results["params"]["steps"], et + 15))
    for ag_name, color in zip(series.keys(), colors):
        s = series[ag_name]
        ax.plot([s[t]["time"] for t in window],
                [s[t]["novelty"] for t in window],
                "o-", label=ag_name, color=color, markersize=4)
    ax.set(xlabel="Time", ylabel="Novelty",
           title="Novelty Window Around Event")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
