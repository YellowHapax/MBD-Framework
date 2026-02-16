"""Phantom Limb Syndrome — Unquenched Prediction Error

Paper 1: Memory as Baseline Deviation (10.5281/zenodo.17381536)

The integrated limb baseline persists after amputation.  The gap
between the high baseline and null sensory input produces a large,
persistent prediction error experienced as phantom pain.

Ported from MemoryLab-OS/phenomena.tsx — PhantomLimbModel.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 1
PAPER_TITLE = "Memory as Baseline Deviation"
PAPER_DOI = "10.5281/zenodo.17381536"
LAB_TITLE = "Phenomenon: Phantom Limb Syndrome"

THESIS = (
    "The persistent baseline representation of the limb, integrated over a "
    "lifetime, creates a powerful prediction. When sensory input ceases "
    "(amputation), the resulting large, unquenched prediction error between "
    "the high baseline and null input is experienced as the phantom "
    "sensation. Pain is the raw signal of this drastic baseline-input "
    "mismatch."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    baseline_limb: float = 0.80,
    input_pre: float = 0.80,
    input_post: float = 0.00,
    t_amputation: int = 50,
    steps: int = 100,
) -> Dict[str, Any]:
    """Simulate phantom limb prediction error.

    Parameters
    ----------
    baseline_limb : float  Long-term integrated limb baseline.
    input_pre : float      Sensory input while limb exists.
    input_post : float     Sensory input after amputation (≈0).
    t_amputation : int     Timestep of amputation.
    steps : int            Total simulation length.
    """
    history: List[Dict[str, float]] = []

    for t in range(steps):
        inp = input_pre if t < t_amputation else input_post
        error = abs(baseline_limb - inp)
        history.append(dict(
            time=t,
            baseline=round(baseline_limb, 4),
            input=round(inp, 4),
            prediction_error=round(error, 4),
        ))

    return dict(
        timeseries=history,
        summary=dict(
            baseline=baseline_limb,
            error_pre=round(abs(baseline_limb - input_pre), 4),
            error_post=round(abs(baseline_limb - input_post), 4),
            t_amputation=t_amputation,
        ),
        params=dict(
            baseline_limb=baseline_limb, input_pre=input_pre,
            input_post=input_post, t_amputation=t_amputation,
            steps=steps,
        ),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt

    if results is None:
        results = run(**kw)
    ts = results["timeseries"]
    s = results["summary"]

    t = [d["time"] for d in ts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    ax1.plot(t, [d["baseline"] for d in ts], label="Limb Baseline (B)", color="#8884d8")
    ax1.plot(t, [d["input"] for d in ts], label="Sensory Input (I)",
             color="#82ca9d", drawstyle="steps-post")
    ax1.plot(t, [d["prediction_error"] for d in ts], label="Prediction Error (Pain)",
             color="#ff8042", drawstyle="steps-post")
    ax1.axvline(s["t_amputation"], color="red", linestyle="--", alpha=0.6, label="Amputation")
    ax1.set(xlabel="Time", ylabel="Signal", title="Temporal Mismatch (Pain Signal)",
            ylim=(-0.05, 1.05))
    ax1.legend()

    labels = ["Pre-Amputation", "Post-Amputation"]
    long_term = [s["baseline"], s["baseline"]]
    working = [s["baseline"], s["baseline"] - s["error_post"]]
    x = range(len(labels))
    w = 0.35
    ax2.bar([i - w / 2 for i in x], long_term, w, label="B_long (The 'Slow')", color="#8884d8")
    ax2.bar([i + w / 2 for i in x], [s["baseline"], s["error_post"]],
            w, label="B_work / Error", color="#ffc658")
    ax2.set(xticks=list(x), xticklabels=labels, ylabel="Value",
            title='Baseline Divergence ("Encoding Error")', ylim=(0, 1.1))
    ax2.legend()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
