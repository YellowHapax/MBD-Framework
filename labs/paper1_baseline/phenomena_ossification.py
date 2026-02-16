"""Baseline Ossification — Fossilised Predictive Scaffold

Paper 1: Memory as Baseline Deviation (10.5281/zenodo.17381536)

Models a pathological state where λ → 0 over the lifespan,
fossilising the baseline.  New experiences fail to shift the
integral.  The agent becomes a perfect predictor of the past
but incapable of genuine learning.

Ported from MemoryLab-OS/phenomena.tsx — BaselineOssificationModel.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 1
PAPER_TITLE = "Memory as Baseline Deviation"
PAPER_DOI = "10.5281/zenodo.17381536"
LAB_TITLE = "Phenomenon: Baseline Ossification"

THESIS = (
    "A pathological state where an agent's baseline decay rate (λ) approaches "
    "zero, 'fossilising' the baseline. New experiences fail to produce a "
    "meaningful gradient to alter the integral. The agent becomes a perfect "
    "predictor of their past but is incapable of genuine learning. The rate "
    "of change of the self (dB/dt) approaches zero."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    lambda_initial: float = 0.50,
    lambda_floor: float = 0.01,
    decay_rate: float = 0.05,
    constant_novelty: float = 10.0,
    steps: int = 100,
) -> Dict[str, Any]:
    """Simulate baseline ossification.

    Parameters
    ----------
    lambda_initial : float  Starting plasticity.
    lambda_floor : float    Minimum λ value.
    decay_rate : float      Exponential decay constant for λ.
    constant_novelty : float  Assumed environmental novelty (held constant
                              to isolate the λ effect).
    steps : int             Number of timesteps.

    Returns
    -------
    dict with timeseries (time, lambda, dB_dt), summary, params.
    """
    history: List[Dict[str, float]] = []

    for t in range(steps):
        lam = lambda_initial * math.exp(-decay_rate * t) + lambda_floor
        db_dt = constant_novelty * lam
        history.append(dict(
            time=t,
            lambda_val=round(lam, 6),
            dB_dt=round(db_dt, 6),
        ))

    p_write_early = min(1.0, history[0]["lambda_val"] * 2.0)
    p_write_late = min(1.0, history[-1]["lambda_val"] * 2.0)

    return dict(
        timeseries=history,
        summary=dict(
            lambda_early=round(history[0]["lambda_val"], 4),
            lambda_late=round(history[-1]["lambda_val"], 4),
            dB_dt_early=round(history[0]["dB_dt"], 4),
            dB_dt_late=round(history[-1]["dB_dt"], 4),
            p_write_early=round(p_write_early, 4),
            p_write_late=round(p_write_late, 4),
        ),
        params=dict(
            lambda_initial=lambda_initial, lambda_floor=lambda_floor,
            decay_rate=decay_rate, constant_novelty=constant_novelty,
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

    ax1b = ax1.twinx()
    ax1.plot(t, [d["lambda_val"] for d in ts], label="λ (Decay Rate)", color="#8884d8")
    ax1b.plot(t, [d["dB_dt"] for d in ts], label="dB/dt (Ontological Shift)",
              color="#ff8042", linestyle="--")
    ax1.set(xlabel="Lifespan", ylabel="λ Rate", title="λ Decay & Ontological Shift")
    ax1b.set_ylabel("dB/dt")
    ax1.legend(loc="upper right")
    ax1b.legend(loc="center right")

    labels = ["Early Life\n(High λ)", "Late Life\n(Low λ)"]
    values = [s["p_write_early"], s["p_write_late"]]
    colors = ["#82ca9d", "#ff8042"]
    ax2.bar(labels, values, color=colors, width=0.5)
    ax2.set(ylabel="P(WRITE)", title="Resulting Plasticity", ylim=(0, 1.1))
    for i, v in enumerate(values):
        ax2.text(i, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
