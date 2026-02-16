"""Strategic Baseline Simplification (SBS)

Paper 1: Memory as Baseline Deviation (10.5281/zenodo.17381536)

Models a cognitive coping mechanism: faced with chronic high-error
environments, the agent raises λ to minimise prediction error at
the cost of long-term adaptation capacity.

Ported from MemoryLab-OS/phenomena.tsx — SBSModel.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 1
PAPER_TITLE = "Memory as Baseline Deviation"
PAPER_DOI = "10.5281/zenodo.17381536"
LAB_TITLE = "Phenomenon: Strategic Baseline Simplification (SBS)"

THESIS = (
    "SBS is a cognitive coping mechanism in which an agent, faced with a "
    "chronically high-error environment (e.g., traumatic, chaotic, or "
    "invalidating), intentionally 'dumbs down' its own working baseline to "
    "minimise prediction error and reduce metabolic cost. This is achieved "
    "by increasing the baseline's decay rate (λ) and reducing its "
    "dimensionality. While this provides short-term relief, it "
    "catastrophically sabotages the 'long-slow' process of deep baseline "
    "integration, preventing long-term adaptation and growth."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    error_rise_rate: float = 0.10,
    error_floor: float = 0.10,
    error_ceil: float = 0.80,
    lambda_base: float = 0.05,
    lambda_scale: float = 0.70,
    steps: int = 100,
) -> Dict[str, Any]:
    """Simulate the vicious cycle of SBS.

    Chronic error rises → λ increases → dB/dt collapses → plasticity dies.
    """
    history: List[Dict[str, float]] = []

    for t in range(steps):
        error = error_ceil * (1.0 - math.exp(-error_rise_rate * t)) + error_floor
        lam = lambda_base + lambda_scale * (error - error_floor)
        db_dt = 1.0 / (1.0 + lam * 5.0)
        history.append(dict(
            time=t,
            chronic_error=round(error, 6),
            lambda_val=round(lam, 6),
            dB_dt=round(db_dt, 6),
        ))

    p_write_high = min(1.0, 1.0 / (1.0 + history[0]["lambda_val"] * 5.0))
    p_write_sbs = min(1.0, 1.0 / (1.0 + history[-1]["lambda_val"] * 5.0))

    return dict(
        timeseries=history,
        summary=dict(
            lambda_early=round(history[0]["lambda_val"], 4),
            lambda_late=round(history[-1]["lambda_val"], 4),
            dB_dt_early=round(history[0]["dB_dt"], 4),
            dB_dt_late=round(history[-1]["dB_dt"], 4),
            p_write_healthy=round(p_write_high, 4),
            p_write_sbs=round(p_write_sbs, 4),
        ),
        params=dict(
            error_rise_rate=error_rise_rate, error_floor=error_floor,
            error_ceil=error_ceil, lambda_base=lambda_base,
            lambda_scale=lambda_scale, steps=steps,
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

    ax1.plot(t, [d["lambda_val"] for d in ts], label="λ (Decay Rate)", color="#ff8042")
    ax1.plot(t, [d["dB_dt"] for d in ts], label="dB/dt (Ontological Shift)", color="#8884d8")
    ax1.set(xlabel="Lifespan in High-Error Environment", ylabel="Rates",
            title="The Vicious Cycle")
    ax1.legend()

    labels = ["High Plasticity\n(Low λ)", "SBS State\n(High λ)"]
    values = [s["p_write_healthy"], s["p_write_sbs"]]
    colors = ["#82ca9d", "#ff8042"]
    ax2.bar(labels, values, color=colors, width=0.5)
    ax2.set(ylabel="P(WRITE)", title="Resulting Plasticity Collapse", ylim=(0, 1.1))
    for i, v in enumerate(values):
        ax2.text(i, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
