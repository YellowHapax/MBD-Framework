"""ADHD Phenomenon — High Baseline Plasticity

Paper 1: Memory as Baseline Deviation (10.5281/zenodo.17381536)

Models ADHD as an excessively high decay rate (λ).  The predictive
scaffold never stabilises, so most inputs are processed as highly
novel, producing perpetual distractibility and epistemic foraging.

Ported from MemoryLab-OS/phenomena.tsx — ADHDModel.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 1
PAPER_TITLE = "Memory as Baseline Deviation"
PAPER_DOI = "10.5281/zenodo.17381536"
LAB_TITLE = "Phenomenon: ADHD (High-λ Baseline Decay)"

THESIS = (
    "ADHD can be modelled as a working baseline with an excessively high "
    "decay rate (λ). The predictive scaffold required to 'quiet' incoming "
    "sensory data never stabilises. Consequently, most inputs are processed "
    "as highly novel, leading to perpetual distractibility. The system is "
    "trapped in a state of high epistemic foraging."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    lambda_typical: float = 0.10,
    lambda_adhd: float = 0.80,
    adhd_extra_decay: float = 0.85,
    input_events: Optional[List[Dict]] = None,
    steps: int = 100,
) -> Dict[str, Any]:
    """Compare a typical agent with an ADHD-model agent.

    Parameters
    ----------
    lambda_typical : float   Typical baseline plasticity.
    lambda_adhd : float      ADHD baseline plasticity (high).
    adhd_extra_decay : float Multiplicative decay applied per step to ADHD baseline.
    input_events : list      [{t: int, v: float}] sensory events.  Defaults to
                             sparse bursts at t=20,40,60,80.
    steps : int              Simulation length.

    Returns
    -------
    dict with timeseries, summary, params.
    """
    if input_events is None:
        input_events = [
            {"t": 20, "v": 1.0},
            {"t": 40, "v": 0.8},
            {"t": 60, "v": 1.2},
            {"t": 80, "v": 0.9},
        ]
    event_map = {e["t"]: e["v"] for e in input_events}

    typical_b = 0.0
    adhd_b = 0.0
    history: List[Dict[str, float]] = []

    for i in range(steps):
        inp = event_map.get(i, 0.0)
        prev_typical = typical_b
        prev_adhd = adhd_b

        typical_b = typical_b * (1.0 - lambda_typical) + inp * lambda_typical
        adhd_b = adhd_b * (1.0 - lambda_adhd) + inp * lambda_adhd
        adhd_b *= adhd_extra_decay

        novelty_t = abs(inp - prev_typical) if inp else 0.0
        novelty_a = abs(inp - prev_adhd) if inp else 0.0

        history.append(dict(
            time=i, input=inp,
            typical=round(typical_b, 6),
            adhd=round(adhd_b, 6),
            novelty_typical=round(novelty_t, 6),
            novelty_adhd=round(novelty_a, 6),
        ))

    # Summary: average novelty for event steps only
    event_steps = [h for h in history if h["input"] != 0]
    avg_novelty_t = np.mean([h["novelty_typical"] for h in event_steps]) if event_steps else 0
    avg_novelty_a = np.mean([h["novelty_adhd"] for h in event_steps]) if event_steps else 0

    return dict(
        timeseries=history,
        summary=dict(
            avg_novelty_typical=round(float(avg_novelty_t), 4),
            avg_novelty_adhd=round(float(avg_novelty_a), 4),
            novelty_ratio=round(float(avg_novelty_a / avg_novelty_t), 2) if avg_novelty_t > 0 else float("inf"),
        ),
        params=dict(
            lambda_typical=lambda_typical, lambda_adhd=lambda_adhd,
            adhd_extra_decay=adhd_extra_decay, steps=steps,
        ),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results: Optional[Dict[str, Any]] = None, **run_kwargs):
    import matplotlib.pyplot as plt

    if results is None:
        results = run(**run_kwargs)
    ts = results["timeseries"]

    t = [d["time"] for d in ts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    # Baseline comparison
    ax1.plot(t, [d["typical"] for d in ts], label="Typical (low λ)", color="#8884d8")
    ax1.plot(t, [d["adhd"] for d in ts], label="ADHD (high λ)", color="#ff8042")
    ax1.set(xlabel="Time", ylabel="Baseline", title="Baseline Decay Comparison")
    ax1.legend()

    # Novelty bars
    event_ts = [d for d in ts if d["input"] != 0]
    x = [d["time"] for d in event_ts]
    w = 1.5
    ax2.bar([xi - w / 2 for xi in x], [d["novelty_typical"] for d in event_ts],
            width=w, label="Typical Novelty", color="#8884d8", alpha=0.7)
    ax2.bar([xi + w / 2 for xi in x], [d["novelty_adhd"] for d in event_ts],
            width=w, label="ADHD Novelty", color="#ff8042", alpha=0.7)
    ax2.set(xlabel="Time", ylabel="Novelty (N)", title="Resulting Novelty Signal")
    ax2.legend()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
