"""Borderline Personality (BPD) — Coupling Instability

Paper 4: The Coupling Asymmetry (10.5281/zenodo.18519187)

κ oscillates wildly between near-total fusion (idealisation) and
near-zero (devaluation).  The baseline is volatile and the sense of
self fractured, because the very constitution of 'being' is in flux.

Ported from MemoryLab-OS/phenomena.tsx — BPDModel.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

import numpy as np

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 4
PAPER_TITLE = "The Coupling Asymmetry"
PAPER_DOI = "10.5281/zenodo.18519187"
LAB_TITLE = "Phenomenon: Borderline Personality (BPD)"

THESIS = (
    "BPD is modelled as a pathology of coupling dynamics. The coupling "
    "parameter, κ, is highly unstable, oscillating between near-total "
    "fusion (idealisation) and near-zero (devaluation) in response to "
    "perceived relational threats. This creates a volatile baseline and a "
    "fractured, unstable sense of self, as the very constitution of 'being' "
    "is in constant flux."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    kappa_typical: float = 0.70,
    kappa_schedule: Optional[List[Dict]] = None,
    lambda_val: float = 0.10,
    steps: int = 71,
) -> Dict[str, Any]:
    """Simulate BPD coupling instability.

    Parameters
    ----------
    kappa_typical : float   Stable κ for the control agent.
    kappa_schedule : list   [{time: int, kappa: float}] BPD κ keyframes.
                            Defaults to the canonical idealisation/devaluation cycle.
    lambda_val : float      Baseline decay rate.
    steps : int             Simulation length.
    """
    if kappa_schedule is None:
        kappa_schedule = [
            {"time": 0, "kappa": 0.50}, {"time": 10, "kappa": 0.80},
            {"time": 20, "kappa": 0.85}, {"time": 30, "kappa": 0.20},
            {"time": 40, "kappa": 0.10}, {"time": 50, "kappa": 0.90},
            {"time": 60, "kappa": 0.30}, {"time": 70, "kappa": 0.70},
        ]

    sched_map = {e["time"]: e["kappa"] for e in kappa_schedule}
    last_kappa_bpd = kappa_schedule[0]["kappa"]

    bpd_b = 0.0
    typ_b = 0.0
    history: List[Dict[str, float]] = []

    for t in range(steps):
        if t in sched_map:
            last_kappa_bpd = sched_map[t]

        D = math.sin(t / 10.0)  # External relational signal

        prev_bpd = bpd_b
        bpd_b = bpd_b * (1.0 - lambda_val) + lambda_val * (D * last_kappa_bpd)
        bpd_db = abs(bpd_b - prev_bpd)

        prev_typ = typ_b
        typ_b = typ_b * (1.0 - lambda_val) + lambda_val * (D * kappa_typical)
        typ_db = abs(typ_b - prev_typ)

        history.append(dict(
            time=t,
            kappa_bpd=round(last_kappa_bpd, 4),
            kappa_typical=round(kappa_typical, 4),
            bpd_dB=round(bpd_db, 6),
            typical_dB=round(typ_db, 6),
            bpd_baseline=round(bpd_b, 6),
            typical_baseline=round(typ_b, 6),
        ))

    kappas = [d["kappa_bpd"] for d in history]
    return dict(
        timeseries=history,
        summary=dict(
            kappa_mean=round(float(np.mean(kappas)), 4),
            kappa_std=round(float(np.std(kappas)), 4),
            kappa_range=round(max(kappas) - min(kappas), 4),
            avg_bpd_dB=round(float(np.mean([d["bpd_dB"] for d in history])), 6),
            avg_typical_dB=round(float(np.mean([d["typical_dB"] for d in history])), 6),
        ),
        params=dict(kappa_typical=kappa_typical, lambda_val=lambda_val, steps=steps),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt

    if results is None:
        results = run(**kw)
    ts = results["timeseries"]

    t = [d["time"] for d in ts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    ax1.plot(t, [d["kappa_bpd"] for d in ts], "o-", markersize=4,
             color="#ff4d4d", label="BPD κ(t)")
    ax1.axhline(results["params"]["kappa_typical"], color="#ccc",
                linestyle="--", label="Typical κ")
    ax1.set(xlabel="Time", ylabel="κ", title="Coupling (κ) Instability", ylim=(0, 1))
    ax1.legend()

    ax2.plot(t, [d["bpd_dB"] for d in ts], color="#ff4d4d", label="BPD |dB/dt|")
    ax2.plot(t, [d["typical_dB"] for d in ts], color="#ccc", label="Typical |dB/dt|")
    ax2.set(xlabel="Time", ylabel="Rate of Change", title="Resulting Self-Process (dB/dt)")
    ax2.legend()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
