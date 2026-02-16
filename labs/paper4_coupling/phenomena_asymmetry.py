"""κ-Asymmetry — Relational Parasitism

Paper 4: The Coupling Asymmetry (10.5281/zenodo.18519187)

Coupling is not mutual: Agent A (Host) has high κ to Agent B
(Parasite), but B maintains low κ to A.  A's baseline becomes a
distorted reflection of B while B remains independent.  Models
exploitation, narcissistic, and certain empathic dynamics.

Ported from MemoryLab-OS/phenomena.tsx — KAsymmetryModel.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 4
PAPER_TITLE = "The Coupling Asymmetry"
PAPER_DOI = "10.5281/zenodo.18519187"
LAB_TITLE = "Phenomenon: κ-Asymmetry (Relational Parasitism)"

THESIS = (
    "A dyadic pathology where coupling is not mutual: Agent A has high κ to "
    "B, but B maintains low κ to A. This creates a one-way flow of "
    "constitutive influence. A's baseline becomes a distorted reflection of "
    "B, while B's baseline remains independent. Models exploitation and "
    "certain narcissistic/empathic dynamics."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    kappa_host_max: float = 0.90,
    kappa_parasite_base: float = 0.10,
    host_growth_rate: float = 0.05,
    parasite_decay_rate: float = 0.05,
    steps: int = 100,
) -> Dict[str, Any]:
    """Simulate asymmetric coupling dynamics.

    The Host's κ grows toward kappa_host_max; the Parasite's κ decays toward
    kappa_parasite_base.  A sinusoidal signal from the Parasite demonstrates
    how the Host's baseline is dragged along while the Parasite is unaffected.
    """
    history: List[Dict[str, float]] = []

    for t in range(steps):
        k_host = kappa_host_max * (1.0 - math.exp(-host_growth_rate * t))
        k_par = kappa_parasite_base + 0.15 * math.exp(-parasite_decay_rate * t)
        parasite_signal = math.sin(t / 5.0)
        host_baseline = parasite_signal * k_host

        history.append(dict(
            time=t,
            kappa_host=round(k_host, 6),
            kappa_parasite=round(k_par, 6),
            parasite_state=round(parasite_signal, 6),
            host_baseline=round(host_baseline, 6),
        ))

    return dict(
        timeseries=history,
        influence=dict(
            host_to_parasite_shift=round(abs(history[-1]["kappa_parasite"]) * 0.3, 4),
            parasite_to_host_shift=round(abs(history[-1]["kappa_host"]) * 0.85, 4),
        ),
        summary=dict(
            final_kappa_host=history[-1]["kappa_host"],
            final_kappa_parasite=history[-1]["kappa_parasite"],
        ),
        params=dict(
            kappa_host_max=kappa_host_max, kappa_parasite_base=kappa_parasite_base,
            steps=steps,
        ),
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

    ax1.plot(t, [d["kappa_host"] for d in ts], label="κ Host → Parasite", color="#8884d8")
    ax1.plot(t, [d["kappa_parasite"] for d in ts], label="κ Parasite → Host", color="#ff8042")
    ax1.set(xlabel="Time", ylabel="κ",
            title="Asymmetric Coupling Evolution", ylim=(0, 1))
    ax1.legend()

    ax2.plot(t, [d["parasite_state"] for d in ts], label="Parasite's State",
             color="#ff8042")
    ax2.plot(t, [d["host_baseline"] for d in ts], label="Host's Baseline",
             color="#8884d8")
    ax2.set(xlabel="Time", ylabel="Baseline",
            title="Resulting Baseline Constitution", ylim=(-1, 1))
    ax2.legend()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
