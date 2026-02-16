"""Baseline Fragmentation — Dissociative Multiple Baselines

Paper 4: The Coupling Asymmetry (10.5281/zenodo.18519187)

Models DID / dissociative amnesia: the system partitions into multiple
semi-independent baselines (B₁, B₂).  Traces encoded while B₁ is
active are inaccessible when B₂ is active.  κ becomes state-dependent.

Ported from MemoryLab-OS/phenomena.tsx — BaselineFragmentationModel.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 4
PAPER_TITLE = "The Coupling Asymmetry"
PAPER_DOI = "10.5281/zenodo.18519187"
LAB_TITLE = "Phenomenon: Baseline Fragmentation"

THESIS = (
    "A trauma-response model where the system partitions itself into "
    "multiple, semi-independent baselines (B₁, B₂, ...). Each baseline "
    "integrates a different class of experience. Traces encoded while B₁ is "
    "active are inaccessible when B₂ is active, modelling dissociative "
    "amnesia. Coupling (κ) becomes state-dependent, explaining different "
    "relational styles."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(**_kw) -> Dict[str, Any]:
    """Return pre-computed fragmentation data.

    This lab is primarily illustrative: it shows how encoding access and
    recall probability differ across fragmented baseline states, rather
    than running a continuous simulation.
    """
    encoding_state1 = [
        dict(label="Safe Memory", p_write=0.90),
        dict(label="Trauma Memory", p_write=0.05),
    ]
    encoding_state2 = [
        dict(label="Safe Memory", p_write=0.10),
        dict(label="Trauma Memory", p_write=0.95),
    ]
    cross_state_recall = [
        dict(scenario="Recall Safe Mem in Safe State", p_recall=0.95),
        dict(scenario="Recall Safe Mem in Trauma State", p_recall=0.02),
        dict(scenario="Recall Trauma Mem in Trauma State", p_recall=0.98),
        dict(scenario="Recall Trauma Mem in Safe State", p_recall=0.05),
    ]

    return dict(
        encoding_safe=encoding_state1,
        encoding_trauma=encoding_state2,
        cross_state_recall=cross_state_recall,
        summary=dict(
            safe_in_safe=0.95,
            safe_in_trauma=0.02,
            trauma_in_trauma=0.98,
            trauma_in_safe=0.05,
            amnesia_severity=round(1.0 - (0.02 + 0.05) / 2.0, 3),
        ),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt
    import numpy as np

    if results is None:
        results = run(**kw)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    # State 1: Safe
    es = results["encoding_safe"]
    ax = axes[0]
    ax.bar([d["label"] for d in es], [d["p_write"] for d in es],
           color="#82ca9d")
    ax.set(ylabel="P(WRITE)", title="State 1: 'Safe' Baseline Active", ylim=(0, 1.1))

    # State 2: Trauma
    et = results["encoding_trauma"]
    ax = axes[1]
    ax.bar([d["label"] for d in et], [d["p_write"] for d in et],
           color="#ff8042")
    ax.set(ylabel="P(WRITE)", title="State 2: 'Trauma' Baseline Active", ylim=(0, 1.1))

    # Cross-state recall
    csr = results["cross_state_recall"]
    ax = axes[2]
    scenarios = [d["scenario"] for d in csr]
    probs = [d["p_recall"] for d in csr]
    y = np.arange(len(scenarios))
    ax.barh(y, probs, color="#8884d8")
    ax.set(yticks=y, yticklabels=[s.replace(" in ", "\nin ") for s in scenarios],
           xlabel="P(Recall)", title="Cross-State Retrieval Failure", xlim=(0, 1.1))

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
