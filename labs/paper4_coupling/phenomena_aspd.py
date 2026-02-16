"""Antisocial Personality (ASPD) — Deontological Immunity

Paper 4: The Coupling Asymmetry (10.5281/zenodo.18519187)

κ locked near zero.  The agent's baseline is constituted solely by
self-experience (I), immune to relational constitution (D).  This is
not a missing theory of mind but a functional solipsism.

Ported from MemoryLab-OS/phenomena.tsx — ASPDModel.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 4
PAPER_TITLE = "The Coupling Asymmetry"
PAPER_DOI = "10.5281/zenodo.18519187"
LAB_TITLE = "Phenomenon: Antisocial Personality (ASPD)"

THESIS = (
    "Antisocial Personality is a failure of the deontological pathway. The "
    "coupling parameter, κ, remains locked near zero. The agent's baseline "
    "is constituted solely by self-experience (I), not by the constitutive "
    "influence of others (D). This is not a lack of a 'theory of mind', but "
    "an immunity to relational constitution, resulting in a functional "
    "solipsism."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    kappa_typical: float = 0.60,
    kappa_aspd: float = 0.05,
    lambda_val: float = 0.10,
    social_event_t: int = 2,
    social_event_d: float = -0.4,
    steps: int = 5,
) -> Dict[str, Any]:
    """Compare typical vs ASPD response to a social event.

    At t=social_event_t, a social input D shifts the Other's demand.
    The typical agent integrates it; the ASPD agent barely registers it.
    """
    typical_b = 0.5
    aspd_b = 0.5
    history: List[Dict[str, float]] = []

    for t in range(steps):
        d = social_event_d if t == social_event_t else 0.0
        prev_typ, prev_aspd = typical_b, aspd_b

        typical_b = typical_b + kappa_typical * d * lambda_val
        aspd_b = aspd_b + kappa_aspd * d * lambda_val

        history.append(dict(
            time=t,
            typical=round(typical_b, 6),
            aspd=round(aspd_b, 6),
            social_input=round(d, 4),
        ))

    constitution = [
        dict(agent="Typical Agent", self_pct=round((1 - kappa_typical) * 100, 1),
             other_pct=round(kappa_typical * 100, 1)),
        dict(agent="ASPD Model", self_pct=round((1 - kappa_aspd) * 100, 1),
             other_pct=round(kappa_aspd * 100, 1)),
    ]

    return dict(
        timeseries=history,
        constitution=constitution,
        summary=dict(
            typical_shift=round(abs(history[-1]["typical"] - 0.5), 4),
            aspd_shift=round(abs(history[-1]["aspd"] - 0.5), 4),
            kappa_typical=kappa_typical,
            kappa_aspd=kappa_aspd,
        ),
        params=dict(kappa_typical=kappa_typical, kappa_aspd=kappa_aspd,
                    lambda_val=lambda_val, social_event_t=social_event_t,
                    social_event_d=social_event_d, steps=steps),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt
    import numpy as np

    if results is None:
        results = run(**kw)
    ts = results["timeseries"]
    con = results["constitution"]

    t = [d["time"] for d in ts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    # Baseline constitution
    agents = [c["agent"] for c in con]
    self_pct = [c["self_pct"] for c in con]
    other_pct = [c["other_pct"] for c in con]
    y = np.arange(len(agents))
    ax1.barh(y, self_pct, color="#82ca9d", label="Self-Input (I)")
    ax1.barh(y, other_pct, left=self_pct, color="#8884d8", label="Other-Input (κD)")
    ax1.set(yticks=y, yticklabels=agents, xlabel="%",
            title="Baseline Constitution")
    ax1.legend()

    # Response to social event
    ax2.plot(t, [d["typical"] for d in ts], "o-", label="Typical Baseline",
             color="#4a90e2", drawstyle="steps-post")
    ax2.plot(t, [d["aspd"] for d in ts], "o-", label="ASPD Baseline",
             color="#ff8042", drawstyle="steps-post")
    se = results["params"]["social_event_t"]
    ax2.axvline(se, color="red", linestyle="--", alpha=0.6, label="Social Event")
    ax2.set(xlabel="Time", ylabel="Baseline", title="Response to Social Input",
            ylim=(0, 1))
    ax2.legend()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
