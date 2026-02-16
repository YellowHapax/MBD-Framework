"""Deontological Echo — The Internalised Other

Paper 4: The Coupling Asymmetry (10.5281/zenodo.18519187)

A past, powerful relational baseline becomes an internalised 'Other'
exerting continuous deontological influence (D_self).  The agent is
constitutively coupled to a ghost of a past relationship.

Ported from MemoryLab-OS/phenomena.tsx — DeontologicalEchoModel.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 4
PAPER_TITLE = "The Coupling Asymmetry"
PAPER_DOI = "10.5281/zenodo.18519187"
LAB_TITLE = "Phenomenon: Deontological Echo"

THESIS = (
    "A phenomenon where a past, powerful relational baseline becomes an "
    "internalised 'Other' exerting continuous deontological influence "
    "(D_self). The agent is constitutively coupled to a ghost of a past "
    "relationship or self, creating a persistent internal tension or 'drag' "
    "on their current baseline, tethering them to the past."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    d_self: float = 0.80,
    kappa_self: float = 0.70,
    lambda_val: float = 0.10,
    env_start: int = 20,
    env_end: int = 80,
    env_value: float = -0.50,
    steps: int = 100,
) -> Dict[str, Any]:
    """Simulate deontological echo tethering.

    Parameters
    ----------
    d_self : float       The internalised ghost baseline value.
    kappa_self : float   Coupling strength to the ghost.
    lambda_val : float   Baseline plasticity.
    env_start, env_end : int  Window of new environmental input.
    env_value : float    Value of the new environment.
    steps : int          Simulation length.
    """
    echo_b = 0.0
    control_b = 0.0
    history: List[Dict[str, float]] = []

    for t in range(steps):
        I = env_value if env_start < t < env_end else 0.0

        echo_b = (echo_b * (1.0 - lambda_val)
                  + (I + kappa_self * (d_self - echo_b)) * lambda_val)
        control_b = control_b * (1.0 - lambda_val) + I * lambda_val

        history.append(dict(
            time=t,
            echo=round(echo_b, 6),
            control=round(control_b, 6),
            env_input=round(I, 4),
        ))

    return dict(
        timeseries=history,
        summary=dict(
            echo_final=round(echo_b, 4),
            control_final=round(control_b, 4),
            echo_tether_strength=round(abs(echo_b - control_b), 4),
            d_self=d_self,
        ),
        attractor_points=[
            dict(label="Start", value=0.0),
            dict(label="D_self (Ghost)", value=d_self),
            dict(label="New Environment", value=env_value),
            dict(label="Control Final", value=round(control_b, 4)),
            dict(label="Echo Final", value=round(echo_b, 4)),
        ],
        params=dict(
            d_self=d_self, kappa_self=kappa_self, lambda_val=lambda_val,
            env_start=env_start, env_end=env_end, env_value=env_value,
            steps=steps,
        ),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt

    if results is None:
        results = run(**kw)
    ts = results["timeseries"]
    p = results["params"]
    att = results["attractor_points"]

    t = [d["time"] for d in ts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    ax1.plot(t, [d["echo"] for d in ts], label="Echo Agent", color="#ff8042")
    ax1.plot(t, [d["control"] for d in ts], label="Control Agent", color="#8884d8")
    ax1.axhline(p["d_self"], color="#ff8042", linestyle="--", alpha=0.5,
                label=f"D_self ({p['d_self']})")
    ax1.axhline(p["env_value"], color="#ccc", linestyle="--", alpha=0.5,
                label=f"New Input ({p['env_value']})")
    ax1.set(xlabel="Time", ylabel="Baseline",
            title="Baseline Trajectory Tethering", ylim=(-0.7, 1.0))
    ax1.legend(fontsize=8)

    # Attractor points
    labels = [a["label"] for a in att]
    values = [a["value"] for a in att]
    colors = ["gray", "#ff8042", "#ccc", "#8884d8", "#ff8042"]
    ax2.barh(range(len(labels)), values, color=colors)
    ax2.set(yticks=range(len(labels)), yticklabels=labels,
            xlabel="Baseline Value", title="Attractor Space")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
