"""Zeta (ζ) Lab — Deontological Attention Filter

Paper 6: The Resonant Gate (10.5281/zenodo.17352481)

ζ modulates the balance between self-input (I) and other-input (D).
At low ζ the agent is world-focused; at high ζ the agent becomes
fixated on the relational Other and tunes out world novelty.
Comparative trials at ζ = 0.1, 0.5, 0.9 demonstrate:
  • κ-acceleration (faster coupling growth at high ζ)
  • Baseline constitution shift (D overtakes I)
  • Novelty focus shift (relational N dominates world N)

Ported from MemoryLab-OS/zetaLab.tsx (V2 calibrated protocol).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

import numpy as np

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 6
PAPER_TITLE = "The Resonant Gate"
PAPER_DOI = "10.5281/zenodo.17352481"
LAB_TITLE = "Zeta (ζ) Lab — Deontological Attention Filter"

THESIS = (
    "ζ (zeta) is the deontological attention filter. At low ζ, the agent's "
    "baseline is constituted primarily by its own sensory experience (I). "
    "At high ζ, the agent tunes out world input and becomes hyper-sensitive "
    "to the Other's signals (D). This lab runs comparative trials at "
    "ζ = 0.1, 0.5, 0.9 to visualise κ-acceleration, baseline constitution "
    "shift, and novelty focus shift."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    zeta_values: Optional[List[float]] = None,
    sim_steps: int = 100,
    omega: float = 0.10,
    zeta_noise: float = 0.10,
    alpha: float = 0.05,
    beta: float = 0.02,
    lambda_val: float = 0.10,
    gamma_L: float = 0.50,
    seed: int = 7,
) -> Dict[str, Any]:
    """Run comparative ζ trials.

    Equations (from zetaLab V2):
        ζ_eff = clamp(0, 1, ζ + noise)
        N_world = |I - b_obs|
        obs_input = ((1 - ζ_eff)(1 - ω) + ω)·I + (κ + ζ_eff·γ_L)·D
        b_obs ← b_obs·(1-λ) + obs_input·λ
        b_tar ← b_tar·(1-λ) + (I - D)·λ
        α_eff = α·(1 - tanh(N_world))
        dκ = α_eff·(1 + ζ_eff)·(1 - N_mutual²) - β·κ
    """
    if zeta_values is None:
        zeta_values = [0.1, 0.5, 0.9]

    rng = np.random.default_rng(seed)
    labels = [f"ζ={z:.1f}" for z in zeta_values]

    kappa_data: Dict[str, List[Dict]] = {lb: [] for lb in labels}
    baseline_data: Dict[str, List[Dict]] = {lb: [] for lb in labels}
    novelty_data: Dict[str, List[Dict]] = {lb: [] for lb in labels}

    for z_val, label in zip(zeta_values, labels):
        b_obs, b_tar = 0.8, -0.8
        kappa = 0.1

        for t in range(sim_steps):
            I = 1.0 if 30 < t < 40 else -0.1   # World Event
            D_target_base = 1.0 if 70 < t < 80 else 0.0  # Relational Event
            D = (b_tar - b_obs) + D_target_base
            z_eff = max(0.0, min(1.0, z_val + (rng.random() - 0.5) * zeta_noise))

            N_world = abs(I - b_obs)

            i_weight = (1.0 - z_eff) * (1.0 - omega) + omega
            d_weight = kappa + z_eff * gamma_L
            obs_input = i_weight * I + d_weight * D
            new_b_obs = b_obs * (1.0 - lambda_val) + obs_input * lambda_val
            new_b_tar = b_tar * (1.0 - lambda_val) + (I - D) * lambda_val

            N_mutual_sq = ((b_obs - b_tar) ** 2) / 4.0
            alpha_eff = alpha * (1.0 - math.tanh(N_world))
            dk = alpha_eff * (1.0 + z_eff) * (1.0 - N_mutual_sq) - beta * kappa
            new_kappa = max(0.0, min(1.0, kappa + dk))

            kappa_data[label].append(dict(turn=t, kappa=round(new_kappa, 5)))
            baseline_data[label].append(dict(
                turn=t,
                I_contribution=round(abs(i_weight * I), 5),
                D_contribution=round(abs(d_weight * D), 5),
            ))
            novelty_data[label].append(dict(
                turn=t,
                world_N=round((1.0 - z_eff) * abs(I - b_obs), 5),
                relational_N=round((1.0 + z_eff) * abs(D_target_base), 5),
            ))

            b_obs, b_tar, kappa = new_b_obs, new_b_tar, new_kappa

    return dict(
        kappa=kappa_data,
        baseline=baseline_data,
        novelty=novelty_data,
        labels=labels,
        summary=dict(
            final_kappas={lb: kappa_data[lb][-1]["kappa"] for lb in labels},
        ),
        params=dict(
            zeta_values=zeta_values, sim_steps=sim_steps, omega=omega,
            zeta_noise=zeta_noise, alpha=alpha, beta=beta,
            lambda_val=lambda_val, gamma_L=gamma_L, seed=seed,
        ),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt

    if results is None:
        results = run(**kw)
    labels = results["labels"]
    colors = ["#8884d8", "#82ca9d", "#ff8042"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    # κ acceleration
    ax = axes[0]
    for lb, c in zip(labels, colors):
        data = results["kappa"][lb]
        ax.plot([d["turn"] for d in data], [d["kappa"] for d in data],
                label=lb, color=c)
    ax.set(xlabel="Turn", ylabel="κ", title="Coupling (κ) Acceleration",
           ylim=(0, 1))
    ax.legend()

    # Baseline constitution (use mid-ζ trial)
    ax = axes[1]
    mid = labels[1]
    data = results["baseline"][mid]
    t = [d["turn"] for d in data]
    ax.fill_between(t, [d["I_contribution"] for d in data],
                     label="Self-Input (I)", color="#8884d8", alpha=0.6)
    ax.fill_between(t, [d["I_contribution"] for d in data],
                     [d["I_contribution"] + d["D_contribution"] for d in data],
                     label="Other-Input (D)", color="#ff8042", alpha=0.6)
    ax.set(xlabel="Turn", ylabel="Contribution Magnitude",
           title=f"Baseline Constitution ({mid})")
    ax.legend()

    # Novelty focus shift (use high-ζ trial)
    ax = axes[2]
    hi = labels[-1]
    data = results["novelty"][hi]
    t = [d["turn"] for d in data]
    ax.plot(t, [d["world_N"] for d in data], label="World Novelty", color="#82ca9d")
    ax.plot(t, [d["relational_N"] for d in data], label="Relational Novelty",
            color="#ff8042")
    ax.axvline(35, color="gray", linestyle="--", alpha=0.4, label="World Event")
    ax.axvline(75, color="gray", linestyle=":", alpha=0.4, label="Relational Event")
    ax.set(xlabel="Turn", ylabel="Perceived Novelty",
           title=f"Novelty Focus Shift ({hi})")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
