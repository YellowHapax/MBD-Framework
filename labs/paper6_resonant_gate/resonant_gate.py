"""The Resonant Gate — Coupling-Amplified Encoding

Paper 6: The Resonant Gate (10.5281/zenodo.17352481)

The same insight is presented to an agent twice: once early (low κ)
and once late (high κ).  The resonant state amplifies the novelty
signal and dramatically increases P(WRITE).  *Who* we are with
changes *what* we learn.

Ported from MemoryLab-OS/resonantGate.tsx (V4).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List

import numpy as np

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 6
PAPER_TITLE = "The Resonant Gate"
PAPER_DOI = "10.5281/zenodo.17352481"
LAB_TITLE = "The Resonant Gate"

THESIS = (
    "The Resonant Gate demonstrates that interpersonal coupling (κ) acts "
    "as a precision gain on prediction error. An identical external insight "
    "presented at low-κ (early) produces weak novelty and minimal encoding. "
    "The same insight at high-κ (late) produces amplified novelty that "
    "crosses the encoding threshold. Who we are with changes what we learn."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    n_timesteps: int = 200,
    t_insight_early: int = 30,
    t_insight_late: int = 150,
    normal_input_std: float = 1.0,
    novel_insight_value: float = 8.0,
    lambda_b: float = 0.10,
    alpha_k: float = 0.02,
    beta_k: float = 0.025,
    precision_gain: float = 2.5,
    theta_gate: float = 17.0,
    k_gate: float = 2.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Reproduce the original resonantGate V4 simulation.

    Two agents (Alex & Ben) receive random inputs.  Alex receives the
    novel insight at t_insight_early and t_insight_late.  Ben's Novelty
    and P(WRITE) are tracked to show the resonance amplification effect.
    """
    rng = np.random.default_rng(seed)
    baselines_A = np.zeros(n_timesteps)
    baselines_B = np.zeros(n_timesteps)
    kappa = np.zeros(n_timesteps)
    novelty_B = np.zeros(n_timesteps)
    p_write_B = np.zeros(n_timesteps)
    kappa[0] = 0.05

    for t in range(1, n_timesteps):
        inp_A = rng.normal(0, normal_input_std)
        if t == t_insight_early or t == t_insight_late:
            inp_A = novel_insight_value
        inp_B = rng.normal(0, normal_input_std)

        precision = 1.0 + precision_gain * kappa[t - 1]
        pe_A = abs(inp_B - baselines_A[t - 1])
        pe_B = abs(inp_A - baselines_B[t - 1])
        novelty_B[t] = precision * pe_B

        mutual_pe = min(1.5, max(0, (pe_A + pe_B) / theta_gate))
        dk = alpha_k * (1 - mutual_pe ** 2) - beta_k * kappa[t - 1]
        kappa[t] = max(0.0, min(1.0, kappa[t - 1] + dk))

        baselines_A[t] = (
            (1 - lambda_b) * baselines_A[t - 1]
            + lambda_b * (inp_A + kappa[t] * inp_B)
        )
        baselines_B[t] = (
            (1 - lambda_b) * baselines_B[t - 1]
            + lambda_b * (inp_B + kappa[t] * inp_A)
        )
        p_write_B[t] = 1.0 / (1.0 + math.exp(-k_gate * (novelty_B[t] - theta_gate)))

    timeseries = [
        dict(
            time=int(t),
            kappa=round(float(kappa[t]), 5),
            baselineA=round(float(baselines_A[t]), 5),
            baselineB=round(float(baselines_B[t]), 5),
            noveltyB=round(float(novelty_B[t]), 5),
            pWriteB=round(float(p_write_B[t]), 5),
        )
        for t in range(n_timesteps)
    ]

    n_early = float(novelty_B[t_insight_early])
    n_late = float(novelty_B[t_insight_late])
    pw_early = float(p_write_B[t_insight_early])
    pw_late = float(p_write_B[t_insight_late])

    return dict(
        timeseries=timeseries,
        summary=dict(
            kappa_early=round(float(kappa[t_insight_early]), 4),
            kappa_late=round(float(kappa[t_insight_late]), 4),
            novelty_early=round(n_early, 4),
            novelty_late=round(n_late, 4),
            p_write_early=round(pw_early, 4),
            p_write_late=round(pw_late, 4),
            novelty_amplification=round(n_late / max(n_early, 1e-9), 2),
            p_write_amplification=round(pw_late / max(pw_early, 1e-9), 2),
        ),
        params=dict(
            n_timesteps=n_timesteps, t_insight_early=t_insight_early,
            t_insight_late=t_insight_late, lambda_b=lambda_b,
            alpha_k=alpha_k, beta_k=beta_k, precision_gain=precision_gain,
            theta_gate=theta_gate, k_gate=k_gate, seed=seed,
        ),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt

    if results is None:
        results = run(**kw)
    ts = results["timeseries"]
    s = results["summary"]
    p = results["params"]

    t = [d["time"] for d in ts]
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"{LAB_TITLE} (V4)", fontsize=14, fontweight="bold")

    te, tl = p["t_insight_early"], p["t_insight_late"]

    # κ(t)
    ax = axes[0]
    ax.plot(t, [d["kappa"] for d in ts], color="purple")
    ax.axvline(te, color="red", linestyle="--", alpha=0.6, label=f"Early (t={te})")
    ax.axvline(tl, color="green", linestyle="--", alpha=0.6, label=f"Late (t={tl})")
    ax.axvspan(0, 75, alpha=0.05, color="gray", label="High-Inertia")
    ax.axvspan(75, p["n_timesteps"], alpha=0.05, color="blue", label="Resonant")
    ax.set(ylabel="κ(t)")
    ax.legend(fontsize=7)

    # Baselines
    ax = axes[1]
    ax.plot(t, [d["baselineA"] for d in ts], label="Alex", color="#8884d8")
    ax.plot(t, [d["baselineB"] for d in ts], label="Ben", color="#82ca9d")
    ax.axvline(te, color="red", linestyle="--", alpha=0.4)
    ax.axvline(tl, color="green", linestyle="--", alpha=0.4)
    ax.set(ylabel="Baseline")
    ax.legend(fontsize=7)

    # Novelty
    ax = axes[2]
    ax.plot(t, [d["noveltyB"] for d in ts], color="orange", label="Ben's Novelty")
    ax.axhline(p["theta_gate"], color="black", linestyle=":", alpha=0.5, label=f"θ_h={p['theta_gate']}")
    ax.axvline(te, color="red", linestyle="--", alpha=0.4)
    ax.axvline(tl, color="green", linestyle="--", alpha=0.4)
    ax.set(ylabel="Novelty N_h")
    ax.legend(fontsize=7)

    # P(WRITE)
    ax = axes[3]
    ax.plot(t, [d["pWriteB"] for d in ts], color="green", label="Ben's P(WRITE)")
    ax.axvline(te, color="red", linestyle="--", alpha=0.4)
    ax.axvline(tl, color="green", linestyle="--", alpha=0.4)
    ax.set(xlabel="Time", ylabel="P(WRITE)", ylim=(0, 1.05))
    ax.legend(fontsize=7)

    # Annotation
    txt = (
        f"Early (t={te}): κ={s['kappa_early']:.3f}  N={s['novelty_early']:.1f}  "
        f"P(W)={s['p_write_early']:.3f}\n"
        f"Late  (t={tl}): κ={s['kappa_late']:.3f}  N={s['novelty_late']:.1f}  "
        f"P(W)={s['p_write_late']:.3f}\n"
        f"Novelty Amp: {s['novelty_amplification']}×    "
        f"P(WRITE) Amp: {s['p_write_amplification']}×"
    )
    fig.text(0.12, 0.01, txt, fontsize=9, fontfamily="monospace",
             bbox=dict(facecolor="lightyellow", edgecolor="gray"))

    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
