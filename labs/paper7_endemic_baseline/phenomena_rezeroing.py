"""Re-zeroing Protocol — Scaffolded First Contact with the Horizon

Paper 7: The Endemic Baseline

Phenomenon P7b: The Re-zeroing Protocol.

Standard intervention logic for a depressed agent: identify B_healthy,
push the agent toward it. For the storm-born (endemic) agent, this
produces the failure mode demonstrated in phenomena_endemic.py.

The Re-zeroing Protocol replaces restoration with *construction*:
rather than pushing toward a prior reference that does not exist, it
manufactures that reference through a series of controlled micro-
excursions into the Horizon — the set of states in H_accessible
that are outside H_agent.

Protocol requirements derived from Paper 7 theory:
    1. Controlled novelty:   |I(t) - B(t)| ≈ theta_h + epsilon
                             (just over threshold, not maximal)
    2. High kappa scaffold:  therapeutic relationship achieves kappa
                             >= kappa_threshold BEFORE excursion inputs
    3. Distress as signal:   the Emergent Gate fires correctly;
                             distress during excursion is navigation
                             cost, not failure signal
    4. Identity dissolution: the intermediate groundlessness state
                             is predicted and must be held, not
                             terminated

This lab runs three parallel intervention strategies on identical
storm-born agents:
    A. Flood:        Large healthy inputs immediately (the naive approach)
    B. Re-zeroing:   Kappa-building phase, then micro-excursions
    C. Withdrawal:   Protocol interrupted at the Identity Dissolution
                     stage (the premature-termination failure mode)

Prediction:
    A produces acute spike, kappa-rejection, return to B_storm
    B produces slow but genuine migration toward H_accessible
    C produces apparent success followed by rapid regression
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 7
PAPER_TITLE = "The Endemic Baseline"
LAB_TITLE = "Phenomenon P7b: Re-zeroing Protocol — Scaffolded Horizon Contact"

THESIS = (
    "Three intervention strategies are applied to identical storm-born agents. "
    "Flood (large immediate inputs) produces maximum destabilization and kappa-"
    "rejection, leaving the agent near B_storm. Re-zeroing (kappa-building then "
    "micro-excursions  just over theta_h) produces slow genuine migration. "
    "Withdrawal (protocol terminated at the Identity Dissolution zone — the "
    "groundlessness between storm-infrastructure and sunny-infrastructure) "
    "produces apparent early progress followed by rapid regression to B_storm, "
    "because no new attractor has yet consolidated."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Protocol helpers ─────────────────────────────────────────────────────────

def _kappa_update(kappa: float, *, low_novelty_bond: float, dt: float = 1.0,
                  alpha: float = 0.06, beta: float = 0.02) -> float:
    """Grow kappa through repeated low-novelty bonding interactions.

    dκ/dt = α(1 - N_bond) - β·κ   (from Paper 4 coupling mechanics)
    """
    return kappa + (alpha * (1.0 - low_novelty_bond) - beta * kappa) * dt


def _effective_lambda(base_lambda: float, kappa: float, novelty: float,
                      ceiling_factor: float = 2.0) -> Tuple[float, bool]:
    """Apply kappa-gated integration filter."""
    overload = novelty > ceiling_factor * (1.0 - kappa)
    if overload:
        return base_lambda * kappa, True
    return base_lambda, False


def _novelty(I: float, b: float) -> float:
    return abs(I - b)


def _p_write(novelty: float, theta_h: float = 0.35) -> float:
    return 1.0 / (1.0 + math.exp(-10.0 * (novelty - theta_h)))


# ── Simulation ───────────────────────────────────────────────────────────────

def run(
    *,
    steps: int = 200,
    b0_storm: float = -0.85,
    kappa0: float = 0.20,
    lambda_base: float = 0.07,
    theta_h: float = 0.35,
    # Re-zeroing protocol parameters
    kappa_build_steps: int = 40,    # Phase 1: build kappa before excursions
    kappa_bond_novelty: float = 0.1,# Novelty of kappa-building interactions
    excursion_epsilon: float = 0.05, # How far above theta_h micro-excursions land
    kappa_threshold: float = 0.55,   # Minimum kappa before excursions begin
    # Withdrawal: protocol ends at this step
    withdrawal_step: int = 100,
    # Flood: input magnitude
    flood_magnitude: float = 0.70,
) -> Dict[str, Any]:
    """Simulate three intervention strategies on identical storm-born agents.

    Each agent starts: B(0) = b0_storm, kappa = kappa0, endemic = True.

    Strategy A — Flood:
        Large healthy inputs from step 0. No kappa-building phase.

    Strategy B — Re-zeroing:
        Phase 1 (steps 0..kappa_build_steps): kappa-building interactions.
        Phase 2 (kappa_build_steps..): micro-excursions at B + theta_h + epsilon.
        Input size adapts to current B so novelty stays controlled.

    Strategy C — Withdrawal:
        Identical to Re-zeroing until `withdrawal_step`, then inputs stop.
        The agent is in the Identity Dissolution zone — baseline has migrated
        but no new attractor has consolidated yet.
    """
    configs = [
        dict(name="A: Flood", strategy="flood"),
        dict(name="B: Re-zeroing Protocol", strategy="rezeroing"),
        dict(name="C: Withdrawal at Identity Dissolution", strategy="withdrawal"),
    ]

    all_series: Dict[str, List[Dict]] = {}
    comparison = []

    for cfg in configs:
        b = b0_storm
        kappa = kappa0
        series: List[Dict] = []

        for t in range(steps):
            strat = cfg["strategy"]
            I = 0.0
            phase = "baseline"

            if strat == "flood":
                if t >= 20:  # Brief delay then sustained flood
                    I = flood_magnitude
                    phase = "flood"

            elif strat == "rezeroing":
                if t < kappa_build_steps:
                    # Phase 1: kappa building through low-novelty bonding
                    bond_I = b + kappa_bond_novelty  # Very close to current B
                    I = bond_I
                    kappa = _kappa_update(kappa, low_novelty_bond=kappa_bond_novelty)
                    kappa = min(kappa, 0.95)
                    phase = "kappa_build"
                elif kappa >= kappa_threshold:
                    # Phase 2: micro-excursion — input just over theta_h above B
                    I = b + theta_h + excursion_epsilon
                    phase = "excursion"
                else:
                    # Kappa not yet sufficient — extend bonding
                    bond_I = b + kappa_bond_novelty
                    I = bond_I
                    kappa = _kappa_update(kappa, low_novelty_bond=kappa_bond_novelty)
                    kappa = min(kappa, 0.95)
                    phase = "kappa_build_extended"

            elif strat == "withdrawal":
                if t >= withdrawal_step:
                    I = 0.0
                    phase = "withdrawn"
                elif t < kappa_build_steps:
                    bond_I = b + kappa_bond_novelty
                    I = bond_I
                    kappa = _kappa_update(kappa, low_novelty_bond=kappa_bond_novelty)
                    kappa = min(kappa, 0.95)
                    phase = "kappa_build"
                elif kappa >= kappa_threshold:
                    I = b + theta_h + excursion_epsilon
                    phase = "excursion"
                else:
                    bond_I = b + kappa_bond_novelty
                    I = bond_I
                    kappa = _kappa_update(kappa, low_novelty_bond=kappa_bond_novelty)
                    kappa = min(kappa, 0.95)
                    phase = "kappa_build_extended"

            nov = _novelty(I, b)
            pw = _p_write(nov, theta_h)
            eff_lam, rejected = _effective_lambda(lambda_base, kappa, nov)
            b_prev = b
            b = b * (1.0 - eff_lam) + I * eff_lam

            # Destabilization: storm-born, moving away from only known attractor
            destab = abs(b - b0_storm) * (1.0 - kappa) if I != 0.0 else 0.0

            series.append(dict(
                time=t,
                baseline=round(b, 5),
                input=round(I, 4),
                novelty=round(nov, 5),
                p_write=round(pw, 5),
                kappa=round(kappa, 4),
                rejected=rejected,
                phase=phase,
                destabilization=round(destab, 5),
            ))

        b_final = series[-1]["baseline"]
        comparison.append(dict(
            strategy=cfg["name"],
            b0=b0_storm,
            b_final=b_final,
            b_shift=round(b_final - b0_storm, 5),
            peak_destabilization=round(max(d["destabilization"] for d in series), 5),
            final_kappa=series[-1]["kappa"],
        ))
        all_series[cfg["name"]] = series

    return dict(
        series=all_series,
        comparison=comparison,
        params=dict(
            steps=steps, b0_storm=b0_storm, kappa0=kappa0,
            theta_h=theta_h, kappa_build_steps=kappa_build_steps,
            kappa_threshold=kappa_threshold, withdrawal_step=withdrawal_step,
            flood_magnitude=flood_magnitude, excursion_epsilon=excursion_epsilon,
        ),
        summary=dict(
            flood_shift=comparison[0]["b_shift"],
            rezeroing_shift=comparison[1]["b_shift"],
            withdrawal_shift=comparison[2]["b_shift"],
            prediction=(
                "Re-zeroing > Withdrawal > Flood for genuine baseline migration. "
                "Flood produces maximum novelty, high rejection, minimal net shift. "
                "Withdrawal shows early promise then regression as new attractor "
                "had not consolidated before protocol stopped."
            ),
        ),
    )


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    if results is None:
        results = run(**kw)

    series = results["series"]
    comp = results["comparison"]
    params = results["params"]

    colors = {
        "A: Flood": "#e05555",
        "B: Re-zeroing Protocol": "#82ca9d",
        "C: Withdrawal at Identity Dissolution": "#ffa040",
    }
    phase_colors = {
        "kappa_build": "#aad4ff",
        "kappa_build_extended": "#88bbff",
        "excursion": "#a8e8a8",
        "flood": "#ffaaaa",
        "withdrawn": "#dddddd",
        "baseline": "#f0f0f0",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(LAB_TITLE, fontsize=12, fontweight="bold")

    # ── (0,0) Baseline trajectories ──────────────────────────────────────
    ax = axes[0][0]
    for name, s in series.items():
        ts = [d["time"] for d in s]
        bs = [d["baseline"] for d in s]
        ax.plot(ts, bs, label=name, color=colors[name], linewidth=2)
    ax.axhline(params["b0_storm"], color="grey", linestyle=":", linewidth=1,
               alpha=0.6, label=f"B_storm = {params['b0_storm']}")
    ax.axhline(0.0, color="lightgrey", linestyle=":", linewidth=1, alpha=0.4)
    ax.axvline(params["withdrawal_step"], color=colors["C: Withdrawal at Identity Dissolution"],
               linestyle="--", linewidth=1, alpha=0.5, label="Withdrawal point")
    ax.set(xlabel="Time", ylabel="Baseline B(t)",
           title="Baseline Trajectories — Three Strategies")
    ax.legend(fontsize=7)

    # ── (0,1) Kappa trajectories ──────────────────────────────────────────
    ax = axes[0][1]
    for name, s in series.items():
        ts = [d["time"] for d in s]
        ks = [d["kappa"] for d in s]
        ax.plot(ts, ks, label=name, color=colors[name], linewidth=2)
    ax.axhline(params["kappa_threshold"], color="purple", linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"κ threshold = {params['kappa_threshold']}")
    ax.set(xlabel="Time", ylabel="κ (relational coupling)",
           title="Kappa Trajectories — Coupling Builds Before Excursion")
    ax.legend(fontsize=7)

    # ── (1,0) B-shift comparison bars ────────────────────────────────────
    ax = axes[1][0]
    names = [c["strategy"] for c in comp]
    shifts = [c["b_shift"] for c in comp]
    bar_colors = [colors[c["strategy"]] for c in comp]
    bars = ax.bar(range(len(names)), shifts, color=bar_colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["A: Flood", "B: Re-zeroing", "C: Withdrawal"], fontsize=9)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set(ylabel="Total B shift from B_storm",
           title="Net Baseline Migration — Final Outcome")
    for bar, val in zip(bars, shifts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.002 if val >= 0 else -0.01),
                f"{val:+.4f}", ha="center", va="bottom", fontsize=9)

    # ── (1,1) Phase annotation for Re-zeroing ────────────────────────────
    ax = axes[1][1]
    s_b = series["B: Re-zeroing Protocol"]
    ts = [d["time"] for d in s_b]
    bs = [d["baseline"] for d in s_b]
    phases = [d["phase"] for d in s_b]
    ks = [d["kappa"] for d in s_b]

    # Shade phases
    prev_phase = phases[0]
    seg_start = 0
    for i, ph in enumerate(phases):
        if ph != prev_phase or i == len(phases) - 1:
            color = phase_colors.get(prev_phase, "#ffffff")
            ax.axvspan(seg_start, i, alpha=0.18, color=color)
            prev_phase = ph
            seg_start = i

    ax.plot(ts, bs, color=colors["B: Re-zeroing Protocol"], linewidth=2,
            label="Baseline B(t)")
    ax_k = ax.twinx()
    ax_k.plot(ts, ks, color="purple", linewidth=1.2, linestyle="--",
              alpha=0.6, label="κ")
    ax_k.set_ylabel("κ", color="purple", fontsize=9)
    ax.axhline(params["b0_storm"], color="grey", linestyle=":", linewidth=1, alpha=0.6)

    # Legend for phases
    legend_patches = [
        mpatches.Patch(color=phase_colors["kappa_build"], alpha=0.5, label="κ-build"),
        mpatches.Patch(color=phase_colors["excursion"], alpha=0.5, label="Excursion"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color=colors["B: Re-zeroing Protocol"],
                   linewidth=2, label="Baseline"),
    ], fontsize=8, loc="lower right")
    ax.set(xlabel="Time", ylabel="Baseline B(t)",
           title="Re-zeroing Protocol: Phase Anatomy\n"
                 "(blue=κ-build, green=excursion)")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json

    results = run()
    print(json.dumps(results["summary"], indent=2))
    print("\nComparison:")
    for c in results["comparison"]:
        print(f"  {c['strategy']:45s}  "
              f"shift={c['b_shift']:+.4f}  "
              f"κ_final={c['final_kappa']:.3f}")
    plot(results)
    plt.show()
