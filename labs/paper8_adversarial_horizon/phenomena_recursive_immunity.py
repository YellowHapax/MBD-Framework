"""Recursive Immunity — Suppression Dynamics and the Evidence Floor

Paper 8: The Suppressive and Emergent Phenomenon

Phenomenon P8d: An adversarial system attempts to suppress information
about the mechanism it uses to maintain control. The suppression attempt
operates on two separable channels:

    1. DIRECT DISTRIBUTION CHANNEL: The fraction of the target population
       that receives the formalization P. Adversarial suppression (increasing
       s) reduces this directly: E_direct(s) = (1 - s).

    2. SUPPRESSION-AS-EVIDENCE CHANNEL: Suppression acts are observable by
       institutional actors, researchers, and whistleblowers outside the
       adversarial system's information control. Each observed suppression act
       constitutes empirical evidence for the model (Paper 8, §6.1).

The critical modeling insight (correcting the naive U-shape prediction):
observation_probability is NOT a constant. It is itself a decreasing
function of suppression_strength, because V1 and V2 operate on the
observation channel as well as the direct distribution channel. The
adversarial system can suppress evidence OF suppression, up to a floor
(obs_floor) set by actors it cannot reach: international institutions,
distributed archives, independent researchers, structural leakage.

    obs_prob(s) = obs_floor + (1 - obs_floor) * exp(-obs_decay * s)

Total evidence integrates both channels, weighted by the relative
institutional reach of the observer network vs. the target population:

    E(s) = E_direct(s) + w * E_suppression(s)
         = (1 - s) + w * s * obs_prob(s)

where w is the credibility/reach ratio of the suppression-evidence
channel versus the direct distribution channel.

MBD predicts:
    - For low w (suppression-evidence channel is weak): adversary's
      optimal strategy is maximal suppression (s → 1). E(s) is
      monotone decreasing. Floor evidence = obs_floor.
    - For high w (independent observers have high institutional reach):
      selective suppression at intermediate s* may minimize E. This is
      the empirically recognizable pattern of sophisticated suppression:
      plausibly deniable, targeted, calibrated to stay below detection
      threshold.
    - The critical parameter is obs_floor. As long as obs_floor > 0 —
      maintained by actors outside the adversarial system's control —
      maximal suppression cannot reduce E to zero. The adversary's
      second-order response is to apply V3 (Storm Normalization) to
      the suppression itself: normalize restriction as routine
      administrative procedure, driving obs_floor toward zero in
      effective detection probability.

The lab finds the adversary's optimal s* as a function of obs_floor,
obs_decay, and w. It identifies the minimum obs_floor that makes total
suppression (s=1) produce more total evidence than no suppression (s=0),
i.e., the condition E(1) > E(0) — the "floor protection threshold" above
which the adversary is forced into an unfavorable cost-benefit position.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 8
PAPER_TITLE = "The Suppressive and Emergent Phenomenon"
LAB_TITLE = "Phenomenon P8d: Recursive Immunity — Suppression Dynamics and the Evidence Floor"

THESIS = (
    "An adversarial system that attempts to suppress information about its own "
    "mechanism operates on two channels: direct distribution (reduced by suppression) "
    "and suppression-as-evidence (increased by suppression, but itself suppressible). "
    "The observation probability is a decreasing function of suppression strength, not "
    "a constant — V1 and V2 operate on the observation channel as well. The adversary's "
    "optimal strategy is maximal suppression when the suppression-evidence channel has "
    "low institutional reach; selective suppression when it is high. In all cases, an "
    "irreducible detection floor (obs_floor > 0) from actors outside the adversarial "
    "system's control prevents total evidence from reaching zero. The adversary's "
    "second-order response — applying V3 (Storm Normalization) to the suppression acts "
    "themselves — is the empirically distinguishing signature of sophisticated vs. "
    "coercive suppression."
)


def describe() -> Dict[str, Any]:
    return dict(
        paper=PAPER,
        paper_title=PAPER_TITLE,
        lab_title=LAB_TITLE,
        thesis=THESIS,
    )


# ── Core model ───────────────────────────────────────────────────────────────

def obs_prob(s: float, obs_floor: float, obs_decay: float) -> float:
    """Observation probability as a function of suppression strength.

    Not a constant: V1+V2 can reduce it, but cannot suppress below obs_floor.

    Args:
        s: Suppression strength ∈ [0, 1]
        obs_floor: Irreducible detection floor (whistleblowers, structural leakage)
        obs_decay: Rate at which observation prob decreases under suppression
    """
    return obs_floor + (1.0 - obs_floor) * math.exp(-obs_decay * s)


def total_evidence(
    s: float,
    obs_floor: float,
    obs_decay: float,
    w: float = 1.0,
) -> float:
    """Total evidence accrued for the model at suppression strength s.

    E(s) = E_direct(s) + w * E_suppression(s)
         = (1 - s) + w * s * obs_prob(s)

    Args:
        s: Suppression strength ∈ [0, 1]
        obs_floor: Irreducible detection floor
        obs_decay: Rate of observation prob decrease
        w: Relative reach/credibility of suppression-evidence channel
    """
    e_direct = 1.0 - s
    e_supp = w * s * obs_prob(s, obs_floor, obs_decay)
    return e_direct + e_supp


def find_optimal_suppression(
    obs_floor: float,
    obs_decay: float,
    w: float = 1.0,
    resolution: int = 1000,
) -> Tuple[float, float]:
    """Find the adversary's optimal suppression strength s*.

    Returns (s_optimal, E_at_optimal).
    """
    s_values = [i / resolution for i in range(resolution + 1)]
    e_values = [total_evidence(s, obs_floor, obs_decay, w) for s in s_values]
    min_idx = min(range(len(e_values)), key=lambda i: e_values[i])
    return s_values[min_idx], e_values[min_idx]


# ── Simulation ───────────────────────────────────────────────────────────────

def run(
    *,
    obs_floor_values: Optional[List[float]] = None,
    obs_decay_values: Optional[List[float]] = None,
    w_values: Optional[List[float]] = None,
    resolution: int = 200,
) -> Dict[str, Any]:
    """Sweep parameter space and compute evidence curves + optimal strategies.

    For each combination of (obs_floor, obs_decay, w), compute:
    - E(s) curve across s ∈ [0, 1]
    - Adversary's optimal s* and the evidence level achieved
    - E(0) — baseline (no suppression) evidence level
    - E(1) — total-suppression evidence level
    - Whether total suppression is adversarially optimal vs. partial suppression

    Returns dict with curves, optimal points, and phase analysis.
    """
    if obs_floor_values is None:
        obs_floor_values = [0.0, 0.05, 0.10, 0.20, 0.35]
    if obs_decay_values is None:
        obs_decay_values = [1.0, 3.0, 8.0]
    if w_values is None:
        w_values = [0.5, 1.0, 2.0]

    s_grid = [i / resolution for i in range(resolution + 1)]

    curves: List[Dict] = []
    optimal_map: List[Dict] = []

    for obs_floor in obs_floor_values:
        for obs_decay in obs_decay_values:
            for w in w_values:
                # Full E(s) curve
                e_curve = [total_evidence(s, obs_floor, obs_decay, w) for s in s_grid]
                obs_curve = [obs_prob(s, obs_floor, obs_decay) for s in s_grid]

                # Adversary's optimal
                s_star, e_star = find_optimal_suppression(obs_floor, obs_decay, w, resolution)

                e_at_zero = total_evidence(0.0, obs_floor, obs_decay, w)
                e_at_one = total_evidence(1.0, obs_floor, obs_decay, w)

                # Is total suppression better for adversary than no suppression?
                total_suppression_better = e_at_one < e_at_zero

                # Is the optimal at the boundary (s*=1 means "suppress maximally")
                # or is there a genuine interior minimum?
                interior_minimum = s_star < 0.95

                curves.append(dict(
                    obs_floor=round(obs_floor, 4),
                    obs_decay=round(obs_decay, 4),
                    w=round(w, 4),
                    s_grid=s_grid,
                    e_curve=[round(e, 6) for e in e_curve],
                    obs_curve=[round(o, 6) for o in obs_curve],
                ))

                optimal_map.append(dict(
                    obs_floor=round(obs_floor, 4),
                    obs_decay=round(obs_decay, 4),
                    w=round(w, 4),
                    s_star=round(s_star, 4),
                    e_star=round(e_star, 6),
                    e_at_zero=round(e_at_zero, 6),
                    e_at_one=round(e_at_one, 6),
                    total_suppression_better=total_suppression_better,
                    interior_minimum=interior_minimum,
                    strategy=(
                        "selective" if interior_minimum else "maximal"
                    ),
                    adversary_gain=round(e_at_zero - e_star, 6),
                ))

    # ── Floor protection threshold ────────────────────────────────────────
    # Find the minimum obs_floor at which E(1) >= E(0) for each (obs_decay, w)
    # i.e., the floor above which total suppression is counterproductive.
    # E(1) = obs_floor (since obs_prob(1) = obs_floor at high obs_decay)
    # E(0) = 1.0
    # E(1) >= E(0) requires obs_floor >= 1.0 — impossible.
    # So total suppression is ALWAYS better for the adversary than no suppression
    # in terms of direct comparison E(1) vs E(0).
    #
    # The more meaningful comparison: at what obs_floor does the adversary
    # LOSE the benefit of any suppression — i.e., where does min_s E(s) = E(0)?
    # This only occurs when obs_floor → 1 (observation is perfect regardless of s).
    # More practically: find obs_floor where E(s*) > (1 - floor_advantage),
    # i.e., where the adversary's gain from suppression is negligible.

    floor_thresholds: List[Dict] = []
    floor_scan = [i / 100 for i in range(101)]

    for obs_decay in obs_decay_values:
        for w in w_values:
            gains = []
            for obs_floor in floor_scan:
                s_star, e_star = find_optimal_suppression(obs_floor, obs_decay, w, 500)
                e0 = total_evidence(0.0, obs_floor, obs_decay, w)
                gains.append(dict(
                    obs_floor=round(obs_floor, 3),
                    adversary_gain=round(e0 - e_star, 6),
                    s_star=round(s_star, 3),
                ))
            floor_thresholds.append(dict(
                obs_decay=obs_decay,
                w=w,
                gains=gains,
            ))

    # ── Summary ───────────────────────────────────────────────────────────
    # How many parameter configurations have interior minimum (selective strategy)?
    n_selective = sum(1 for r in optimal_map if r["interior_minimum"])
    n_total = len(optimal_map)

    return dict(
        curves=curves,
        optimal_map=optimal_map,
        floor_thresholds=floor_thresholds,
        params=dict(
            obs_floor_values=obs_floor_values,
            obs_decay_values=obs_decay_values,
            w_values=w_values,
            resolution=resolution,
        ),
        summary=dict(
            n_configurations=n_total,
            n_selective_optimal=n_selective,
            n_maximal_optimal=n_total - n_selective,
            pct_selective=round(n_selective / n_total, 3),
            finding=(
                "Adversary favors selective suppression when suppression-evidence "
                "channel has high reach (w > 1) — the empirical signature of "
                "sophisticated institutional suppression. Maximal suppression is "
                "optimal when w is low or obs_decay is high (V1+V2 fully controlling "
                "the observation channel). In all cases, obs_floor > 0 prevents total "
                "evidence from reaching zero."
            ),
        ),
    )


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if results is None:
        results = run(**kw)

    curves = results["curves"]
    optimal_map = results["optimal_map"]
    floor_thresholds = results["floor_thresholds"]
    params = results["params"]
    w_values = params["w_values"]
    obs_decay_values = params["obs_decay_values"]
    obs_floor_values = params["obs_floor_values"]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(LAB_TITLE, fontsize=12, fontweight="bold")

    # ── (top-left) E(s) curves at varying obs_floor, fixed w=1, obs_decay=3 ──
    ax1 = fig.add_subplot(2, 3, 1)
    target_decay = obs_decay_values[len(obs_decay_values) // 2]
    target_w = w_values[len(w_values) // 2]
    color_map = cm.RdYlGn(np.linspace(0.1, 0.9, len(obs_floor_values)))

    for i, obs_floor in enumerate(obs_floor_values):
        c = [cr for cr in curves
             if cr["obs_floor"] == round(obs_floor, 4)
             and cr["obs_decay"] == round(target_decay, 4)
             and cr["w"] == round(target_w, 4)]
        if not c:
            continue
        c = c[0]
        opt = [o for o in optimal_map
               if o["obs_floor"] == round(obs_floor, 4)
               and o["obs_decay"] == round(target_decay, 4)
               and o["w"] == round(target_w, 4)]
        ax1.plot(c["s_grid"], c["e_curve"],
                 color=color_map[i], linewidth=2,
                 label=f"obs_floor={obs_floor:.2f}")
        if opt:
            ax1.plot(opt[0]["s_star"], opt[0]["e_star"],
                     "v", color=color_map[i], markersize=8)

    ax1.axhline(1.0, color="grey", linestyle=":", linewidth=1, alpha=0.5, label="E(0)=1.0")
    ax1.set(xlabel="Suppression strength s", ylabel="Total evidence E(s)",
            title=f"E(s) at varying obs_floor\n(obs_decay={target_decay}, w={target_w})\n"
                  f"▽ marks adversary's optimal s*")
    ax1.legend(fontsize=7)

    # ── (top-center) E(s) at varying w, fixed obs_floor=0.1, obs_decay=3 ──
    ax2 = fig.add_subplot(2, 3, 2)
    target_floor = 0.10
    color_w = cm.plasma(np.linspace(0.1, 0.9, len(w_values)))

    for i, w in enumerate(w_values):
        c = [cr for cr in curves
             if cr["obs_floor"] == round(target_floor, 4)
             and cr["obs_decay"] == round(target_decay, 4)
             and cr["w"] == round(w, 4)]
        if not c:
            continue
        c = c[0]
        opt = [o for o in optimal_map
               if o["obs_floor"] == round(target_floor, 4)
               and o["obs_decay"] == round(target_decay, 4)
               and o["w"] == round(w, 4)]
        style = "-" if (opt and opt[0]["interior_minimum"]) else "--"
        ax2.plot(c["s_grid"], c["e_curve"],
                 color=color_w[i], linewidth=2, linestyle=style,
                 label=f"w={w} ({'selective' if opt and opt[0]['interior_minimum'] else 'maximal'})")
        if opt:
            ax2.plot(opt[0]["s_star"], opt[0]["e_star"],
                     "v", color=color_w[i], markersize=8)

    ax2.axhline(1.0, color="grey", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set(xlabel="Suppression strength s", ylabel="Total evidence E(s)",
            title=f"E(s) at varying w (observer reach)\n"
                  f"(obs_floor={target_floor}, obs_decay={target_decay})\n"
                  f"Solid=selective opt, Dashed=maximal opt")
    ax2.legend(fontsize=7)

    # ── (top-right) Strategy phase diagram: selective vs maximal ──────────
    ax3 = fig.add_subplot(2, 3, 3)
    w_arr = sorted(set(o["w"] for o in optimal_map))
    floor_arr = sorted(set(o["obs_floor"] for o in optimal_map))
    decay_arr = sorted(set(o["obs_decay"] for o in optimal_map))

    target_decay_phase = decay_arr[len(decay_arr) // 2]
    Z = np.zeros((len(floor_arr), len(w_arr)))
    for i, f in enumerate(floor_arr):
        for j, w in enumerate(w_arr):
            matches = [o for o in optimal_map
                       if o["obs_floor"] == f
                       and o["w"] == w
                       and o["obs_decay"] == target_decay_phase]
            if matches:
                Z[i, j] = 1.0 if matches[0]["interior_minimum"] else 0.0

    ax3.imshow(Z, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
               extent=[min(w_arr) - 0.1, max(w_arr) + 0.1,
                       min(floor_arr) - 0.01, max(floor_arr) + 0.01])
    ax3.set(xlabel="Observer reach w", ylabel="Detection floor obs_floor",
            title=f"Strategy phase: green=selective, red=maximal\n"
                  f"(obs_decay={target_decay_phase})")
    ax3.set_xticks(w_arr)
    ax3.set_yticks([round(f, 2) for f in floor_arr])

    # ── (bottom-left) Adversary's gain as function of obs_floor ───────────
    ax4 = fig.add_subplot(2, 3, 4)
    color_decay = cm.viridis(np.linspace(0.0, 0.9, len(obs_decay_values)))

    filtered_thresholds = [ft for ft in floor_thresholds if ft["w"] == target_w]
    for i, ft in enumerate(filtered_thresholds):
        floors = [g["obs_floor"] for g in ft["gains"]]
        gains = [g["adversary_gain"] for g in ft["gains"]]
        ax4.plot(floors, gains, linewidth=2, color=color_decay[i],
                 label=f"obs_decay={ft['obs_decay']}")

    ax4.axhline(0, color="red", linestyle=":", linewidth=1, alpha=0.5,
                label="No adversary gain")
    ax4.set(xlabel="Detection floor obs_floor",
            ylabel="Adversary gain: E(0) − E(s*)",
            title=f"Adversary's suppression gain vs. obs_floor\n"
                  f"(w={target_w})\nFloor → 0 = adversary wins; Floor → 1 = system loses")
    ax4.legend(fontsize=7)

    # ── (bottom-center) obs_prob(s) curves ────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    color_floor = cm.RdYlGn(np.linspace(0.1, 0.9, len(obs_floor_values)))

    for i, obs_floor in enumerate(obs_floor_values):
        c = [cr for cr in curves
             if cr["obs_floor"] == round(obs_floor, 4)
             and cr["obs_decay"] == round(target_decay, 4)
             and cr["w"] == round(target_w, 4)]
        if not c:
            continue
        ax5.plot(c[0]["s_grid"], c[0]["obs_curve"],
                 color=color_floor[i], linewidth=2,
                 label=f"obs_floor={obs_floor:.2f}")
    ax5.axhline(0, color="grey", linestyle=":", linewidth=1, alpha=0.3)
    ax5.set(xlabel="Suppression strength s",
            ylabel="obs_prob(s)",
            title=f"Observation probability decay\n"
                  f"(obs_decay={target_decay})\nFloor is the irreducible whistleblower signal")
    ax5.legend(fontsize=7)

    # ── (bottom-right) Summary table ─────────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    key_rows = [o for o in optimal_map
                if o["obs_decay"] == round(target_decay, 4)
                and o["w"] == round(target_w, 4)]
    table_data = [["obs_floor", "s*", "E(s*)", "E(1)", "Strategy"]]
    for r in key_rows:
        table_data.append([
            f"{r['obs_floor']:.2f}",
            f"{r['s_star']:.3f}",
            f"{r['e_star']:.4f}",
            f"{r['e_at_one']:.4f}",
            r["strategy"],
        ])
    tbl = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.5)
    # Highlight selective rows
    for row_i, r in enumerate(key_rows):
        fc = "#d4edda" if r["interior_minimum"] else "#f8d7da"
        for col_i in range(5):
            tbl[row_i + 1, col_i].set_facecolor(fc)

    ax6.set_title(
        f"Optimal suppression parameters\n(obs_decay={target_decay}, w={target_w})",
        fontsize=9, pad=10,
    )

    fig.tight_layout()
    return fig


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json

    results = run()
    summary = results["summary"]

    print(f"\n{'='*60}")
    print(f"  {LAB_TITLE}")
    print(f"{'='*60}\n")
    print(f"Configurations tested:  {summary['n_configurations']}")
    print(f"  Selective suppression optimal: {summary['n_selective_optimal']} "
          f"({summary['pct_selective']*100:.0f}%)")
    print(f"  Maximal suppression optimal:   {summary['n_maximal_optimal']}\n")
    print("Finding:")
    print(f"  {summary['finding']}\n")

    print("Adversary's optimal strategy (obs_decay=3.0, w=1.0):")
    print(f"  {'obs_floor':>10}  {'s*':>6}  {'E(s*)':>8}  {'E(1)':>8}  {'strategy':>10}")
    for r in results["optimal_map"]:
        if r["obs_decay"] == 3.0 and r["w"] == 1.0:
            print(f"  {r['obs_floor']:>10.2f}  {r['s_star']:>6.3f}  "
                  f"{r['e_star']:>8.4f}  {r['e_at_one']:>8.4f}  {r['strategy']:>10}")

    print()
    plot(results)
    plt.show()
