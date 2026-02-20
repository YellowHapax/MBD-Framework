"""Endemic Baseline — Calibration Failure and Healthy-Input Rejection

Paper 7: The Endemic Baseline

Phenomenon P7a: An agent whose B(0) was set during chronic disruption
(the "storm-born" agent) receives healthy inputs identical to those
that successfully move a neurotypical agent toward health.

MBD predicts three things in the endemic case:
    1. HIGH NOVELTY AMPLIFICATION: The same positive input registers as
       maximally novel (|I - B_storm| >> theta_h), far exceeding the
       incongruence seen in Paper 5's mood-incongruent case. Per the
       Emergent Gate logic, maximum novelty means maximum encoding
       *weight*, but without a target attractor basin to land in, this
       produces destabilization rather than integration.

    2. KAPPA REJECTION: Under low baseline kappa (endemic agents adapt
       to chronic disruption by maintaining low relational coupling),
       inputs above a novelty ceiling are rejected rather than integrated.
       The system protects the only stable attractor it knows: B_storm.

    3. IDENTITY DISRUPTION WITHOUT LANDING ZONE: Even when the Emergent
       Gate fires and the baseline *does* shift, the shift is away from
       B_storm and into H_accessible \\ H_agent — uncharted territory
       with no attractor structure, producing acute groundlessness.

The lab compares three agents receiving identical positive inputs:
    - Neurotypical:  B(0) = -0.1 (mild weather, reference exists)
    - Storm-born:    B(0) = -0.85 (endemic; entire reference is storm)
    - Acute-onset:   B(0) = -0.85 with prior B_ref = 0.4 (has been sunny before)

Only the storm-born agent lacks a prior B_ref. The acute-onset agent
with the same current baseline integrates the same inputs successfully
because recovery has a *target* — the prior healthy encoding.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 7
PAPER_TITLE = "The Endemic Baseline"
LAB_TITLE = "Phenomenon P7a: Endemic Baseline — Healthy-Input Rejection"

THESIS = (
    "A storm-born agent (B(0) set during chronic disruption; no prior healthy "
    "encoding) receives the same positive inputs that successfully move a "
    "neurotypical agent toward health. MBD predicts the storm-born agent will "
    "experience maximum novelty (far exceeding mood-incongruence in Paper 5), "
    "activate the kappa-rejection mechanism, and fail to build a stable new "
    "attractor. The neurotypical and acute-onset agents integrate. The endemic "
    "agent is destabilized without landing. The failure is structural, not "
    "motivational."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Core dataclass ───────────────────────────────────────────────────────────

@dataclass
class Agent:
    name: str
    b0: float
    kappa: float              # relational coupling; governs integration weight
    b_ref_healthy: Optional[float]  # None iff endemic (the critical flag)
    lambda_val: float = 0.08

    def __post_init__(self):
        self.b = self.b0
        self.endemic = self.b_ref_healthy is None

    @property
    def has_sunny_reference(self) -> bool:
        return self.b_ref_healthy is not None


# ── Simulation ───────────────────────────────────────────────────────────────

def run(
    *,
    steps: int = 120,
    event_start: int = 40,
    event_duration: int = 40,
    healthy_input: float = 0.70,
    theta_h: float = 0.35,
    # Kappa rejection: inputs with novelty > kappa_ceiling * (1 - kappa) are
    # partially rejected; integration weight scales with kappa.
    kappa_ceiling_factor: float = 2.0,
) -> Dict[str, Any]:
    """Simulate three agents receiving identical sustained positive input.

    The key experimental distinction:
      - Neurotypical and Acute-onset agents differ only in current B, not in
        whether a prior B_ref exists.
      - Storm-born agent has no B_ref — the endemic flag is set.

    Returns a dict with per-agent time-series and a summary comparison.
    """
    agents = [
        Agent(
            name="Neurotypical",
            b0=-0.10,
            kappa=0.65,
            b_ref_healthy=0.40,
        ),
        Agent(
            name="Acute-onset (same B, prior reference exists)",
            b0=-0.85,
            kappa=0.65,
            b_ref_healthy=0.40,   # Has been sunny before; reference is real
        ),
        Agent(
            name="Storm-born (endemic; B_ref = None)",
            b0=-0.85,
            kappa=0.25,           # Low kappa: rational adaptation to chronic disruption
            b_ref_healthy=None,   # ← The endemic flag. No prior healthy encoding.
        ),
    ]

    series: Dict[str, List[Dict]] = {ag.name: [] for ag in agents}

    for t in range(steps):
        in_event = event_start <= t < event_start + event_duration
        I = healthy_input if in_event else 0.0

        for ag in agents:
            novelty = abs(I - ag.b)
            p_write = 1.0 / (1.0 + math.exp(-10.0 * (novelty - theta_h)))

            # Kappa-gated integration:
            # High kappa → input is integrated at face value (λ unchanged).
            # Low kappa + high novelty → partial rejection; effective_λ reduced.
            novelty_overload = novelty > kappa_ceiling_factor * (1.0 - ag.kappa)
            if novelty_overload:
                # The system protects its attractor by muting integration.
                effective_lambda = ag.lambda_val * ag.kappa
                rejected = True
            else:
                effective_lambda = ag.lambda_val
                rejected = False

            b_prev = ag.b
            ag.b = ag.b * (1.0 - effective_lambda) + I * effective_lambda

            # Destabilization index: how far has the agent moved without
            # an attractor to land on? For endemic agents in H_accessible \ H_agent,
            # every step away from B_storm is into uncharted space.
            if ag.endemic and I != 0.0:
                # No prior encoding exists near the new position.
                # Destabilization ∝ distance from storm × (1 - kappa).
                destabilization = abs(ag.b - ag.b0) * (1.0 - ag.kappa)
            else:
                # Non-endemic agents have a target attractor; movement is
                # toward something, not into void.
                destabilization = 0.0

            series[ag.name].append(dict(
                time=t,
                baseline=round(ag.b, 5),
                input=round(I, 4),
                novelty=round(novelty, 5),
                p_write=round(p_write, 5),
                kappa=ag.kappa,
                rejected=rejected,
                destabilization=round(destabilization, 5),
                endemic=ag.endemic,
            ))

    # Summary: compare final baselines and peak destabilization
    comparison = []
    for ag in agents:
        s = series[ag.name]
        event_steps = [d for d in s if d["input"] > 0]
        comparison.append(dict(
            agent=ag.name,
            endemic=ag.endemic,
            b0=ag.b0,
            b_final=s[-1]["baseline"],
            b_shift=round(s[-1]["baseline"] - ag.b0, 5),
            peak_novelty=round(max(d["novelty"] for d in s), 5),
            peak_destabilization=round(max(d["destabilization"] for d in s), 5),
            rejections=sum(1 for d in event_steps if d["rejected"]),
            event_steps=len(event_steps),
            pct_rejected=round(
                sum(1 for d in event_steps if d["rejected"]) / max(len(event_steps), 1),
                3,
            ),
        ))

    return dict(
        series=series,
        comparison=comparison,
        params=dict(
            steps=steps,
            event_start=event_start,
            event_duration=event_duration,
            healthy_input=healthy_input,
            theta_h=theta_h,
        ),
        summary=dict(
            neurotypical_shift=comparison[0]["b_shift"],
            acute_onset_shift=comparison[1]["b_shift"],
            endemic_shift=comparison[2]["b_shift"],
            endemic_peak_destabilization=comparison[2]["peak_destabilization"],
            endemic_rejection_rate=comparison[2]["pct_rejected"],
        ),
    )


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if results is None:
        results = run(**kw)

    series = results["series"]
    comp = results["comparison"]
    params = results["params"]

    colors = {
        "Neurotypical": "#82ca9d",
        "Acute-onset (same B, prior reference exists)": "#ffa040",
        "Storm-born (endemic; B_ref = None)": "#e05555",
    }
    short_names = {
        "Neurotypical": "Neurotypical",
        "Acute-onset (same B, prior reference exists)": "Acute-onset",
        "Storm-born (endemic; B_ref = None)": "Storm-born (endemic)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(LAB_TITLE, fontsize=13, fontweight="bold")

    # ── (0,0) Baseline trajectories ──────────────────────────────────────
    ax = axes[0][0]
    for name, s in series.items():
        ts = [d["time"] for d in s]
        bs = [d["baseline"] for d in s]
        ax.plot(ts, bs, label=short_names[name], color=colors[name], linewidth=2)
    ax.axvspan(params["event_start"],
               params["event_start"] + params["event_duration"],
               alpha=0.08, color="green", label="Healthy-input window")
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=1, alpha=0.5)
    ax.set(xlabel="Time", ylabel="Baseline B(t)",
           title="Baseline Trajectories Under Healthy Input")
    ax.legend(fontsize=8)

    # ── (0,1) Peak novelty comparison ────────────────────────────────────
    ax = axes[0][1]
    names_short = [short_names[c["agent"]] for c in comp]
    novelties = [c["peak_novelty"] for c in comp]
    bar_colors = [colors[c["agent"]] for c in comp]
    bars = ax.bar(names_short, novelties, color=bar_colors)
    ax.axhline(params["theta_h"], color="red", linestyle="--",
               alpha=0.6, label=f"θ_h = {params['theta_h']}")
    ax.set(ylabel="Peak Novelty", title="Peak Novelty — Same Input, Different Reference")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, novelties):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # ── (1,0) Destabilization index (endemic agent only) ─────────────────
    ax = axes[1][0]
    for name, s in series.items():
        if not series[name][0]["endemic"]:
            continue
        ts = [d["time"] for d in s]
        ds = [d["destabilization"] for d in s]
        ax.fill_between(ts, ds, alpha=0.4, color=colors[name])
        ax.plot(ts, ds, color=colors[name], linewidth=1.5,
                label=short_names[name])
    ax.axvspan(params["event_start"],
               params["event_start"] + params["event_duration"],
               alpha=0.08, color="green")
    ax.set(xlabel="Time", ylabel="Destabilization Index",
           title="Storm-born: Destabilization Without Landing Zone\n"
                 "(movement away from only known attractor, no target basin)")
    ax.legend(fontsize=8)

    # ── (1,1) Rejection rate table ───────────────────────────────────────
    ax = axes[1][1]
    ax.axis("off")
    table_data = [
        ["Agent", "B₀", "B final", "Shift", "Rejection rate"],
    ]
    for c in comp:
        table_data.append([
            short_names[c["agent"]],
            f"{c['b0']:.2f}",
            f"{c['b_final']:.3f}",
            f"{c['b_shift']:+.3f}",
            f"{c['pct_rejected']*100:.0f}%{'  ← endemic' if c['endemic'] else ''}",
        ])
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    ax.set_title("Integration Outcomes — Identical Input, Structural Difference",
                 fontsize=10, pad=12)
    # Highlight endemic row
    for col in range(5):
        tbl[3, col].set_facecolor("#ffe0e0")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json

    results = run()
    print(json.dumps(results["summary"], indent=2))
    print("\nComparison:")
    for c in results["comparison"]:
        flag = " ← ENDEMIC" if c["endemic"] else ""
        print(f"  {c['agent'][:35]:35s}  "
              f"shift={c['b_shift']:+.4f}  "
              f"peak_novelty={c['peak_novelty']:.4f}  "
              f"rejected={c['pct_rejected']*100:.0f}%{flag}")
    plot(results)
    plt.show()
