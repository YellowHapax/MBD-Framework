"""Deontological Tests — P8 / P9 / P10

Paper 6: The Resonant Gate (10.5281/zenodo.17352481)

Three testable predictions from the resonant-gate formalism:

P8  Deontological Blindness: Very high κ creates a shortcut where the
    agent uses its own baseline to predict the Other.  This succeeds
    for familiar topics but fails for genuinely novel private
    experience, producing surprisingly large prediction errors.

P9  Neurological Dissociation of Error Signals: "World glitch" (I-error)
    and "social glitch" (D-error) are computationally distinct.  MBD
    predicts they activate different neural substrates.

P10 Deontological Immunity: κ ≈ 0 renders the agent immune to social
    pressure to believe something false, while a typical agent's
    baseline is corrupted by relational influence.

Extracted from MemoryLab-OS/elucidation.tsx simulation contexts.
"""

from __future__ import annotations
from typing import Any, Dict, List

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 6
PAPER_TITLE = "The Resonant Gate"
PAPER_DOI = "10.5281/zenodo.17352481"
LAB_TITLE = "Deontological Tests (P8 / P9 / P10)"

THESIS = (
    "Three testable predictions: (P8) high-κ agents become 'blind' to the "
    "Other's truly novel private experience because they use their own "
    "baseline as a shortcut; (P9) world-prediction error and social-"
    "prediction error are computationally distinct and should activate "
    "different brain networks; (P10) an agent with κ ≈ 0 is immune to "
    "social pressure, remaining anchored to objective fact while a coupled "
    "agent's baseline is corrupted."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── P8: Deontological Blindness ────────────────────────────────────────────

def _run_p8(
    *,
    kappa_low: float = 0.10,
    kappa_high: float = 0.90,
    lambda_val: float = 0.10,
    shared_topics: int = 20,
    novel_topic_t: int = 25,
    novel_private_value: float = 1.50,
    steps: int = 40,
) -> Dict[str, Any]:
    """P8: prediction-error spike for high-κ agent on novel private topic."""
    results: Dict[str, List[Dict]] = {"low_k": [], "high_k": []}

    for label, kappa in [("low_k", kappa_low), ("high_k", kappa_high)]:
        b_self, b_model_other = 0.0, 0.0
        for t in range(steps):
            # Shared experience most of the time
            other_true = 0.5 if t < shared_topics else 0.5
            if t == novel_topic_t:
                other_true = novel_private_value  # Genuinely novel private exp

            # Agent predicts Other ≈ self (shortcut scales with κ)
            prediction = b_self * kappa + b_model_other * (1.0 - kappa)
            pe = abs(other_true - prediction)

            b_self = b_self * (1.0 - lambda_val) + other_true * lambda_val * kappa
            b_model_other = b_model_other * (1.0 - lambda_val) + other_true * lambda_val

            results[label].append(dict(time=t, prediction_error=round(pe, 5),
                                       prediction=round(prediction, 5)))

    return dict(
        timeseries=results,
        summary=dict(
            blindness_spike_low=results["low_k"][novel_topic_t]["prediction_error"],
            blindness_spike_high=results["high_k"][novel_topic_t]["prediction_error"],
        ),
    )


# ── P9: Error Dissociation ─────────────────────────────────────────────────

def _run_p9() -> Dict[str, Any]:
    """P9: world-error vs social-error are computationally distinct."""
    scenarios = [
        dict(label="World Glitch", source="I-channel",
             description="Coffee cup appears unexpectedly",
             error_type="sensory prediction error",
             predicted_substrate="vmPFC / cerebellum"),
        dict(label="Social Glitch", source="D-channel",
             description="Friend acts irrationally",
             error_type="deontological prediction error",
             predicted_substrate="TPJ / mPFC"),
    ]

    # Illustrative error magnitudes under same stimulus amplitude
    I_error = abs(0.0 - 1.0)  # No cup expected → cup appears
    D_error = abs(0.8 - (-0.5))  # Trusted friend acts hostile
    return dict(
        scenarios=scenarios,
        magnitudes=dict(world_error=round(I_error, 3),
                        social_error=round(D_error, 3)),
        summary=dict(
            distinct=True,
            note=("MBD predicts these are irreducible to a single 'surprise' signal. "
                  "fMRI should show double dissociation between vmPFC and TPJ."),
        ),
    )


# ── P10: Deontological Immunity ────────────────────────────────────────────

def _run_p10(
    *,
    kappa_typical: float = 0.70,
    kappa_immune: float = 0.02,
    lambda_val: float = 0.10,
    truth: float = 1.00,
    social_pressure: float = -0.50,
    pressure_start: int = 20,
    steps: int = 60,
) -> Dict[str, Any]:
    """P10: κ≈0 agent resists social pressure."""
    results: Dict[str, List[Dict]] = {"typical": [], "immune": []}

    for label, kappa in [("typical", kappa_typical), ("immune", kappa_immune)]:
        b = truth
        for t in range(steps):
            D = social_pressure if t >= pressure_start else 0.0
            b = b * (1.0 - lambda_val) + (b + kappa * D) * lambda_val
            results[label].append(dict(time=t, belief=round(b, 5)))

    return dict(
        timeseries=results,
        summary=dict(
            typical_final=results["typical"][-1]["belief"],
            immune_final=results["immune"][-1]["belief"],
            corruption=round(abs(truth - results["typical"][-1]["belief"]), 4),
            immunity=round(abs(results["immune"][-1]["belief"] - truth) < 0.1, 4),
        ),
    )


# ── Public Interface ────────────────────────────────────────────────────────

def run(**kw) -> Dict[str, Any]:
    """Run all three deontological tests."""
    return dict(
        p8=_run_p8(),
        p9=_run_p9(),
        p10=_run_p10(),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt

    if results is None:
        results = run(**kw)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    # P8
    ax = axes[0]
    p8 = results["p8"]
    for label, color in [("low_k", "#82ca9d"), ("high_k", "#ff4d4d")]:
        data = p8["timeseries"][label]
        ax.plot([d["time"] for d in data], [d["prediction_error"] for d in data],
                label=f"κ={'low' if 'low' in label else 'high'}", color=color)
    ax.set(xlabel="Time", ylabel="Prediction Error",
           title="P8: Deontological Blindness")
    ax.legend()

    # P9
    ax = axes[1]
    p9 = results["p9"]
    ax.barh([0, 1],
            [p9["magnitudes"]["world_error"], p9["magnitudes"]["social_error"]],
            color=["#8884d8", "#ff8042"])
    ax.set(yticks=[0, 1], yticklabels=["World Glitch (I)", "Social Glitch (D)"],
           xlabel="Error Magnitude", title="P9: Error Signal Dissociation")

    # P10
    ax = axes[2]
    p10 = results["p10"]
    for label, color in [("typical", "#ff8042"), ("immune", "#82ca9d")]:
        data = p10["timeseries"][label]
        ax.plot([d["time"] for d in data], [d["belief"] for d in data],
                label=f"{'Typical' if 'typ' in label else 'Immune'} Agent", color=color)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.4, label="Truth")
    ax.axvline(20, color="red", linestyle="--", alpha=0.3, label="Pressure starts")
    ax.set(xlabel="Time", ylabel="Belief Value",
           title="P10: Deontological Immunity")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
