"""Resonant Re-instantiation — Episodic Encoding & Recall

Paper 3: Episodic Recall as Resonant Re-instantiation
(10.5281/zenodo.17374270)

Phase 1 — Encoding: wandering baseline encounters events; only events
whose novelty exceeds θ_h produce an episodic trace via P(WRITE) gating.
Phase 2 — Recall: a context cue is matched against stored traces within
a search radius θ_search.  Demonstrates context-scaffolded retrieval.

Ported from MemoryLab-OS/resonantReinstantiation.tsx (138 lines).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

import numpy as np

# ── Metadata ────────────────────────────────────────────────────────────────

PAPER = 3
PAPER_TITLE = "Episodic Recall as Resonant Re-instantiation"
PAPER_DOI = "10.5281/zenodo.17374270"
LAB_TITLE = "Resonant Re-instantiation: Encoding & Recall"

THESIS = (
    "Episodic memory is not simply 'recording' — it is a novelty-gated "
    "encoding process. Only events that deviate sufficiently from the "
    "current baseline (exceeding threshold θ_h) trigger a state write. "
    "Retrieval is context-scaffolded: a memory can only be accessed if the "
    "current cognitive context is sufficiently similar to the context at "
    "encoding, measured within a search radius θ_search."
)


def describe() -> Dict[str, str]:
    return dict(
        paper=PAPER, paper_title=PAPER_TITLE, paper_doi=PAPER_DOI,
        lab_title=LAB_TITLE, thesis=THESIS,
    )


# ── Simulation ──────────────────────────────────────────────────────────────

def run(
    *,
    n_timesteps: int = 100,
    dims: int = 2,
    lambda_drift: float = 0.05,
    sigma_noise: float = 0.03,
    theta_gate_write: float = 0.35,
    theta_search: float = 0.30,
    t_novel_event: int = 40,
    novel_event_offset: Optional[List[float]] = None,
    t_recall_matched: int = 80,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulate episodic encoding and context-scaffolded recall.

    Parameters
    ----------
    n_timesteps : int          Length of the experience stream.
    dims : int                 Dimensionality of the baseline (2 for visualisation).
    lambda_drift : float       Baseline drift magnitude per step.
    sigma_noise : float        Random noise per step.
    theta_gate_write : float   Write-gating novelty threshold.
    theta_search : float       Retrieval context-match radius (squared distance).
    t_novel_event : int        Timestep of the genuinely novel event.
    novel_event_offset : list  Displacement vector for the novel event (defaults to [0.8, 0.5]).
    t_recall_matched : int     Timestep of the recall attempt.
    seed : int                 RNG seed.
    """
    rng = np.random.RandomState(seed)
    if novel_event_offset is None:
        novel_event_offset = [0.8, 0.5]

    baseline = np.zeros((n_timesteps, dims))
    novelty_signal = np.zeros(n_timesteps)
    p_write_signal = np.zeros(n_timesteps)
    episodic_traces: List[Dict] = []
    log: List[str] = []

    # Build baseline trajectory
    for t in range(1, n_timesteps):
        drift = rng.randn(dims) * lambda_drift
        noise = rng.randn(dims) * sigma_noise
        baseline[t] = baseline[t - 1] + drift + noise

    # Inject novel event
    if t_novel_event < n_timesteps:
        baseline[t_novel_event] = baseline[t_novel_event - 1] + np.array(novel_event_offset[:dims])

    # Compute novelty and encoding
    target_trace = None
    for t in range(1, n_timesteps):
        diff = baseline[t] - baseline[t - 1]
        nov = float(np.linalg.norm(diff))
        novelty_signal[t] = nov

        # Sigmoid-ish P(WRITE) around threshold
        pw = 1.0 / (1.0 + math.exp(-10.0 * (nov - theta_gate_write)))
        p_write_signal[t] = pw

        if nov > theta_gate_write:
            trace = dict(
                time=int(t),
                context=baseline[t].tolist(),
                novelty=round(nov, 4),
            )
            episodic_traces.append(trace)
            if t == t_novel_event:
                target_trace = trace
                log.append(f"t={t}: NOVEL EVENT encoded (N={nov:.3f} > θ={theta_gate_write})")
            else:
                log.append(f"t={t}: trace stored (N={nov:.3f})")

    # Phase 2: Recall
    if target_trace is not None:
        enc_context = np.array(target_trace["context"])
    else:
        enc_context = baseline[t_novel_event]

    # Matched cue: average of recall-time baseline and encoding context
    recall_baseline = baseline[min(t_recall_matched, n_timesteps - 1)]
    matched_cue = (recall_baseline + enc_context) / 2.0

    def attempt_recall(cue: np.ndarray) -> Dict:
        if not episodic_traces:
            return dict(success=False, candidates=0)
        candidates = []
        for tr in episodic_traces:
            ctx = np.array(tr["context"])
            dist_sq = float(np.sum((cue - ctx) ** 2))
            if dist_sq < theta_search:
                candidates.append(tr)
        return dict(success=len(candidates) > 0, candidates=len(candidates))

    recall_result = attempt_recall(matched_cue)
    log.append(f"t={t_recall_matched}: recall attempt → "
               f"{'SUCCESS' if recall_result['success'] else 'FAILED'} "
               f"({recall_result['candidates']} candidates)")

    # Build output data
    baseline_path = [dict(x=float(baseline[t, 0]), y=float(baseline[t, 1]), t=int(t))
                     for t in range(n_timesteps)]
    mundane_traces = [dict(x=tr["context"][0], y=tr["context"][1])
                      for tr in episodic_traces if tr["time"] != t_novel_event]
    target_point = (dict(x=target_trace["context"][0], y=target_trace["context"][1])
                    if target_trace else None)
    matched_cue_point = dict(x=float(matched_cue[0]), y=float(matched_cue[1]))
    time_series = [dict(time=int(t), novelty=round(float(novelty_signal[t]), 4),
                        pWrite=round(float(p_write_signal[t]), 4))
                   for t in range(n_timesteps)]

    return dict(
        timeseries=time_series,
        scatter=dict(
            baseline_path=baseline_path,
            mundane_traces=mundane_traces,
            target_trace=target_point,
            matched_cue=matched_cue_point,
        ),
        log=log,
        recall_result=recall_result,
        summary=dict(
            traces_stored=len(episodic_traces),
            novel_event_encoded=target_trace is not None,
            recall_success=recall_result["success"],
            recall_candidates=recall_result["candidates"],
        ),
        params=dict(
            n_timesteps=n_timesteps, dims=dims, lambda_drift=lambda_drift,
            sigma_noise=sigma_noise, theta_gate_write=theta_gate_write,
            theta_search=theta_search, t_novel_event=t_novel_event,
            t_recall_matched=t_recall_matched,
        ),
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot(results=None, **kw):
    import matplotlib.pyplot as plt

    if results is None:
        results = run(**kw)
    sc = results["scatter"]
    ts = results["timeseries"]
    p = results["params"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(LAB_TITLE, fontsize=14, fontweight="bold")

    # Scatter: baseline path + traces
    bx = [pt["x"] for pt in sc["baseline_path"]]
    by = [pt["y"] for pt in sc["baseline_path"]]
    ax1.plot(bx, by, "-", color="gray", alpha=0.4, label="Baseline Path")

    if sc["mundane_traces"]:
        mx = [pt["x"] for pt in sc["mundane_traces"]]
        my = [pt["y"] for pt in sc["mundane_traces"]]
        ax1.scatter(mx, my, marker="x", color="black", s=40, label="Mundane Traces")

    if sc["target_trace"]:
        ax1.scatter(sc["target_trace"]["x"], sc["target_trace"]["y"],
                    marker="*", color="red", s=200, zorder=5, label="Novel Trace")

    ax1.scatter(sc["matched_cue"]["x"], sc["matched_cue"]["y"],
                marker="^", color="green", s=120, zorder=5, label="Matched Cue")
    ax1.set(xlabel="Dim 1", ylabel="Dim 2", title="Context Space (Encoding & Recall)")
    ax1.legend(fontsize=8)

    # Time series: novelty + P(WRITE)
    t = [d["time"] for d in ts]
    ax2b = ax2.twinx()
    ax2.plot(t, [d["novelty"] for d in ts], color="orange", label="Novelty")
    ax2b.plot(t, [d["pWrite"] for d in ts], color="green", alpha=0.7, label="P(WRITE)")
    ax2.axhline(p["theta_gate_write"], color="black", linestyle="--", alpha=0.5, label="θ_h")
    ax2.axvline(p["t_novel_event"], color="red", linestyle="--", alpha=0.5)
    ax2.axvline(p["t_recall_matched"], color="green", linestyle="--", alpha=0.5)
    ax2.set(xlabel="Time", ylabel="Novelty", title="Novelty & Write Probability")
    ax2b.set_ylabel("P(WRITE)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plot()
    plt.show()
