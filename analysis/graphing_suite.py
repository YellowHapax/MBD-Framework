"""MBD Graphing Suite — Trauma and Collision Analytics.

This module rebuilds the lab's visualization pipeline. It reads the
``MBD_LAB_JOURNAL.jsonl`` stream, aggregates trauma tensors and collision
vectors, renders analytical plots, and emits a Markdown report that captures
both the quantitative metrics and the governing formulae.
"""

from __future__ import annotations

import json
import textwrap
from collections import Counter, defaultdict
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, cast

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - scripts should degrade gracefully without plots
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore

def _null_profiles():
    """Stub: returns empty profile list when no external data source is configured."""
    return []

list_profiles = _null_profiles

JOURNAL_PATH = Path("MBD_LAB_JOURNAL.jsonl")
REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "figures"


@dataclass
class TraumaRecord:
    index: int
    room: str
    character: str
    tcpb_delta: Dict[str, float]
    energy_l2: float
    compression: float
    expansion: float
    gravity: Optional[float]
    zone_pressure: Optional[float]


@dataclass
class CollisionRecord:
    index: int
    drylab: str
    turn: int
    agent_name: str
    other_agent_name: str
    baseline_delta: Dict[str, float]
    interference: Dict[str, float]
    novelty: float
    kappa: float
    consolidation_window_hours: float


@dataclass
class EcologyRecord:
    index: int
    tier: str
    pattern: str
    intensity: float
    entropy: Optional[float]
    participants: List[str]


@dataclass
class ResonanceSnapshot:
    index: int
    room: str
    character: str
    occupant_count: int
    baseline_before: Dict[str, float]
    baseline_after: Dict[str, float]
    bio_modulation: Dict[str, float]
    delta_pre_zone: Dict[str, float]
    delta_post_zone: Dict[str, float]
    zone_ideology: Optional[Dict[str, float]]
    gravity: float
    tactile_temperature: float
    tactile_luminosity: float
    delta_magnitude: float


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_event_count(count: int) -> str:
    unit = "event" if count == 1 else "events"
    return f"{count} {unit}"


def load_journal(path: Path = JOURNAL_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def extract_trauma(entries: Iterable[Dict[str, Any]]) -> List[TraumaRecord]:
    trauma_records: List[TraumaRecord] = []
    for idx, entry in enumerate(entries):
        if entry.get("event") != "trauma_tensor":
            continue
        tensor = entry.get("trauma_tensor", {})
        trauma_records.append(
            TraumaRecord(
                index=len(trauma_records),
                room=str(entry.get("room", "?")),
                character=str(entry.get("character", "?")),
                tcpb_delta={k: _safe_float(v) for k, v in entry.get("tcpb_delta", {}).items()},
                energy_l2=_safe_float(tensor.get("energy_l2")),
                compression=_safe_float(tensor.get("compression")),
                expansion=_safe_float(tensor.get("expansion")),
                gravity=None if tensor.get("gravity") is None else _safe_float(tensor.get("gravity")),
                zone_pressure=None if tensor.get("zone_pressure") is None else _safe_float(tensor.get("zone_pressure")),
            )
        )
    return trauma_records


def extract_collisions(entries: Iterable[Dict[str, Any]]) -> List[CollisionRecord]:
    collision_records: List[CollisionRecord] = []
    for idx, entry in enumerate(entries):
        if entry.get("event") != "collision_tensor":
            continue
        tensor = entry.get("collision_tensor", {})
        collision_records.append(
            CollisionRecord(
                index=len(collision_records),
                drylab=str(entry.get("drylab", "?")),
                turn=int(entry.get("turn", 0)),
                agent_name=str(entry.get("agent_name", "?")),
                other_agent_name=str(entry.get("other_agent_name", "?")),
                baseline_delta={k: _safe_float(v) for k, v in entry.get("baseline_delta", {}).items()},
                interference={k: _safe_float(v) for k, v in entry.get("interference", {}).items()},
                novelty=_safe_float(entry.get("novelty")),
                kappa=_safe_float(entry.get("kappa")),
                consolidation_window_hours=_safe_float(entry.get("consolidation_window_hours", tensor.get("consolidation_window_hours"))),
            )
        )
    return collision_records


def _pick_field(entry: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in entry and entry[key] is not None:
            return entry[key]
    return None


def extract_ecology(entries: Iterable[Dict[str, Any]]) -> List[EcologyRecord]:
    ecology_records: List[EcologyRecord] = []
    for entry in entries:
        event_type = entry.get("event")
        if event_type not in {"ecology_event", "ecology_snapshot", "ecosystem_event"}:
            continue
        payload = entry.get("ecology") if isinstance(entry.get("ecology"), dict) else {}
        tier = (_pick_field(entry, "tier", "structure_tier")
                or _pick_field(payload or {}, "tier", "structure_tier")
                or "unclassified")
        pattern = (_pick_field(entry, "pattern", "ecology_type", "dynamic")
                   or _pick_field(payload or {}, "pattern", "type", "dynamic")
                   or "niche")
        intensity = _safe_float(_pick_field(entry, "intensity", "magnitude")
                                or _pick_field(payload or {}, "intensity", "magnitude"))
        entropy = _pick_field(entry, "entropy") or _pick_field(payload or {}, "entropy")
        entropy_val = None if entropy is None else _safe_float(entropy)
        participants_field = _pick_field(entry, "participants", "actors")
        if participants_field is None:
            participants_field = _pick_field(payload or {}, "participants", "actors", "entities")
        if isinstance(participants_field, str):
            participants = [participants_field]
        elif isinstance(participants_field, IterableABC):
            participants = [str(item) for item in participants_field]
        else:
            participants = []
        ecology_records.append(
            EcologyRecord(
                index=len(ecology_records),
                tier=str(tier).lower(),
                pattern=str(pattern).lower().replace(" ", "_"),
                intensity=intensity,
                entropy=entropy_val,
                participants=participants,
            )
        )
    return ecology_records


def extract_resonance(entries: Iterable[Dict[str, Any]]) -> List[ResonanceSnapshot]:
    resonance_records: List[ResonanceSnapshot] = []
    for entry in entries:
        if entry.get("event") != "resonance_snapshot":
            continue
        baseline_before_raw = entry.get("baseline_before", {}) or {}
        baseline_after_raw = entry.get("baseline_after", {}) or {}
        bio_modulation_raw = entry.get("bio_modulation", {}) or {}
        delta_pre_raw = entry.get("tcpb_delta_pre_zone", {}) or {}
        delta_post_raw = entry.get("tcpb_delta_post_zone", {}) or {}
        zone_raw = entry.get("zone_ideology")
        zone_dict: Optional[Dict[str, float]]
        if isinstance(zone_raw, dict):
            zone_dict = {k: _safe_float(v) for k, v in zone_raw.items()}
        else:
            zone_dict = None

        tactile_raw = entry.get("tactile_state", {}) or {}
        resonance_records.append(
            ResonanceSnapshot(
                index=len(resonance_records),
                room=str(entry.get("room", "?")),
                character=str(entry.get("character", "?")),
                occupant_count=int(entry.get("occupant_count", 0) or 0),
                baseline_before={k: _safe_float(v) for k, v in baseline_before_raw.items()},
                baseline_after={k: _safe_float(v) for k, v in baseline_after_raw.items()},
                bio_modulation={k: _safe_float(v) for k, v in bio_modulation_raw.items()},
                delta_pre_zone={k: _safe_float(v) for k, v in delta_pre_raw.items()},
                delta_post_zone={k: _safe_float(v) for k, v in delta_post_raw.items()},
                zone_ideology=zone_dict,
                gravity=_safe_float(entry.get("gravity")),
                tactile_temperature=_safe_float(tactile_raw.get("temperature")),
                tactile_luminosity=_safe_float(tactile_raw.get("luminosity")),
                delta_magnitude=_safe_float(entry.get("delta_magnitude")),
            )
        )
    return resonance_records


def ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_trauma_energy(records: List[TraumaRecord]) -> Optional[Path]:
    if not records:
        return None
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return None
    ensure_output_dirs()
    plt_mod = cast(Any, plt)
    fig, ax = plt_mod.subplots(figsize=(8, 4.5))
    indices = [r.index for r in records]
    energies = [r.energy_l2 for r in records]
    compression = [r.compression for r in records]
    expansion = [r.expansion for r in records]
    ax.plot(indices, energies, label="Energy L2", color="tab:blue")
    ax.plot(indices, compression, label="Compression", color="tab:red", linestyle="--")
    ax.plot(indices, expansion, label="Expansion", color="tab:green", linestyle=":")
    ax.set_xlabel("Trauma Event Index")
    ax.set_ylabel("Magnitude")
    ax.set_title("Trauma Tensor Energy Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_path = PLOTS_DIR / "trauma_energy.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt_mod.close(fig)
    return plot_path


def plot_collision_4d(records: List[CollisionRecord]) -> Optional[Path]:
    if not records:
        return None
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return None
    ensure_output_dirs()
    plt_mod = cast(Any, plt)
    fig = plt_mod.figure(figsize=(7.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    x = [rec.baseline_delta.get("trust", 0.0) for rec in records]
    y = [rec.baseline_delta.get("curiosity", 0.0) for rec in records]
    z = [rec.baseline_delta.get("playfulness", 0.0) for rec in records]
    c = [rec.baseline_delta.get("boldness", 0.0) for rec in records]
    sizes = [20 + abs(val) * 40 for val in c]
    sc = cast(Any, ax).scatter(xs=x, ys=y, zs=z, c=c, cmap="coolwarm", s=sizes, alpha=0.85)
    ax.set_xlabel("Trust Delta")
    ax.set_ylabel("Curiosity Delta")
    ax.set_zlabel("Playfulness Delta")
    ax.set_title("Collision Baseline Delta (4D via color/size)")
    colorbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
    colorbar.set_label("Boldness Delta")
    plot_path = PLOTS_DIR / "collision_4d.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt_mod.close(fig)
    return plot_path


def summarize(records: Iterable[TraumaRecord]) -> Dict[str, float]:
    data = list(records)
    if not data:
        return {"count": 0}
    return {
        "count": len(data),
        "energy_mean": mean(r.energy_l2 for r in data),
        "compression_mean": mean(r.compression for r in data),
        "expansion_mean": mean(r.expansion for r in data),
    }


def summarize_collisions(records: Iterable[CollisionRecord]) -> Dict[str, float]:
    data = list(records)
    if not data:
        return {"count": 0}
    return {
        "count": len(data),
        "novelty_mean": mean(r.novelty for r in data),
        "kappa_mean": mean(r.kappa for r in data),
        "consolidation_mean": mean(r.consolidation_window_hours for r in data),
    }


TIERS = ["social", "economic", "governmental", "personal"]
PATTERNS = ["niche", "symbiosis", "predation", "act_of_god", "freak_accident"]


def plot_ecology_heatmap(records: List[EcologyRecord]) -> Optional[Path]:
    if not records:
        return None
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return None
    ensure_output_dirs()
    matrix = [[0.0 for _ in PATTERNS] for _ in TIERS]
    counts = [[0 for _ in PATTERNS] for _ in TIERS]
    index_lookup = {tier: idx for idx, tier in enumerate(TIERS)}
    pattern_lookup = {pattern: idx for idx, pattern in enumerate(PATTERNS)}
    for record in records:
        tier_idx = index_lookup.get(record.tier, None)
        pattern_idx = pattern_lookup.get(record.pattern, None)
        if tier_idx is None or pattern_idx is None:
            continue
        matrix[tier_idx][pattern_idx] += record.intensity or 0.0
        counts[tier_idx][pattern_idx] += 1
    for tier_idx, pattern_arr in enumerate(matrix):
        for pattern_idx, value in enumerate(pattern_arr):
            count = counts[tier_idx][pattern_idx]
            if count:
                matrix[tier_idx][pattern_idx] = value / count
    plt_mod = cast(Any, plt)
    fig, ax = plt_mod.subplots(figsize=(8, 4.5))
    heatmap = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(PATTERNS)))
    ax.set_xticklabels([label.replace("_", " ").title() for label in PATTERNS], rotation=35, ha="right")
    ax.set_yticks(range(len(TIERS)))
    ax.set_yticklabels([label.title() for label in TIERS])
    ax.set_xlabel("Ecological Pattern")
    ax.set_ylabel("Structural Tier")
    ax.set_title("Mean Intensity by Tier × Ecological Pattern")
    cbar = fig.colorbar(heatmap, ax=ax, shrink=0.85)
    cbar.set_label("Mean Intensity")
    plot_path = PLOTS_DIR / "ecology_heatmap.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt_mod.close(fig)
    return plot_path


def summarize_ecology(records: Iterable[EcologyRecord]) -> Dict[str, Any]:
    data = list(records)
    if not data:
        return {"count": 0, "tiers": {}, "pattern_counts": {}, "top_patterns": [], "top_participants": []}
    tiers: Dict[str, Dict[str, Any]] = {}
    for tier in TIERS:
        tiers[tier] = {
            "count": 0,
            "mean_intensity": 0.0,
            "dominant_pattern": None,
            "participant_samples": set(),
        }
    tier_pattern_counter: Dict[str, Counter[str]] = defaultdict(Counter)
    pattern_counts: Counter[str] = Counter()
    participant_counts: Counter[str] = Counter()
    for record in data:
        tier_key = record.tier if record.tier in tiers else "unclassified"
        if tier_key not in tiers:
            tiers[tier_key] = {
                "count": 0,
                "mean_intensity": 0.0,
                "dominant_pattern": None,
                "participant_samples": set(),
            }
        tier_entry = tiers[tier_key]
        tier_entry["count"] += 1
        tier_entry["mean_intensity"] += record.intensity
        tier_entry["participant_samples"].update(record.participants[:2])
        tier_pattern_counter[tier_key][record.pattern] += 1
        pattern_counts[record.pattern] += 1
        participant_counts.update(record.participants)
    for tier_key, entry in tiers.items():
        if entry["count"]:
            entry["mean_intensity"] = entry["mean_intensity"] / entry["count"]
            entry["dominant_pattern"] = tier_pattern_counter[tier_key].most_common(1)[0][0]
        entry["participant_samples"] = sorted(entry["participant_samples"])
    top_patterns = [
        {"pattern": name, "count": count}
        for name, count in pattern_counts.most_common(5)
    ]
    top_participants = [
        {"participant": name, "count": count}
        for name, count in participant_counts.most_common(5)
    ]
    return {
        "count": len(data),
        "tiers": tiers,
        "pattern_counts": dict(pattern_counts),
        "top_patterns": top_patterns,
        "top_participants": top_participants,
    }


def plot_resonance_deltas(records: List[ResonanceSnapshot]) -> Optional[Path]:
    if not records:
        return None
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return None
    ensure_output_dirs()
    plt_mod = cast(Any, plt)
    fig, ax1 = plt_mod.subplots(figsize=(8, 4.5))
    indices = [snap.index for snap in records]
    delta_values = [snap.delta_magnitude for snap in records]
    ax1.plot(indices, delta_values, color="tab:purple", label="‖Δb‖₂")
    ax1.set_xlabel("Snapshot Index")
    ax1.set_ylabel("‖Δb‖₂ Magnitude", color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")

    zone_pressures = [
        snap.zone_ideology["pressure"]
        for snap in records
        if snap.zone_ideology and snap.zone_ideology.get("pressure") is not None
    ]
    if zone_pressures:
        aligned_pressures: List[float] = []
        pressure_indices: List[int] = []
        for snap in records:
            if snap.zone_ideology and snap.zone_ideology.get("pressure") is not None:
                pressure_indices.append(snap.index)
                aligned_pressures.append(float(snap.zone_ideology["pressure"]))
        ax2 = ax1.twinx()
        ax2.plot(pressure_indices, aligned_pressures, color="tab:orange", label="Zone Pressure")
        ax2.set_ylabel("Zone Pressure", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        # Compose combined legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    fig.tight_layout()
    plot_path = PLOTS_DIR / "resonance_deltas.png"
    fig.savefig(plot_path, dpi=200)
    plt_mod.close(fig)
    return plot_path


def summarize_resonance(records: Iterable[ResonanceSnapshot]) -> Dict[str, Any]:
    data = list(records)
    if not data:
        return {
            "count": 0,
            "mean_delta": 0.0,
            "mean_zone_pressure": 0.0,
            "mean_gravity": 0.0,
            "mean_temperature": 0.0,
            "mean_luminosity": 0.0,
            "mean_bio_intensity": 0.0,
            "top_characters": [],
        }
    delta_values = [snap.delta_magnitude for snap in data]
    gravity_values = [snap.gravity for snap in data]
    temperature_values = [snap.tactile_temperature for snap in data]
    luminosity_values = [snap.tactile_luminosity for snap in data]
    zone_pressures = [
        float(snap.zone_ideology["pressure"])
        for snap in data
        if snap.zone_ideology and snap.zone_ideology.get("pressure") is not None
    ]
    bio_intensities = [
        sum(abs(val) for val in snap.bio_modulation.values())
        for snap in data
    ]
    character_counts = Counter(snap.character for snap in data if snap.character)
    top_characters = [
        {"name": name, "count": count}
        for name, count in character_counts.most_common(5)
    ]
    return {
        "count": len(data),
        "mean_delta": mean(delta_values) if delta_values else 0.0,
        "mean_zone_pressure": mean(zone_pressures) if zone_pressures else 0.0,
        "mean_gravity": mean(gravity_values) if gravity_values else 0.0,
        "mean_temperature": mean(temperature_values) if temperature_values else 0.0,
        "mean_luminosity": mean(luminosity_values) if luminosity_values else 0.0,
        "mean_bio_intensity": mean(bio_intensities) if bio_intensities else 0.0,
        "top_characters": top_characters,
    }


FORMULAE_BLOCK = textwrap.dedent(
    """
    Governing Formulae (MBD Lab):

    - Coupling Dynamics: dκ/dt = α(1 - μ²) - βκ
    - Baseline Drift: Δb = λκ(u - b)
    - Trauma Energy: ‖Δb‖₂ = sqrt(Σ (Δbᵢ)²)
    - Compression / Expansion: Σ max(0, -Δbᵢ) / Σ max(0, Δbᵢ)
    - Ecological Pressure Gradient: Πᵗ = Σ (wᵢ × intensityᵢ) across tiers t ∈ {social, economic, governmental, personal}
    - Consolidation Window Hypothesis: PTSD risk ∝ novelty × compression × κ within 12–48 h
    """
)


def build_report(
    trauma_summary: Dict[str, Any],
    collision_summary: Dict[str, Any],
    ecology_summary: Dict[str, Any],
    resonance_summary: Dict[str, Any],
    trauma_plot: Optional[Path],
    collision_plot: Optional[Path],
    ecology_plot: Optional[Path],
    resonance_plot: Optional[Path],
    generated_at: str,
) -> str:
    profiles = list_profiles()

    lines: List[str] = []
    lines.append("# MBD Laboratory Report")
    lines.append("")
    lines.append("Generated by `analysis/mbd_graphing_suite.py`.")
    lines.append(f"Timestamp: {generated_at}")
    lines.append("")
    lines.append(FORMULAE_BLOCK)
    lines.append("")

    lines.append("## Trauma Tensor Overview")
    lines.append("")
    if trauma_summary["count"] == 0:
        lines.append("No trauma tensor entries were detected in the journal.")
    else:
        lines.append(f"Events captured: {trauma_summary['count']}")
        lines.append(
            f"Mean energy ‖Δb‖₂ = {trauma_summary['energy_mean']:.3f}, "
            f"compression = {trauma_summary['compression_mean']:.3f}, "
            f"expansion = {trauma_summary['expansion_mean']:.3f}."
        )
        if trauma_plot:
            lines.append("")
            lines.append(f"![Trauma Energy Plot]({trauma_plot.as_posix()})")
    lines.append("")

    lines.append("## Collision Tensor Overview")
    lines.append("")
    if collision_summary["count"] == 0:
        lines.append("No collision tensor entries were detected in the journal.")
    else:
        lines.append(f"Events captured: {collision_summary['count']}")
        lines.append(
            f"Mean novelty = {collision_summary['novelty_mean']:.3f}, "
            f"mean κ = {collision_summary['kappa_mean']:.3f}, "
            f"avg consolidation window = {collision_summary['consolidation_mean']:.1f} h."
        )
        if collision_plot:
            lines.append("")
            lines.append(f"![Collision 4D Scatter]({collision_plot.as_posix()})")
    lines.append("")

    lines.append("## Ecological Tier Dynamics")
    lines.append("")
    if ecology_summary["count"] == 0:
        lines.append("No ecological tier events have been recorded yet. The matrix awaits first contact.")
    else:
        tiers = ecology_summary["tiers"]
        lines.append("| Tier | Events | Mean Intensity | Dominant Pattern | Participant Samples |")
        lines.append("| ---- | ------ | -------------- | ---------------- | ------------------- |")
        for tier_name, entry in tiers.items():
            samples = ", ".join(entry["participant_samples"]) if entry["participant_samples"] else "—"
            dominant = entry["dominant_pattern"].replace("_", " ").title() if entry["dominant_pattern"] else "—"
            lines.append(
                f"| {tier_name.title()} | {entry['count']} | {entry['mean_intensity']:.2f} | "
                f"{dominant} | {samples} |"
            )
        if ecology_plot:
            lines.append("")
            lines.append(f"![Ecology Tier Heatmap]({ecology_plot.as_posix()})")
        top_patterns = ecology_summary.get("top_patterns", [])
        top_participants = ecology_summary.get("top_participants", [])
        if top_patterns or top_participants:
            lines.append("")
            lines.append("### Ecological Highlights")
            lines.append("")
            if top_patterns:
                dominant_entry = top_patterns[0]
                dominant_name = dominant_entry["pattern"].replace("_", " ").title()
                lines.append(
                    f"- Dominant pattern overall: {dominant_name} ({_format_event_count(dominant_entry['count'])})."
                )
                if len(top_patterns) > 1:
                    runners = ", ".join(
                        f"{entry['pattern'].replace('_', ' ').title()} ({_format_event_count(entry['count'])})"
                        for entry in top_patterns[1:3]
                    )
                    if runners:
                        lines.append(f"- Runners-up: {runners}.")
            if top_participants:
                participant_line = ", ".join(
                    f"{entry['participant']} ({_format_event_count(entry['count'])})"
                    for entry in top_participants[:3]
                )
                lines.append(f"- Most active participants: {participant_line}.")
    lines.append("")

    lines.append("## Resonance Snapshot Diagnostics")
    lines.append("")
    if resonance_summary["count"] == 0:
        lines.append("No resonance snapshots captured; chamber telemetry feed is silent.")
    else:
        lines.append(f"Snapshots captured: {_format_event_count(resonance_summary['count'])}.")
        lines.append(
            f"Mean ‖Δb‖₂ = {resonance_summary['mean_delta']:.3f}, "
            f"mean zone pressure = {resonance_summary['mean_zone_pressure']:.2f}, "
            f"gravity = {resonance_summary['mean_gravity']:.2f}g."
        )
        lines.append(
            f"Thermal field ≈ {resonance_summary['mean_temperature']:.2f}°C;"
            f" luminosity ≈ {resonance_summary['mean_luminosity']:.1%}."
        )
        lines.append(
            f"Bio modulation intensity (Σ|Δ|) averages {resonance_summary['mean_bio_intensity']:.3f}."
        )
        top_characters = resonance_summary.get("top_characters", [])
        if top_characters:
            char_line = ", ".join(
                f"{entry['name']} ({_format_event_count(entry['count'])})"
                for entry in top_characters[:3]
            )
            lines.append(f"Most frequently sampled characters: {char_line}.")
        if resonance_plot:
            lines.append("")
            lines.append(f"![Resonance Δb / Zone Pressure]({resonance_plot.as_posix()})")
    lines.append("")

    lines.append("## Phenomenon Library Snapshot")
    lines.append("")
    lines.append("| Key | Title | Tags | Consolidation (h) | Summary |")
    lines.append("| --- | ----- | ---- | ----------------- | ------- |")
    for profile in profiles:
        tags = ", ".join(profile.tags)
        summary = profile.summary.replace("|", "/")
        lines.append(
            f"| {profile.key} | {profile.title} | {tags} | "
            f"{profile.consolidation_window_hours:.1f} | {summary} |"
        )
    lines.append("")

    lines.append("## Descriptor Notes")
    lines.append("")
    lines.append(
        "- *Compression* measures inward impulses (anti-waveform pressure)."\
        " High values correlate with destabilizing safety re-evaluations."
    )
    lines.append(
        "- *Expansion* tracks outward impulses (confidence surges)."\
        " Differential between compression and expansion informs plasticity."
    )
    lines.append(
        "- *Novelty* registers Fokker-Planck interference magnitude; coupling κ"\
        " modulates whether the wave collapses into long-term memory." 
    )
    lines.append("")

    return "\n".join(lines)


def generate_report() -> Path:
    entries = load_journal()
    trauma_records = extract_trauma(entries)
    collision_records = extract_collisions(entries)
    ecology_records = extract_ecology(entries)
    resonance_records = extract_resonance(entries)

    trauma_plot = plot_trauma_energy(trauma_records)
    collision_plot = plot_collision_4d(collision_records)
    ecology_plot = plot_ecology_heatmap(ecology_records)
    resonance_plot = plot_resonance_deltas(resonance_records)

    trauma_summary = summarize(trauma_records)
    collision_summary = summarize_collisions(collision_records)
    ecology_summary = summarize_ecology(ecology_records)
    resonance_summary = summarize_resonance(resonance_records)

    now = datetime.now()
    timestamp_display = now.isoformat(timespec="seconds")
    timestamp_slug = now.strftime("%Y%m%d_%H%M%S")

    report_text = build_report(
        trauma_summary,
        collision_summary,
        ecology_summary,
        resonance_summary,
        trauma_plot,
        collision_plot,
        ecology_plot,
        resonance_plot,
        timestamp_display,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"mbd_lab_report_{timestamp_slug}.md"
    report_path.write_text(report_text, encoding="utf-8")

    latest_path = REPORTS_DIR / "mbd_lab_report_latest.md"
    latest_path.write_text(report_text, encoding="utf-8")

    summary_payload = {
        "timestamp": timestamp_display,
        "report_slug": report_path.name,
        "trauma": trauma_summary,
        "collision": collision_summary,
        "ecology": {
            "count": ecology_summary.get("count", 0),
            "tiers": ecology_summary.get("tiers", {}),
            "pattern_counts": ecology_summary.get("pattern_counts", {}),
            "top_patterns": ecology_summary.get("top_patterns", []),
            "top_participants": ecology_summary.get("top_participants", []),
        },
        "resonance": resonance_summary,
    }

    summary_text = json.dumps(summary_payload, indent=2, sort_keys=True)
    summary_path = REPORTS_DIR / f"mbd_lab_summary_{timestamp_slug}.json"
    summary_path.write_text(summary_text, encoding="utf-8")

    latest_summary_path = REPORTS_DIR / "mbd_lab_summary_latest.json"
    latest_summary_path.write_text(summary_text, encoding="utf-8")

    return report_path


if __name__ == "__main__":
    path = generate_report()
    print(f"Report written to {path}")
