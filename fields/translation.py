"""
Standalone affective <-> field translation functions.
Extracted from resonance_chamber.py for independent use.
"""

from typing import Optional, Dict, Any, List


AFFECTIVE_FIELD_BLUEPRINT: Dict[str, Dict[str, Any]] = {
    "trust": {
        "field": "buoyancy",
        "positive": {
            "field_effect": "Lighter gravity; buoyant lift",
            "somatic": "Weightlessness and crisp air ease movement.",
            "agency": "Actions feel effortless; confidence increases.",
            "prompt": "Gravity eases, a buoyant lift answering your trust.",
        },
        "negative": {
            "field_effect": "Heavier gravity; viscous drag",
            "somatic": "Crushing pressure pools in your limbs.",
            "agency": "Effort spikes; movement feels costly.",
            "prompt": "Gravity drags, distrust thickening the air around you.",
        },
    },
    "curiosity": {
        "field": "luminosity",
        "positive": {
            "field_effect": "Sharpened light; highlighted vectors",
            "somatic": "Details pop with inviting clarity.",
            "agency": "Discovery pathways unfold; perception heightens.",
            "prompt": "Light clarifies; curiosity pulls hidden paths into focus.",
        },
        "negative": {
            "field_effect": "Fogged light; blurred edges",
            "somatic": "Dim haze mutes form and direction.",
            "agency": "Uncertainty slows searching; hesitation grows.",
            "prompt": "Mist gathers; hesitance blurs the chamber's edges.",
        },
    },
    "playfulness": {
        "field": "tactile_response",
        "positive": {
            "field_effect": "Warm, responsive surfaces",
            "somatic": "Textures glow with gentle warmth.",
            "agency": "Experimentation is rewarded; touch feels safe.",
            "prompt": "Surfaces warm to you; playfulness invites experimentation.",
        },
        "negative": {
            "field_effect": "Rigid, cold surfaces",
            "somatic": "Frosted pressure resists your touch.",
            "agency": "Contact feels punitive; improvisation stalls.",
            "prompt": "Surfaces harden; sternness answers with austere resistance.",
        },
    },
    "boldness": {
        "field": "resonant_harmonics",
        "positive": {
            "field_effect": "Expansive harmonics; open vectors",
            "somatic": "Sound carries bright and clear in generous space.",
            "agency": "Decisive motion feels supported and amplified.",
            "prompt": "Space opens wide; boldness threads ringing harmonics through the chamber.",
        },
        "negative": {
            "field_effect": "Dampened resonance; constricted space",
            "somatic": "Silence presses close with shallow breath.",
            "agency": "Actions shrink; fear urges quiet, careful steps.",
            "prompt": "Sound dulls; fear tightens the walls into a narrow corridor.",
        },
    },
}


def translate_affective_to_field(tcpb_delta: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Map affective deltas onto field descriptors for resonance rendering."""
    field_state: Dict[str, Dict[str, Any]] = {}
    for pole, blueprint in AFFECTIVE_FIELD_BLUEPRINT.items():
        raw_value = float(tcpb_delta.get(pole, 0.0))
        clamped = max(-5.0, min(5.0, raw_value))
        normalized = clamped / 5.0  # scale to [-1, 1]
        descriptor = blueprint["positive"] if normalized >= 0 else blueprint["negative"]
        field_state[pole] = {
            "value": normalized,
            "magnitude": abs(normalized),
            "field": blueprint["field"],
            "field_effect": descriptor["field_effect"],
            "somatic": descriptor["somatic"],
            "agency": descriptor["agency"],
            "prompt": descriptor["prompt"],
        }
    return field_state


def translate_field_to_affective(field_state: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Recover approximate affective deltas from stored field state."""
    return {
        pole: max(-1.0, min(1.0, details.get("value", 0.0))) * 5.0
        for pole, details in field_state.items()
    }


def render_affective_prompt(field_state: Dict[str, Dict[str, Any]], gravity: Optional[float] = None) -> str:
    """Compose a narrative prompt for the AI based on active field descriptors."""
    segments: List[str] = []
    if gravity is not None:
        segments.append(f"Gravity shifts to {gravity:.2f}g.")
    dominant = sorted(field_state.items(), key=lambda item: item[1]["magnitude"], reverse=True)
    for pole, details in dominant:
        if details["magnitude"] < 0.15:
            continue
        segments.append(details["prompt"])
    if not segments:
        return "Gravity steadies in a neutral equilibrium."
    return " ".join(segments)
