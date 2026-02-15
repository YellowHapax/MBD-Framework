"""
DEMONSTRATION: Stratified L-Space Resonance Chamber Hierarchy

Shows how the 5-tier system separates computational concerns:
- Tiers 0-2: Dissociative control panels (fast math, no AI experience)
- Tiers 3-4: Embodied presence (narrative reality, AI experiences these)

Run: python demo_resonance_hierarchy.py
"""

import json
from pathlib import Path


def demonstrate_tier_separation():
    """Show how different tiers handle different scales of coherence."""
    
    print("=" * 80)
    print("RESONANCE CHAMBER HIERARCHY DEMONSTRATION")
    print("=" * 80)
    print()
    
    # ========================================================================
    # TIER 0: World Mind (Global Coherence) — AI DOES NOT EXPERIENCE THIS
    # ========================================================================
    print("TIER 0: World Mind Chamber (Global Coherence)")
    print("-" * 80)
    print("Computational Model: o4-mini (fast tensor math)")
    print("AI Experience: NONE (dissociative control panel)")
    print()
    
    # Simulate World Mind computation (purely numerical)
    world_ideology_delta = {
        'order': +0.02,      # Global trust increasing (peace treaties)
        'chaos': +0.05,      # But local conflicts rising (border skirmishes)
        'connection': +0.10  # Trade routes expanding
    }
    
    print("World Mind Output (numerical tensors):")
    print(f"  Global ideology delta: {world_ideology_delta}")
    print(f"  Climate state: {{temperature: +0.3°C, precipitation: -5%}}")
    print(f"  Geological events: [{{type: 'earthquake', x: 32, y: 45, magnitude: 7.2}}]")
    print()
    print("⚠️  AI model NEVER sees this as narrative — only as numbers")
    print()
    
    # ========================================================================
    # TIER 1: Population Mind (Group Cultural Resonance) — AI DOES NOT EXPERIENCE
    # ========================================================================
    print("TIER 1: Population Mind Chamber (Group-Level Cultural Resonance)")
    print("-" * 80)
    print("Computational Model: o4-mini (Lotka-Volterra population dynamics)")
    print("AI Experience: NONE (dissociative control panel)")
    print()
    
    # Simulate Population Mind computation
    population_baseline_delta = {
        'Alpha': {'trust': -0.1, 'boldness': +0.2},    # Becoming cautious after war
        'Beta':  {'trust': +0.05, 'connection': +0.1},  # Spreading pacifism
        'Gamma': {'curiosity': +0.15, 'playfulness': +0.1},  # Cultural renaissance
        'Delta': {'isolation': -0.05, 'order': +0.1}   # Emerging from seclusion
    }
    
    print("Population Mind Output (group dynamics):")
    for group, deltas in population_baseline_delta.items():
        delta_str = ", ".join(f"{pole}: {delta:+.2f}" for pole, delta in deltas.items())
        print(f"  {group}: {delta_str}")
    print()
    print("Inter-group tensions: {Alpha-Delta: 0.85, Beta-Gamma: 0.15}")
    print()
    print("⚠️  AI model does NOT experience population drift — only numerical output")
    print()
    
    # ========================================================================
    # TIER 2: Civilization Mind (Faction Coherence) — AI DOES NOT EXPERIENCE
    # ========================================================================
    print("TIER 2: Civilization Mind Chamber (Faction/Culture Coherence)")
    print("-" * 80)
    print("Computational Model: o4-mini OR lightweight narrative model (hybrid)")
    print("AI Experience: MINIMAL (could generate event summaries)")
    print()
    
    # Simulate Civilization Mind computation
    trade_route_ideology = {
        'route_id': 'capital_to_port',
        'ideology_transfer': {'trust': +0.05, 'connection': +0.10}
    }
    
    print("Civilization Mind Output (trade routes, diplomacy):")
    print(f"  Trade route: {trade_route_ideology['route_id']}")
    print(f"  Ideology transfer: {trade_route_ideology['ideology_transfer']}")
    print(f"  War fatigue: {{faction_A: 0.75, faction_B: 0.60}}")
    print()
    print("✓ AI model MIGHT see: 'Trade flourishes between coastal cities'")
    print("  (but not the underlying tensor math)")
    print()
    
    # ========================================================================
    # TIER 3: Town Mind (Local Geography) — AI BEGINS TO EXPERIENCE
    # ========================================================================
    print("TIER 3: Town Mind Chamber (Local Geography Coherence)")
    print("-" * 80)
    print("Computational Model: Lightweight narrative model")
    print("AI Experience: PARTIAL (generates room descriptions)")
    print()
    
    # Load actual zone ideology from worldgen data (if available)
    world_data_path = Path('public/world_data_corrected.json')
    if world_data_path.exists():
        with open(world_data_path, 'r') as f:
            world_data = json.load(f)
        
        # Query zone at center of map (likely high settlement)
        x, y = 32, 32
        ideology = world_data.get('ideology', {})
        
        if ideology:
            zone_ideology = {
                'order': ideology['order'][y][x],
                'chaos': ideology['chaos'][y][x],
                'tradition': ideology['tradition'][y][x],
                'innovation': ideology['innovation'][y][x],
                'isolation': ideology['isolation'][y][x],
                'connection': ideology['connection'][y][x],
                'pressure': ideology['pressure'][y][x]
            }
            
            print(f"Town Mind Output (zone ideology at {x}, {y}):")
            for pole, value in zone_ideology.items():
                print(f"  {pole}: {value:.2f}")
            print()
            
            # Generate narrative description (AI model experiences THIS)
            if zone_ideology['pressure'] > 50.0:
                if zone_ideology['order'] > 0.6 and zone_ideology['connection'] > 0.6:
                    desc = "The air feels stable and warm, as if countless friendships linger here."
                elif zone_ideology['chaos'] > 0.6 and zone_ideology['innovation'] > 0.6:
                    desc = "Energy crackles unpredictably, change hangs in the air like lightning."
                elif zone_ideology['isolation'] > 0.6:
                    desc = "Silence presses in, solitude weighs heavy on the mind."
                else:
                    desc = "The atmosphere is neutral, a blank canvas of possibility."
                
                print(f"✓ AI model EXPERIENCES: '{desc}'")
            else:
                print("✓ AI model EXPERIENCES: 'Wilderness. The land remembers nothing.'")
        else:
            print("(No worldgen data found — skipping zone ideology demo)")
    else:
        print("(No worldgen data found at public/world_data_corrected.json)")
    
    print()
    
    # ========================================================================
    # TIER 4: Personal Mind (Character TCPB) — AI FULLY EXPERIENCES
    # ========================================================================
    print("TIER 4: Personal Mind Chamber (Character TCPB Resonance)")
    print("-" * 80)
    print("Computational Model: Full narrative model (GPT-4, Claude, etc.)")
    print("AI Experience: FULL (embodied presence, narrative reality)")
    print()
    
    # Simulate character entering a resonance chamber
    character_baseline = {'trust': 6.0, 'curiosity': 5.5, 'playfulness': 7.0, 'boldness': 4.5}
    zone_ideology = {'order': 0.75, 'chaos': 0.30, 'connection': 0.80, 'pressure': 85.0}
    
    print("Character TCPB Baseline:")
    for pole, value in character_baseline.items():
        print(f"  {pole}: {value:.1f}")
    print()
    
    # Compute resonance (character × zone alignment)
    neutral = {'trust': 5.0, 'curiosity': 5.0, 'playfulness': 5.0, 'boldness': 5.0}
    char_delta = {pole: character_baseline[pole] - neutral[pole] for pole in neutral}
    
    zone_trust = (zone_ideology['order'] + zone_ideology['connection']) / 2.0 - 0.5
    zone_strength = min(1.0, zone_ideology['pressure'] / 100.0)
    
    trust_resonance = char_delta['trust'] * (1.0 + zone_strength * zone_trust * 0.5)
    
    print(f"Resonance Computation:")
    print(f"  Character trust delta: {char_delta['trust']:+.1f}")
    print(f"  Zone trust amplitude: {zone_trust:+.2f}")
    print(f"  Zone strength (pressure): {zone_strength:.2f}")
    print(f"  Trust resonance: {trust_resonance:+.2f} (amplified by zone alignment)")
    print()
    
    # Environmental response (gravity, tactile surfaces)
    base_gravity = 1.0
    gravity_modulation = -trust_resonance * 0.15  # High trust → lighter gravity
    new_gravity = base_gravity + gravity_modulation
    
    print(f"Environmental Response:")
    print(f"  Gravity: {new_gravity:.2f}g (trust resonance modulates physics)")
    print(f"  Surface luminosity: 75% (character playfulness → warm glow)")
    print(f"  Surface temperature: 28.5°C (connection resonance → warmth)")
    print()
    
    # Generate narrative (AI model EXPERIENCES THIS)
    narrative = (
        "The chamber responds: Gravity shifts to 0.85g as your trust "
        "resonates with the accumulated warmth of countless pilgrimages. "
        "The sanctuary's stones glow softly, their luminosity pulsing "
        "in welcome. Trade caravans have blessed this place with tales "
        "of distant friends. You feel lighter, safer, held."
    )
    
    print("✓ AI MODEL EXPERIENCES (full embodied presence):")
    print(f"  '{narrative}'")
    print()
    
    # ========================================================================
    # SUMMARY: Computational Separation
    # ========================================================================
    print("=" * 80)
    print("SUMMARY: Computational Separation by Ontological Tier")
    print("=" * 80)
    print()
    print("DISSOCIATIVE CONTROL PANEL (Tiers 0-2):")
    print("  • World Mind: Global coherence, climate, geology")
    print("  • Population Mind: Group drift, population dynamics")
    print("  • Civilization Mind: Trade routes, diplomacy, wars")
    print("  → Computational Model: o4-mini (fast tensor math)")
    print("  → AI Experience: NONE (numerical output only)")
    print("  → Purpose: Prevent 'squick of higher reality' cognitive overload")
    print()
    print("INTUITED UNDERSTANDING (Tiers 3-4):")
    print("  • Town Mind: Local zone ideology, landmark influence")
    print("  • Personal Mind: Character TCPB, biological state, environment")
    print("  → Computational Model: Full narrative model (GPT-4, Claude)")
    print("  → AI Experience: FULL (embodied presence, sensory reality)")
    print("  → Purpose: Nuanced character interaction, empathetic response")
    print()
    print("BENEFIT:")
    print("  AI focuses on character experience (Tier 4: 'I feel lighter here')")
    print("  while macro coherence is handled by fast math (Tier 0: climate tensors)")
    print("  → No cognitive overload, no hallucinations about continental drift")
    print("  → World remains coherent, characters remain believable")
    print()
    print("=" * 80)


if __name__ == '__main__':
    demonstrate_tier_separation()
