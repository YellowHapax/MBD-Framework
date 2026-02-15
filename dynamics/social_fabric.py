"""
social_fabric_simulator.py

This script runs a discrete-time simulation to generate a rich history of
social interactions between agents in the world. It is designed to be run
as a standalone process, which generates events and commits them to the
central `world_data.json` file via the `history_aggregator`.

The simulation operates in ticks, with each tick representing a small
increment of time (e.g., an hour). In each tick, it updates agent's internal
states (like desire and frustration) and then calculates the probability of
interaction between all pairs of agents, generating specific social events.
"""
import os
import json
import random
import math
import time
from typing import Dict, Any, List, Tuple

# Event logging stub (standalone version; the full system uses history_aggregator)
def commit_event(event_dict):
    pass  # No-op in standalone mode

# --- Constants ---
WORLD_DATA_PATH = os.path.join(os.path.dirname(__file__), "public", "world_data.json")
# Allow quick override for validation runs: in PowerShell -> $env:SIM_TICKS=48; python social_fabric_simulator.py
SIMULATION_TICKS = int(os.environ.get("SIM_TICKS", 24 * 90))  # Simulate 90 days by default
BASE_INTERACTION_PROB = 0.005  # Lower base prob, as it will be amplified by desire

# --- Population Archetype Configuration ---
# Four lifecycle profiles spanning different longevity scales.
# These are abstract archetypes; concrete populations reference them by key.
POPULATION_PROFILES = {
    "standard":    {"avg_lifespan": 80,   "fertility_start": 16,  "fertility_end": 45,  "adulthood": 18},
    "extended":    {"avg_lifespan": 250,  "fertility_start": 30,  "fertility_end": 150, "adulthood": 40},
    "long_lived":  {"avg_lifespan": 500,  "fertility_start": 50,  "fertility_end": 300, "adulthood": 60},
    "geological":  {"avg_lifespan": 1000, "fertility_start": 100, "fertility_end": 800, "adulthood": 120},
    "default":     {"avg_lifespan": 100,  "fertility_start": 18,  "fertility_end": 50,  "adulthood": 20},
}

# Map group labels to their lifecycle profile
GROUP_PROFILE_MAP = {
    "alpha": "standard",
    "beta":  "extended",
    "gamma": "long_lived",
    "delta": "geological",
}

# --- Helper Functions ---

def _load_world_data() -> Dict[str, Any]:
    """Loads the main world data file."""
    if not os.path.exists(WORLD_DATA_PATH):
        return {"agents": [], "meta": {"relationship_graph": {"nodes": [], "edges": []}, "world_traces": []}}
    try:
        with open(WORLD_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure agents list exists at the top level
            if 'agents' not in data:
                data['agents'] = []
            return data
    except (json.JSONDecodeError, IOError):
        return {"agents": [], "meta": {"relationship_graph": {"nodes": [], "edges": []}, "world_traces": []}}


def _synthesize_agents_and_edges(world_data: Dict[str, Any], per_race: int = 8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fallback: create a small synthetic agent set and lightweight relationship edges when
    no canonical agents/edges are present in world_data.json.

    Strategy:
    - Use `capitals` groups as canonical populations.
    - Create `per_race` agents per group with plausible defaults for fields this simulator uses.
    - Create intra-group edges with low initial pressures; a few inter-group edges to seed variety.
    """
    capitals = world_data.get("capitals", {})
    races = list(capitals.keys()) or ["Alpha", "Beta", "Gamma", "Delta"]

    agents: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    def rand_sex() -> str:
        return "female" if random.random() < 0.5 else "male"

    def init_agent(race: str, idx: int) -> Dict[str, Any]:
        race_key = race.lower()
        # Resolve lifecycle profile via group map, then fall back to direct key match
        profile_key = GROUP_PROFILE_MAP.get(race_key, race_key)
        profile = POPULATION_PROFILES.get(profile_key, POPULATION_PROFILES["default"])
        adulthood = profile.get("adulthood", 20)
        fert_end = profile.get("fertility_end", 50)
        age = max(12, min(fert_end, int(random.gauss(mu=(adulthood + fert_end) / 2, sigma=(fert_end - adulthood) / 4))))
        return {
            "id": f"{race[:3].upper()}-{idx}",
            "name": f"{race} {idx}",
            "race": race_key,
            "sex": rand_sex(),
            "body_morph": {
                "age": age,
                "fertility": random.random(),
            },
            "psyche": {
                "trust": random.uniform(0.2, 0.8),
                "playful": random.uniform(0.2, 0.8),
                "aggression": random.uniform(0.1, 0.4),
                "reproductive_drive": random.uniform(0.05, 0.35),
                "bonding_capacity": random.uniform(0.05, 0.25)
            },
            "pressures": {
                "frustration": random.uniform(0.0, 0.2)
            },
            "needs": {
                "hunger": random.uniform(0.0, 0.3)
            },
            "inventory": {
                "food": random.randint(1, 5)
            }
        }

    # Build agents grouped by race
    group_members: Dict[str, List[str]] = {}
    for race in races:
        members = []
        for i in range(per_race):
            a = init_agent(race, i)
            agents.append(a)
            members.append(a["id"])
        group_members[race] = members

    # Intra-group edges (friendships/intimacy potential)
    for race, members in group_members.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if random.random() < 0.15:  # sparse
                    edges.append({
                        "a": members[i],
                        "b": members[j],
                        "intimacy": random.uniform(0.05, 0.25),
                        "love": random.uniform(0.0, 0.2),
                        "conflict": random.uniform(0.0, 0.2),
                        "pair_bonding": random.uniform(0.0, 0.1),
                    })

    # A few inter-group cross links
    all_ids = [a["id"] for a in agents]
    for _ in range(max(3, len(all_ids) // 8)):
        a, b = random.sample(all_ids, 2)
        edges.append({
            "a": a,
            "b": b,
            "intimacy": random.uniform(0.02, 0.15),
            "love": random.uniform(0.0, 0.1),
            "conflict": random.uniform(0.0, 0.25),
            "pair_bonding": random.uniform(0.0, 0.08),
        })

    return agents, edges

def _get_agent_by_id(agent_id: str, agents: List[Dict]) -> Dict:
    """Finds an agent in the list by their ID."""
    for agent in agents:
        if agent["id"] == agent_id:
            return agent
    return {}

def _update_reproductive_drive(agent: Dict, all_agents: List[Dict]):
    """
    Updates an agent's internal state for reproductive_drive and frustration based
    on age, race, and demographic pressures.
    """
    race = agent.get("race", "default")
    profile_key = GROUP_PROFILE_MAP.get(race, race)
    lifecycle = POPULATION_PROFILES.get(profile_key, POPULATION_PROFILES["default"])
    age = agent.get("body_morph", {}).get("age", lifecycle["adulthood"])

    # 1. Age Gating
    if age <= 10:
        agent["psyche"]["reproductive_drive"] = 0
        return
    elif 11 <= age <= 15:
        # Curiosity phase: desire can be primed but is capped
        agent["psyche"]["reproductive_drive"] = min(agent["psyche"].get("reproductive_drive", 0), 0.3)
    
    # 2. Reproductive Pressure (Time-gated biological drive)
    fertility_window = lifecycle["fertility_end"] - lifecycle["fertility_start"]
    if lifecycle["fertility_start"] <= age <= lifecycle["fertility_end"]:
        # Parabolic curve, peaks in the middle of the fertility window
        mid_point = lifecycle["fertility_start"] + fertility_window / 2
        urgency = 1 - (abs(age - mid_point) / (fertility_window / 2))**2
        agent["psyche"]["reproductive_drive"] += urgency * 0.01 # Small nudge each tick
    
    # 3. Demographic Pressure (Subconscious heuristic)
    elderly_count = sum(1 for a in all_agents if a.get("body_morph", {}).get("age", 0) > lifecycle["avg_lifespan"] * 0.7)
    young_adult_count = sum(1 for a in all_agents if lifecycle["adulthood"] <= a.get("body_morph", {}).get("age", 0) <= lifecycle["fertility_end"])
    if young_adult_count > 0 and elderly_count / young_adult_count > 0.5: # If elders are >50% of young adults
        agent["psyche"]["reproductive_drive"] += 0.02

    # 4. Frustration
    if agent["psyche"]["reproductive_drive"] > 0.7 and agent.get("pressures", {}).get("pair_bonding", 0) < 0.1:
        agent["pressures"]["frustration"] = min(agent["pressures"].get("frustration", 0) + 0.05, 1.0)
    else:
        # Decay frustration naturally
        agent["pressures"]["frustration"] = max(agent["pressures"].get("frustration", 0) * 0.95, 0.0)

    # Clamp desire to [0, 1]
    agent["psyche"]["reproductive_drive"] = min(max(agent["psyche"]["reproductive_drive"], 0), 1.0)


def _update_agent_needs(agent: Dict):
    """Updates agent's hunger and handles eating."""
    # 1. Hunger increases every tick
    agent["needs"]["hunger"] = min(agent["needs"].get("hunger", 0) + 0.02, 1.0)

    # 2. If hungry and has food, eat
    if agent["needs"]["hunger"] > 0.6 and agent["inventory"].get("food", 0) > 0:
        agent["inventory"]["food"] -= 1
        agent["needs"]["hunger"] = 0
        # This could be an event, but for now it's an internal state change
        # print(f"  - {agent['id']} ate food.")


def _build_relationship_matrix(edges: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Transforms the edge list into a nested dictionary for O(1) lookups."""
    matrix = {}
    for edge in edges:
        a_id = edge["a"]
        b_id = edge["b"]
        
        pressures = {
            "intimacy": edge.get("intimacy", 0),
            "love": edge.get("love", 0),
            "conflict": edge.get("conflict", 0),
            "pair_bonding": edge.get("pair_bonding", 0),
        }
        
        if a_id not in matrix:
            matrix[a_id] = {}
        if b_id not in matrix:
            matrix[b_id] = {}
            
        matrix[a_id][b_id] = pressures
        matrix[b_id][a_id] = pressures
        
    return matrix

def get_current_pressures(agent1_id: str, agent2_id: str, matrix: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    """Gets the current pressure values between two agents from the pre-built matrix."""
    try:
        return matrix[agent1_id][agent2_id]
    except KeyError:
        # No direct edge exists, return a default neutral state.
        return {"intimacy": 0, "love": 0, "conflict": 0, "pair_bonding": 0}

def calculate_interaction_prob(agent1: Dict, agent2: Dict, pressures: Dict[str, float]) -> float:
    """Calculates the probability of an interaction, now heavily influenced by desire."""
    # Base attraction on psyche similarity (simplified)
    psyche1, psyche2 = agent1.get("psyche", {}), agent2.get("psyche", {})
    attraction = 0.5 + 0.5 * (1 - abs(psyche1.get("trust", 0) - psyche2.get("trust", 0)))

    # Pull from positive pressures
    positive_pull = 1 + math.tanh(pressures["intimacy"] + pressures["love"])
    
    # Push from desire and frustration
    desire_push = 1 + agent1["psyche"]["reproductive_drive"] + agent2["psyche"]["reproductive_drive"]
    frustration_push = 1 + agent1["pressures"].get("frustration", 0) + agent2["pressures"].get("frustration", 0)

    prob = BASE_INTERACTION_PROB * attraction * positive_pull * desire_push * frustration_push
    return min(prob, 1.0)

def generate_event_type(agent1: Dict, agent2: Dict, pressures: Dict[str, float]) -> str:
    """
    Decides the type of event based on a hierarchy of needs and desires.
    This now functions as the agent's "decision-making" core for interactions.
    """
    # 1. Primary Need: Hunger
    hunger1 = agent1["needs"].get("hunger", 0)
    if hunger1 > 0.7: # If agent1 is very hungry
        aggression1 = agent1["psyche"].get("aggression", 0.3)
        # Chance to steal is hunger * aggression
        if random.random() < hunger1 * aggression1 and agent2["inventory"].get("food", 0) > 0:
            return "theft"

    # 2. Sexual Desire
    total_desire = agent1["psyche"]["reproductive_drive"] + agent2["psyche"]["reproductive_drive"]
    if total_desire > 1.0 and pressures["conflict"] < 0.5:
        if random.random() < total_desire / 2.0:
            return "pair_bonding"

    # 3. Social Pressures (Conflict and Love)
    if pressures["conflict"] > 0.5 and random.random() < pressures["conflict"]:
        return "conflict"
        
    if pressures["love"] > 0.3 and random.random() < pressures["love"]:
        return "intimacy"

    # 4. Default Action: Forage for food if a bit hungry, otherwise socialize
    if hunger1 > 0.4 and random.random() < 0.2:
        return "forage"

    return "intimacy" # Default to a neutral/small positive interaction


def _handle_theft(thief: Dict, victim: Dict, tick: int, start_time: float):
    """Handles the logic for a theft event."""
    stolen_amount = min(victim["inventory"].get("food", 0), random.randint(1, 3))
    if stolen_amount > 0:
        thief["inventory"]["food"] = thief["inventory"].get("food", 0) + stolen_amount
        victim["inventory"]["food"] -= stolen_amount
        
        event_time = int(start_time) + tick
        note = f"{thief['id']} stole {stolen_amount} food from {victim['id']}."
        commit_event({
            "type": "theft", "actors": [thief["id"], victim["id"]],
            "magnitude": stolen_amount, "time": event_time, "note": note
        })
        print(f"  - Tick {tick}: ⚔️ THEFT! {note}")
        
        # Consequence: Increase conflict pressure
        # This requires modifying the relationship matrix in-memory for the simulation to be reactive.
        # For now, we just log the event. Future work can make this dynamic.

def _handle_forage(agent: Dict, tick: int, start_time: float):
    """Handles the logic for a foraging event."""
    found_amount = random.randint(0, 2) # Can find nothing
    if found_amount > 0:
        agent["inventory"]["food"] = agent["inventory"].get("food", 0) + found_amount
        event_time = int(start_time) + tick
        note = f"{agent['id']} foraged and found {found_amount} food."
        commit_event({
            "type": "forage", "actors": [agent["id"]],
            "magnitude": found_amount, "time": event_time, "note": note
        })
        print(f"  - Tick {tick}: 🧺 Forage. {note}")

def _handle_pair_bonding(agent1: Dict, agent2: Dict, tick: int, start_time: float):
    """Handles the logic for a pair-bonding event."""
    # 1. Calculate experience score
    skill1 = agent1["psyche"].get("bonding_capacity", 0.1)
    skill2 = agent2["psyche"].get("bonding_capacity", 0.1)
    avg_skill = (skill1 + skill2) / 2
    compatibility = 0.5 + 0.5 * (1 - abs(agent1["psyche"].get("playful", 0) - agent2["psyche"].get("playful", 0)))
    experience_score = min(avg_skill * 0.6 + compatibility * 0.4 + random.uniform(-0.1, 0.1), 1.0)

    # 2. Update skills ("get better with practice")
    agent1["psyche"]["bonding_capacity"] = min(skill1 + (experience_score - skill1) * 0.1, 1.0)
    agent2["psyche"]["bonding_capacity"] = min(skill2 + (experience_score - skill2) * 0.1, 1.0)

    # 3. Commit the event
    event_time = int(start_time) + tick
    event_note = f"A pair-bonding event occurred. Experience score: {experience_score:.2f}"
    commit_event({
        "type": "pair_bonding", "actors": [agent1["id"], agent2["id"]],
        "magnitude": experience_score, "time": event_time, "note": event_note
    })
    print(f"  - Tick {tick}: Committed pair_bonding between {agent1['id']} and {agent2['id']}")

    # 4. Handle Conception
    race1, race2 = agent1.get("race"), agent2.get("race")
    sex1, sex2 = agent1.get("sex"), agent2.get("sex")
    if race1 == race2 and sex1 != sex2 and experience_score > 0.5:
        female_agent = agent1 if sex1 == "female" else agent2
        fertility = female_agent.get("body_morph", {}).get("fertility", 0)
        if random.random() < fertility * 0.1: # 10% of fertility score is chance
            conception_note = f"{female_agent['id']} has conceived a child with {agent1['id'] if sex1 != 'female' else agent2['id']}."
            commit_event({
                "type": "conception", "actors": [agent1["id"], agent2["id"]],
                "magnitude": 1.0, "time": event_time, "note": conception_note
            })
            print(f"  - Tick {tick}: ✨ CONCEPTION! {conception_note}")


# --- Main Simulation Logic ---

def run_simulation():
    """Main function to run the social fabric simulation."""
    print("Starting Social Fabric Simulation v2...")
    
    world_data = _load_world_data()
    # Preferred sources
    agents = world_data.get("agents", [])
    current_edges = world_data.get("relationship_matrix", {}).get("edges", [])
    # Historical/legacy fallback
    if not agents:
        agents = world_data.get("meta", {}).get("relationship_graph", {}).get("nodes", [])
    if not current_edges:
        current_edges = world_data.get("meta", {}).get("relationship_graph", {}).get("edges", [])

    # Last-resort synthetic fallback
    if not agents:
        print("[WARN] No agents present in world_data.json. Activating synthetic agent fallback.")
        agents, current_edges = _synthesize_agents_and_edges(world_data, per_race=8)
        print(f"[INFO] Synthesized {len(agents)} agents and {len(current_edges)} edges for this simulation run.")
    
    if not agents:
        print("No agents found or synthesized. Aborting simulation.")
        return

    # Build the relationship matrix for efficient lookups
    relationship_matrix = _build_relationship_matrix(current_edges)
    print(f"Built relationship matrix for {len(agents)} agents.")

    # Ensure agents have the new data structures
    for agent in agents:
        if "psyche" not in agent: agent["psyche"] = {}
        if "body_morph" not in agent: agent["body_morph"] = {}
        if "pressures" not in agent: agent["pressures"] = {}
        if "needs" not in agent: agent["needs"] = {}
        if "inventory" not in agent: agent["inventory"] = {}
        agent["psyche"].setdefault("reproductive_drive", 0.0)
        agent["psyche"].setdefault("bonding_capacity", 0.1)
        agent["psyche"].setdefault("aggression", 0.2)
        agent["pressures"].setdefault("frustration", 0.0)
        agent["needs"].setdefault("hunger", 0.0)
        agent["inventory"].setdefault("food", 1)

    print(f"Found {len(agents)} agents. Beginning simulation for {SIMULATION_TICKS} ticks.")
    start_time = time.time()
    
    for tick in range(SIMULATION_TICKS):
        # 1. Update internal states for all agents
        for agent in agents:
            _update_reproductive_drive(agent, agents)
            _update_agent_needs(agent)

        # 2. Process interactions
        for i in range(len(agents)):
            # Solo actions (like foraging) can happen outside of interactions
            agent1 = agents[i]
            if agent1["needs"].get("hunger", 0) > 0.5 and random.random() < 0.1:
                 _handle_forage(agent1, tick, start_time)

            for j in range(i + 1, len(agents)):
                agent2 = agents[j]
                
                pressures = get_current_pressures(agent1["id"], agent2["id"], relationship_matrix)
                interaction_prob = calculate_interaction_prob(agent1, agent2, pressures)
                
                if random.random() < interaction_prob:
                    event_type = generate_event_type(agent1, agent2, pressures)
                    
                    if event_type == "pair_bonding":
                        _handle_pair_bonding(agent1, agent2, tick, start_time)
                    elif event_type == "theft":
                        _handle_theft(agent1, agent2, tick, start_time)
                    else:
                        # Handle other events
                        magnitude = random.uniform(0.1, 0.5)
                        event_time = int(start_time) + tick
                        commit_event({
                            "type": event_type, "actors": [agent1["id"], agent2["id"]],
                            "magnitude": magnitude, "time": event_time,
                            "note": f"Simulated {event_type} event at tick {tick}."
                        })
                        print(f"  - Tick {tick}: Committed {event_type} between {agent1['id']} and {agent2['id']}")

        if tick > 0 and tick % 100 == 0:
            print(f"Progress: Tick {tick}/{SIMULATION_TICKS} complete.")

    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
    print("Social fabric history has been enriched with complex behaviors.")

if __name__ == "__main__":
    run_simulation()
