import json
import math
import os
import hashlib
import random
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np

# Manifold engine integration is optional and not included in this distribution.
# This standalone version operates without it.
MANIFOLD_AVAILABLE = False

WORLD_PATH = os.path.join('public', 'world_data.json')

# Movement directions (dx, dy) with diagonal factor
DIRS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))
]


def load_world(path=WORLD_PATH) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_world(data: Dict, path=WORLD_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def compute_slope(elev: np.ndarray) -> np.ndarray:
    sx = np.abs(np.roll(elev, -1, 1) - elev)
    sy = np.abs(np.roll(elev, -1, 0) - elev)
    slope = np.sqrt(sx * sx + sy * sy)
    if slope.max() > 0:
        slope = slope / slope.max()
    return slope


def compute_conductance(biome_map: List[List[str]], flow: np.ndarray, elev: np.ndarray, sea_level: float) -> np.ndarray:
    h, w = elev.shape
    cond = np.ones((h, w), dtype=float) * 0.8
    # Base conductance per biome (aligned with generator)
    biome_to_cond = {
        'ocean': 0.0,
        'river': 1.15,
        'plains': 1.0,
        'farmland': 1.0,
        'savanna': 0.9,
        'forest': 0.75,
        'montane_forest': 0.55,
        'hills': 0.6,
        'mountain': 0.25,
        'alpine': 0.35,
        'snow': 0.2,
        'desert': 0.5,
        'beach': 0.8,
        'coastal_scrub': 0.8,
        'coastal_scrub_oak': 0.75,
        'mangrove_wetland': 0.6,
        'palm_canopy': 0.85,
        'wetland': 0.55,
    }
    for y in range(h):
        for x in range(w):
            b = biome_map[y][x]
            cond[y, x] = biome_to_cond.get(b, cond[y, x])
    # Rivers facilitate movement
    cond += 0.35 * np.clip(flow, 0, 1)
    # Impede by slope
    slope = compute_slope(elev)
    cond *= (1.0 - 0.65 * slope)
    # Ocean blocks
    cond[elev <= sea_level] = 0.0
    return np.clip(cond, 0.0, 1.2)


def make_roads_mask(roads: List[List[List[int]]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for path in roads or []:
        for x, y in path:
            if 0 <= x < w and 0 <= y < h:
                mask[y, x] = True
    return mask


def a_star(start: Tuple[int, int], goal: Tuple[int, int], cost_grid: np.ndarray) -> List[Tuple[int, int]]:
    """A* on a small grid; cost_grid contains per-tile base costs (>=0, inf to block)."""
    h, w = cost_grid.shape
    sx, sy = start
    gx, gy = goal

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < w and 0 <= y < h

    def heuristic(x: int, y: int) -> float:
        dx = x - gx
        dy = y - gy
        return math.hypot(dx, dy)

    INF = 1e18
    gscore = np.full((h, w), INF, dtype=float)
    fscore = np.full((h, w), INF, dtype=float)
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}

    gscore[sy, sx] = 0.0
    fscore[sy, sx] = heuristic(sx, sy)
    open_set: set[Tuple[int, int]] = {(sx, sy)}

    while open_set:
        # pick node with smallest fscore
        current = min(open_set, key=lambda p: fscore[p[1], p[0]])
        cx, cy = current
        if current == (gx, gy):
            # reconstruct path
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            path.reverse()
            return path

        open_set.remove((cx, cy))
        cbase = cost_grid[cy, cx]
        if cbase >= INF:
            continue
        for dx, dy, diag in DIRS:
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny):
                continue
            step_cost = cost_grid[ny, nx]
            if step_cost >= INF:
                continue
            tentative = gscore[cy, cx] + step_cost * diag
            if tentative < gscore[ny, nx]:
                came[(nx, ny)] = (cx, cy)
                gscore[ny, nx] = tentative
                fscore[ny, nx] = tentative + heuristic(nx, ny)
                open_set.add((nx, ny))

    return []


def pick_village_sites(race: str, capital: Tuple[int, int], elev: np.ndarray, moisture: np.ndarray,
                       biome_map: List[List[str]], flow: np.ndarray, sea_level: float,
                       existing: List[Tuple[int, int]], count: int = 2) -> List[Tuple[int, int]]:
    """Score tiles and pick a few good village sites around the capital with spacing constraints."""
    h, w = elev.shape
    cx, cy = capital
    slope = compute_slope(elev)

    # Simple biome preferences for villages
    base_pref = {
        'plains': 1.0, 'savanna': 0.85, 'forest': 0.8, 'coastal_scrub': 0.75, 'coastal_scrub_oak': 0.75,
        'palm_canopy': 0.7, 'wetland': 0.55, 'beach': 0.6, 'hills': 0.55,
        'montane_forest': 0.3, 'mountain': 0.1, 'alpine': 0.1, 'snow': 0.05, 'desert': 0.3,
        'river': 0.9, 'mangrove_wetland': 0.4, 'ocean': 0.0
    }

    score = np.zeros((h, w), dtype=float)
    for y in range(h):
        for x in range(w):
            if elev[y, x] <= sea_level:
                continue
            b = biome_map[y][x]
            s = base_pref.get(b, 0.5)
            # Prefer near rivers and gentle slopes
            s += 0.25 * min(1.0, max(flow[y, x],
                                      flow[y, x-1] if x > 0 else 0.0,
                                      flow[y, x+1] if x < w-1 else 0.0,
                                      flow[y-1, x] if y > 0 else 0.0,
                                      flow[y+1, x] if y < h-1 else 0.0))
            s *= (1.0 - 0.5 * slope[y, x])
            # Distance preference: not too close, not too far
            dx = x - cx; dy = y - cy
            dist = math.hypot(dx, dy)
            if dist < max(3, min(h, w) * 0.05) or dist > min(h, w) * 0.35:
                s *= 0.2
            score[y, x] = s

    picks: List[Tuple[int, int]] = []
    flat_idx = np.argsort(score, axis=None)[::-1]
    min_spacing = max(3, int(min(h, w) * 0.06))
    for idx in flat_idx:
        y, x = divmod(int(idx), w)
        if score[y, x] <= 0:
            break
        ok = True
        for (vx, vy) in existing + picks:
            if math.hypot(x - vx, y - vy) < min_spacing:
                ok = False
                break
        if ok:
            picks.append((x, y))
            if len(picks) >= count:
                break
    return picks


def build_cost_grid(cond: np.ndarray, elev: np.ndarray, sea_level: float, roads_mask: np.ndarray) -> np.ndarray:
    # Convert conductance to cost; encourage following existing roads
    base = np.where(cond > 0, 1.0 / (1e-3 + cond), 1e18)
    # reward existing roads by lowering cost
    base = np.where(roads_mask, np.maximum(0.15, base * 0.35), base)
    # disallow ocean
    base = np.where(elev <= sea_level, 1e18, base)
    return base


# --- Deterministic PRNG helpers for persistent object behavior ---
def _deterministic_seed(*parts: object) -> int:
    """Create a stable 31-bit seed from arbitrary parts using blake2b."""
    s = "|".join(str(p) for p in parts)
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) & 0x7FFFFFFF


def _ensure_settlement_traits(lm: Dict, world_seed: int) -> Dict:
    """Ensure a settlement landmark has persistent trait parameters; return the traits dict."""
    traits = lm.get('traits')
    if isinstance(traits, dict) and traits:
        return traits

    name = lm.get('name', 'Settlement')
    pos = lm.get('position', [0, 0])
    sx, sy = int(pos[0]), int(pos[1])
    seed = _deterministic_seed(world_seed, 'traits', name, sx, sy)
    rng = random.Random(seed)

    traits = {
        # Multiplier applied to race baseline r
        'r_factor': rng.uniform(0.85, 1.15),
        # Carrying capacity contribution per farmland tile
        'farm_cap_per_tile': rng.uniform(130.0, 180.0),
        # How strongly stress reduces r and K
        'stress_r_coeff': rng.uniform(0.55, 0.85),  # default ~0.7
        'stress_k_coeff': rng.uniform(0.30, 0.50),  # default ~0.4
        # Stochastic shocks/boons
        'shock_rate': rng.uniform(0.002, 0.010),
        'shock_max': rng.uniform(0.05, 0.12),
        'boon_rate': rng.uniform(0.005, 0.020),
    }
    lm['traits'] = traits
    return traits


# --- Cohort Temporal Profiles (mirrored from social_fabric.py) ---
# Dimensionless parameters — see social_fabric.py for full documentation.
DEFAULT_EPOCH_TICKS: int = 100

COHORT_PROFILES = {
    "default":  {"epoch_scale": 1.0, "maturation": 0.20, "bond_onset": 0.20, "bond_offset": 0.56, "senescence": 0.75},
    "fast":     {"epoch_scale": 0.8, "maturation": 0.22, "bond_onset": 0.20, "bond_offset": 0.56, "senescence": 0.70},
    "moderate": {"epoch_scale": 1.5, "maturation": 0.16, "bond_onset": 0.12, "bond_offset": 0.60, "senescence": 0.78},
    "slow":     {"epoch_scale": 3.0, "maturation": 0.10, "bond_onset": 0.10, "bond_offset": 0.60, "senescence": 0.85},
}


def _resolve_profile(group_label: str) -> dict:
    """Resolve a group label to its cohort profile with computed tick values."""
    key = group_label.lower()
    profile = COHORT_PROFILES.get(key, COHORT_PROFILES["default"])
    epoch = DEFAULT_EPOCH_TICKS * profile["epoch_scale"]
    return {
        "epoch":       epoch,
        "maturation":  profile["maturation"] * epoch,
        "bond_onset":  profile["bond_onset"] * epoch,
        "bond_offset": profile["bond_offset"] * epoch,
        "senescence":  profile["senescence"] * epoch,
    }


def _generate_agents_and_relationships(landmarks: List[Dict], world_seed: int) -> Tuple[List[Dict], Dict]:
    """
    Generates a canonical list of agents based on settlement populations and establishes
    a baseline relationship matrix. This should replace any synthetic generation.
    """
    agents: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    agent_count = 0

    # Use a deterministic RNG for agent creation
    master_rng = random.Random(_deterministic_seed(world_seed, 'agent_generation'))

    for settlement in landmarks:
        if settlement.get('type') not in ['village', 'town', 'city']:
            continue

        population = int(settlement.get('population', 0))
        if population <= 0:
            continue

        settlement_name = settlement.get('name', 'Unknown Settlement')
        settlement_race = settlement.get('race', 'Human').lower()
        mean_age = settlement.get('mean_age', 30)
        
        # Determine how many agents to generate (e.g., 1 agent per 10 people for performance)
        num_agents_to_create = max(1, population // 10)
        
        settlement_agents = []
        for i in range(num_agents_to_create):
            agent_id = f"{settlement_race[:3].upper()}-{agent_count}"
            agent_count += 1
            
            # Use settlement properties to inform agent creation
            profile = _resolve_profile(settlement_race)
            age = int(master_rng.gauss(mu=mean_age, sigma=15))
            age = max(1, min(int(profile['epoch']), age))  # Clamp to epoch

            agent = {
                "id": agent_id,
                "name": f"{settlement_race.capitalize()} {i} of {settlement_name}",
                "race": settlement_race,
                "sex": "female" if master_rng.random() < 0.5 else "male",
                "body_morph": {
                    "age": age,
                    "fertility": master_rng.random(),
                },
                "psyche": {
                    "trust": master_rng.uniform(0.2, 0.8),
                    "playful": master_rng.uniform(0.2, 0.8),
                    "reproductive_drive": master_rng.uniform(0.05, 0.35),
                    "bonding_capacity": master_rng.uniform(0.05, 0.25)
                },
                "pressures": {
                    "frustration": master_rng.uniform(0.0, 0.2)
                },
                "home_settlement": settlement_name,
                # UNIFIED FIELD THEORY: Initialize biological drivers + circadian rhythm
                "biological_state": {
                    "cortisol_analog": master_rng.uniform(4.0, 6.0),
                    "oxytocin_analog": master_rng.uniform(4.0, 6.0),
                    "dopamine_analog": master_rng.uniform(4.0, 6.0),
                    "serotonin_analog": master_rng.uniform(4.0, 6.0),
                },
                "circadian_state": {
                    "circadian_phase": master_rng.uniform(0.0, 24.0),
                    "sleep_pressure": master_rng.uniform(0.0, 3.0),
                    "wake_duration": master_rng.uniform(0.0, 12.0),
                },
                "need_state": {
                    "physical_intensity": master_rng.uniform(0.1, 0.4),
                    "emotional_intensity": master_rng.uniform(0.1, 0.4),
                    "cognitive_intensity": master_rng.uniform(0.1, 0.4),
                }
            }
            agents.append(agent)
            settlement_agents.append(agent)

        # Create baseline relationships within the settlement
        for i in range(len(settlement_agents)):
            for j in range(i + 1, len(settlement_agents)):
                # Create sparse connections
                if master_rng.random() < 0.2:
                    edges.append({
                        "a": settlement_agents[i]["id"],
                        "b": settlement_agents[j]["id"],
                        "intimacy": master_rng.uniform(0.05, 0.25),
                        "love": master_rng.uniform(0.0, 0.2),
                        "conflict": master_rng.uniform(0.0, 0.2),
                        "pair_bonding": master_rng.uniform(0.0, 0.1),
                    })

    relationship_matrix = {"edges": edges}
    return agents, relationship_matrix


def evolve_once(data: Dict) -> Dict:
    shape = data.get('shape', {})
    w = int(shape.get('width', 0))
    h = int(shape.get('height', 0))
    if w == 0 or h == 0:
        return data

    # Ensure a persistent world seed exists for deterministic stochasticity
    meta = data.get('meta') or {}
    world_seed = meta.get('world_seed')
    if world_seed is None:
        world_seed = random.Random(os.getpid()).randint(1, 0x7FFFFFFF)
        meta['world_seed'] = int(world_seed)
        data['meta'] = meta
    current_step = int(meta.get('evolution_steps', 0)) + 1

    terrain = data.get('terrain') or {}
    elev = np.array(terrain.get('elevation', [[0]*w for _ in range(h)]), dtype=float)
    moisture = np.array(terrain.get('moisture', [[0]*w for _ in range(h)]), dtype=float)
    biome_map: List[List[str]] = terrain.get('biome_map', [['plains']*w for _ in range(h)])
    sea_level = float(terrain.get('sea_level', 0.42))
    flow = np.array(terrain.get('rivers', [[0]*w for _ in range(h)]), dtype=float)

    capitals: Dict[str, List[int]] = data.get('capitals', {})
    landmarks: List[Dict] = data.get('landmarks', [])
    roads: List[List[List[int]]] = data.get('roads', [])
    stress_map = np.array(data.get('stress_map', [[0]*w for _ in range(h)]), dtype=float)

    cond = compute_conductance(biome_map, flow, elev, sea_level)
    roads_mask = make_roads_mask(roads, h, w)
    cost_grid = build_cost_grid(cond, elev, sea_level, roads_mask)

    # Gather existing village positions
    existing_villages = [(lm['position'][0], lm['position'][1]) for lm in landmarks if lm.get('type') == 'village']

    # --- Population dynamics parameters to avoid exponential runaway ---
    # Group-specific baseline growth and mortality (dimensionless per step), and life expectancy (steps)
    population_params: Dict[str, Dict[str, float]] = {
        'Alpha': {'r': 0.04, 'm': 0.010, 'L': 70.0, 'Kmul': 1.00},
        'Beta':  {'r': 0.020, 'm': 0.005, 'L': 200.0, 'Kmul': 0.95},
        'Gamma': {'r': 0.018, 'm': 0.006, 'L': 120.0, 'Kmul': 0.90},
        'Delta': {'r': 0.010, 'm': 0.004, 'L': 150.0, 'Kmul': 0.85},
    }

    # Group-specific preferences for biomes (for stress generation)
    # Higher value means converting this biome causes more stress for the group.
    GROUP_BIOME_PREFERENCE = {
        'Gamma': {'forest': 0.15, 'rainforest': 0.2, 'boreal_forest': 0.15, 'palm_canopy': 0.1, 'montane_forest': 0.1},
        'Beta': {'mountain': 0.1, 'hills': 0.05, 'rock': 0.08},
        'Alpha': {},  # Adaptable, less attached to specific biomes
        'Delta': {},  # Indifferent to surface ecology
    }

    # Helper: local suitability (0..1) and carrying capacity K based on biome/slope/river/roads/stress
    slope = compute_slope(elev)
    biome_map_arr = np.array(biome_map)

    def local_suitability(x: int, y: int) -> float:
        if elev[y, x] <= sea_level:
            return 0.0
        b = biome_map[y][x]
        base_pref = {
            'plains': 1.0, 'farmland': 1.2, 'savanna': 0.85, 'forest': 0.8, 'coastal_scrub': 0.75, 'coastal_scrub_oak': 0.75,
            'palm_canopy': 0.7, 'wetland': 0.65, 'beach': 0.6, 'hills': 0.55,
            'montane_forest': 0.35, 'mountain': 0.2, 'alpine': 0.25, 'snow': 0.15, 'desert': 0.35,
            'river': 0.9, 'mangrove_wetland': 0.5, 'ocean': 0.0
        }
        s = base_pref.get(b, 0.5)
        # Favor gentle slope and river proximity
        fn = flow[y, x]
        if fn <= 0.12:
            fn = max(
                flow[y, x],
                flow[y, x-1] if x > 0 else 0.0,
                flow[y, x+1] if x < w-1 else 0.0,
                flow[y-1, x] if y > 0 else 0.0,
                flow[y+1, x] if y < h-1 else 0.0
            )
        s += 0.2 * min(1.0, fn)
        s *= (1.0 - 0.4 * slope[y, x])
        return float(np.clip(s, 0.0, 1.0))

    roads_mask = make_roads_mask(roads, h, w)
    def local_capacity(x: int, y: int, race: str, settlement_farms: int,
                       farm_cap_per_tile: float = 150.0,
                       stress_k_coeff: float = 0.4) -> float:
        # Carrying capacity is now primarily a function of farmland
        # Each farmland tile can support a certain number of people.
        K = 50.0 + settlement_farms * float(farm_cap_per_tile)

        # Road connectivity still provides a bonus (trade, access to other resources)
        deg = 0
        if roads:
            for path in roads:
                for rx, ry in path:
                    if rx == x and ry == y:
                        deg += 1
                        break
        K += 60.0 * deg
        
        stress_local = float(np.clip(stress_map[y, x], 0.0, 1.0))
        K *= (1.0 - float(stress_k_coeff) * stress_local)
        K *= population_params.get(race, {}).get('Kmul', 1.0)
        return max(50.0, K)  # minimum carrying capacity

    # --- Update existing villages: logistic growth with mortality & age ---
    founder_default = int(data.get('meta', {}).get('evolution_steps', 0))
    
    # Ensure all settlements have persistent traits
    for lm in landmarks:
        if lm.get('type') in ['village', 'town', 'city']:
            _ensure_settlement_traits(lm, int(world_seed))

    # --- Farmland Expansion ---
    # For each settlement, convert nearby suitable land to farmland based on population
    for lm in landmarks:
        if lm.get('type') not in ['village', 'town', 'city']:
            continue
        
        pop = float(lm.get('population', 0))
        if pop <= 0:
            continue

        vx, vy = int(lm['position'][0]), int(lm['position'][1])
        traits = _ensure_settlement_traits(lm, int(world_seed))
        
        # Each person requires a certain amount of farmland.
        # Use settlement-specific farm capacity per tile.
        required_farm_tiles = int(math.ceil(pop / max(1e-6, float(traits['farm_cap_per_tile']))))

        # Find existing farmland for this settlement
        # A simple approach: search in a radius around the settlement
        search_radius = int(5 + math.sqrt(pop) / 10)
        min_x, max_x = max(0, vx - search_radius), min(w, vx + search_radius + 1)
        min_y, max_y = max(0, vy - search_radius), min(h, vy + search_radius + 1)
        
        settlement_farms = np.sum(biome_map_arr[min_y:max_y, min_x:max_x] == 'farmland')

        if settlement_farms < required_farm_tiles:
            tiles_to_convert = required_farm_tiles - settlement_farms
            
            # Find suitable tiles to convert
            potential_tiles = []
            for r in range(1, search_radius + 1):
                for i in range(-r, r + 1):
                    for j in [j for j in [-r, r] if i != -r and i != r] + ([i] if i == -r or i == r else []):
                        px, py = vx + i, vy + j
                        if 0 <= px < w and 0 <= py < h:
                            biome = biome_map[py][px]
                            # Can convert plains and forests, but plains are easier
                            if biome in ['plains', 'forest', 'savanna', 'grassland']:
                                score = 1.0 if biome == 'plains' else 0.6
                                score *= (1.0 - slope[py][px]) # Prefer flat land
                                potential_tiles.append(((px, py), score))
            
            potential_tiles.sort(key=lambda item: item[1], reverse=True)
            
            territory_map = np.array(data.get('territory_map', [[0]*w for _ in range(h)]))
            race_id_map = {i + 1: r for i, r in enumerate(capitals.keys())}

            converted_count = 0
            for (px, py), score in potential_tiles:
                if converted_count >= tiles_to_convert:
                    break
                
                original_biome = biome_map[py][px]
                if original_biome != 'farmland':
                    # Check for ecological grudge
                    if original_biome in ['forest', 'rainforest', 'boreal_forest', 'montane_forest', 'palm_canopy']:
                        territory_id = int(territory_map[py, px]) # Cast to python int
                        if territory_id > 0 and territory_id in race_id_map:
                            affected_race = race_id_map[territory_id]
                            prefs = GROUP_BIOME_PREFERENCE.get(affected_race, {})
                            if original_biome in prefs:
                                stress_increase = prefs[original_biome]
                                # Increase stress in a small radius
                                for dy in range(-1, 2):
                                    for dx in range(-1, 2):
                                        ny, nx = py + dy, px + dx
                                        if 0 <= ny < h and 0 <= nx < w:
                                            stress_map[ny, nx] = min(1.0, stress_map[ny, nx] + stress_increase / (1.0 + abs(dx) + abs(dy)))


                    biome_map[py][px] = 'farmland'
                    settlement_farms += 1
                    converted_count += 1

    for lm in landmarks:
        if lm.get('type') not in ['village', 'town', 'city']:
            continue
        vx, vy = int(lm['position'][0]), int(lm['position'][1])
        race = lm.get('race') or 'Alpha'
        params = population_params.get(race, population_params['Alpha'])
        traits = _ensure_settlement_traits(lm, int(world_seed))
        
        # Recalculate settlement farms after conversion phase
        pop = max(10.0, float(lm.get('population') or 100))
        search_radius = int(5 + math.sqrt(pop) / 10)
        min_x, max_x = max(0, vx - search_radius), min(w, vx + search_radius + 1)
        min_y, max_y = max(0, vy - search_radius), min(h, vy + search_radius + 1)
        settlement_farms = np.sum(np.array(biome_map)[min_y:max_y, min_x:max_x] == 'farmland')

        # Initialize demographics
        mean_age = float(lm.get('mean_age') or 24.0)
        founder_step = int(lm.get('founder_step') or founder_default)
        
        # Suitability & capacity
        suit = local_suitability(vx, vy)
        K = local_capacity(vx, vy, race, settlement_farms,
                          farm_cap_per_tile=float(traits['farm_cap_per_tile']),
                          stress_k_coeff=float(traits['stress_k_coeff']))
        stress_local = float(np.clip(stress_map[vy, vx], 0.0, 1.0))
        
        # Effective growth & mortality with traits and pressures
        r0 = params['r'] * float(traits['r_factor'])
        # Positive pressure from farmland surplus and road connectivity
        required_tiles = int(math.ceil(pop / max(1e-6, float(traits['farm_cap_per_tile']))))
        surplus_ratio = (float(settlement_farms) - float(required_tiles)) / max(1.0, float(required_tiles))
        # Road connectivity degree at this node
        deg = 0
        if roads:
            for path in roads:
                for rx, ry in path:
                    if rx == vx and ry == vy:
                        deg += 1
                        break
        pos_boost = 0.15 * float(np.clip(surplus_ratio, -1.0, 1.0)) + 0.03 * min(3, deg)
        r_eff = r0 * (0.5 + 0.5 * suit) * (1.0 - float(traits['stress_r_coeff']) * stress_local)
        r_eff *= (1.0 + pos_boost)
        m_eff = params['m'] * (1.0 + 0.5 * stress_local)
        
        # Extra age-related mortality as sigmoid beyond ~80% of life expectancy
        L = params['L']
        age_excess = (mean_age - 0.8 * L) / max(1e-6, 0.1 * L)
        age_mort = 0.03 * (1.0 / (1.0 + math.exp(-age_excess)))
        m_eff += max(0.0, age_mort)
        
        # Logistic births and deaths
        births = max(0.0, r_eff * pop * (1.0 - pop / K))
        deaths = m_eff * pop
        new_pop = max(0.0, pop + births - deaths)

        # Stochastic shocks (decay) and boons (growth), deterministic per step & settlement
        rng = random.Random(_deterministic_seed(world_seed, 'shock_boon', lm.get('name', ''), vx, vy, current_step))
        # Negative shock chance increases with stress
        if rng.random() < float(traits['shock_rate']) * (0.25 + 0.75 * stress_local):
            shock_fraction = rng.uniform(0.0, float(traits['shock_max'])) * max(0.1, stress_local)
            new_pop *= max(0.0, 1.0 - shock_fraction)
        # Positive boon more likely with low stress and surplus
        if rng.random() < float(traits['boon_rate']) * (1.0 - stress_local) * (0.2 + max(0.0, min(1.0, surplus_ratio))):
            boon_fraction = rng.uniform(0.01, 0.04)
            new_pop *= (1.0 + boon_fraction)
        
        # Update mean age (coarse): everyone ages +1, newborns at age 0
        total_age = mean_age * pop + 0.0 * births
        total_people = max(1e-6, pop + births)
        mean_age = min(L * 2.0, (total_age / total_people) + 1.0)  # cap at 2x L to avoid blow-up
        
        # Apply updates
        lm['population'] = int(round(new_pop))
        lm['mean_age'] = float(round(mean_age, 2))
        lm['founder_step'] = founder_step
        
        # Promote/demote settlement type based on population tiers
        if new_pop >= 5000:
            lm['type'] = 'city'
        elif new_pop >= 1000:
            lm['type'] = 'town'
        else:
            lm['type'] = 'village'

    # --- Place a couple villages per race per evolution step ---
    id_to_race = list(capitals.keys())
    for race in id_to_race:
        cx, cy = capitals[race]
        new_sites = pick_village_sites(race, (cx, cy), elev, moisture, biome_map, flow, sea_level, existing_villages, count=2)
        for (vx, vy) in new_sites:
            idx = sum(1 for lm in landmarks if lm.get('type') == 'village' and lm.get('race') == race) + 1
            name = f"{race} Village #{idx}"
            landmarks.append({
                'name': name,
                'type': 'village',
                'race': race,
                'position': [int(vx), int(vy)],
                'founded_at': datetime.utcnow().isoformat() + 'Z',
                'population': int(80 + np.random.default_rng().integers(0, 120)),
                'mean_age': float(22 + np.random.default_rng().integers(0, 10)),
                'founder_step': int(data.get('meta', {}).get('evolution_steps', 0)) + 1,
            })
            existing_villages.append((vx, vy))

            # Connect to nearest network node (capital or existing village of same race)
            cpos = capitals[race]
            cxy: Tuple[int, int] = (int(cpos[0]), int(cpos[1]))
            network_nodes: List[Tuple[int, int]] = [cxy] + [
                (lm['position'][0], lm['position'][1])
                for lm in landmarks if lm.get('type') == 'village' and lm.get('race') == race and (lm['position'][0], lm['position'][1]) != (vx, vy)
            ]
            if network_nodes:
                # choose the closest by straight-line distance first; we'll still pathfind
                nx, ny = min(network_nodes, key=lambda p: math.hypot(p[0]-vx, p[1]-vy))
                path = a_star((vx, vy), (nx, ny), cost_grid)
                if len(path) >= 2:
                    # dedupe trivial repeats
                    dedup: List[List[int]] = []
                    last = None
                    for (px, py) in path:
                        if last != (px, py):
                            dedup.append([int(px), int(py)])
                            last = (px, py)
                    roads.append(dedup)
                    # update roads mask/costs to bias future routes
                    for (px, py) in path:
                        if 0 <= px < w and 0 <= py < h:
                            roads_mask[py, px] = True
                    cost_grid = build_cost_grid(cond, elev, sea_level, roads_mask)

    # Update metadata for evolution bookkeeping
    meta = data.get('meta') or {}
    steps = int(meta.get('evolution_steps', 0)) + 1
    meta['evolution_steps'] = steps
    meta['last_evolved_at'] = datetime.utcnow().isoformat() + 'Z'
    data['meta'] = meta

    data['landmarks'] = landmarks
    data['roads'] = roads
    data['terrain']['biome_map'] = biome_map

    # --- Generate Structures (Quadrafoil of Influence) ---
    structures = data.get('structures', [])
    if not any(s['name'] == 'Central Sanctuary' for s in structures):
        # Find the highest point on the map
        max_elev_val = elev.max()
        highest_points = np.argwhere(elev == max_elev_val)
        # Select one deterministically
        tower_pos_idx = _deterministic_seed(world_seed, 'sanctuary_pos') % len(highest_points)
        ty, tx = highest_points[tower_pos_idx]
        
        structures.append({
            "id": "struct_central_sanctuary",
            "name": "Central Sanctuary",
            "type": "sanctuary",
            "position": [int(tx), int(ty)],
            "influence": {
                "trust": 0.2,
                "aggression": -0.3,
                "playfulness": 0.1
            }
        })
        print(f"Placed Central Sanctuary at highest point: ({tx}, {ty})")

    data['structures'] = structures

    # --- Generate Agents and Relationships ---
    # This runs *after* population evolution, so agents reflect the latest world state.
    agents, relationship_matrix = _generate_agents_and_relationships(landmarks, int(world_seed))
    data['agents'] = agents
    data['relationship_matrix'] = relationship_matrix

    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Incrementally evolve the world: add villages and roads.')
    parser.add_argument('--steps', type=int, default=1, help='Number of evolution steps to apply')
    args = parser.parse_args()

    world = load_world()
    for _ in range(max(1, args.steps)):
        world = evolve_once(world)
    save_world(world)
    print(f"Evolution complete. Steps applied: {args.steps}. Updated -> {WORLD_PATH}")
