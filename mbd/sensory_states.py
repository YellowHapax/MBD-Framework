# mbd/sensory_states.py — Sensory perception interface

class SensoryStates:
    """
    Represents the sensory interface of the agent to the world.
    It queries the world state to gather information within the agent's perception range.
    This forms the "sensory" part of the Markov Blanket.
    """
    def __init__(self, agent_id, world_state):
        self.agent_id = agent_id
        self.perception_radius = 5 # Example radius

    def perceive(self, agent_id, world_state):
        """
        Gathers sensory information from the world_state for a given agent.
        """
        percepts = {
            'nearby_agents': [],
            'visible_structures': [],
            'local_terrain': None,
        }
        
        # Placeholder logic: find agent's position and check for nearby entities
        # This will be replaced with more sophisticated queries.
        agent_data = world_state['agents'].get(agent_id)
        if not agent_data:
            return percepts

        agent_pos = agent_data.get('pos')
        if not agent_pos:
            return percepts

        # Example: find nearby agents
        for other_id, other_data in world_state['agents'].items():
            if other_id == agent_id:
                continue
            other_pos = other_data.get('pos')
            if other_pos:
                # A simple distance check
                dist_sq = (agent_pos[0] - other_pos[0])**2 + (agent_pos[1] - other_pos[1])**2
                if dist_sq <= self.perception_radius**2:
                    percepts['nearby_agents'].append(other_data)

        return percepts

    def __repr__(self):
        return f"<SensoryStates for Agent {self.agent_id}>"
