# mbd/active_states.py — Action selection and motor commands

class ActiveStates:
    """
    Represents the action interface of the agent to the world.
    It determines which action to take based on the agent's internal states.
    This forms the "active" part of the Markov Blanket.
    """
    def __init__(self, agent_id, world_state):
        self.agent_id = agent_id
        self.available_actions = ['idle', 'move', 'forage', 'attack']

    def choose_action(self, internal_states):
        """
        Selects an action based on the agent's current beliefs, needs, and goals.
        """
        # Placeholder for decision-making logic (e.g., utility-based, FSM, etc.)
        # Simple example: if hungry, forage. If unsafe, move.
        if internal_states.needs.get('hunger', 0) > 0.7:
            return {'type': 'forage', 'target': None}
        
        if internal_states.needs.get('safety', 1.0) < 0.5:
            # Choose a random direction to move away
            import random
            direction = random.choice(['north', 'south', 'east', 'west'])
            return {'type': 'move', 'direction': direction}

        return {'type': 'idle'}

    def __repr__(self):
        return f"<ActiveStates for Agent {self.agent_id}>"
