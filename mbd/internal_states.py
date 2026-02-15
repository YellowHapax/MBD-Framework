# mbd/internal_states.py — Internal belief state management

class InternalStates:
    """
    Represents the internal state (beliefs, goals, needs) of an agent.
    Beliefs are now structured as models with a value and a certainty.
    This is the "mind" of the agent, hidden from the external world by the Markov Blanket.
    """
    def __init__(self, agent_id, **kwargs):
        self.agent_id = agent_id
        # Core beliefs are now models with value and certainty
        self.beliefs = {
            'location': {'value': kwargs.get('location', None), 'certainty': 1.0},
            'race': {'value': kwargs.get('race', 'unknown'), 'certainty': 1.0},
            'faction': {'value': kwargs.get('faction', 'unaligned'), 'certainty': 1.0},
        }
        # Dynamic needs
        self.needs = {
            'hunger': kwargs.get('hunger', 0.0),
            'safety': kwargs.get('safety', 1.0),
        }
        # Agent's goals or intentions
        self.goals = []

    def update(self, percepts):
        """
        Update internal states based on sensory percepts.
        This is where the agent's beliefs are revised based on direct observation.
        """
        # Placeholder: a simple update logic
        if 'nearby_agents' in percepts and len(percepts['nearby_agents']) > 0:
            self.needs['safety'] -= 0.1
        else:
            self.needs['safety'] += 0.05
        
        # Clamp values
        self.needs['safety'] = max(0, min(1, self.needs['safety']))

    def update_belief_from_interpolation(self, belief_key, objective_model):
        """
        Updates a belief using a higher-order model interpolated from the Hypercube.
        This is how an agent's perspective is refined by the collective.
        """
        if not objective_model:
            return

        current_belief = self.beliefs.get(belief_key, {'value': None, 'certainty': 0.0})
        
        # A simple weighted average based on certainty. More complex Bayesian updates could be used here.
        current_weight = current_belief['certainty']
        objective_weight = objective_model['certainty']
        total_weight = current_weight + objective_weight

        if total_weight > 0:
            # This is a simplification. A real system might need to handle different data types for 'value'.
            # For now, we assume numeric or overwritable values.
            if isinstance(current_belief['value'], (int, float)) and isinstance(objective_model['value'], (int, float)):
                 new_value = ((current_belief['value'] * current_weight) + (objective_model['value'] * objective_weight)) / total_weight
            else:
                # If not numeric, the higher certainty model wins
                new_value = objective_model['value'] if objective_weight > current_weight else current_belief['value']
            
            new_certainty = min(1.0, total_weight) # Certainty increases, capped at 1.0

            self.beliefs[belief_key] = {'value': new_value, 'certainty': new_certainty}
            # print(f"Agent {self.agent_id} updated belief '{belief_key}' via interpolation. New certainty: {new_certainty:.2f}")

    def __repr__(self):
        return f"<InternalStates for Agent {self.agent_id}>"
