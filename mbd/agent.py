# mbd/agent.py — Core MBD agent with Markov Blanket architecture

from .internal_states import InternalStates
from .sensory_states import SensoryStates
from .active_states import ActiveStates

class Agent:
    """
    Represents a single agent operating under the principles of a Markov Blanket,
    now capable of participating in a Markov Hypercube.
    It integrates internal, sensory, and active states to produce behavior.
    """
    def __init__(self, agent_id, world_state, hypercube_ref, **kwargs):
        self.id = agent_id
        self.hypercube = hypercube_ref # A reference to the global Hypercube
        self.internal_states = InternalStates(agent_id=self.id, **kwargs)
        self.sensory_states = SensoryStates(agent_id=self.id, world_state=world_state)
        self.active_states = ActiveStates(agent_id=self.id, world_state=world_state)

    def tick(self, world_state):
        """
        Executes one cycle of the agent's perception-action loop.
        1. Perceive the world (update sensory states).
        2. Update beliefs from direct perception.
        3. Interpolate objective models from the Hypercube.
        4. Update beliefs from interpolation.
        5. Choose an action based on refined beliefs.
        """
        # 1. Perceive the world and update sensory states
        percepts = self.sensory_states.perceive(self.id, world_state)

        # 2. Update internal states (beliefs) based on new sensory information
        self.internal_states.update(percepts)

        # 3. Query the Hypercube for a higher-order model of an external state.
        # As a placeholder, let's try to get a better model of a target agent's faction.
        if percepts.get('nearby_agents'):
            target_agent_id = percepts['nearby_agents'][0]['id']
            belief_key = f"agent_{target_agent_id}_faction"
            
            # Initialize a belief about the target if one doesn't exist
            if belief_key not in self.internal_states.beliefs:
                self.internal_states.beliefs[belief_key] = {'value': 'unknown', 'certainty': 0.1}

            # 4. Interpolate and update belief
            objective_model = self.hypercube.interpolate_objective_model(belief_key, self.id)
            self.internal_states.update_belief_from_interpolation(belief_key, objective_model)

        # 5. Choose an action based on updated internal states
        action = self.active_states.choose_action(self.internal_states)
        
        return action

    def __repr__(self):
        return f"<Agent {self.id}>"
