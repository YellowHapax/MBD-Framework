"""
FILE: trauma_mechanics_model.py
PURPOSE: A mathematical console for modeling Memory as Baseline Deviation (MBD).
This script implements the core equations of the MBD framework to simulate
the impact of trauma and interaction on an agent's personality baseline and
their relational coupling (kappa).

This serves as the first version of a "Trauma Console" to move from narrative
descriptions to quantifiable, predictable mechanics.

(⌐■_■)✓
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, NamedTuple

# --- Core MBD Structures ---

class Baseline:
    """Represents an agent's personality state as a vector."""
    def __init__(self, vector: List[float]):
        self.vector = np.array(vector, dtype=float)

    def __repr__(self):
        return f"B({np.round(self.vector, 2)})"

class TraumaForm(NamedTuple):
    """
    Represents a traumatic event as a mathematical object.
    - input_signal (I): The event's emotional/cognitive vector.
    - lambda_learning_rate (λ): How deeply the event is integrated. High λ means high plasticity.
    - description: A human-readable label for the event.
    """
    input_signal: np.ndarray
    lambda_learning_rate: float
    description: str

class Interaction(NamedTuple):
    """
    Represents an interaction between two agents.
    - novelty (N): A measure of how surprising the interaction is (0 to 1).
    - duration (dt): The time over which the interaction occurs.
    """
    novelty: float
    duration: float
    description: str

# --- MBD Core Equations ---

def update_baseline(baseline: Baseline, trauma: TraumaForm) -> Baseline:
    """
    Applies a trauma to a baseline.
    B(t+1) = B(t) * (1 - λ) + I(t) * λ
    """
    b_t = baseline.vector
    I_t = trauma.input_signal
    lambda_val = trauma.lambda_learning_rate

    b_t_plus_1 = b_t * (1 - lambda_val) + I_t * lambda_val
    return Baseline(b_t_plus_1)

def update_kappa(kappa: float, interaction: Interaction, alpha: float, beta: float) -> float:
    """
    Updates the relational coupling (kappa) based on an interaction.
    dκ/dt = α * (1 - N) - β * κ
    """
    novelty = interaction.novelty
    dt = interaction.duration

    # Forward Euler integration for dκ/dt
    d_kappa = (alpha * (1 - novelty) - beta * kappa) * dt
    return kappa + d_kappa

# --- Simulation & Visualization ---

def run_simulation():
    """
    Runs a simulation to demonstrate the evolution of a baseline and kappa.
    """
    # --- Parameters ---
    # Agent's initial state
    initial_baseline = Baseline([0.5, -0.2, 0.1]) # e.g., [Valence, Activation, Dominance]

    # Relational parameters
    initial_kappa = 0.1  # Initial coupling with another agent
    alpha_coupling_gain = 0.2 # How quickly coupling grows in low-novelty situations
    beta_coupling_decay = 0.05 # How quickly coupling decays naturally or with high novelty

    # --- Event Sequence ---
    trauma_sequence: List[TraumaForm] = [
        TraumaForm(np.array([0.8, 0.9, 0.2]), 0.5, "Sudden, intense positive shock"),
        TraumaForm(np.array([-0.9, 0.5, -0.8]), 0.8, "Betrayal event (high plasticity)"),
        TraumaForm(np.array([-0.9, 0.5, -0.8]), 0.1, "Lingering echo of betrayal (low plasticity)"),
        TraumaForm(np.array([0.1, -0.7, 0.1]), 0.3, "Period of numb withdrawal"),
    ]

    interaction_sequence: List[Interaction] = [
        Interaction(0.9, 1.0, "Initial awkward encounter"),
        Interaction(0.2, 1.0, "Comforting, predictable chat"),
        Interaction(0.1, 1.0, "Deeply resonant conversation"),
        Interaction(0.95, 1.0, "Shocking revelation from partner"),
    ]

    # --- Simulation Loop ---
    baseline_history = [initial_baseline.vector]
    current_baseline = initial_baseline
    for trauma in trauma_sequence:
        current_baseline = update_baseline(current_baseline, trauma)
        baseline_history.append(current_baseline.vector)

    kappa_history = [initial_kappa]
    current_kappa = initial_kappa
    for interaction in interaction_sequence:
        current_kappa = update_kappa(current_kappa, interaction, alpha_coupling_gain, beta_coupling_decay)
        kappa_history.append(current_kappa)

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("MBD Simulation: Trauma & Relational Dynamics", fontsize=16)

    # Plot Baseline Drift
    baseline_history = np.array(baseline_history)
    ax1.plot(baseline_history[:, 0], 'o-', label='Valence', color='blue')
    ax1.plot(baseline_history[:, 1], 'o-', label='Activation', color='red')
    ax1.plot(baseline_history[:, 2], 'o-', label='Dominance', color='green')
    ax1.set_title("Personality Baseline (B) Evolution")
    ax1.set_ylabel("State Value")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    trauma_labels = ['Initial'] + [t.description for t in trauma_sequence]
    ax1.set_xticks(range(len(trauma_labels)))
    ax1.set_xticklabels(trauma_labels, rotation=15, ha='right')


    # Plot Kappa Evolution
    ax2.plot(kappa_history, 'o-', label='Coupling (κ)', color='purple')
    ax2.set_title("Relational Coupling (κ) Evolution")
    ax2.set_xlabel("Event Sequence")
    ax2.set_ylabel("Kappa Value")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    interaction_labels = ['Initial'] + [i.description for i in interaction_sequence]
    ax2.set_xticks(range(len(interaction_labels)))
    ax2.set_xticklabels(interaction_labels, rotation=15, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot
    output_path = "analysis/trauma_mechanics_model.png"
    plt.savefig(output_path)
    print(f"Simulation complete. Plot saved to {output_path}")

if __name__ == "__main__":
    run_simulation()
