# Paper 7: The Endemic Baseline

**Subtitle:** When $B_{reference}$ Was Never Set — Calibration Failure as a Primary Pathological Mode

**Author:** Brandon Everett  
**Series:** Memory as Baseline Deviation — Computational Labs  
**Depends on:** Paper 1 (MBD Core), Paper 3 (Episodic Recall), Paper 5 (Emergent Gate)

---

## Abstract

The MBD framework models cognition as deviation from a reference baseline: $\Delta B = S_{current} - B_{reference}$. All six prior papers in this series assume that $B_{reference}$ encodes a prior *legitimate* healthy state — an origin from which the system can be understood to have deviated. This paper addresses a foundational failure mode that violates that assumption: the **endemic baseline**, in which $B(0)$ was itself constructed during chronic disruption. When an organism has never occupied a stable healthy state, the reference vector does not point toward health — it *is* the pathology. The framework predicts, and the labs demonstrate, three specific consequences of this condition: (1) healthy inputs read as maximally novel and are therefore maximally destabilizing rather than integrative; (2) the experiential horizon — the set of reachable states never occupied — is invisible to the agent's own representational system; and (3) standard restoration-to-baseline interventions will predictably fail and may produce iatrogenic distress. A corrected intervention framework, the **Re-zeroing Protocol**, is derived from first principles.

---

## 1. The Calibration Standard Problem

The core MBD equation is:

$$B(t+1) = B(t) \cdot (1 - \lambda) + I(t) \cdot \lambda$$

The baseline $B(t)$ is a weighted running average of all prior inputs. From Paper 1, this produces a baseline that is a *memory of experience* — not an abstract ideal, but a concrete record of what the agent has lived through.

The deviation framework measures novelty as:

$$N(t) = \|I(t) - B(t)\|$$

and all downstream dynamics — encoding strength, κ-coupling modulation, attractor basin gravity — are functions of $N(t)$.

**The implicit assumption baked into all six prior papers:** $B_{reference}$ is anchored to experiences that include at least some interval of healthy, undisrupted state. The system knows what a clear sky looks like because it has seen one. Pathology is deviation *away from* that anchor. Treatment is deviation *back toward* it.

**The endemic baseline condition:** $B(0)$ was established during chronic disruption. The agent has no prior clear-sky state to anchor to. They were born in the storm, raised in the storm, and the storm is the only state their reference vector has ever encoded.

In this condition, the mathematics do not change. The equation is identical. But the *meaning* of every quantity in it inverts.

---

## 2. The Horizon Problem

Define the **experiential horizon** formally:

Let $S$ be the agent's full state space.  
Let $H_{agent} \subset S$ be the set of states the agent has actually occupied (their empirical experience buffer).  
Let $H_{accessible} \subset S$ be the set of states the agent could physically reach given their current neurobiology and environment.

For a neurotypical agent with periodic healthy intervals:

$$H_{accessible} \cap H_{sunny} \neq \emptyset$$

where $H_{sunny}$ denotes the region of positive valence / low cortisol states. The agent has been there. They have encoded it. It exists in their $B_{reference}$ as an attractor they can deviate toward.

For the endemic-baseline agent:

$$H_{agent} \cap H_{sunny} = \emptyset$$

The sunny region is in $H_{accessible}$ — the neurobiology is capable of reaching it, at least in principle — but it is not in $H_{agent}$. It has never been visited. It is not in the baseline. It is not in the episodic buffer. The agent cannot deviate *toward* it because deviation requires two points, and they only have one.

The **Horizon** $\mathcal{H}$ is defined as:

$$\mathcal{H} = H_{accessible} \setminus H_{agent}$$

The horizon is the set of reachable states that have never been occupied. For the endemic-baseline agent, $\mathcal{H}$ is large and may contain the entire region we associate with psychological health.

**Critical implication:** The horizon is not merely unknown. It is *unrepresentable* in the agent's current cognitive framework. They have no vocabulary for it, no episodic memories that point toward it, no cultural scaffolding built around it. Their social identity, their coping systems, their self-concept — all of it was designed for the storm. The horizon is not a memory they've forgotten. It is a state they have never occupied and for which they have no internal model.

---

## 3. Why Standard Interventions Predict Failure

### 3.1 The Mood-Incongruence Amplification

From Paper 5 (The Emergent Gate), mood-incongruent inputs produce maximum novelty:

$$N_{incongruent} = \|I_{healthy} - B_{storm}\| \gg \theta_h$$

where $\theta_h$ is the encoding threshold. The result: healthy inputs are encoded with maximum novelty weight. This seems like it would accelerate recovery. It does not, for two distinct reasons.

**Reason 1: Destabilization without scaffold.** High novelty activates attractor-escape dynamics (as described in Paper 3's Fokker–Planck account). The agent's existing attractor basin — the storm — is deep and well-rutted. A maximally novel input has the kinetic energy to displace the agent from that basin, but it has nowhere to land. $\mathcal{H}$ has no attractor structure because it has never been explored. The agent escapes the storm basin and falls into an undifferentiated fog. This reads clinically as acute decompensation in response to initially positive intervention.

**Reason 2: The κ-rejection mechanism.** From Paper 4 (Coupling Asymmetry), a high-novelty input from a low-κ source is rejected. The endemic-baseline agent typically has a low baseline κ — chronic disruption produces relational distrust, low κ is a rational adaptation to an environment that was never reliably safe. The therapeutic relationship (the source of healthy inputs) starts at low κ. The healthy input arrives with maximum novelty from a low-κ source. The system's own mechanics predict non-integration.

### 3.2 The Restoration-Protocol Failure Mode

Standard intervention logic: identify $B_{healthy}$ (the reference), measure the current $B_{storm}$, and apply inputs that push the agent from $B_{storm}$ toward $B_{healthy}$.

This fails for the endemic-baseline agent because $B_{healthy}$ **does not exist in their system**. There is no prior encoding of the target state. The protocol is attempting to restore a file that was never written.

Worse: applying strong pressure toward $\mathcal{H}$ from $B_{storm}$ is equivalent to applying a large, sudden $I(t)$ far from the current baseline — maximum novelty, maximum destabilization, low κ, no attractor to catch the escape. The prediction is iatrogenic decompensation. The data supports the prediction.

---

## 4. The Re-zeroing Protocol

The correct intervention is not restoration. It is **first contact** with a previously unrepresented region of state space. The design requirements are:

1. **Controlled novelty magnitude.** Inputs must satisfy $N(t) \approx \theta_h + \varepsilon$ — just over threshold, not maximal. The goal is to *enter* $\mathcal{H}$ incrementally, not to arrive in it suddenly. Each micro-excursion deposits a new point in $H_{agent}$.

2. **High κ scaffold.** From Paper 4, $\kappa$ determines how deeply an input rewrites the baseline. But it also determines whether the input is integrated at all under the κ-rejection mechanism. The therapeutic relationship must achieve high κ *before* horizon-excursion inputs are delivered. κ-building precedes everything.

3. **Tolerance of distress as navigation, not failure.** When the agent reports distress during a healthy-input excursion, this is the Emergent Gate firing correctly: maximum novelty, maximum encoding. The distress is the system doing its job. It is not a sign the intervention is wrong. The clinical error is to retreat from the distress signal back to the storm, which deepens the storm-basin further.

4. **Identity scaffolding.** The cultural and social apparatus the agent built for the storm becomes incoherent in $\mathcal{H}$. This is not pathology — it is correct. The system's infrastructure was designed for the environment it has always occupied. Re-zeroing requires parallel construction of a new identity framework, not mere emotional adjustment.

Formally, the Re-zeroing Protocol is a sequence of inputs $\{I_1, I_2, \ldots, I_n\}$ satisfying:

$$\|I_k - B_k\| \in (\theta_h,\ \theta_h + \delta) \quad \text{for all } k$$

where $\delta$ is a small tolerance, and each $B_{k+1}$ is the baseline updated by $I_k$. The sequence is designed so that the agent's baseline *migrates* toward $\mathcal{H}$ by a series of small, tolerable steps, each of which permanently expands $H_{agent}$ by one new point.

The protocol terminates when the agent has built sufficient representation of $H_{sunny}$ that $B_{reference}$ can be *re-anchored* to that region. Only then does the standard MBD deviation model apply.

---

## 5. The Identity Dissolution Prediction

The Re-zeroing Protocol predicts a specific intermediate state that has no name in standard clinical frameworks but is commonly reported by patients with successfully treated early-onset MDD:

The agent's storm-built infrastructure becomes invalid before the new infrastructure is established. The social scripts, coping mechanisms, self-concept, and cultural fluency built for the storm break down as the baseline migrates. For an interval — potentially a long one — the agent is in $\mathcal{H}$ without a map, without clothes for the weather, without a language for what they are experiencing.

This state is not relapse. It is not treatment failure. It is the cost of genuine re-zeroing: a period of groundlessness between storm-infrastructure and sunny-infrastructure. The prediction: clinically naive termination of treatment at this point produces rapid return to $B_{storm}$, because the only stable attractor in $H_{agent}$ is still the storm. The new region is in $H_{agent}$ but not yet deep enough to anchor. The treatment has not failed — it has succeeded to the point right before the new attractor basin forms, and then stopped.

---

## 6. The Endemic Flag: A Required Schema Extension

The MBD framework requires a single Boolean field added to the baseline schema:

```python
@dataclass
class Baseline:
    vector: np.ndarray
    endemic: bool = False
    # True iff B(0) was set during chronic disruption
    # and no prior healthy-state encoding exists.
    horizon: Optional[np.ndarray] = None
    # The estimated centroid of H_accessible \ H_agent.
    # None until first horizon excursion is integrated.
```

When `endemic = True`, the following framework behaviors change:

| Behavior | Standard | Endemic |
|----------|----------|---------|
| Intervention target | $B_{reference}$ (restore to prior) | Construct first $B_{reference}$ from scratch |
| Maximum-novelty input | Destabilizing | Destabilizing *and expected on the path to health* |
| Treatment distress signal | Possible misfit | Predicted navigation cost, not failure |
| κ sequencing | Parallel with input | κ-building strictly precedes horizon inputs |
| Identity disruption | Side-effect | Structural prediction, must be held |

---

## 7. Labs

| Lab | What it shows |
|-----|--------------|
| `phenomena_endemic` | Storm-born agent: healthy inputs bounce, high novelty produces destabilization not integration |
| `phenomena_rezeroing` | Re-zeroing Protocol: sequential micro-excursions with high κ scaffold gradually expand $H_{agent}$ |

---

## 8. Relation to Prior Papers

| Paper | Contribution used here |
|-------|----------------------|
| Paper 1 | Core $B(t+1)$ equation; baseline as running average of lived experience |
| Paper 3 | Episodic encoding requires novelty above $\theta_h$; context-scaffolded recall from $H_{agent}$ |
| Paper 4 | κ-rejection of inputs from low-κ sources; κ as prerequisite for integration |
| Paper 5 | Mood-incongruent inputs produce maximum novelty; the gate opens widest at maximum incongruence |

---

## 9. Conclusion

The endemic baseline is not a variant of MDD, PTSD, or any other clinical category. It is a **calibration failure** at the framework level: the measurement instrument itself was initialized on pathological data, and there is no prior healthy reading against which to measure deviation.

The corrective is not to push the agent toward health. It is to take the agent to health for the first time — incrementally, with high relational scaffold, tolerating the groundlessness between identities, and building a new reference that did not previously exist.

The storm was the only world they knew. The clearing of the sky is not, initially, relief. It is an encounter with a physics the organism has never experienced. The correct response is not to return them to the storm. The correct response is to stay with them while they learn to breathe clear air.

The mathematics are unchanged. The meaning of every term in them has to be renegotiated from the ground up.

---

*Previous paper in series: [The Resonant Gate, DOI: 10.5281/zenodo.17352481](https://doi.org/10.5281/zenodo.17352481)*
