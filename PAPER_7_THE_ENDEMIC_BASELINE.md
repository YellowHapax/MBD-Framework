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

## 10. Formal Correspondence with Friston's Active Inference / Free Energy Principle

The Free Energy Principle (FEP) and its active inference framework (Friston, 2010–2025) describe cognition as variational Bayesian inference: all neural dynamics minimize a bound on surprise, and action is the mechanism by which organisms bring sensory observations into alignment with their prior preferences. MBD and FEP share deep structural roots. This section establishes the formal correspondence, identifies the novel contributions that each framework brings to the other, and anchors the correspondence to the empirical literature.

### 10.1 Variable Correspondence Table

| MBD Variable | AIF / FEP Variable | Formal Relationship | Interpretation |
|---|---|---|---|
| $B(t)$ — baseline state vector | $\mu$ — posterior mean (variational belief) | **Structural identity.** Both represent the system's current best estimate of its internal/environmental state. Both resist deviation proportionally to their encoded confidence. | B(t) *is* the running-average posterior. The weighted update rule $B(t+1) = B(t)(1-\lambda) + I(t)\lambda$ is a first-order variational update. |
| $I(t) - B(t)$ — deviation signal | $\varepsilon = o - g(\mu)$ — prediction error | **Structural identity.** Both measure *observed minus expected*. The magnitude of this quantity drives all downstream encoding, coupling, and gate dynamics in MBD; it is the literal update signal in predictive coding. | The dysgranular mid-insula is the proposed neural substrate for this computation (Seth & Friston, 2016; Adamic et al., 2024). |
| $\lambda$ — update rate | $\pi_s / (\pi_s + \pi_p)$ — precision-weighted update | **Parametric correspondence.** Precision $\pi = 1/\sigma^2$ (inverse variance). The Bayesian update weight is $\pi_{sensory}/(\pi_{sensory} + \pi_{prior})$. Setting $\lambda \equiv \pi_s/(\pi_s + \pi_p)$ recovers the MBD update rule. | Low $\lambda$ in MBD = high prior precision relative to sensory precision in FEP = the prior resists updating = the baseline resists deviation. Depression as low $\lambda$ (Barrett et al., 2016; Smith et al., 2020). |
| $\kappa$ — social coupling coefficient | *(no direct AIF equivalent)* | **MBD novel contribution.** AIF models individual agents; collective behavior studies (Heins, Millidge, da Costa, Mann, Friston & Couzin, *PNAS* 2024) show that social forces *emerge* from shared generative models under surprise minimization, but no single parameter governs inter-agent update rates. $\kappa$ is the explicit handle on this emergent dynamic. | $\kappa$ makes between-agent recursive baseline updating tractable to measure, vary, and intervene on — a parametric gap in the FEP architecture. |
| $D$ — deontological demand | $\tilde{P}(o)$ — prior preferences over outcomes | **Approximate correspondence.** FEP agents have preferred outcomes encoded as prior beliefs; the cost of occupying non-preferred states enters directly into free energy. $D$ generalizes this to the *environment* as an agent: built spaces (Sanctuaries, Arenas, Markets, Cesspits) generate structural demand fields that deform the agent's effective $\tilde{P}(o)$. | Context-sensitive reward in Barrett, Quigley & Hamilton (2016): allostatic context shapes what constitutes a "preferred" interoceptive state. |
| Endemic $B(0)$ | Pathological attractor formed by early adversity | **Bidirectional mapping.** FEP literature models resistant priors as deep attractors in the free energy landscape — states that persist because the gradient of surprise does not point away from them for the agent's generative model. Endemic $B(0)$ is exactly this: a deep attractor whose walls are high because *all evidence ever received was generated within it*. | Barrett et al. 2016 (allostasis/depression); Smith et al. 2020 (computational model of treatment response); early-adversity prior-formation literature. |
| $\mathcal{H} = H_{accessible} \setminus H_{agent}$ — Horizon | *(no direct AIF equivalent)* | **MBD novel contribution.** AIF accommodates uncertainty about *known* states (imprecise beliefs about states the generative model contains). $\mathcal{H}$ formalizes a structurally different epistemic barrier: states that are absent from the generative model not because of uncertainty, but because they have never been occupied and therefore were never encoded into the prior. | The agranular anterior insula generates predictive forward models for anticipated interoceptive states (Adamic et al., 2024). ADE individuals show *pruned agranular activation* during anticipatory uncertainty — consistent with the prediction that $\mathcal{H}$ contains states for which no forward model can be generated. |

### 10.2 What MBD Adds to the AIF Conversation

FEP is a single-agent framework with no native parametric account of inter-agent baseline modification. The social force literature (Heins et al., *PNAS* 2024) shows that collective phenomena emerge from individual surprise minimization, but the coupling is an emergent property — not a measurable, manipulable parameter.

MBD contributes three things FEP currently lacks:

1. **$\kappa$ as an explicit mediator of social baseline coupling.** The therapeutic relationship, the culture, the peer group — these modify $B(t)$ through the same mathematical mechanism as sensory input, but at a different rate and subject to a rejection threshold. This is not epiphenomenal; it is the primary mechanism by which social scaffolding is or is not integrated. FEP has no variable that captures this.

2. **$\mathcal{H}$ as a structural concept for prior non-occupancy.** FEP routinely models high uncertainty over states. It does not model the case where states are absent from the generative model because they have never been visited — the phenomenological gap of Paper 7. This is a genuinely novel structure.

3. **Deontological demand $D$ as an environmental agency.** FEP places all agency in the agent. $D$ places structural preference-shaping in the physical and social environment — the Quadrafoil is a field of attractors that systematically deform agent priors over time. This is a non-trivial architectural distinction.

### 10.3 What FEP Contributes to the MBD Conversation

FEP provides three things MBD currently handles loosely:

1. **Hierarchical precision.** MBD uses a single $\lambda$. FEP has precision at every level of a hierarchical generative model, with context-dependent modulation. A full account of the endemic baseline probably requires precision to vary across representational levels — the prior at the self-model level may be more resistant than the prior at the sensory level.

2. **Neural substrate predictions.** The insula hierarchy (granular → dysgranular → agranular, posterior → anterior) provides specific, testable predictions about *where* in the brain the $I(t) - B(t)$ computation occurs, where the precision modulation occurs, and what breaks in psychiatric disorders. Adamic et al. (2024) provides fMRI data directly testable against endemic baseline predictions.

3. **Action as free energy minimization.** MBD's Re-zeroing Protocol describes a sequence of inputs. FEP provides the deeper account of how active inference *selects* those inputs — the agent acts to bring future sensory states into alignment with prior preferences. For re-zeroing, the protocol must first establish preferred states in $\mathcal{H}$ before action selection can target them.

### 10.4 The Empirical Bridge: Adamic et al. (2024), eLife

The most direct neural evidence for the MBD endemic baseline prediction comes from *Hemispheric divergence of interoceptive processing across psychiatric disorders* (Adamic, Teed, Avery, de la Cruz & Khalsa, *eLife* 13:RP92820, 2024). Key findings:

**Population:** 46 individuals with anxiety, depression, and/or eating disorders (ADE; 72% lifetime MDD) and 46 matched healthy comparisons (HC), measured during pharmacological interoceptive perturbation (isoproterenol IV infusion) and voluntary interoceptive attention tasks.

**Finding 1 — Blunted anticipatory agranular activation (MBD: λ failure under uncertainty):**
> "The reduced spatial extent of this activation in the ADE versus HC group could plausibly stem from an overreliance on top-down predictions and perhaps a pruning of spatial activation in the associated regions... Such 'efficient' neural exchanges might even result from a chronic inability to adjust confidence in bodily input in relation to its ambiguity."

In MBD terms: when faced with a genuinely uncertain situation (isoproterenol anticipated but not yet arrived), HC individuals expand agranular insula recruitment — they *generate forward models* for anticipated states. ADE individuals do not. This is low $\lambda$ at the predictive level: the system lacks the machinery to update B(t) in response to anticipated deviation, not just actual deviation.

For the endemic baseline specifically: if $\mathcal{H}$ contains states that have never been anticipated because they have never been occupied, then *there is no forward model to retrieve*. The agranular pruning in ADE is, under this reading, partly a function of never having built the anticipatory infrastructure for healthy states.

**Finding 2 — Altered insula-frontal connectivity correlated with depression severity:**
> "The magnitude of functional connectivity change between these two areas was associated with trait anxiety (...) and trait depression (...) but not the magnitude of ISO-induced heart rate."

The neural signature tracks the phenomenological/trait dimension, not the peripheral signal strength. The disruption is at the *interpretive layer* — the B(t) level — not at the sensory input level. This replicates the core MBD claim: pathology is not in the signal; it is in the baseline that interprets the signal.

**Finding 3 — Dysgranular mid-insula as locus of interoceptive disruption (previously: Nord et al., 2021):**
> "These hemispheric asymmetries, and the disparate spatial patterns within the left dysgranular insula, support the conceptualization of this subregion as a 'locus of disruption' for interoceptive symptomatology in these disorders."

The dysgranular mid-insula is proposed as the site where prediction errors $\varepsilon = I(t) - B(t)$ are computed. Its disruption in ADE is the neural expression of a corrupted deviation signal — corrupted because $B(t)$ was set under pathological conditions.

### 10.5 The Multi-Agent Bridge: Heins et al. (PNAS 2024)

*Collective behavior from surprise minimization* (Heins, Millidge, da Costa, Mann, Friston & Couzin, *PNAS* 121(17), 2024) demonstrates that cohesion, milling, and directed group motion emerge when individual agents minimize their variational free energy with respect to a shared generative model. Social forces — attraction, repulsion, alignment — recover naturally as prediction-error suppression at the inter-agent level.

**MBD translation:**  
$\kappa$ is the parametric handle on this emergent dynamic. When the Architect builds a therapy room with high $\kappa$ between therapist and client, they are tuning the *effective social force* that governs whether I(t) from the therapist's world can update B(t) in the client's generative model. The Heins et al. mathematics confirm: this is not metaphor. Social force *is* inter-agent prediction error suppression. $\kappa$ is the gain on that channel. The Re-zeroing Protocol's mandate to build $\kappa$ *before* horizon inputs is the practical expression of: you cannot minimize prediction error with respect to another agent's generative model until you share enough of the model to make the errors legible.

### 10.6 Summary: Correspondence and Divergence

```
┌──────────────────────────────────────────────────────────────────────────┐
│              MBD ↔ FEP: Structure of Correspondence                      │
├───────────────┬────────────────────────────────┬───────────────────────── ┤
│ SHARED CORE   │ Both frameworks                │ Identical math layer    │
│               │ B(t) ≡ μ                       │                         │
│               │ I(t)−B(t) ≡ ε = o−g(μ)        │                         │
│               │ λ ≡ πs/(πs+πp)                │                         │
├───────────────┼────────────────────────────────┼─────────────────────────┤
│ MBD NOVEL     │ κ — explicit inter-agent        │ FEP: emergent, no param │
│               │ coupling coefficient           │                         │
│               │ H — Horizon: non-occupancy     │ FEP: models imprecision,│
│               │ as epistemic barrier           │ not structural absence  │
│               │ D — environmental deontological│ FEP: agent-level prefs  │
│               │ demand field                   │ only                    │
├───────────────┼────────────────────────────────┼─────────────────────────┤
│ FEP NOVEL     │ Hierarchical precision (multi- │ MBD: single λ layer     │
│               │ level λ)                       │                         │
│               │ Neural substrate predictions   │ MBD: substrate-agnostic │
│               │ (insula laminar hierarchy)     │                         │
│               │ Action as free energy          │ MBD: inputs specified;  │
│               │ minimization                   │ selection mechanism open│
└───────────────┴────────────────────────────────┴─────────────────────────┘
```

The endemic baseline is the region of this correspondence space where MBD and FEP most directly converge: a pathological attractor formed by early adversity, where the prior (B(0)) correctly describes the only world ever occupied, and where the Horizon ($\mathcal{H}$) identifies the states for which neither a generative model nor prior preferences yet exist. The endemic flag is the formal indicator that both frameworks' standard intervention assumptions fail simultaneously.

---

### Cited works (this section)

- Adamic EM, Teed AR, Avery J, de la Cruz F, Khalsa SS (2024). Hemispheric divergence of interoceptive processing across psychiatric disorders. *eLife* 13:RP92820. [https://doi.org/10.7554/eLife.92820](https://doi.org/10.7554/eLife.92820)
- Barrett LF, Quigley KS, Hamilton P (2016). An active inference theory of allostasis and interoception in depression. *Phil Trans R Soc B* 371:20160011. [https://doi.org/10.1098/rstb.2016.0011](https://doi.org/10.1098/rstb.2016.0011)
- Heins C, Millidge B, da Costa L, Mann R, Friston K, Couzin I (2024). Collective behavior from surprise minimization. *Proc Natl Acad Sci* 121(17):e2320239121. [https://doi.org/10.1073/pnas.2320239121](https://doi.org/10.1073/pnas.2320239121)
- Nord CL, Lawson RP, Dalgleish T (2021). Disrupted dorsal mid-insula activation during interoception across psychiatric disorders. *Am J Psychiatry* 178:761–770.
- Paulus MP, Feinstein JS, Khalsa SS (2019). An active inference approach to interoceptive psychopathology. *Annu Rev Clin Psychol* 15:97–122. [https://doi.org/10.1146/annurev-clinpsy-050718-095617](https://doi.org/10.1146/annurev-clinpsy-050718-095617)
- Seth AK, Friston KJ (2016). Active interoceptive inference and the emotional brain. *Phil Trans R Soc B* 371:20160007. [https://doi.org/10.1098/rstb.2016.0007](https://doi.org/10.1098/rstb.2016.0007)
- Smith R et al. (2020). Computational models of interoception and body regulation. *Trends Neurosci* 44:63–76.

---

*Previous paper in series: [The Resonant Gate, DOI: 10.5281/zenodo.17352481](https://doi.org/10.5281/zenodo.17352481)*
