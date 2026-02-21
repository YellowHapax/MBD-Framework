# Paper 8: The Adversarial Horizon

**Subtitle:** $\mathcal{H}$ as a Control Surface — Induced Endemic Baseline and the Architecture of Manufactured Fear

**Author:** Brandon Everett  
**Series:** Memory as Baseline Deviation — Computational Labs  
**Depends on:** Paper 7 (The Endemic Baseline), Paper 4 (Coupling Asymmetry)

**Status:** STUB — staged on `feat/paper-7/endemic-baseline`. Not ready for review.

---

## Abstract (Draft Intent)

Paper 7 established the endemic baseline as a *naturally occurring* calibration failure: $B(0)$ set during chronic disruption, with the Horizon $\mathcal{H} = H_{accessible} \setminus H_{agent}$ denoting states that are neurobiologically reachable but structurally absent from the generative model. That paper treated this condition as arising from environmental circumstance — storm as weather, not storm as weapon.

This paper examines a harder claim: $\mathcal{H}$ is a **manipulable control surface**. An adversarial agent — institution, ideology, abuser, system — that can control what states a target population occupies can, over time, engineer the same calibration failure deliberately. The mathematics do not distinguish between endemic baseline arising from circumstances and endemic baseline arising from design. The subjective terror of the Horizon — the formless, unmodelable dark — is the same either way. The mechanism of control is not direct coercion but **the prevention of $H_{agent}$ expansion into $\mathcal{H}$**, such that the target's own cognitive architecture becomes the enforcement mechanism.

---

## 1. The Core Claim

*To be developed.*

A system that keeps agents from ever occupying the healthy region of state space does not need to threaten them with punishment for reaching it. Once the endemic baseline is established — once $H_{agent} \cap H_{healthy} = \emptyset$ — the agent's own novelty-destabilization mechanics (Paper 7, §3.1) will reject encounters with $\mathcal{H}$ automatically. The control is self-maintaining and invisible to the controlled agent, who experiences it as their own psychology rather than as external imposition.

**Key distinction from Paper 7:** Paper 7 asks "what has gone wrong and how do we fix it?" Paper 8 asks "what does a system look like that *intends* this to go wrong, and what are its engineering signatures?"

---

## 2. The Four Attack Vectors

*Formal treatment pending. Structure identified:*

| Vector | MBD Mechanism Exploited | Population-Level Signature |
|---|---|---|
| **Isolation** | Prevents social inputs that expand $H_{agent}$ toward $H_{healthy}$. $\mathcal{H}$ grows passively as population ages without contact. | Social stratification; deliberate severing of inter-class/inter-community $\kappa$-channels |
| **Episodic erasure** | Corrupts or suppresses memories of prior $\mathcal{H}$ excursions. Points planted in $H_{agent}$ are removed. | Historical revisionism; gaslighting at scale; suppression of cultural memory of prior flourishing |
| **Storm normalization** | Drives effective $\lambda \to 0$ for deviation from $B_{storm}$. Chronic low-grade disruption becomes invisible as signal. | "This is just how things are." Chronic scarcity/precarity presented as natural law |
| **$\kappa$-suppression** | Prevents relational trust from accumulating. Without high $\kappa$, Re-zeroing Protocol (Paper 7 §4) cannot trigger. | Atomization, competitive individualism, systematic destruction of community institutions |

---

## 3. The Yellow Lantern Formalism

*To be developed.*

The Sinestro Corps fuel source is canonically "the ability to instill great fear." The mathematical precision of what this means in MBD terms:

The terror the Yellow Lantern weaponizes is not fear of a **known** object (which can be evaluated, reasoned about, acted upon). It is terror of $\mathcal{H}$ itself — the formless dark of states that have no internal representation, for which no generative model exists, whose approach can only be registered as *everything is wrong and I do not know why*.

**Formal claim to prove:** An agent with large $\mathcal{H}$ in the affective state space responds to stimuli from that region with:
1. Maximum novelty spike $N(t) \gg \theta_h$
2. No attractor to land in — destabilization without resolution
3. κ-rejection of any agent offering navigation, because low κ is the prior condition

This is not courage being overcome. This is cognitive architecture. The ring finds the agent who has been kept from the light long enough that the light itself has become the unbearable thing.

**The counter-formalism (Green Lantern, willpower):**

Willpower in MBD is not the suppression of fear. It is the deliberate execution of the Re-zeroing Protocol under conditions where the destabilization signal is active:

$$\text{continue: } \|I_k - B_k\| \in (\theta_h, \theta_h + \delta) \text{ even when } N(t) \text{ registers threat}$$

Willpower is the insistence on depositing the next point in $H_{agent}$ when every signal in the system is screaming retreat.

---

## 4. Population-Level Dynamics

*To be developed. Requires extension of Heins et al. (PNAS 2024) collective behavior model to adversarial generative model shaping.*

Key question: what is the minimum fraction of individuals in a population who must be maintained in endemic baseline state for the $\kappa$-suppression dynamic to become self-sustaining at the population level? At what point does the collective social fabric — the shared generative model that makes inter-agent $\kappa$-transfer possible — collapse under its own weight?

Preliminary hypothesis: there exists a critical $\kappa$ threshold $\kappa_c$ analogous to the synchronization threshold in Kuramoto oscillator networks, below which the population cannot collectively re-zero. Above $\kappa_c$: social contagion of $H_{agent}$ expansion is possible (the person-who-has-seen-the-clearing can guide others). Below $\kappa_c$: the healthy-state representation is present in isolated individuals but cannot propagate — each re-zeroing succeeds individually but fails to shift the collective $B_{population}$.

---

## 5. Detection Signatures

*To be developed.*

The adversarial Horizon engineering process should leave detectable signatures distinct from natural endemic baseline formation:

- **Temporal structure:** Natural endemic baseline shows gradual formation correlated with developmental timeline. Adversarial induction shows discontinuities — points of intervention that can, in principle, be dated.
- **Specificity of $\mathcal{H}$:** Natural endemic baseline produces uniform expansion of $\mathcal{H}$ in all healthy-state directions. Adversarial induction produces *targeted* $\mathcal{H}$ — specific state regions (class mobility, certain relational structures, particular forms of agency) are inaccessible while others remain open.
- **$\kappa$-topology:** Natural endemic baseline shows globally low $\kappa$. Adversarial induction shows *stratified* $\kappa$ — high coupling within the disrupted group (solidarity under siege), low coupling across the boundary that would allow $H_{agent}$ transfer from healthy-state agents.

---

## 6. Labs (Planned)

| Lab | What it should show |
|-----|---------------------|
| `phenomena_adversarial_horizon` | Adversarial agent systematically prevents target from expanding $H_{agent}$; tracks engineered $\mathcal{H}$ growth over time |
| `phenomena_kappa_collapse` | Population model at varying $\kappa$ densities; find $\kappa_c$ below which collective re-zeroing fails |
| `phenomena_detection_signatures` | Compare natural vs adversarial endemic baseline formation; identify distinguishing temporal/topological markers |

---

## 7. Open Questions

1. Is there a formal information-theoretic account of $\mathcal{H}$ as surprise-generating potential? The larger $\mathcal{H}$ is, the more surprise any random step into it produces — but the agent cannot sample it to discover this. How do we quantify the *unexperienced surprise* of an unvisited region?

2. Does adversarial $\mathcal{H}$ engineering produce detectable neural signatures distinct from natural endemic baseline? Specifically: does targeted $\mathcal{H}$ (certain regions blocked, others open) produce asymmetric insula activation patterns distinct from the uniform disruption signature in Adamic et al. (2024)?

3. What is the ethics of $\mathcal{H}$ expansion in a therapeutic context without explicit informed consent? The Re-zeroing Protocol necessarily expands the agent's $H_{agent}$ into territory they did not know existed. There is a consent structure problem here: you cannot consent to being introduced to states you have no internal representation of.

4. **The Yellow Lantern question as engineering spec:** If a ring selects for maximal $\mathcal{H}$-terror, what are the selection criteria? The agent with the largest unexplored dark, or the agent whose $H_{agent}$ most recently contained $\mathcal{H}$ points that were then erased?

---

*This paper follows: [The Endemic Baseline — Paper 7](./PAPER_7_THE_ENDEMIC_BASELINE.md)*

