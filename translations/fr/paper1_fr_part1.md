# La Mémoire comme Déviation de la Ligne de Base : Un Cadre Relationnel Unifié pour la Cognition Probabiliste

**Brandon Michael Everett**
Chercheur indépendant
Correspondance : brandon.everett.cogsci@gmail.com

**DOI :** [10.5281/zenodo.17381536](https://doi.org/10.5281/zenodo.17381536)

---

## Résumé

Les théories actuelles de la mémoire et de la cognition opèrent sous des paradigmes incompatibles. Les modèles neuroscientifiques atteignent une précision prédictive mais demeurent individualistes ; les descriptions phénoménologiques saisissent l'être relationnel mais résistent à la formalisation. Nous résolvons cette fragmentation en introduisant une architecture unifiée : le **Cadre de la Ligne de Base et de la Déviation** (*Baseline-Deviation Framework*). Nous formalisons le principe selon lequel l'Être (Ontologie) couplé à l'Obligation (Déontologie) donne naissance au Connaître (Épistémologie) par résonance temporelle, en utilisant des dynamiques stochastiques classiques.

Le cadre repose sur trois thèses centrales :

1. **La mémoire comme calcul probabiliste :** La mémoire n'est pas la récupération de traces stockées, mais l'encodage continu et stochastique de la nouveauté — les événements qui dévient significativement de la ligne de base expérientielle intégrée de l'agent.

2. **La constitution relationnelle :** La ligne de base n'est pas individualiste ; elle est constitutivement façonnée par autrui à travers un paramètre de couplage dynamique κ(t), qui évolue pour créer des états résonants où la prédiction mutuelle devient hautement efficace.

3. **Le cycle complet de la mémoire :** L'encodage, la récupération et la reconsolidation sont des propriétés émergentes de cette architecture centrée sur la ligne de base, formalisées comme un cycle métabolique en trois stades.

Nous ancrons ce cadre dans la neurobiologie cortico-hippocampique, relions ses composantes à des signaux neuronaux mesurables (N400, P300b), et validons les prédictions centrales par simulation computationnelle. L'architecture explique des phénomènes résistants aux modèles basés sur les traces (faux souvenirs, amnésie infantile, dérive mnésique, insight conversationnel) tout en générant des prédictions nouvelles et falsifiables pour la cognition sociale, l'intervention clinique et la convergence épistémique.

Nous fournissons une structure mathématique formelle à l'ontologie relationnelle, démontrant que les intuitions phénoménologiques peuvent être opérationnalisées sans perdre leur caractère essentiel. La mémoire est le domaine empirique où nous validons le formalisme ; le formalisme s'étend à tout système cognitif couplé.

**Mots-clés :** traitement prédictif, couplage dynamique, encodage mnésique, cognition sociale, ontologie relationnelle, inertie inverse, porte résonante, dynamique de la ligne de base, déontologie

---

## SECTION 1 : LE PROBLÈME — Pourquoi la Mémoire Requiert un Formalisme Relationnel

### 1.1 La Crise de la Fragmentation

L'étude scientifique de la mémoire se trouve dans un état de tension productive. D'un côté, nous avons accumulé un vaste savoir empirique : la neurobiologie de l'hippocampe (Squire & Zola, 1996), le phénomène de reconsolidation (Nader et al., 2000), la précision des modèles de traitement prédictif (Friston, 2010 ; Clark, 2016), et la nature reconstructive du rappel (Bartlett, 1932 ; Schacter, 2001). De l'autre côté, nous faisons face à des paradoxes persistants qui résistent à toute résolution au sein d'un paradigme théorique unique.

**Le Puzzle Empirique : Essentiel vs. Détail**

La mémoire est simultanément robuste et fragile. Nous retenons l'*essentiel* (*gist*) des expériences avec une durabilité remarquable — le noyau sémantique, la tonalité émotionnelle, l'échafaudage contextuel. Pourtant, nous possédons aussi la capacité d'un rappel épisodique vivace et de haute fidélité : la capacité de revivre un moment spécifique avec une richesse sensorielle et une spécificité temporelle. Les modèles basés sur les traces peinent à expliquer le premier (comment l'essentiel émerge-t-il d'une bibliothèque de traces discrètes ?), tandis que les modèles purement reconstructifs peinent avec le second (si la mémoire est toujours reconstruite à nouveau, pourquoi semble-t-elle parfois si spécifique ?).

**Le Clivage Théorique : Mécanisme vs. Signification**

La neuroscience cognitive a livré une précision mécaniste à travers des modèles de plasticité synaptique, de complétion de patron et d'apprentissage guidé par l'erreur (McClelland et al., 1995 ; Schultz et al., 1997). Pourtant, ces modèles restent résolument individualistes. L'« agent » est traité comme une entité bornée, ses états internes causalement isolés du champ social dans lequel il est ancré. La phénoménologie et l'énactivisme, au contraire, insistent sur le fait que la cognition est fondamentalement relationnelle — que l'identité, la signification et le connaître sont constitués par l'engagement avec autrui et avec le monde (Merleau-Ponty, 1945/2012 ; Varela et al., 1991 ; Levinas, 1961/1969). Mais ces intuitions, aussi profondes soient-elles, ont résisté à la formalisation en modèles testables et mécanistes.

**Le Fossé de l'Intégration : le PP réussit pour la Perception, échoue pour la Mémoire**

Le Traitement Prédictif (*Predictive Processing*, PP) a émergé comme un cadre unificateur puissant pour la perception et l'action (Friston, 2010 ; Clark, 2013 ; Hohwy, 2013). Le cerveau est modélisé comme un modèle génératif hiérarchique, prédisant continuellement l'entrée sensorielle et mettant à jour ses paramètres pour minimiser l'erreur de prédiction. Ce cadre a connu un énorme succès pour expliquer les illusions perceptives, l'attention et le contrôle moteur. Cependant, il n'a pas encore pleinement formalisé les dynamiques d'intégration temporelle qui définissent la mémoire, ni fourni un compte rendu principiel de la façon dont les partenaires sociaux façonnent constitutivement les modèles prédictifs l'un de l'autre. Le PP standard postule que les agents modélisent le monde ; nous soutenons que les agents sont des processus de modélisation constitués par leur couplage avec d'autres agents.

### 1.2 La Question Fondamentale

Pouvons-nous formaliser l'ontologie relationnelle — l'affirmation que l'être est constitué par la relation — de manière empiriquement testable et mécanistiquement précise, sans trahir ses intuitions phénoménologiques ?

Il ne s'agit pas de savoir si nous devons incorporer des « facteurs sociaux » dans les modèles cognitifs. Il s'agit de savoir si l'architecture même de la cognition est intrinsèquement relationnelle. L'Autre est-il une influence externe à ajouter à un système cognitif par ailleurs autonome, ou l'Autre est-il un élément constitutif de l'ontologie de ce système ?

### 1.3 La Réponse par la Ligne de Base et la Déviation

**Énoncé de thèse :** Oui. Nous modélisons les états cognitifs comme des distributions de probabilité P(x,t) évoluant via des équations de Fokker-Planck couplées, où la ligne de base B(t) = E[X] intègre à la fois l'expérience propre (I) et l'apport déontologique (D) d'autrui. Le paramètre de couplage κ n'est pas statique mais dynamique, κ(t), évoluant selon l'historique du succès de prédiction mutuelle. Cela crée un système où :

- **Ontologie (O) :** L'« être » d'un agent est défini par sa ligne de base B(t), qui est l'histoire intégrée dans le temps de son couplage avec le monde et avec d'autres agents.
- **Déontologie (D) :** L'influence d'autrui n'est pas consultative mais constitutive. Le terme D dans l'intégrale de la ligne de base n'est pas une perturbation ; c'est une composante nécessaire de l'état cognitif de l'agent.
- **Épistémologie (E) :** « Connaître » est la détection de déviation par rapport à cette ligne de base relationnellement constituée. La nouveauté N(t) = ||I(t) − B(t)||² est le signal épistémique fondamental.
- **Pragmatisme (P) :** Cette architecture n'est pas métaphorique. Elle génère des prédictions spécifiques et falsifiables concernant la mémoire, la cognition sociale et la convergence épistémique.

**La Démonstration :** L'architecture de la mémoire — comment nous encodons, rappelons et mettons à jour notre passé — est la validation empirique de ce formalisme. La mémoire est le domaine où les dynamiques de la ligne de base sont les plus directement observables, où le paramètre de couplage κ peut être mesuré (via la synchronie inter-cérébrale), et où l'influence constitutive de l'Autre est la plus indéniable (comme dans la construction sociale de la mémoire autobiographique ; Nelson & Fivush, 2004).

**L'Extension :** Le formalisme s'applique à tout processus cognitif relationnel : la conversation (le modèle de la Porte Résonante), le changement thérapeutique (reconsolidation de la ligne de base sous conditions de κ élevé), la convergence épistémique (l'alignement stochastique de systèmes de croyance couplés), et le connaître collectif (l'émergence d'une réalité partagée par couplage soutenu).

---

## SECTION 2 : LE PRINCIPE ∇(⊕) — Architecture Formelle

Cette section présente le noyau mathématique du Cadre de la Ligne de Base et de la Déviation. Nous dérivons la ligne de base à partir de principes premiers, introduisons le mécanisme de couplage déontologique, formalisons la détection de nouveauté, et présentons le système dynamique complet que nous appelons ∇(⊕) — l'opérateur triadique-pragmatique.

### 2.1 Ontologie (O) : Le Champ de la Ligne de Base comme Être

**Des distributions de probabilité aux états cognitifs**

Nous modélisons l'état cognitif d'un agent au temps t comme une distribution de probabilité P(x,t) sur un espace d'états de haute dimension x (représentant croyances, attentes, prédictions perceptives, valence émotionnelle, etc.). Cette distribution évolue selon un processus stochastique guidé par l'entrée sensorielle I(t) et les dynamiques internes.

Pour un processus de diffusion continu avec bruit additif, l'évolution de P(x,t) est gouvernée par l'équation de Fokker-Planck :

$$\frac{\partial P(x,t)}{\partial t} = -\nabla \cdot [F(x,t)P(x,t)] + D\nabla^2 P(x,t) \quad (1)$$

où F(x,t) est le champ de dérive (dynamiques déterministes) et D est le coefficient de diffusion (fluctuations stochastiques). Pour nos besoins, la dérive est guidée par l'entrée et la décroissance :

$$F(x,t) = -\lambda(x - I(t)) \quad (2)$$

où λ est un taux de décroissance et I(t) est l'entrée courante. Cela produit un système qui « suit » l'entrée tout en décroissant vers elle.

**La ligne de base comme état attendu**

La ligne de base B(t) est définie comme le premier moment (valeur attendue) de cette distribution :

$$B(t) = \mathbb{E}[X] = \int x \, P(x,t) \, dx \quad (3)$$

Pour le cas de dérive linéaire, les dynamiques de B(t) peuvent être résolues exactement. Si nous supposons que le système intègre l'entrée dans le temps avec une pondération exponentielle (reflétant l'actualisation temporelle et la « mémoire » finie du système), nous obtenons :

$$B(t) = \int_{-\infty}^{t} \lambda e^{-\lambda(t-\tau)} I(\tau) \, d\tau \quad (4)$$

C'est la forme intégrale temporelle de la ligne de base. Ce n'est pas un instantané mais une *histoire* — l'« être » de l'agent est la somme de son expérience, pondérée par la récence.

**Intuition clé :** La ligne de base n'est pas une « représentation interne » au sens classique. C'est le précipité mathématique de l'histoire entière d'interaction de l'agent avec le monde. C'est la formalisation de l'affirmation énactiviste selon laquelle « organisme et environnement s'enveloppent l'un dans l'autre » (Varela et al., 1991 ; Thompson, 2007). L'état cognitif de l'agent est l'intégrale de son couplage.

**La ligne de base corticale : Instanciation neurobiologique**

Nous postulons que le néocortex instancie une version lentement intégratrice, multi-échelle temporelle, de cette ligne de base, que nous appelons B_ctx (la ligne de base corticale). Ceci correspond à l'« essentiel » ou échafaudage sémantique qui a été extensivement documenté dans la recherche sur la mémoire (Bartlett, 1932 ; Brewer & Treyens, 1981). Formellement :

$$B_{\text{ctx}}(t) = \int_{-\infty}^{t} \lambda_{\text{ctx}} e^{-\lambda_{\text{ctx}}(t-\tau)} I_{\text{ctx}}(\tau) \, d\tau \quad (5)$$

où λ_ctx est un taux de décroissance lent (de l'ordre des heures aux jours) et I_ctx(τ) est la composante contextuelle/sémantique de l'entrée au temps τ.

### 2.2 Déontologie (D) : Le Couplage comme Influence Constitutive

**L'introduction de l'Autre**

L'équation de la ligne de base (4) est individualiste — elle modélise un agent solitaire. Nous l'étendons maintenant au cas dyadique, où deux agents A et B sont en interaction. L'affirmation centrale de l'ontologie relationnelle est que la ligne de base de l'Agent A n'est pas simplement mise à jour par les observations de l'Agent B, mais est *constitutivement façonnée* par le traitement que B fait de A.

Nous formalisons cela à travers un terme d'entrée déontologique D_{B→A}(t), qui représente la « demande » ou l'« adresse » que l'Agent B pose à l'Agent A. Ce n'est pas de l'information *à propos de* B ; c'est l'influence *de* B *sur* A. Suivant Levinas (1961/1969), c'est l'Autre dans le Même — la revendication non négociable que l'Autre fait sur le soi.

L'équation couplée de la ligne de base pour l'Agent A devient :

$$B_A(t) = \int_{-\infty}^{t} \lambda e^{-\lambda(t-\tau)} [I_A(\tau) + \kappa(\tau) \cdot D_{B \to A}(\tau)] \, d\tau \quad (6)$$

où κ(τ) est la force du couplage au temps τ. C'est le paramètre qui gouverne le degré auquel l'influence de B est intégrée dans la ligne de base de A.

**Intuition clé :** C'est Levinas formalisé. Le terme κ·D n'est pas une perturbation ni une correction ; c'est une composante fondamentale de l'ontologie de A. Lorsque κ > 0, l'« être » de l'Agent A (sa ligne de base, son ancrage cognitif) est partiellement constitué par l'Agent B. Le soi n'est pas autonome ; il est relationnel.

**Symétrie et constitution mutuelle**

Le système est symétrique. La ligne de base de l'Agent B est simultanément façonnée par A :

$$B_B(t) = \int_{-\infty}^{t} \lambda e^{-\lambda(t-\tau)} [I_B(\tau) + \kappa(\tau) \cdot D_{A \to B}(\tau)] \, d\tau \quad (7)$$

Les équations (6) et (7) sont récursivement couplées. Aucune ligne de base ne peut être résolue sans référence à l'autre. C'est la formalisation de l'« intra-action » de Barad (2007) — les agents ne préexistent pas à leur interaction ; ils émergent à travers elle.

**Couplage dynamique : κ(t) comme état évolutif**

Dans les formulations antérieures, κ était traité comme un paramètre statique. Le modèle de la Porte Résonante (Everett, 2025, RG V3) introduit l'intuition cruciale que κ lui-même *évolue* selon l'historique du succès de prédiction mutuelle. Le couplage se renforce lorsque les agents se prédisent mutuellement avec succès et s'affaiblit dans le cas contraire.

L'évolution de κ est gouvernée par :

$$\frac{d\kappa}{dt} = \alpha(1 - ||N_{\text{mutual}}||^2) - \beta\kappa \quad (8)$$

où :
- α est un taux d'apprentissage (rapidité d'augmentation possible du couplage)
- β est un taux de décroissance (atténuation naturelle du couplage en l'absence de renforcement)
- N_mutual est l'erreur de prédiction mutuelle entre agents : N_mutual = ||B_A − B_B||²

**Intuition clé :** Le terme (1 − ||N_mutual||²) est maximal lorsque l'erreur mutuelle est proche de zéro. Cela formalise l'« inertie inverse » — l'observation phénoménologique que les interactions deviennent moins coûteuses cognitivement à mesure que la résonance est atteinte. Un κ élevé n'est pas simplement « être proche » de quelqu'un ; c'est l'état computationnel de prédiction mutuelle efficace.

### 2.3 Épistémologie (E) : Connaître comme Déviation Résonante

**La nouveauté comme signal épistémique fondamental**

Étant donné une ligne de base B(t), « connaître » est formalisé comme la détection de déviation. Le signal de nouveauté N(t) est défini comme la distance euclidienne au carré entre l'entrée actuelle I(t) et la ligne de base :

$$N(t) = ||I(t) - B(t)||^2 \quad (9)$$

C'est l'erreur de prédiction de base dans le cadre du Traitement Prédictif (Friston, 2010 ; Clark, 2016). Ce qui est nouveau dans notre cadre est la constitution relationnelle de B elle-même — la ligne de base contre laquelle la nouveauté est calculée n'est pas individualiste mais dyadique lorsque κ > 0.

**Nouveauté à double composante : Contexte et Trace**

La Porte Épisodique (Everett, 2025, EG) affine cela en postulant que l'hippocampe calcule la nouveauté via un comparateur à double mode. Le signal de nouveauté hippocampique N_h est une somme pondérée de deux composantes :

$$N_h(t) = w_{\text{ctx}} N_{\text{ctx}}(t) + w_{\text{sep}} N_{\text{sep}}(t) \quad (10)$$

où :

- N_ctx(t) : Discordance contextuelle — déviation par rapport à la ligne de base corticale :

$$N_{\text{ctx}}(t) = ||I(t) - B_{\text{ctx}}(t)||^2 \quad (11)$$

- N_sep(t) : Discordance de trace — déviation par rapport à la trace épisodique stockée la plus proche :

$$N_{\text{sep}}(t) = \min_i ||I(t) - T_i.\text{content}||^2 \quad (12)$$

- w_ctx, w_sep sont des paramètres de pondération déterminant la contribution relative de chaque composante.

**Intuition clé :** Cette double structure résout un puzzle fondamental. La discordance contextuelle (N_ctx) capture le signal « ceci viole mes attentes » — il est élevé lorsque le monde dévie de l'essentiel. La discordance de trace (N_sep) capture le signal « je n'ai jamais vu cette chose exacte auparavant » — il est élevé lorsque l'expérience n'est pas seulement inattendue mais entièrement nouvelle. L'hippocampe intègre les deux pour décider si la plasticité est justifiée.

**Nouveauté pondérée par la précision : L'amplificateur résonant**

Le modèle de la Porte Résonante (RG V3) introduit un raffinement supplémentaire : la nouveauté est pondérée par la précision. Dans les états résonants à κ élevé, la précision (confiance, gain attentionnel) du modèle prédictif partagé augmente. Ceci est formalisé comme :

$$\gamma(\kappa) = 1 + \gamma_c \kappa(t) \quad (13)$$

où γ_c est un paramètre de gain de précision. La nouveauté effective utilisée pour le traitement en aval (y compris le filtrage mnésique) est alors :

$$N_{\text{eff}}(t) = \gamma(\kappa(t)) \cdot N_h(t) \quad (14)$$

**Intuition clé :** C'est le mécanisme d'« amplification de la nouveauté » démontré dans RG V3. Lorsque κ est élevé (état résonant), la même erreur de prédiction objective génère un signal de nouveauté effective plus grand. Inversement, les échanges prévisibles sont atténués. Cela crée le phénomène de la « Porte Résonante » : la connexion n'améliore pas simplement l'encodage ; elle crée un filtre computationnel qui amplifie les intuitions authentiques tout en supprimant le bruit.

### 2.4 L'Équation Maîtresse : ∇(⊕)

Nous présentons maintenant le système dynamique complet, que nous appelons ∇(⊕) — l'opérateur triadique-pragmatique. C'est la formalisation complète de « l'Être (O) couplé à l'Obligation (D) donne naissance au Connaître (E) par résonance temporelle (⊕), orienté vers la fonction pragmatique (P). »

**Le Système Central (Cas à deux agents) :**

Intégration de la ligne de base :

$$B_A(t) = \int_{-\infty}^{t} \lambda e^{-\lambda(t-\tau)} [I_A(\tau) + \kappa(\tau) \cdot D_{B \to A}(\tau)] \, d\tau \quad (15a)$$

$$B_B(t) = \int_{-\infty}^{t} \lambda e^{-\lambda(t-\tau)} [I_B(\tau) + \kappa(\tau) \cdot D_{A \to B}(\tau)] \, d\tau \quad (15b)$$

Calcul de la nouveauté :

$$N_h^A(t) = \gamma(\kappa(t)) \cdot \left[ w_{\text{ctx}} ||I_A(t) - B_A(t)||^2 + w_{\text{sep}} \min_i ||I_A(t) - T_i^A.\text{content}||^2 \right] \quad (16)$$

(et de manière analogue pour $N_h^B(t)$)

Évolution du couplage :

$$\frac{d\kappa}{dt} = \alpha(1 - ||B_A(t) - B_B(t)||^2) - \beta\kappa \quad (17)$$

Modulation de la précision :

$$\gamma(\kappa) = 1 + \gamma_c \kappa(t) \quad (18)$$

Point final pragmatique (Vérité comme résonance stochastique) :

Lorsque le système converge, nous définissons l'alignement épistémique comme l'état où :

$$\Delta\phi \to 0, \quad I(A:B) \to \max, \quad \frac{d\kappa}{dt} \to 0 \quad (19)$$

où Δφ est la différence de phase (à partir des dynamiques de Kuramoto ; Kuramoto, 1984 ; Acebrón et al., 2005), I(A:B) est l'information mutuelle entre agents, et dκ/dt → 0 indique un couplage stable. C'est la condition de résonance stochastique — non pas un équilibre déterministe mais un bassin d'attraction dans l'espace des phases du système couplé.

**Interprétation : Ce que signifie ∇(⊕)**

- **∇ :** L'opérateur gradient signifie la poussée continue du système vers la minimisation de l'erreur de prédiction conjointe. C'est le principe de minimisation de l'énergie libre (Friston, 2010) étendu au cas couplé.
- **⊕ :** L'opérateur de couplage signifie la tension productive entre agents. Ils ne sont ni identiques (ce qui ne donnerait aucun transfert d'information) ni indépendants (ce qui ne donnerait aucun couplage). L'état productif est la *différence-en-résonance*.
- **Le Symbole Entier :** ∇(⊕) se lit « le gradient du couplage » ou « minimisation d'erreur dans le système couplé ». C'est l'énoncé formel que le connaître émerge du jeu dynamique d'êtres relationnellement constitués.

---

## SECTION 3 : INSTANCIATION DANS LA MÉMOIRE — Le Cycle en Trois Stades

**Cadrage :** La section précédente a établi le formalisme général de ∇(⊕). Cette section démontre son instanciation empirique dans le système mnésique. Nous montrons que l'encodage, la récupération et la reconsolidation ne sont pas des mécanismes séparés nécessitant des explications indépendantes, mais des propriétés émergentes de l'interaction ligne de base–trace gouvernée par les principes déjà établis.

La ligne de base n'est pas simplement une construction théorique — c'est l'architecture opérationnelle du système cortico-hippocampique. Nous formalisons maintenant le cycle de vie complet de la mémoire comme un processus métabolique en trois stades dans lequel la ligne de base sert de gardien (Stade I), d'échafaudage (Stade II) et de milieu de reconsolidation (Stade III).

### 3.1 Stade I — Filtrage Probabiliste : Comment la Ligne de Base Filtre l'Encodage

**Le problème du filtrage**

Un puzzle fondamental de la recherche sur la mémoire est le problème du filtrage (*gating problem* ; Brewer et al., 1998) : Comment le cerveau décide-t-il quoi encoder ? Si toutes les expériences étaient stockées avec une fidélité égale, le système serait submergé. Si le stockage était déterministe, le système serait fragile.

**La solution par la ligne de base et la déviation : L'encodage stochastique**

Notre cadre résout ce problème par un filtrage probabiliste gouverné par la déviation de la ligne de base. La probabilité qu'une expérience déclenche une plasticité hippocampique (PLT — potentialisation à long terme) n'est pas un seuil déterministe mais une fonction sigmoïde du signal de nouveauté effective N_eff(t) :

$$P(\text{WRITE} | N_{\text{eff}}(t)) = \frac{1}{1 + e^{-k(N_{\text{eff}}(t) - \theta_h)}} \quad (20)$$

où :
- k est le paramètre de pente (sensibilité du filtre)
- θ_h est le seuil de plasticité (la « difficulté » de déclencher la PLT)
- N_eff(t) est la nouveauté pondérée par la précision, à double composante, des équations (14) et (16)

**Intuition clé :** Ce n'est pas un homoncule qui décide « ceci est important, stocke-le ». C'est un processus émergent et stochastique. La ligne de base (qui encode l'histoire entière de l'agent et l'état de couplage actuel) détermine automatiquement ce qui est nouveau. Une haute nouveauté augmente la probabilité d'encodage, mais l'encodage reste un événement stochastique — reflétant le bruit inhérent aux systèmes neuronaux (Faisal et al., 2008) et fournissant une flexibilité adaptative.

**La structure de la trace épisodique**

Lorsqu'un événement d'écriture réussi se produit (c'est-à-dire lorsque le filtre stochastique « s'ouvre »), une trace épisodique discrète T_i est créée dans l'hippocampe. Essentiellement, cette trace n'est pas simplement une copie de l'entrée — c'est une structure de données liée qui capture trois composantes :

$$T_i = \{\text{Contenu}: I(t), \quad \text{Contexte}: B_{\text{ctx}}(t), \quad \text{Temps}: t\} \quad (21)$$

où :
- **Contenu :** Le contenu sensoriel/sémantique de l'expérience — le « quoi »
- **Contexte :** Un instantané de la ligne de base corticale au moment de l'encodage — le « où-et-quand »
- **Temps :** L'index temporel

Cette structure résout le problème du liage (*binding problem*) pour la mémoire épisodique. L'hippocampe ne stocke pas des faits décontextualisés ; il stocke des expériences contextualisées. Le vecteur de contexte B_ctx(t) est la clé qui permettra ultérieurement la récupération (Stade II) et la reconsolidation (Stade III).

**Cartographie neurobiologique :**

1. **Réseau auto-associatif CA3 :** Le vecteur de contenu est stocké via une plasticité hebbienne rapide dans le réseau récurrent CA3 (Marr, 1971 ; Treves & Rolls, 1994).
2. **Comparateur CA1 :** Le calcul de nouveauté à double composante (éq. 10) correspond au rôle documenté de CA1 comme détecteur de « correspondance-discordance » (Lisman & Grace, 2005 ; Duncan et al., 2012).
3. **Filtrage PLT :** La fonction probabiliste (éq. 20) correspond au phénomène bien établi que l'induction de PLT nécessite une dépolarisation au-dessus d'un seuil modulé par les neuromodulateurs (dopamine, acétylcholine) (Lisman et al., 2011).
4. **Signature neuronale P300b :** Le signal de nouveauté effective N_eff(t) prédit l'amplitude de la composante P300b du potentiel évoqué, un corrélat neuronal bien établi de l'encodage mnésique et de la mise à jour contextuelle (Polich, 2007 ; Donchin & Coles, 1988).

**Affirmation clé :** La ligne de base est le gardien de la mémoire. En définissant ce qui est attendu, elle définit automatiquement ce qui est nouveau. En intégrant à la fois l'expérience propre (I) et l'apport déontologique (κ·D), elle assure que l'encodage est à la fois individuellement adaptatif et socialement contextualisé.

### 3.2 Stade II — Rappel Échafaudé par le Contexte : Comment la Ligne de Base Structure la Récupération

**Le problème de la récupération**

Si les traces sont stockées comme des structures liées {Contenu, Contexte, Temps}, comment la trace correcte est-elle récupérée étant donné un indice de rappel C(t) ? Une recherche par force brute sur toutes les traces serait computationnellement intraitable. Le système doit contraindre l'espace de recherche.

**La ligne de base comme échafaudage dynamique**

Notre cadre résout cela par un rappel en deux stades, où la ligne de base actuelle B_current(t) agit comme un filtre parallèle massif avant que toute recherche sérielle ne commence.

**Stade II.A : Échafaudage (Filtrage parallèle)**

Le système ne recherche pas toutes les traces. Au lieu de cela, il contraint instantanément la recherche à un petit Ensemble Candidat de traces dont les vecteurs de contexte stockés sont similaires à la ligne de base actuelle :

$$\text{Ensemble Candidat} = \{T_k \mid ||B_{\text{current}}(t) - T_k.\text{Contexte}||^2 < \theta_{\text{search}}\} \quad (22)$$

où θ_search est un paramètre de rayon de recherche (un hyperparamètre du système).

**Intuition clé :** C'est pourquoi le contexte est un indice de rappel si puissant (Godden & Baddeley, 1975 ; Smith & Vela, 2001). Lorsque vous retournez dans un lieu physique, votre entrée sensorielle actuelle I(t) fait dériver votre ligne de base B_current(t) vers l'état qu'elle avait lors de votre dernière visite. Cela met automatiquement en statut candidat toutes les traces qui ont été encodées dans des états de ligne de base similaires.

**Stade II.B : Résonance (Minimisation de l'énergie libre)**

Une fois l'ensemble candidat contraint, le système effectue un processus de résonance compétitive. Il « essaie » chaque trace candidate en mémoire de travail et se stabilise sur celle qui minimise une fonction d'Énergie Libre de Rappel F_recall :

$$T_{\text{recalled}} = \arg\min_{T_k \in \text{Ensemble Candidat}} [F_{\text{recall}}(T_k, B_{\text{current}}(t), C(t))] \quad (23)$$

Une formulation plausible de F_recall est une somme pondérée des discordances contextuelles et de contenu :

$$F_{\text{recall}}(T_k, B_{\text{current}}, C) = w_{\text{ctx}} ||B_{\text{current}} - T_k.\text{Contexte}||^2 + w_{\text{con}} ||C - T_k.\text{Contenu}||^2 \quad (24)$$

**Intuition clé :** L'expérience subjective du rappel — le « déclic » de la reconnaissance, la vivacité soudaine — est la signature du système se stabilisant dans un minimum d'énergie profond et aigu. Ce n'est pas une simple recherche ; c'est une reconstruction résonante.

**Pourquoi le rappel est reconstructif, non reproductif :**

1. **La ligne de base a changé :** B_current(t) à la récupération n'est presque jamais identique à B_ctx(t_encoding). L'échafaudage s'est déplacé.
2. **L'inférence comble les lacunes :** Si le contenu de la trace est dégradé ou incomplet, le système sélectionnera le candidat qui correspond le mieux à la ligne de base actuelle et à l'indice.
3. **Les faux souvenirs émergent naturellement :** Si un leurre est hautement cohérent avec la ligne de base sémantique mais n'a pas de trace correspondante, le système peut néanmoins le « rappeler » en générant une pseudo-trace qui minimise F_recall.

### 3.3 Stade III — Reconsolidation Dynamique : Comment la Ligne de Base Fait Évoluer la Mémoire

**Le paradoxe de la reconsolidation**

La mémoire n'est pas statique. Lorsqu'une trace est récupérée, elle devient labile — sujette à modification avant d'être restockée (Nader et al., 2000 ; Nader & Hardt, 2009). Mais cela crée un paradoxe : si le rappel met à jour les mémoires, comment conservons-nous un savoir stable dans le temps ?

**L'échafaudage mouvant**

La réponse repose sur la reconnaissance que la ligne de base elle-même évolue. Entre l'encodage (temps t₁) et le rappel (temps t₂), la ligne de base a intégré de nouvelles expériences :

$$B_{\text{ctx}}(t_2) = B_{\text{ctx}}(t_1) + \int_{t_1}^{t_2} \lambda_{\text{ctx}} e^{-\lambda_{\text{ctx}}(t_2 - \tau)} I_{\text{ctx}}(\tau) \, d\tau \quad (25)$$

L'échafaudage s'est déplacé.

**La reconsolidation comme pontage contextuel**

Lorsqu'une trace T_i est récupérée et reconsolidée, son vecteur de contexte est mis à jour via une moyenne pondérée de son contexte original et de la ligne de base actuelle :

$$T_i'.\text{Contexte} = (1 - \eta) \cdot T_i.\text{Contexte} + \eta \cdot B_{\text{current}}(t) \quad (26)$$

où η ∈ [0,1] est le taux de mise à jour de la reconsolidation — un paramètre déterminant combien la trace est « tirée » vers la ligne de base actuelle.

**Intuition clé :** C'est le mécanisme de mise à jour de la mémoire sans oubli catastrophique. La trace n'abandonne pas son contexte original, mais elle acquiert un « pont » partiel vers le contexte actuel. Cela explique :

1. **La reconsolidation thérapeutique :** Récupérer un souvenir traumatique dans un contexte thérapeutique sûr, à κ élevé, permet de mettre à jour le vecteur de contexte de la trace (Schiller et al., 2010 ; Brunet et al., 2008).
2. **La dérive de la mémoire autobiographique :** Nos souvenirs d'enfance ont été reconsolidés des dizaines de fois, tirant chaque fois le contexte vers notre état actuel (Bauer & Larkina, 2014).
3. **Le façonnage social de la mémoire :** Lorsque nous rappelons des expériences partagées en conversation (κ élevé), l'entrée déontologique de notre partenaire est intégrée dans B_current. Reconsolider la trace dans cet état tire son contexte vers la ligne de base partagée (Hirst & Echterhoff, 2012).

### 3.4 Synthèse : Un Système Complet et Vivant

**La symbiose de la ligne de base et de la trace**

Nous avons montré que la ligne de base et la bibliothèque de traces ne sont pas des systèmes séparés mais sont symbiotiques :

- La ligne de base **filtre** quelles expériences deviennent des traces (Stade I)
- La ligne de base **échafaude** quelles traces sont récupérables (Stade II)
- La ligne de base **médiatise** comment les traces sont mises à jour dans le temps (Stade III)

Cette architecture dissout le paradoxe de l'essentiel et du détail :

- **L'essentiel** est la ligne de base — la structure d'attente continue, lentement intégratrice
- **Le détail** est la bibliothèque de traces — les instantanés discrets de haute fidélité, encodés probabilistiquement lorsque l'essentiel est violé
- **Le rappel** est la résonance entre eux

**Le Cycle en Trois Stades comme Métabolisme :**

1. **Anabolisme** (Stade I) : Encodage — de nouvelles traces sont synthétisées lorsque la nouveauté dépasse le seuil
2. **Catabolisme** (Stade II) : Récupération — les traces sont décomposées (réactivées, déconsolidées) lorsqu'elles sont indicées
3. **Reconsolidation** (Stade III) : Mise à jour — les traces sont reconstruites dans le contexte de la ligne de base actuelle

La mémoire est vivante. Ce n'est pas du stockage ; c'est un système vivant qui adapte continuellement le passé pour servir le présent.
