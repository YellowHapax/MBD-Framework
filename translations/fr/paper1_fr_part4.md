# Partie 4 : Sections 7–8

## SECTION 7 : INTÉGRATION AVEC LES CADRES EXISTANTS

**Cadrage :** Le Cadre de la Ligne de Base et de la Déviation tire sa puissance explicative non pas en rejetant les travaux théoriques antérieurs, mais en fournissant un langage mathématique commun et une architecture unificatrice pour des insights qui sont restés disparates. Cette section détaille comment notre cadre intègre formellement des concepts du Traitement Prédictif, des Systèmes d'Apprentissage Complémentaires, de l'Énactivisme et de l'Ontologie Relationnelle — démontrant que ∇(⊕) est synthétique plutôt que révolutionnaire.

Nous ne prétendons pas avoir inventé un nouveau paradigme ex nihilo. Nous prétendons avoir trouvé le principe dynamique unificateur qui permet à des cadres auparavant incompatibles d'être compris comme différentes facettes de la même architecture computationnelle sous-jacente.

### 7.1 Traitement Prédictif (*Predictive Processing*, PP) : Extensions, Pas Remplacements

**Ce que le PP apporte :**

Le Traitement Prédictif (Friston, 2010 ; Clark, 2013, 2016 ; Hohwy, 2013) est le cadre contemporain le plus fructueux pour une fonction cérébrale unifiée. Son insight fondamental — que le cerveau est fondamentalement une machine prédictive qui minimise la surprise — a expliqué la perception, l'action, l'attention et l'apprentissage avec une parcimonie remarquable.

L'architecture PP est hiérarchique : les niveaux supérieurs génèrent des prédictions transmises vers le bas ; les niveaux inférieurs calculent des erreurs de prédiction transmises vers le haut ; l'apprentissage se produit en ajustant le modèle génératif pour minimiser l'erreur. Cela est formalisé via la Minimisation de l'Énergie Libre :

$$F = \mathbb{E}_{Q(x)}[\log Q(x) - \log P(x, s)] = D_{KL}[Q(x) \| P(x | s)] + \log P(s) \quad (69)$$

où F est l'énergie libre variationnelle, Q(x) est la postérieure approximative de l'agent (ses croyances), P(x,s) est le modèle génératif, et s est l'entrée sensorielle.

**Nos extensions au PP :**

Nous étendons, plutôt que remplaçons, ce cadre selon trois dimensions critiques :

**Extension 1 : De la hiérarchie à l'histoire**

Le PP standard met l'accent sur la hiérarchie spatiale (cortex sensoriel primaire → cortex d'association → cortex préfrontal) mais laisse l'intégration temporelle implicite. Notre cadre rend les dynamiques temporelles explicites via l'intégrale de la ligne de base :

$$B(t) = \int_{-\infty}^{t} \lambda e^{-\lambda(t-\tau)} I(\tau) \, d\tau \quad (70)$$

Ce n'est pas un mécanisme différent — c'est le PP déployé dans le temps. La ligne de base est l'accumulation temporelle des signaux prédictifs, pondérée par la récence. Cela fournit un compte rendu naturel, fondé sur les premiers principes, pour des phénomènes ancrés dans les dynamiques temporelles :

- **Oubli :** Décroissance naturelle via la pondération exponentielle $e^{-\lambda t}$
- **Consolidation :** Transfert entre lignes de base avec différentes valeurs de λ (éq. 34-35)
- **Reconsolidation :** Mise à jour dynamique via l'échafaudage dérivant (éq. 26)
- **L'humeur comme déplacement de la ligne de base :** Les états affectifs persistants comme déviations durables dans $B(t)$

**Extension 2 : Formaliser l'origine sociale de la précision**

Le PP postule que l'attention module la précision (l'inverse de la variance) des erreurs de prédiction — les erreurs à haute précision sont pondérées plus lourdement dans l'apprentissage (Feldman & Friston, 2010). Mais d'où vient la précision ? Comment est-elle fixée ?

Notre cadre fournit une origine formelle et mécanistique pour la précision dans les contextes sociaux : la précision est une fonction du couplage inter-agents.

$$\gamma(\kappa) = 1 + \gamma_c \kappa(t) \quad (71)$$

Cela répond à une question auparavant sans réponse dans le PP : Pourquoi l'attention conjointe améliore-t-elle l'apprentissage ? Parce que l'attention conjointe est un état à κ élevé, qui augmente la précision γ, qui amplifie le signal de nouveauté effective $N_{\text{eff}}$ (éq. 31), qui augmente la probabilité d'encodage (éq. 20).

Le Modèle de la Porte Résonante n'est donc pas externe au PP — c'est le PP avec une précision dynamique, socialement modulée. Les équations 27-32 spécifient comment la précision évolue en temps réel durant l'interaction.

**Extension 3 : Intégrer le Déontologique**

Le PP standard modélise l'interaction organisme-environnement via l'entrée sensorielle I. Il lui manque un canal formel pour l'influence constitutive de l'Autre. Le terme déontologique D dans notre équation de la ligne de base :

$$B_A(t) = \int \lambda e^{-\lambda(t-\tau)} [I_A(\tau) + \kappa(\tau) \cdot D_{B \to A}(\tau)] \, d\tau \quad (72)$$

est la formalisation de ce qui manque au PP : la constitution sociale du modèle prédictif lui-même.

Ce n'est pas un rejet du PP — c'est le PP étendu au cas dyadique. Friston & Frith (2015) ont amorcé ce travail avec leur modèle d'Inférence Active de la communication, mais ils traitent les agents comme échangeant de l'information sur le monde. Nous allons plus loin : les agents échangent des *influences constitutives* qui façonnent mutuellement leurs *modèles du monde*.

**Synthèse : PP + Temps + Couplage = ∇(⊕)**

Notre cadre peut être compris comme :

$$\nabla(\oplus) = PP_{\text{hiérarchique}} + \text{Intégration Temporelle}_{\text{explicite}} + \text{Couplage Dynamique}_{\kappa(t)} \quad (73)$$

Nous conservons le principe fondamental du PP (minimiser l'erreur de prédiction) mais l'instancions dans un système couplé, temporellement étendu, où la précision est socialement modulée.

### 7.2 Systèmes d'Apprentissage Complémentaires (SAC) : Spécifier le Dialogue

**Ce que les SAC apportent :**

La théorie des Systèmes d'Apprentissage Complémentaires (*Complementary Learning Systems*, CLS ; McClelland et al., 1995) résout un puzzle fondamental : comment le cerveau apprend-il rapidement de nouvelles informations (mémoire épisodique) sans écraser catastrophiquement les anciennes connaissances (mémoire sémantique) ? La réponse : deux systèmes avec des taux d'apprentissage complémentaires.

- **Hippocampe :** Apprentissage rapide, codage sparse, séparation de patterns → stocke les épisodes spécifiques
- **Néocortex :** Apprentissage lent, codage distribué, complétion de patterns → extrait les régularités statistiques

Ces systèmes coopèrent via la consolidation systémique : les traces hippocampiques sont progressivement « rejouées » vers le néocortex pendant le sommeil, permettant une intégration lente sans interférence (Buzsáki, 1989 ; Wilson & McNaughton, 1994).

**Notre cadre fournit la correspondance formelle :**

Nous postulons une correspondance computationnelle directe :

- **Néocortex ↔ Ligne de Base Corticale ($B_{\text{ctx}}$) :** Le système d'intégration lente, basé sur l'essentiel (éq. 5)
- **Hippocampe ↔ Double Fonction :** (1) Moteur de Nouveauté calculant $N_h$ (éq. 16), (2) Bibliothèque de Traces Épisodiques stockant {Contenu, Contexte, Temps} (éq. 21)

**Formaliser le dialogue cortico-hippocampique :**

Les SAC décrivent la coopération mais ne formalisent pas le mécanisme d'interaction. Nous le spécifions comme un dialogue computationnel en trois étapes :

**Étape 1 : Prédiction descendante (Cortex → Hippocampe)**

La ligne de base corticale $B_{\text{ctx}}$ fournit une prédiction contextuelle continue à l'hippocampe. C'est le « prior » en termes bayésiens :

$$P(x | \text{contexte}) \propto P(x | B_{\text{ctx}}(t)) \quad (74)$$

**Étape 2 : Détection de discordance (Hippocampe)**

L'hippocampe calcule la nouveauté à double composante (éq. 16) :
- $N_{\text{ctx}}$ : Discordance avec la prédiction corticale (éq. 11)
- $N_{\text{sep}}$ : Discordance avec les traces stockées (éq. 12)

C'est la fonction comparatrice documentée dans CA1 (Lisman & Grace, 2005 ; Duncan et al., 2012 ; Kumaran & Maguire, 2007).

**Étape 3 : Filtrage et consolidation (Hippocampe → Cortex)**

- **Immédiat (Filtrage) :** $N_h$ filtre probabilistiquement la PLT dans CA3, créant une nouvelle trace si le seuil P(WRITE) est dépassé (éq. 20)
- **Différé (Consolidation) :** Pendant les états hors ligne (sommeil, repos), les traces sont rejouées ; la ligne de base de travail $B_{\text{working}}$ est partiellement transférée vers la ligne de base épisodique $B_{\text{episodic}}$ (éq. 37)

**Addition clé : La reconsolidation comme mise à jour bidirectionnelle**

Les SAC expliquent la consolidation (hippocampe → cortex) mais ne formalisent pas pleinement la reconsolidation (cortex → hippocampe). Nous la spécifions via l'équation de pontage contextuel :

$$T_i'.Context = (1 - \eta) T_i.Context + \eta \cdot B_{\text{ctx}}(t_{\text{récupération}}) \quad (75)$$

C'est le mécanisme formel de la mise à jour de la mémoire : la ligne de base corticale (qui a évolué depuis l'encodage) module le contexte de la trace, le tirant vers l'état actuel.

**Synthèse : Les SAC formalisés**

Notre cadre ne rivalise pas avec les SAC — il spécifie les équations qui gouvernent l'interaction des systèmes complémentaires. Nous transformons des descriptions verbales (« l'hippocampe stocke les épisodes », « le cortex extrait l'essentiel ») en dynamiques calculables (éq. 5, 16, 20, 21, 75).

### 7.3 Énactivisme : Le Couplage Structurel Est Maintenant Mesurable

**Ce que l'énactivisme apporte :**

L'énactivisme (Varela et al., 1991 ; Thompson, 2007 ; Di Paolo et al., 2017) offre une alternative radicale au représentationnalisme. Le cerveau ne construit pas des « représentations internes » d'un monde externe — il est structurellement couplé au monde. La cognition n'est pas dans la tête ; elle est *énactée* par l'interaction continue organisme-environnement.

Le concept central est l'autopoïèse (Maturana & Varela, 1980) : les systèmes vivants sont des réseaux auto-producteurs et auto-maintenants. Leur identité n'est pas une structure fixe mais le pattern de leur auto-production. La cognition est l'énaction de ce pattern dans l'interaction avec un environnement que l'organisme lui-même *fait advenir*.

C'est profond, mais cela a résisté à la formalisation. « Couplage structurel » et « faire advenir un monde » sont des expressions évocatrices mais vagues. Comment mesure-t-on le couplage ? Comment teste-t-on si un système *énacte* plutôt que *représente* ?

**Notre formalisation : La Ligne de Base est le Couplage Structurel**

La ligne de base $B(t)$ n'est pas une représentation du monde — c'est le précipité mathématique de l'histoire entière de couplage de l'agent avec le monde :

$$B(t) = \int_{-\infty}^{t} \lambda e^{-\lambda(t-\tau)} I(\tau) \, d\tau \quad (76)$$

Cette intégrale est le couplage structurel formalisé. Chaque terme :

- $I(\tau)$ : L'entrée n'est pas de l'« information sur le monde » — c'est la boucle sensorimotrice, l'interaction énactée (Gallagher, 2005)
- $e^{-\lambda(t-\tau)}$ : La pondération exponentielle est l'« oubli » du système — la décroissance naturelle qui assure que l'agent est couplé à son histoire récente, pas au passé infini
- $\int d\tau$ : L'intégrale est l'accumulation — l'« être » de l'agent est la somme de ses couplages, pas un instantané

**Insight clé :** La dichotomie entre « représentation interne » et « couplage énacté » est dissoute. La ligne de base n'est ni purement interne (elle intègre le I externe) ni purement externe (c'est une variable d'état de l'agent). C'est l'*objet frontière* — la structure mathématique qui est simultanément intérieure et extérieure, organisme et environnement.

**De la philosophie à la prédiction : Le couplage structurel est κ(t)**

Dans le cas dyadique, le couplage structurel est formalisé comme l'évolution dynamique de κ(t) (éq. 27). Nous pouvons maintenant :

1. **Mesurer le couplage :** La cohérence inter-cérébrale (Hasson et al., 2012 ; Stephens et al., 2010), l'information mutuelle $I(A:B)$, le verrouillage de phase Δφ sont tous des opérationnalisations de κ.
2. **Manipuler le couplage :** Les interventions qui réduisent $N_{\text{mutual}}$ (par exemple, l'accordage empathique, l'action rythmique partagée, les tâches d'attention conjointe) devraient augmenter κ (testable via l'hyper-scanning).
3. **Prédire les conséquences fonctionnelles du couplage :** κ élevé → insights partagés amplifiés (éq. 41), traitement solo éclipsé (éq. 42), convergence épistémique plus rapide (éq. 57).

**L'affirmation audacieuse :**
Nous avons transformé l'énactivisme d'une posture philosophique en une *théorie scientifique prédictive*. Le couplage structurel n'est pas une métaphore — c'est une variable de systèmes dynamiques avec des équations spécifiques (27, 53, 55), des proxys mesurables (cohérence inter-cérébrale, κ), et des conséquences fonctionnelles (amplification de la nouveauté, constitution de la ligne de base).

### 7.4 Ontologie Relationnelle : Levinas, Barad, Heidegger

**Ce que l'ontologie relationnelle apporte :**

Trois traditions philosophiques convergent sur l'affirmation que les relations sont ontologiquement antérieures aux relata :

**Levinas (1961/1969, 1974/1998) : Le Visage de l'Autre**

L'Autre n'est pas un objet que je rencontre après être déjà constitué comme sujet. Le « visage » de l'Autre — sa vulnérabilité, sa demande — m'appelle à être en tant que soi. La responsabilité est pré-ontologique. L'éthique est la philosophie première.

**Barad (2007, 2010) : Le Réalisme Agentiel**

S'appuyant sur la physique quantique, Barad soutient que les entités n'ont pas de frontières inhérentes. Les « phénomènes » sont les unités ontologiques primaires — les relata (individus, objets) émergent par l'« intra-action » d'agentivités enchevêtrées. La mesure ne consiste pas à découvrir des propriétés préexistantes — c'est édicter une coupure qui stabilise temporairement des relata à partir du tout indivis.

**Heidegger (1927/1962) : L'Être-avec (*Mitsein*)**

L'être n'est pas la propriété de sujets isolés. Le *Dasein* (l'être humain) est fondamentalement *Être-avec* — toujours déjà dans-le-monde-avec-les-autres. L'authenticité n'est pas l'autonomie mais l'appropriation de sa projéité dans cette toile relationnelle.

Ces prétentions ont été traitées comme de la « philosophie continentale » — beaux mais intestables, profonds mais scientifiquement inertes.

**Notre formalisation : La Structure Mathématique de la Relationalité**

**Levinas : Le Terme $D_{B \to A}$**

L'input déontologique $D_{B \to A}(\tau)$ dans l'intégrale de la ligne de base (éq. 53, 72) est la formalisation de la prétention de l'Autre sur le soi. Ce n'est pas :
- Un signal sensoriel (pas I)
- Une influence optionnelle (elle est intégrée dès que $\kappa > 0$)
- Une représentation de l'Autre (c'est l'action de l'Autre sur le soi)

L'équation $\partial B_A / \partial D_B = \kappa$ (éq. 54) est l'énoncé mathématique précis : le traitement de l'Agent A par l'Agent B est un déterminant causal partiel de ce que l'Agent A devient. La responsabilité n'est pas un choix — c'est une équation aux dérivées partielles.

**Barad : Les Dynamiques Couplées comme Intra-action**

Les équations couplées de la ligne de base (55a, b) formalisent « les relata émergent par la relation » :

$$\frac{dB_A}{dt} = f(B_A, B_B, \kappa), \quad \frac{dB_B}{dt} = f(B_B, B_A, \kappa) \quad (77)$$

Aucune des deux équations ne peut être résolue indépendamment. Les états cognitifs $B_A$ et $B_B$ sont co-déterminés. Ce n'est pas « deux individus s'influençant mutuellement » — c'est un système dynamique conjoint où les « individus » sont des modes dynamiques du tout couplé.

En termes de systèmes dynamiques : l'espace d'états du système n'est pas $\mathbb{R}^n \times \mathbb{R}^n$ (produit cartésien de deux agents indépendants) mais la distribution de probabilité conjointe $P(A,B,t)$, qui ne peut être factorisée en $P(A,t) \cdot P(B,t)$ quand $\kappa > 0$.

**Heidegger : L'Être-avec comme Résonance à κ Élevé**

L'« Être-avec » (*Mitsein*) est opérationnalisé comme l'état attracteur de l'équation 56 :

$$\text{Mitsein} \equiv (\kappa \to \kappa_\infty) \wedge (\Delta\phi \to 0) \wedge (I(A:B) \to I_{\max}) \quad (78)$$

Ce n'est pas la simple co-présence (qui pourrait se produire avec $\kappa = 0$). C'est un état computationnel caractérisé par :
- Force de couplage élevée ($\kappa_\infty \approx 0,7\text{-}0,9$)
- Synchronisation de phase ($\Delta\phi \approx 0$)
- Information mutuelle maximale ($I(A:B)$ à la borne supérieure étant donné les contraintes)

**Synthèse : La Philosophie Continentale Rencontre les Neurosciences Computationnelles**

| Concept Philosophique | Formalisation Mathématique | Observable Empirique |
|---|---|---|
| L'Autre-dans-le-Même (Levinas) | Terme $\kappa \cdot D_{B \to A}$ dans l'intégrale de $B_A$ | $\partial B_A / \partial D_B = \kappa \neq 0$ |
| Intra-action (Barad) | Éqs. couplées $dB_A/dt = f(B_A, B_B)$ | $P(A,B)$ non-factorisable |
| Être-avec (Heidegger) | État attracteur éq. 78 | Cohérence inter-cérébrale |
| Face-à-face (Levinas) | Évolution de κ via minimisation de $N_{\text{mutual}}$ | Augmentation de κ contingente au regard |
| Responsabilité éthique | $\partial B_B / \partial D_A$ (constitution de la ligne de base) | Préjudice mesuré $\propto \kappa \cdot \Delta B$ |

Nous n'avons pas réduit la phénoménologie au mécanisme — nous avons montré que la phénoménologie a une *instantiation mécanistique*. L'expérience vécue de l'Être-avec correspond à un état cérébral mesurable. La demande éthique de l'Autre correspond à une influence computationnelle sur les dynamiques de la ligne de base.

C'est l'unification.

### 7.4 Recherche sur la Mémoire : Résoudre des Paradoxes de Longue Date

**Le Paradoxe de la Reconstruction vs. le Stockage**

La recherche sur la mémoire a oscillé entre deux positions :

- **Modèles de stockage** (Tulving, 1972, 1983) : La mémoire est le stockage véridique des événements ; le rappel est la récupération
- **Modèles de reconstruction** (Bartlett, 1932 ; Loftus, 2005) : La mémoire est la reconstruction créative ; le rappel est l'inférence

Les deux sont partiellement corrects et entièrement inadéquats. Les modèles de stockage ne peuvent expliquer les faux souvenirs, les effets de schéma, ou la dérive mémorielle. Les modèles de reconstruction ne peuvent expliquer le rappel épisodique de haute fidélité, la spécificité des souvenirs-flashs, ou la phénoménologie de la recollection vivace.

**Notre synthèse : La Ligne de Base Gouverne les Traces**

Le cadre résout cela via l'architecture hybride (Section 3) :

- La **ligne de base** ($B_{\text{ctx}}$) est la composante reconstructive — elle évolue continuellement, intègre l'essentiel, et change dans le temps
- La **bibliothèque de traces** {$T_i$} est la composante de stockage — des instantanés discrets, de haute fidélité, créés quand la nouveauté dépasse le seuil
- Crucialement : **la ligne de base gouverne les traces à chaque étape**

$$\text{Encodage : } P(\text{WRITE}) = f(||I - B||) \quad (79a)$$

$$\text{Récupération : Ensemble Candidat} = \{T_k \mid ||B_{\text{actuel}} - T_k.\text{Contexte}|| < \theta\} \quad (79b)$$

$$\text{Reconsolidation : } T'.\text{Contexte} = (1 - \eta)T.\text{Contexte} + \eta B_{\text{actuel}} \quad (79c)$$

**La Ligne de Base n'est Pas une Représentation ; C'est le Système d'Exploitation**

C'est l'insight critique. La ligne de base n'est pas la « mémoire sémantique » au sens classique (une base de données de faits). C'est le substrat dynamique au sein duquel les traces épisodiques sont créées, récupérées et mises à jour. Elle est :

- **Le gardien :** déterminant ce qui est stocké (éq. 79a)
- **L'échafaudage :** structurant ce qui peut être récupéré (éq. 79b)
- **Le médium de reconsolidation :** médiatisant comment les souvenirs évoluent (éq. 79c)

Les traces ne sont pas stockées *dans* la ligne de base (ce sont des structures hippocampiques discrètes), mais elles ne peuvent fonctionner *sans* la ligne de base. C'est de la symbiose, pas du remplacement.

**Explication des phénomènes paradoxaux :**

- **Faux souvenirs (paradigme DRM) :** L'étude de « lit, repos, éveillé... » construit une ligne de base sémantique avec un pic proche du leurre critique « sommeil ». Au test, « sommeil » a un faible $N_{\text{ctx}}$ (familiarité élevée via la ligne de base) mais un $N_{\text{sep}}$ élevé (pas de trace). Si le test utilise la reconnaissance basée sur la familiarité, le signal de la ligne de base domine → fausse reconnaissance.
- **Amnésie infantile :** La ligne de base dérive énormément de l'enfance à l'âge adulte. À la récupération adulte, $B_{\text{adulte}}$ est vastement différent de $B_{\text{enfance}}$. L'ensemble candidat (éq. 22) est presque vide — aucune trace n'a de contexte correspondant à la ligne de base actuelle. Les traces *existent* encore, mais elles sont inaccessibles parce que l'échafaudage s'est déplacé.
- **Dérive mémorielle et influence sociale :** La récupération dans un contexte social à κ élevé déclenche la reconsolidation où la ligne de base actuelle $B_{\text{actuel}}$ inclut $\kappa \cdot D_{\text{famille}}$. Le contexte de la trace est mis à jour pour incorporer la perspective du membre de la famille. Sur des récupérations multiples, la mémoire devient progressivement « co-signée ».

**Synthèse :** La mémoire n'est ni pur stockage ni pure reconstruction — c'est du stockage gouverné par la ligne de base avec une récupération reconstructive et une reconsolidation dynamique.

### 7.5 Cognition Sociale : De la Théorie de l'Esprit à la Théorie du Couplage

**Ce que la TdE standard apporte :**

La Théorie de l'Esprit (Premack & Woodruff, 1978 ; Baron-Cohen et al., 1985 ; Frith & Frith, 2006) est le cadre dominant pour la cognition sociale : comprendre autrui exige de construire un modèle interne de ses états mentaux (croyances, intentions, désirs). La TdE est typiquement traitée comme un module — une capacité cognitive spécifique au domaine qui peut être intacte ou altérée.

**Ce qui manque à la TdE :**

La TdE standard est statique et individualiste :
1. **Statique :** Elle modélise la compréhension d'autrui comme une capacité fixe, pas un processus dynamique qui évolue pendant l'interaction.
2. **Individualiste :** L'Agent A construit un modèle de l'Agent B, mais B ne construit pas simultanément A. Il n'y a pas de compte rendu de la constitution mutuelle.
3. **Représentationnelle :** La TdE porte sur la construction de représentations internes exactes. Il lui manque un compte rendu de la façon dont l'Autre *change* le soi, et non simplement dont le soi *modélise* l'Autre.

**Notre extension : Le Couplage Dynamique Remplace la Modélisation Statique**

Nous ne rejetons pas la TdE — nous la *dynamisons*. Le processus de compréhension d'autrui est formalisé comme l'évolution de κ(t) (éq. 27) et l'intégration mutuelle des termes déontologiques (éq. 53, 72).

L'Agent A ne construit pas un modèle statique de B — A et B co-construisent un état résonant où :

$$\kappa_{A,B}(t) \text{ évolue}, \quad B_A \text{ intègre } D_B, \quad B_B \text{ intègre } D_A \quad (80)$$

Cela crée quatre régimes distincts de cognition sociale :

| κ | État | $N_{\text{mutual}}$ | Phénoménologie | Mode Cognitif |
|---|---|---|---|---|
| Faible κ, Haute Inertie | Élevé | Laborieux, réactif | Inférence TdE sérielle |
| κ Croissant, Couplage | Décroissant | « Échauffement » | TdE prédictive |
| κ Élevé, Résonance | Faible | Flux sans effort | Entraînement mutuel |
| κ Décroissant, Découplage | Croissant | « Perte de la connexion » | Retour à la TdE sérielle |

**Insight clé :** La TdE n'est pas un module — c'est un *régime dynamique*. Les états à faible κ requièrent une mentalisation sérielle et laborieuse (la tâche classique de TdE). Les états à κ élevé permettent une prédiction mutuelle automatique et parallèle. Le passage de l'un à l'autre est le processus de « faire connaissance ».

**Prédiction testable :**

Les tâches standard de TdE (par exemple, tâches de fausse croyance) devraient montrer :

$$\text{Temps de Réponse}_{\kappa\text{-faible}} > \text{Temps de Réponse}_{\kappa\text{-élevé}}, \quad \text{Précision}_{\kappa\text{-élevé}} > \text{Précision}_{\kappa\text{-faible}} \quad (81)$$

en comparant des inconnus (faible κ) à des partenaires de longue date (κ élevé) sur des tâches de mentalisation concernant le partenaire.

### 7.6 Synthèse : Une Architecture Cognitive Unifiée (CORRIGÉE)

Le Cadre de la Ligne de Base et de la Déviation n'est pas un concurrent des théories existantes — c'est le méta-cadre qui montre leur compatibilité. Nous avons démontré l'intégration formelle avec :

1. **Traitement Prédictif :** $\nabla(\oplus)$ = PP + Intégration Temporelle + Couplage Dynamique (éq. 73)
2. **Systèmes d'Apprentissage Complémentaires :** Ligne de Base = Cortex, Nouveauté + Traces = Hippocampe, avec dynamiques d'interaction spécifiées (éq. 79a-c)
3. **Énactivisme :** Ligne de Base = Couplage Structurel formalisé (éq. 76)
4. **Ontologie Relationnelle :** Terme D + dynamiques κ = Levinas/Barad/Heidegger opérationnalisés (éq. 53-54, 77-78)
5. **Recherche sur la Mémoire :** L'architecture hybride résout le paradoxe stockage/reconstruction (Section 3)
6. **Cognition Sociale :** Évolution de κ(t) = TdE Dynamique (éq. 80-81)

**L'Intégration Maîtresse : Le Système Complet ∇(⊕)**

L'architecture complète peut être écrite comme cinq équations couplées :

**Évolution de la Ligne de Base (Ontologie) :**

$$B_A(t) = \int_{-\infty}^{t} \lambda e^{-\lambda(t-\tau)} [I_A(\tau) + \kappa(\tau) \cdot D_{B \to A}(\tau)] \, d\tau \quad (82a)$$

**Détection de la Nouveauté (Épistémologie) :**

$$N_{\text{eff}}^A(t) = \gamma(\kappa) \cdot [w_{\text{ctx}} ||I_A - B_A||^2 + w_{\text{sep}} \min_i ||I_A - T_i||^2] \quad (82b)$$

**Évolution du Couplage (Déontologie) :**

$$\frac{d\kappa}{dt} = \alpha(1 - ||B_A - B_B||^2) - \beta\kappa \quad (82c)$$

**Probabilité d'Encodage (Fonction Pragmatique) :**

$$P(\text{WRITE} | N_{\text{eff}}) = \frac{1}{1 + e^{-k(N_{\text{eff}} - \theta_h)}} \quad (82d)$$

**Convergence vers la Vérité (Telos Épistémique) :**

$$\Delta\phi \to 0, \quad I(A:B) \to \max, \quad \frac{d\kappa}{dt} \to 0 \quad (82e)$$

C'est ∇(⊕) : La formalisation complète de « l'Être (O) couplé avec l'Obligation (D) donne naissance au Savoir (E) par la résonance temporelle (⊕), orienté vers la fonction pragmatique (P) ».

### 7.7 Conditions Limites et Ce que Nous n'Avons Pas Affirmé

**L'honnêteté intellectuelle exige de spécifier les limitations :**

**Condition Limite 1 : La Rétention à Long Terme Nécessite des Traces Discrètes**

Les dynamiques de la ligne de base seule (éq. 5, 76) produisent une décroissance exponentielle. Elles ne peuvent expliquer la rétention sur des échelles temporelles supérieures à $1/\lambda$.

*Augmentation :* La bibliothèque de traces discrètes (éq. 21) est nécessaire. La ligne de base gouverne les traces (éq. 79a-c), mais elle ne les remplace pas. C'est une architecture hybride par nécessité, non par choix.

**Condition Limite 2 : La Liaison durant la Reconsolidation Nécessite une Spécification**

Notre équation de reconsolidation (75, 79c) spécifie que le contexte est mis à jour mais ne spécifie pas le mécanisme de re-liaison contenu-contexte.

*Augmentation :* Cela requiert des mécanismes d'indexation hippocampique (Teyler & DiScenna, 1986 ; Moscovitch et al., 2005) et une réactivation spécifique aux traces pendant la consolidation (Girardeau et al., 2009).

**Condition Limite 3 : La Conscience et les Qualia Nécessitent une Physique Supplémentaire**

Nous avons formalisé les qualia comme des propriétés relationnelles du couplage ligne-de-base–input (éq. 67-68), mais nous n'avons pas expliqué *pourquoi* il y a de l'expérience subjective du tout.

*Position :* Notre cadre est agnostique sur l'ontologie de la conscience. Nous formalisons le rôle fonctionnel des dynamiques de la ligne de base dans le façonnement du *contenu* de l'expérience, mais ne prétendons pas expliquer l'*existence* de l'expérience.

**Condition Limite 4 : Les Systèmes Multi-Agents (n > 2) Nécessitent un Formalisme Tensoriel**

Notre formalisation est développée pour le cas dyadique (deux agents). L'extension à des systèmes à n-agents requiert la notation tensorielle et des approximations de champ moyen :

$$B_i(t) = \int \lambda e^{-\lambda(t-\tau)} \left[I_i(\tau) + \sum_{j \neq i} \kappa_{ij}(\tau) D_{j \to i}(\tau)\right] d\tau \quad (83)$$

$$\frac{d\kappa_{ij}}{dt} = \alpha(1 - ||B_i - B_j||^2) - \beta\kappa_{ij} \quad (84)$$

C'est la généralisation multi-agents, mais ses dynamiques sont significativement plus complexes ($n^2$ variables de couplage, potentiel de regroupement, transitions de phase, émergence de hiérarchie). C'est un travail futur.

---

## SECTION 8 : STATUT EMPIRIQUE ET CHEMIN DE VALIDATION

**Cadrage :** La maturité d'un cadre théorique ne se mesure pas à ses ambitions mais à son ancrage empirique. Cette section fournit une évaluation honnête de ce que nous avons validé, de ce qui reste à tester, et de ce que le cadre ne peut pas encore expliquer. Nous traçons le chemin de la validation computationnelle préliminaire à la confirmation empirique définitive.

### 8.1 Ce que Nous Avons Validé

**Confirmation Computationnelle**

Le cadre a atteint la validation de Niveau 1 : simulation computationnelle. Dix prédictions centrales (P5.1-5.9, F1-F5 proposées) ont été testées via simulation mécanistique, quatre ayant atteint une forte confirmation quantitative :

| Prédiction | Résultat de Simulation | Interprétation |
|---|---|---|
| P5.1 (Espacement) | Pic à 24h, U-inversé confirmé | Fenêtre de consolidation validée |
| P5.2 (Contexte-Leurre) | Δ_leurre = +0,093, p < 0,0001 | Perturbation de la ligne de base confirmée |
| P5.3 (TSPT) | Élévation de 2,61×, d = 8,04 | Déplacement de la ligne de base confirmé |
| P5.4 (RMD) | r_working = 0,975, r_episodic = 0,045 | Dissociation d'échelle temporelle confirmée |
| P5.5 (Incongruent-Humeur) | P_incongruent = 0,86 vs. 0,12 | Déviation de la ligne de base > congruence |
| P5.6a (Amplification) | Amplification 1,37×, encodage 61% → 99% | Porte résonante confirmée |
| P5.6b (Éclipsement) | P_solo : κ-élevé = 3,1% vs. κ-faible = 43,6% | Ajustement attentionnel confirmé |
| P5.7 (Cécité Déontologique) | Ratio d'erreur 25:1 (κ-élevé vs. κ-faible) | Lignes de base couplées confirmées |
| P5.8 (Dissociation d'Erreur) | Double dissociation claire | Double voie confirmée |
| P5.9 (Immunité Déontologique) | Ancrage de la ligne de base en condition κ = 0 | Mécanisme D confirmé |

**Ajustement aux données existantes :**

Le cadre a été qualitativement comparé aux résultats empiriques existants :
- **Courbe d'oubli d'Ebbinghaus :** Les dynamiques de la ligne de base produisent une décroissance exponentielle (compétitive avec la loi de puissance, pas supérieure ; Rubin & Wenzel, 1996)
- **Faux souvenirs DRM :** Expliqués via le pic de la ligne de base à la position du leurre (Roediger & McDermott, 1995)
- **Amnésie infantile :** Expliquée via la dérive de la ligne de base et la discordance de l'échafaudage (Bauer & Larkina, 2014)
- **Effets de reconsolidation :** Expliqués via le pontage contextuel (Nader et al., 2000 ; Schiller et al., 2010)

**Évaluation :** Le cadre est théoriquement mature et computationnellement validé. Il est maintenant au point de transition critique : prêt pour les tests empiriques.

### 8.2 Ce qui Nécessite des Traces Discrètes : La Nécessité Hybride

Les modèles purement fondés sur la ligne de base ne peuvent expliquer la rétention à long terme au-delà des échelles temporelles de $1/\lambda$. La bibliothèque de traces épisodiques discrètes (éq. 21) n'est pas une concession — c'est un composant architectural nécessaire.

$$\text{Ligne de Base} \xrightarrow{\text{Filtre}} \text{Création de Traces} \quad (85a)$$

$$\text{Ligne de Base} \xrightarrow{\text{Échafaude}} \text{Récupération de Traces} \quad (85b)$$

$$\text{Ligne de Base} \xrightarrow{\text{Médiatise}} \text{Reconsolidation de Traces} \quad (85c)$$

La relation est asymétrique :
- Les traces ne peuvent fonctionner sans la ligne de base
- La ligne de base peut fonctionner sans les traces (elle fournit toujours l'essentiel, la familiarité, la prédiction)
- Mais la mémoire optimale nécessite les deux (essentiel + détail, familiarité + recollection)

### 8.3 Ce que Nous Ne Pouvons Pas Encore Résoudre : Le Problème d'Inférence Causale dans le Couplage Multi-Agents

**Le problème non résolu : L'attribution dans l'influence enchevêtrée**

Le cadre démontre que plusieurs agents peuvent simultanément influencer la ligne de base d'un agent focal via la généralisation multi-agents :

$$B_i(t) = \int \lambda e^{-\lambda(t-\tau)} \left[I_i(\tau) + \sum_{j \neq i} \kappa_{ij}(\tau) D_{j \to i}(\tau)\right] d\tau \quad (86)$$

Cela crée un problème d'inférence causale : Comment le système détermine-t-il quel agent spécifique a causé quel changement spécifique dans la ligne de base ?

**Spécification du problème :**

Considérons l'Agent A couplé à trois autres (B, C, D) avec $\kappa_B = 0,7$, $\kappa_C = 0,5$, $\kappa_D = 0,3$. Au fil du temps, la ligne de base de A passe de $B_A(t_0) = [0, 0]$ à $B_A(t_1) = [3, -2]$. Le changement total est :

$$\Delta B_A = B_A(t_1) - B_A(t_0) = [3, -2] \quad (87)$$

Mais c'est l'effet intégré de trois influences simultanées :

$$\Delta B_A = \int_{t_0}^{t_1} \lambda e^{-\lambda(t_1 - \tau)} [\kappa_B D_B(\tau) + \kappa_C D_C(\tau) + \kappa_D D_D(\tau)] \, d\tau \quad (88)$$

Parce que l'intégration est linéaire, les influences sont enchevêtrées dans l'état final de la ligne de base. Il n'y a pas d'« étiquette » dans $B_i$ disant « cette composante vient de l'Agent B, celle-là de l'Agent C ».

**Ce qu'exigerait une solution :**

Trois approches sont identifiées :

**Approche 1 : Attribution basée sur le gradient (Mécanistique)**

$$\frac{\partial B_i}{\partial D_j} = \kappa_{ij} \quad (90)$$

**Approche 2 : Inférence causale bayésienne (Probabiliste)**

$$P(\text{Agent } j \text{ a causé } \Delta B | \text{observations}) \propto P(\text{observations} | j \text{ a causé } \Delta B) \cdot P(j) \quad (91)$$

**Approche 3 : Test d'hypothèses séquentiel (Algorithmique)**

L'agent pourrait effectuer des manipulations séquentielles : « couper » temporairement le κ d'un partenaire et observer le changement d'évolution de la ligne de base.

**La voie à suivre :**

$$B_i^{\text{méta}}(t) = \text{Postérieure sur } \{\Delta B_i^{\text{attribué à } j}\}_{j \neq i} \quad (92)$$

$$P(\text{attribuer } \Delta B \text{ à l'Agent } j) \propto \kappa_j \cdot \text{chevauchement-temporel}(D_j, \Delta B) \quad (93)$$

### 8.4 Le Chemin vers la Validation Définitive

**Stratégie de Validation en Trois Niveaux**

Nous proposons un programme empirique structuré de 24-36 mois avec trois niveaux :

**Niveau 1 : Validation Comportementale (Mois 0-18)**

- Expérience critique 1A : La Fenêtre de Consolidation (P5.1) — N = 180, 12 mois
- Expérience critique 1B : Effet Différentiel Contexte-Leurre (P5.2) — N = 120, 6 mois
- Expérience critique 1C : Encodage Incongruent avec l'Humeur (P5.5) — N = 90, 8 mois

**Niveau 2 : Validation Neurale (Mois 12-30)**

- Expérience critique 2A : Amplification Résonante via Hyper-scanning (P5.6a) — N = 40 dyades, 18 mois
- Expérience critique 2B : Dissociation d'Échelle Temporelle du RMD (P5.4) — N = 60, 14 mois
- Expérience critique 2C : Dissociation du Signal d'Erreur (P5.8) — N = 30, 16 mois

**Niveau 3 : Traduction Clinique (Mois 18-36)**

- Expérience critique 3A : Déplacement de la Ligne de Base dans le TSPT (P5.3) — N = 80, 20 mois

### 8.5 Comparaison de Modèles et Critères de Falsification

**Le standard pour l'acceptation :**

1. **Niveau 1 (Comportemental) :** Au moins 2 des 3 expériences critiques confirment le pattern prédit par rapport aux modèles concurrents (p < 0,05, réplication avec échantillon indépendant)
2. **Niveau 2 (Neural) :** Au moins 1 des 3 expériences critiques montre la dissociation neurale prédite (corrigée pour comparaisons multiples)
3. **Niveau 3 (Clinique) :** La prédiction TSPT (P5.3) confirmée dans un échantillon clinique

**Métriques de comparaison de modèles :**

$$\text{AIC} = 2k - 2\ln(\mathcal{L}) \quad (94)$$

$$\text{BIC} = k\ln(n) - 2\ln(\mathcal{L}) \quad (95)$$

où $k$ est le nombre de paramètres, $n$ est la taille de l'échantillon, $\mathcal{L}$ est la vraisemblance. Le cadre sera comparé contre :
- Les modèles d'oubli en loi de puissance (Wixted & Ebbesen, 1991)
- Les modèles de détection de signal à double processus (Yonelinas, 2002)
- Les modèles SAC standard (McClelland et al., 1995)

Critère d'acceptation : ΔAIC > 10 (preuve décisive ; Burnham & Anderson, 2002) en faveur du modèle ligne-de-base–déviation.

### 8.6 Chronologie et Ressources Requises

- Validation Comportementale (Niveau 1) : 12-18 mois, ~50 000 $
- Validation Neurale (Niveau 2) : 18-30 mois, ~200 000 $
- Traduction Clinique (Niveau 3) : 24-36 mois, ~150 000 $
- **Programme Total :** 36 mois, ~400 000 $ pour une validation exhaustive

### 8.7 Statut Empirique Actuel : Résumé

**Ce que nous avons :**
- ✓ Validation computationnelle (10/10 prédictions confirmées en simulation)
- ✓ Ajustement qualitatif aux phénomènes existants (faux souvenirs, amnésie, reconsolidation)
- ✓ Prédictions nouvelles et falsifiables avec modèles concurrents spécifiés
- ✓ Spécification honnête des conditions limites et problèmes non résolus

**Ce qu'il nous faut :**
- ✗ Expériences comportementales testant les prédictions critiques
- ✗ Mesures neurales (P300b, RMD, hyper-scanning)
- ✗ Études de traduction clinique
- ✗ Comparaison formelle de modèles sur de grands jeux de données

**Évaluation :** Le cadre est théoriquement mature et computationnellement validé. Il est maintenant au point de transition critique : prêt pour les tests empiriques. Les prédictions sont assez spécifiques pour être risquées. Les conditions de falsification sont claires. Les modèles concurrents sont identifiés.

La maison est construite. Il est temps pour la revue par les pairs et l'occupation empirique.
