# Partie 2 : Sections 4–5

## SECTION 4 : LA RÉSONANCE — Le Modèle de la Porte Résonante

**Cadrage :** Les sections précédentes ont établi l'architecture formelle (Section 2) et son instanciation dans les mécanismes mnésiques (Section 3). Cette section étend le cadre à la cognition sociale en développant le **Modèle de la Porte Résonante** (*Resonant Gate Model*) — un compte rendu formel de la façon dont le couplage dynamique entre agents crée des états computationnels qui amplifient certains types d'insight tout en atténuant d'autres.

### 4.1 Le Phénomène : L'Amplification Conversationnelle

**L'observation :** Certaines conversations produisent des insights qui n'auraient pu émerger d'aucun des participants seul. Ce n'est pas simplement un « brainstorming » (la génération d'idées par exposition à l'entrée d'autrui). C'est un phénomène computationnel spécifique : l'interaction elle-même crée un filtre qui amplifie certains signaux et en atténue d'autres.

**Le puzzle :** Comment la présence d'un autre esprit change-t-elle le calcul de nouveauté de manière à produire des insights qui ne seraient autrement pas détectés ?

### 4.2 Le Mécanisme : La Porte de la Nouveauté Pondérée par la Précision

**Revue de la nouveauté pondérée par la précision :** Des sections précédentes, la nouveauté effective est :

$$N_{\text{eff}}(t) = \gamma(\kappa(t)) \cdot N_h(t) \quad (27)$$

où $\gamma(\kappa) = 1 + \gamma_c \kappa(t)$ (éq. 18). Le terme clé est la modulation de la précision par $\gamma(\kappa)$.

**Qu'est-ce que la précision dans le contexte social ?**

La précision, dans le cadre du Traitement Prédictif, est la confiance (l'inverse de la variance) dans une prédiction. Une précision élevée signifie que le système est très confiant dans son modèle prédictif ; les erreurs de prédiction à haute précision sont pondérées davantage dans l'apprentissage.

Nous postulons que la précision dans le contexte social augmente avec κ parce que :

1. **La prédiction mutuelle réduit l'incertitude :** Lorsque A prédit B avec précision (et vice versa), le modèle prédictif du monde partagé de chaque agent a une variance plus faible.
2. **L'attention partagée affine les modèles :** L'attention conjointe sur un thème réduit le bruit (l'entrée non pertinente), augmentant le rapport signal-bruit du canal.
3. **Le retour d'information valide les prédictions :** L'acquiescement, l'élaboration et la correction du partenaire fournissent des mises à jour bayésiennes continues qui resserrent les distributions postérieures.

**La porte s'ouvre :** Lorsque κ est élevé, γ est élevé, donc $N_{\text{eff}}$ est élevé pour toute $N_h$ donnée. Cela signifie que des insights faiblement nouveaux (qui ne dépasseraient pas normalement le seuil θ_h pour le filtrage mnésique, éq. 20) sont amplifiés au-delà du seuil dans les états résonants. La Porte Résonante est ouverte : le système est plus sensible à la nouveauté.

**La porte se ferme :** Pour les échanges prévisibles (faible $N_h$), même une haute précision ne produit qu'un faible $N_{\text{eff}}$. Le système ne « surencode » pas les échanges mondains — il amplifie sélectivement ce qui est informatif dans le contexte résonant.

### 4.3 Le Produit : Qu'est-ce que la Résonance Computationnelle ?

**Définition :** La résonance computationnelle est l'état de convergence du système couplé, défini par l'atteinte simultanée de :

$$\Delta\phi \to 0, \quad I(A:B) \to I_{\max}, \quad \frac{d\kappa}{dt} \to 0 \quad (28)$$

C'est un attracteur dans l'espace des phases du système couplé. Ce n'est pas l'accord (les agents pourraient avoir des lignes de base différentes) mais la prédiction mutuelle efficace (chaque agent modélise le modèle de l'autre avec une erreur minimale).

**L'expérience phénoménologique :** Ce que les sujets décrivent comme « connexion », « être sur la même longueur d'onde » ou « flux » dans la conversation correspond à cet état computationnel. Ce n'est pas un sentiment vague — c'est un état mesurable caractérisé par :

- La synchronie inter-cérébrale élevée (mesurable via hyper-scanning EEG ; Hasson et al., 2012 ; Stephens et al., 2010)
- L'information mutuelle maximisée entre les flux comportementaux (tours de parole, gestes, suivi du regard)
- La stabilisation de la trajectoire de κ (la rampe d'apprentissage se stabilise)

**Quatre propriétés de la résonance computationnelle :**

1. **Émergence :** La résonance ne peut être produite par un seul agent. C'est une propriété de la dyade, pas de l'individu. Cela formalise la notion que la connexion n'est pas « dans la tête ».

2. **Fragilité :** La perturbation de Δφ (par exemple, une interruption intempestive, un malentendu) peut faire sortir le système de l'attracteur. La résonance doit être maintenue activement.

3. **Productivité :** L'état résonant est computationnellement productif — il amplifie les insights (via la pondération par la précision) qui sont le plus informatifs étant donné les lignes de base partagées et divergentes des participants.

4. **Bidirectionnalité :** La résonance n'est pas un transfert à sens unique (du connaissant vers l'ignorant). C'est un processus symétrique où les deux lignes de base sont mutuellement modifiées.

### 4.3.1 Application à l'Alliance Thérapeutique

**Le modèle standard :** L'alliance thérapeutique (Bordin, 1979 ; Horvath & Greenberg, 1989) est le facteur prédictif le plus robuste des résultats thérapeutiques, tous types de thérapie confondus. Cependant, sa nature computationnelle demeure sous-spécifiée — c'est un concept descriptif (« accord sur les objectifs, lien émotionnel, consensus sur les tâches ») plutôt qu'un mécanisme formel.

**Notre formalisation :** L'alliance thérapeutique est l'atteinte d'un κ élevé entre le thérapeute et le client. Plus spécifiquement :

$$\kappa_{\text{thérapeutique}}(t) = f(\text{historique de prédiction mutuelle}, \text{confiance}, \text{accord sur les objectifs}) \quad (29)$$

Le κ thérapeutique crée la condition nécessaire pour la reconsolidation de la ligne de base (éq. 26). Le mécanisme est le suivant :

1. Le client récupère un souvenir traumatique ou inadapté (active la trace T_i)
2. La résonance thérapeutique à κ élevé augmente la précision γ (éq. 18)
3. L'insight amplifié par la précision (éq. 27) crée un événement de haute nouveauté effective
4. L'événement de haute nouveauté effective dans le contexte d'un κ thérapeutique élevé permet une reconsolidation avec un taux de mise à jour η accru
5. Le vecteur de contexte de la trace est mis à jour vers la ligne de base actuelle (qui inclut l'entrée déontologique de soutien du thérapeute)

**L'implication testable :** L'alliance thérapeutique mesurée (scores WAI) devrait corréler avec la cohérence inter-cérébrale thérapeute-client (proxy de κ), et les sessions à haute cohérence inter-cérébrale devraient montrer un plus grand changement symptomatique par rapport aux sessions à faible cohérence inter-cérébrale.

### 4.4 L'Inertie Inverse : Le Coût Computationnel de la Connexion

**L'observation :** Initier la connexion est coûteux. Dans chaque conversation, il y a une phase de « réchauffement » — un échange laborieux et fortement prédictif (salutations, bavardage) avant qu'un engagement significatif ne soit possible. Cela ressemble à une inertie : le système résiste au changement d'état de découplé à couplé.

**Le mécanisme :** L'inertie est une propriété émergente de la dynamique de κ (éq. 8). Au début d'une interaction, κ est faible. Le terme (1 − ||N_mutual||²) est faible car N_mutual est élevée (les agents ne se connaissent pas bien). La croissance de κ est donc lente — le système doit accumuler un succès de prédiction mutuelle avant que le couplage ne se renforce.

$$\frac{d\kappa}{dt}\bigg|_{\kappa=0} \approx \alpha(1 - ||N_{\text{mutual,initial}}||^2) - \beta \cdot 0 \approx \alpha \cdot \text{faible} \quad (30)$$

À mesure que κ croît, N_mutual décroît (meilleure prédiction mutuelle), et la croissance de κ s'accélère :

$$\frac{d\kappa}{dt}\bigg|_{\kappa=\kappa_{\text{moyen}}} \approx \alpha(1 - \text{faible}^2) - \beta\kappa_{\text{moyen}} \approx \alpha - \beta\kappa_{\text{moyen}} \quad (31)$$

Cela produit une trajectoire de croissance de κ en forme de sigmoïde — lente au début (inertie), rapide au milieu (« ça a fait tilt »), stabilisée à la fin (résonance).

**L'« Inertie Inverse » :** Le terme « inertie inverse » désigne le phénomène par lequel les interactions deviennent *moins* coûteuses cognitivement à mesure que la résonance est atteinte. C'est l'inverse de la propriété inertielle standard (où la résistance au changement augmente avec la masse). Ici, le « poids » computationnel de l'interaction diminue avec κ :

$$\text{Coût Computationnel} \propto \frac{1}{\kappa(t)} \quad (32)$$

Les états à κ élevé sont « légers » — l'interaction coule avec un effort minimal parce que la prédiction mutuelle est efficace. Les états à faible κ sont « lourds » — chaque tour requiert un traitement sériel coûteux (le raisonnement classique de la Théorie de l'Esprit).

Les raffinements de la modulation de précision et de la dynamique de la porte sont formalisés en totalité dans les équations 13-14 et 17-18 de la Section 2.

---

## SECTION 5 : POUVOIR EXPLICATIF — Dix Prédictions Validées par Simulation

**Cadrage :** Un cadre théorique se mesure à ses prédictions. Cette section présente dix prédictions centrales du Cadre de la Ligne de Base et de la Déviation, chacune : (a) dérivée du formalisme, (b) en conflit avec les modèles dominants, (c) validée par simulation computationnelle, et (d) assortie de conditions de falsification spécifiques.

### 5.1 L'Effet d'Espacement via la Fenêtre de Consolidation

**Le compte rendu standard :** L'apprentissage espacé surpasse l'apprentissage massé (Cepeda et al., 2006). Les modèles de pratique de récupération (Bjork, 1994 ; Bjork & Bjork, 1992) postulent que les intervalles plus longs sont toujours meilleurs (difficulté souhaitable). Les modèles à double processus postulent deux pics (familiarité et recollection à différents intervalles).

**Le compte rendu Ligne de Base–Déviation :** Notre architecture (ligne de base de travail avec décroissance λ_working rapide, plus ligne de base épisodique avec décroissance λ_episodic lente, plus fenêtre de consolidation de 12-48h pour le transfert) génère une prédiction quantitative spécifique : une fonction U-inversé avec un espacement optimal au moment où la consolidation initiale est achevée mais avant que la décroissance excessive ne survienne.

**Mécanisme :** Nous postulons deux systèmes de ligne de base avec différentes échelles temporelles :

- Ligne de base de travail : λ_working = 0,05/heure (demi-vie ≈ 14 heures)

$$B_{\text{working}}(t) = \int_{-\infty}^{t} \lambda_{\text{working}} e^{-\lambda_{\text{working}}(t-\tau)} I(\tau) \, d\tau \quad (34)$$

- Ligne de base épisodique : λ_episodic = 0,0001/heure (demi-vie ≈ 289 jours)

$$B_{\text{episodic}}(t) = \int_{-\infty}^{t} \lambda_{\text{episodic}} e^{-\lambda_{\text{episodic}}(t-\tau)} I(\tau) \, d\tau \quad (35)$$

Pendant le sommeil (12-48h après l'encodage), une fraction de la ligne de base de travail est consolidée (transférée) dans la ligne de base épisodique. L'effet d'espacement émerge de l'interaction de ces échelles temporelles :

- Espacement à 1-6 heures : L'item est encore dans la ligne de base de travail → faible nouveauté à la ré-étude → encodage faible → mauvaise rétention à long terme
- Espacement à 24 heures : Consolidation initiale achevée, l'item est maintenant dans la ligne de base épisodique → nouveauté modérée → encodage fort durant la seconde fenêtre de consolidation → rétention optimale
- Espacement à 96-168 heures : Décroissance excessive de la ligne de base épisodique avant le ré-encodage → faible support de la ligne de base → mauvaise rétention

**Prédiction Nouvelle 5.1 :** R(espacement) a un pic unique à Δt_opt = 24 ± 12 heures (P5.1)

**Validation Computationnelle :** Simulation (MBD Section 7.3, Expérience 2) testant les espacements [1, 6, 24, 48, 96, 168] heures. Valeurs de rétention : [0,005, 0,005, 0,038, 0,025, 0,005, 0,005]. Ajustement quadratique → pic à 24h, coefficient a = -0,0000001 (négatif → U-inversé confirmé).

**Dissociation Empirique :**
- Modèles de pratique de récupération (Bjork, 1994) : prédisent une augmentation monotone
- Modèles à double processus : prédisent deux pics (familiarité à ~6-12h, recollection à ~72-96h)
- Notre modèle : pic unique à 24h guidé par les dynamiques de fenêtre de consolidation

**Condition de Falsification :** Si les données empiriques montrent des effets d'espacement monotones ou des pics doubles, le mécanisme de consolidation-comme-transfert est falsifié.

### 5.2 Effets de Contexte via Perturbation de la Ligne de Base

**Le compte rendu standard :** Le contexte affecte la mémoire via la liaison trace-contexte (Tulving & Thomson, 1973 ; Smith & Vela, 2001) : les caractéristiques contextuelles sont encodées comme partie de la trace ; la correspondance du contexte à la récupération réactive la trace. Cela prédit que le contexte devrait affecter les items anciens (qui ont des traces liées) mais pas les leurres (qui n'ont pas de traces).

**Le compte rendu Ligne de Base–Déviation :** La discordance contextuelle perturbe la ligne de base sémantique via l'injection de bruit, créant une incertitude dans la représentation de l'essentiel. Crucialement, cela affecte davantage les leurres que les items anciens parce que les leurres se situent aux pics de la ligne de base (centroïdes sémantiques) où le bruit a un impact maximal.

**Mécanisme :** Encoder une liste DRM (lit, repos, éveillé, fatigué, rêve, réveil, sieste, couverture) construit une ligne de base sémantique avec un pic proche du leurre critique « sommeil ». À la récupération :

- Même contexte : La ligne de base est stable → reconnaissance modérée du leurre
- Contexte changé : La discordance contextuelle ajoute du bruit gaussien à la ligne de base :

$$B_{\text{test}} = B_{\text{encode}} + \mathcal{N}(0, \sigma_{\text{context}}^2) \quad (36)$$

Le bruit fait dériver/s'étaler le pic de la ligne de base → la position du leurre dans l'espace probabiliste change → la fausse reconnaissance augmente.

**Prédiction Nouvelle 5.2 :** ΔP(reconnaître_leurre)|changement-contexte > ΔP(reconnaître_ancien)|changement-contexte (P5.2)

Le changement de contexte augmente davantage la fausse reconnaissance des leurres que la reconnaissance des items anciens (effet différentiel).

**Validation Computationnelle :** Simulation (MBD Section 7.3, Expérience 3) : Reconnaissance du leurre en même contexte = 0,270, contexte changé = 0,363 (Δ = +0,093, p < 0,0001).

**Condition de Falsification :** Si le changement de contexte affecte les items anciens mais pas les leurres, la liaison trace-contexte est soutenue par rapport à la perturbation de la ligne de base.

### 5.3 Le TSPT comme Déplacement Consolidé de la Ligne de Base

**Le compte rendu standard :** Le TSPT (Trouble de Stress Post-Traumatique) est modélisé comme la sur-consolidation des traces mnésiques traumatiques (Brewin, 2014) ou le conditionnement de la peur aux indices liés au trauma (Rauch et al., 2006). Ces modèles prédisent une réponse élevée aux stimuli liés au trauma spécifiquement.

**Le compte rendu Ligne de Base–Déviation :** Le trauma suivi d'une ré-expérience pendant la fenêtre de consolidation de 12-48h crée un déplacement permanent vers le haut de la ligne de base elle-même. Ce déplacement n'est pas spécifique au stimulus — il élève le calcul de nouveauté pour *tous* les stimuli, y compris les stimuli neutres.

**Mécanisme :**

- Trauma à t = 50h crée un pic important de la ligne de base de travail
- La rumination, les cauchemars ou la ré-exposition pendant la fenêtre de consolidation (62-98h) maintient la ligne de base de travail élevée
- À la fermeture de la fenêtre (t = 100h), une fraction de la ligne de base de travail est consolidée dans la ligne de base épisodique :

$$B_{\text{episodic}}(t > 100\text{h}) = B_{\text{episodic}}^{\text{pre}} + \eta_{\text{trauma}} \cdot B_{\text{working}}(98\text{h}) \quad (37)$$

où η_trauma est un taux de consolidation accru pour les expériences à haute activation (Cahill & McGaugh, 1998).

- Résultat : Tous les calculs de nouveauté ultérieurs sont élevés parce que la ligne de base a un décalage permanent :

$$N_h^{\text{TSPT}}(t) = ||I(t) - (B_{\text{baseline}} + \Delta B_{\text{trauma}})||^2 \quad (38)$$

**Prédiction Nouvelle 5.3 :** $N_h^{\text{TSPT}}(\text{stimulus neutre}) = 2\text{-}5 \times N_h^{\text{contrôle}}(\text{stimulus neutre})$ (P5.3)

Les patients TSPT montrent une erreur de prédiction élevée pour les stimuli neutres et inattendus (non liés au trauma).

**Validation Computationnelle :** Simulation (MBD Section 7.3, Expérience 4) : Contrôles N = 2,448, groupe TSPT N = 6,384 (facteur d'élévation 2,61×, p < 0,000001, d de Cohen = 8,04). L'effet a persisté 51h après la consolidation.

**Condition de Falsification :** Si le TSPT ne montre aucune élévation pour les stimuli neutres, le mécanisme de déplacement de la ligne de base est falsifié.

**Implication Clinique :** L'intervention précoce (restructuration cognitive, blocage pharmacologique) devrait cibler la fenêtre de consolidation de 12-48h pour prévenir le transfert de la ligne de base, et non une intervention immédiate (qui survient avant le début de la consolidation). Ceci s'aligne avec les découvertes récentes sur les thérapies basées sur la reconsolidation (Schiller et al., 2010).

### 5.4 Dissociation des Échelles Temporelles Neuronales : Le RMD Suit la Ligne de Base de Travail, Pas la Ligne de Base Épisodique

**Le compte rendu standard :** L'activité du Réseau du Mode par Défaut (RMD) est associée au « traitement auto-référentiel » ou à la « mémoire autobiographique » (Buckner et al., 2008 ; Andrews-Hanna et al., 2010). Cela suggère que le RMD devrait corréler avec la connaissance de soi à long terme, de type trait, et les événements de vie distants.

**Le compte rendu Ligne de Base–Déviation :** L'activité du RMD reflète la ligne de base de travail (échelle temporelle des heures), et non la ligne de base épisodique (jours-à-mois). Le RMD est le substrat neuronal pour l'intégration en ligne de l'expérience récente, pas pour le stockage hors ligne de souvenirs distants.

**Prédiction Nouvelle 5.4 :** r(B_working, RMD BOLD) > 0,5 ; r(B_episodic, RMD BOLD) < 0,2 (P5.4)

**Validation Computationnelle :** Simulation (MBD Section 7.3, Expérience 5) : r_working = 0,975, r_episodic = 0,045 (t = 23,98, p < 0,000001).

**Condition de Falsification :** Si le BOLD du RMD corrèle également avec les expériences récentes et les souvenirs autobiographiques distants, la dissociation d'échelle temporelle échoue.

### 5.5 Inertie Émotionnelle et Encodage Incongruent avec l'Humeur

**Le compte rendu standard :** Les modèles de mémoire congruente avec l'humeur (Bower, 1981 ; Blaney, 1986) prédisent qu'un état d'humeur donné améliore l'encodage des stimuli émotionnellement congruents via l'amorçage associatif ou l'activation de réseaux affectifs.

**Le compte rendu Ligne de Base–Déviation :** Un état d'humeur persistant crée un déplacement durable de la ligne de base corticale (via l'intégration dans le temps, éq. 5). Par conséquent, les stimuli *incongruents* avec l'humeur génèrent une erreur de prédiction plus grande et ont donc une probabilité d'encodage plus élevée.

**Mécanisme :** Les expériences négatives répétées déplacent la ligne de base vers le négatif :

$$B_{\text{ctx}}^{\text{négatif}} = \int_{t_1}^{t_2} \lambda e^{-\lambda(t_2 - \tau)} I_{\text{neg}}(\tau) \, d\tau < 0 \quad (39)$$

Quand un stimulus positif léger ($I_{\text{pos}}$) est rencontré :

$$N_{\text{incongruent}} = ||I_{\text{pos}} - B_{\text{ctx}}^{\text{négatif}}||^2 > N_{\text{congruent}} = ||I_{\text{neg}} - B_{\text{ctx}}^{\text{négatif}}||^2 \quad (40)$$

Le stimulus incongruent avec l'humeur est plus nouveau contre la ligne de base déplacée.

**Prédiction Nouvelle 5.5 :** P(WRITE | I_incongruent-humeur) > P(WRITE | I_congruent-humeur) (P5.5)

**Validation Computationnelle :** Simulation (Test 6) : Groupe expérimental (ligne de base négative) P(incongruent) = 0,86 vs P(congruent) = 0,12. Groupe contrôle (ligne de base neutre) : encodage égal (0,38 pour les deux).

**Condition de Falsification :** Si l'humeur améliore l'encodage des stimuli congruents, l'amorçage associatif est soutenu par rapport au mécanisme de déviation de la ligne de base.

**Alignement Phénoménologique :** Cela s'aligne avec l'expérience commune qu'un moment de joie inattendue dans une période de tristesse semble particulièrement poignant et mémorable — c'est une grande déviation par rapport à la ligne de base expérientielle actuelle.

### 5.6 La Double Fonction de la Résonance : Amplification et Éclipsement

**Le compte rendu standard :** La présence sociale affecte la mémoire via l'activation (amélioration émotionnelle ; Cahill & McGaugh, 1998) ou le contrôle de la source (les sources sociales sont distinctives ; Johnson et al., 1993). Ceux-ci prédisent des effets sociaux sur la force globale de l'encodage mais pas des effets différentiels pour l'information partagée vs. individuelle.

**Le compte rendu Ligne de Base–Déviation :** Les états résonants à κ élevé servent une double fonction : ils amplifient l'encodage des insights partagés (via la pondération par la précision) tout en supprimant l'encodage de l'information individuelle rencontrée immédiatement après (via l'ajustement attentionnel au contexte dyadique).

**Mécanisme Partie A : Amplification Résonante (Information Partagée)**

Dans un état à κ élevé, la précision γ est élevée (éq. 18). Pour un insight partagé :

$$N_{\text{eff}}^{\kappa_{\text{élevé}}} = (1 + \gamma_c \kappa_{\text{élevé}}) \cdot N_h > N_{\text{eff}}^{\kappa_{\text{faible}}} = (1 + \gamma_c \kappa_{\text{faible}}) \cdot N_h \quad (41)$$

**Mécanisme Partie B : Éclipsement Résonant (Information Individuelle)**

Un κ élevé représente un système attentionnel calibré pour le traitement dyadique. Quand l'interaction se termine et que l'individu rencontre de l'information en solo, le seuil du filtre est temporairement élevé :

$$\theta_{\text{filtre}}^{\text{post-}\kappa_{\text{élevé}}} = \theta_{\text{filtre}}^{\text{base}}(1 + \omega \kappa_{\text{final}}) \quad (42)$$

où ω est un paramètre d'ajustement. Cela supprime l'encodage de l'information qui ne correspond pas au contexte dyadique à haute précision.

**Prédiction Nouvelle 5.6 :**
- P(WRITE_partagé | κ_élevé) > P(WRITE_partagé | κ_faible) (P5.6a)
- P(WRITE_solo-après | κ_élevé) < P(WRITE_solo-après | κ_faible) (P5.6b)

**Validation Computationnelle :** Simulation (Test 7 Unifié) :
- Partie A (Amplification) : Dyade κ-élevé P(insight partagé) = 0,995, Dyade κ-faible = 0,802
- Partie B (Éclipsement) : Dyade κ-élevé P(tâche solo) = 0,031, Dyade κ-faible = 0,436

**Condition de Falsification :** Si la résonance améliore l'encodage uniformément (pas d'interaction entre κ et type d'information), le mécanisme de pondération par la précision est falsifié.

### 5.7 La Cécité Déontologique : La Malédiction de l'Intimité

**Le compte rendu standard :** Les modèles de Théorie de l'Esprit (TdE) (Premack & Woodruff, 1978 ; Baron-Cohen et al., 1985) suggèrent que comprendre autrui implique de construire un modèle interne de ses croyances et intentions. Meilleure TdE → meilleure prédiction. Les relations à long terme devraient améliorer la précision prédictive.

**Le compte rendu Ligne de Base–Déviation :** Dans les dyades à κ élevé et de longue durée, la ligne de base de l'Agent A est si fortement façonnée par l'influence de l'Agent B que A commence à utiliser sa propre ligne de base (B_A) comme proxy inexact de la ligne de base de B (B_B). Cela conduit à la **cécité déontologique** : l'Agent A devient systématiquement *moins bon* pour prédire la réaction de l'Agent B à des sujets véritablement nouveaux qui sortent de leur histoire partagée.

**Mécanisme :** Pour une dyade à κ élevé avec une histoire partagée étendue mais des expériences privées divergentes :

$$B_A = \int [\text{histoire partagée} + \kappa \cdot D_{B \to A}] \, d\tau + \int I_A^{\text{privé}} \, d\tau \quad (43)$$

$$B_B = \int [\text{histoire partagée} + \kappa \cdot D_{A \to B}] \, d\tau + \int I_B^{\text{privé}} \, d\tau \quad (44)$$

L'histoire partagée crée une similarité superficielle, mais les expériences privées créent une divergence cachée. L'erreur de prédiction dans le modèle de A pour B est :

$$\text{Erreur} = |N_{\text{actuel}}^B - N_{\text{prédit}}^{A \to B}| \quad (47)$$

**Prédiction Nouvelle 5.7 :** Erreur_κ-élevé(sujet nouveau) > Erreur_κ-faible(sujet nouveau) (P5.7)

Les partenaires de longue date sont moins bons pour prédire les réactions de l'autre à des sujets véritablement nouveaux hors de leur domaine partagé.

**Validation Computationnelle :** Simulation (Test 8 Révisé) : Erreur de prédiction dyade κ-élevé = 17,37, dyade κ-faible = 0,69 (différence de 25×).

**Condition de Falsification :** Si les partenaires de longue date montrent une précision prédictive universellement améliorée, l'influence D-constitutive est falsifiée.

### 5.8 Dissociation Neurologique des Signaux d'Erreur

**Le compte rendu standard :** L'erreur de prédiction est typiquement traitée comme un signal unitaire calculé dans des réseaux d'erreur indépendants du domaine (cortex cingulaire antérieur, insula ; Ullsperger et al., 2014). Les erreurs sur le monde et les erreurs sur les agents devraient engager les mêmes circuits neuronaux.

**Le compte rendu Ligne de Base–Déviation :** Le mécanisme D affiné postule deux voies distinctes d'erreur de prédiction :

- Erreur_I : Erreur de prédiction sur le monde, calculée dans les cortex sensoriels
- Erreur_D : Erreur de prédiction sur l'agent, calculée dans le réseau TdE (jonction temporo-pariétale, cortex préfrontal médian)

**Mécanisme :** Deux conditions de « bug » (*glitch*) activent sélectivement chaque voie :

- Bug du monde : I_observé ≠ I_prédit, mais B_B,observé ≈ B_B,inféré

$$\text{Erreur}_I = ||I_{\text{observé}} - I_{\text{prédit}}||^2, \quad \text{Erreur}_D \approx 0 \quad (48)$$

- Bug social : I_observé ≈ I_prédit, mais B_B,observé ≠ B_B,inféré

$$\text{Erreur}_I \approx 0, \quad \text{Erreur}_D = ||B_{B,\text{observé}} - B_{B,\text{inféré}}||^2 \quad (49)$$

**Prédiction Nouvelle 5.8 :** Dissociation : Erreur_I_bug-monde ≫ Erreur_D_bug-monde ; Erreur_D_bug-social ≫ Erreur_I_bug-social (P5.8)

**Validation Computationnelle :** Simulation (Test 9) : Bug monde : Erreur_I = 25,0, Erreur_D = 0,0. Bug social : Erreur_I = 0,0, Erreur_D = 16,0. Double dissociation claire confirmée.

**Condition de Falsification :** Si l'IRMf montre une activation chevauchante pour les deux types de bugs, le modèle à double voie est falsifié.

### 5.9 L'Immunité Déontologique : Le Déficit de TdE comme Protection Épistémique

**Le compte rendu standard :** Les déficits de Théorie de l'Esprit (par exemple, dans les conditions du spectre de l'autisme ; Baron-Cohen et al., 1985) sont universellement présentés comme des déficiences — des difficultés de cognition sociale qui réduisent la fonction adaptative.

**Le compte rendu Ligne de Base–Déviation :** Si la voie D est le canal de l'influence sociale, alors un agent avec une voie D compromise (computationnellement, κ ≈ 0) devrait être paradoxalement *immunisé* contre la pression sociale qui contredit des faits établis. Tandis que la ligne de base d'un agent neurotypique sera corrompue par l'influence sociale, la ligne de base de l'agent κ = 0 reste ancrée au signal I initial, basé sur les faits.

**Mécanisme :** Les deux agents initialisés avec une ligne de base fondée sur les faits :

$$B_A(t_0) = B_B(t_0) = I_{\text{fait}} \quad (50)$$

Agent A (contrôle, κ = 0,3) soumis à la pression sociale d'un confédéré affirmant le point de vue opposé (D_confed) :

$$B_A^{\text{contrôle}}(t) = \int \lambda e^{-\lambda(t-\tau)} [I_{\text{fait}} + 0{,}3 \cdot D_{\text{confed}}(\tau)] \, d\tau \quad (51)$$

Agent B (déficit TdE, κ = 0) :

$$B_B^{\text{TdE-déficit}}(t) = \int \lambda e^{-\lambda(t-\tau)} I_{\text{fait}} \, d\tau \quad (52)$$

Au fil du temps : B_A dérive vers D_confed, B_B reste stable.

**Prédiction Nouvelle 5.9 :** ||B_final^TdE-déficit − I_fait|| < ||B_final^contrôle − I_fait|| (P5.9)

Les individus avec des déficits de TdE montrent une plus grande résistance à la corruption des croyances sociales quand la croyance initiale est fondée sur les faits.

**Validation Computationnelle :** Simulation (Test 10) : La ligne de base de l'agent contrôle a dérivé du fait (10,0) vers la position du confédéré (-10,0), terminant à 2,3. La ligne de base de l'agent avec déficit TdE est restée à 8,7 (décroissance mineure seulement, aucune dérive sociale).

**Note Éthique :** Ceci ne vise pas à valoriser les déficits de TdE ni à suggérer que l'immunité à l'influence sociale est globalement adaptative. Cela démontre que le mécanisme κ-D présente des *compromis fonctionnels* — ce qui est adaptatif pour l'apprentissage social devient inadapté quand l'apport social est trompeur.

**Condition de Falsification :** Si les individus avec déficit TdE montrent une susceptibilité égale ou supérieure à l'influence sociale, le mécanisme κ-D est falsifié.

### 5.10 Tableau Récapitulatif des Prédictions Centrales

| Prédiction | Affirmation Centrale | Formule Clé | Statut |
|---|---|---|---|
| P5.1 | L'espacement montre un U-inversé à 24h | R(Δt) culmine à la fenêtre de consolidation | Validé |
| P5.2 | Le changement de contexte affecte davantage les leurres que les items anciens | ΔP(leurre) > ΔP(ancien) | Validé |
| P5.3 | Le TSPT élève l'EP aux stimuli neutres | N_TSPT = 2-5× N_contrôle | Validé |
| P5.4 | Le RMD suit la ligne de base de travail, pas épisodique | r(B_trav, RMD) > 0,5 ; r(B_épi, RMD) < 0,2 | Validé |
| P5.5 | Les stimuli incongruents avec l'humeur sont mieux encodés | P(WRITE\|incongr.) > P(WRITE\|congr.) | Validé |
| P5.6a | La résonance amplifie les insights partagés | N_eff ∝ γ(κ) | Validé |
| P5.6b | La résonance éclipse l'encodage solo | P(solo\|post-κ-élevé) < P(solo\|post-κ-faible) | Validé |
| P5.7 | Les dyades κ-élevé sont moins bonnes en prédiction de sujets nouveaux | Erreur_κ-élevé > Erreur_κ-faible | Validé |
| P5.8 | Les erreurs monde et sociale se dissocient | Erreur_I ⊥ Erreur_D | Dissocié |
| P5.9 | Le déficit TdE protège contre la corruption des croyances | \|\|B_TdE − fait\|\| < \|\|B_ctrl − fait\|\| | Validé |

### 5.11 Tests Futurs Proposés (Avec Résultats Prédits)

**Test F1 : Manipulation de la Fenêtre de Consolidation par Privation de Sommeil**

- **Hypothèse :** Si la consolidation se produit pendant la fenêtre de 12-48h (spécifiquement pendant le sommeil ; Stickgold, 2005), alors la privation de sommeil immédiatement après l'encodage devrait empêcher le transfert de la ligne de base.
- **Design :** Encodage de liste de mots à t=0. Groupe A : sommeil normal. Groupe B : privation de sommeil 0-48h. Les deux groupes ré-étudient à t=24h. Test de rétention à t=168h.
- **Résultat Prédit :** R_A(168h) > R_B(168h), avec R_A ≈ 0,65, R_B ≈ 0,15 (F1)

**Test F2 : Violation Rythmique dans les États de Haute Résonance**

- **Hypothèse :** La structure temporelle (rythme des tours de parole) est prédite dans les états à κ élevé. Les interruptions intempestives devraient être plus perturbantes (Erreur_D plus grande, réponse physiologique accrue) dans les interactions à κ-élevé vs κ-faible.
- **Résultat Prédit :** ΔGSR_haute-cohérence > 2 × ΔGSR_basse-cohérence ; ΔCohérence_haute > ΔCohérence_basse (F2)

**Test F3 : Le Taux de Mise à Jour de la Reconsolidation Varie avec κ**

- **Hypothèse :** Le paramètre de mise à jour de la reconsolidation η (éq. 26) n'est pas fixe mais varie avec la force du couplage à la récupération. Récupération à κ-élevé → η plus grand → plus de mise à jour.
- **Résultat Prédit :** η_solo ≈ 0,1 ; η_thérapeutique ≈ 0,4 ; Distorsion_thérapeutique > 3 × Distorsion_solo (F3)
- **Implication Clinique :** La reconsolidation thérapeutique est plus efficace car les états à κ-élevé permettent une plus grande plasticité mnésique.

**Test F4 : L'Évolution de κ Suit la Fréquence des Signaux de Retour**

- **Hypothèse :** Les signaux de retour conversationnels (« mm-hmm », « d'accord », hochements de tête) sont des signaux explicites qui minimisent N_mutual, accélérant la croissance de κ (éq. 8).
- **Résultat Prédit :** dκ/dt ∝ f_retour ; r(f_retour, κ_final) > 0,6 (F4)

**Test F5 : La Perturbation de la Ligne de Base Augmente le Déjà Vu dans les Contextes de Haute Familiarité**

- **Hypothèse :** Le déjà vu se produit quand les signaux de ligne de base et de trace sont en conflit — la ligne de base dit « familier » (faible N_ctx) mais la bibliothèque de traces dit « nouveau » (N_sep élevé). La perturbation contextuelle devrait augmenter cette dissociation.
- **Résultat Prédit :** P(déjà vu)_contexte-discordant ≈ 0,35 ; P(déjà vu)_contexte-concordant ≈ 0,15 (F5)
