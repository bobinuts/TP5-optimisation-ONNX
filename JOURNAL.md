# JOURNAL — Séance 5 : optimiser et surveiller

Auteur : PELLICANO Théo — M2 Industrialisation de l'IA dans le Cloud — YNOV Campus Montpellier — 2026-2027

## Tableau de décisions

Rempli le 17/11/2026 à 12:23, avant la partie 1 et avant toute mesure de la séance : aucune mesure de variante, de dérive ni de latence n'avait été lue. Les valeurs ne sont pas modifiées ensuite ; toute modification serait datée et déclarée ici.

| Décision | Contrainte | Ma valeur | Pourquoi cette valeur |
|---|---|---|---|
| Suffixe personnel | identique à S1 | `tp42` | Celui du registre `acrdermascantp42`, conservé depuis S1. |
| Écart max. de probabilité, « même modèle » | ONNX contre PyTorch | **1 × 10⁻⁵** (absolu) | C'est l'exigence de concordance que je m'étais fixée en S4 (mesure 16). Le seuil servi vaut 5,8 × 10⁻⁵ : un écart de l'ordre du seuil lui-même ferait basculer des décisions, il doit rester nettement en dessous. |
| Rappel minimal d'une variante | au seuil servi | **0,95** | Le rappel visé depuis S2. Sur les 52 malignes de validation, 50 détectées donnent 0,9615 ; 49 donnent 0,9423 et font refuser la variante : aucune maligne de moins que PyTorch n'est tolérée. |
| Décisions changées max., même version | entier | **0** | Une variante qui change une seule décision n'est plus le même modèle : elle reçoit un nouveau numéro de version et une model card revue. |
| Décisions changées max., variante acceptée sous un nouveau numéro *(ligne ajoutée)* | entier | **40**, sans maligne perdue | Environ 1 % des 4 136 vignettes de validation. Au-delà, le changement de comportement devient de l'ordre de grandeur d'un ré-entraînement et la variante est refusée. |
| Covariable surveillée en priorité | reçue par l'API | `tbp_tile_type` (éclairage) | Le scénario du TP est un centre qui passe en lumière polarisée ; deux modalités seulement, donc peu de bruit, et c'est la dérive que la mesure 20 de S4 a documentée. |
| Statistique de dérive | calculée par `derive.py` | PSI | Imposé. |
| Seuil d'alerte du PSI | avant la mesure | **0,25** | Lecture usuelle du PSI : en dessous de 0,10 la population est stable, entre 0,10 et 0,25 le changement est modéré, au-dessus de 0,25 il est important. C'est une convention de la pratique, pas une loi : je la retiens parce que je n'ai aucune mesure de mon flux pour en proposer une autre. |
| Seuil de surveillance du PSI | sous l'alerte | **0,10** | Premier niveau de la même convention : on regarde, on ne réveille personne. |
| Fenêtre glissante | en inférences | **200** | Assez grande pour que le bruit d'échantillonnage d'une covariable à deux modalités reste faible, assez petite pour qu'un flux dérivé la remplisse vite : 400 appels par phase en local donnent deux fenêtres complètes. À 1 appel par seconde sur Azure, 200 appels couvrent 3 min 20 s, moins que le pas de 5 minutes du tableau de bord. |
| Levier de dérive provoquée | `eclairage`, `age`, `site` | **`eclairage`** ; second levier (mesure 9) : **`age`** | L'éclairage est le scénario du cours. L'âge (70 ans et plus) touche une covariable à nombreuses modalités et peut déplacer la sortie du modèle, ce qui permet de comparer entrées et sortie (question 5.3). |

## Relevés de début de séance

| Relevé | Valeur |
|---|---|
| Crédit Azure (Cost Management, 17/11/2026 à 12:21) | 86 € disponibles sur 90 € ; coûts de novembre 2,56 € ; expire le 06/10/2027 (323 jours) |
| Registre `acrdermascantp42` | tags `1.0.0` (`sha256:9269a14b…`) et `2.0.0` (`sha256:6170676b…`) présents |
| Poste | Windows, `amd64`, 12 CPU logiques, Docker Desktop 29.8.2 avec 12 CPU et 7,4 Gio de mémoire, 80,9 Go de disque libres |
| Paquet S4 | `shasum` de `modele.pt` = `sha256_poids` du meta (`1ab1761fb5bce32c…`) |

## Question 0.1 — Pourquoi fixer le seuil d'alerte avant de regarder la moindre mesure de dérive ?

Un analyste qui choisit son seuil après avoir vu ses courbes place sans le vouloir la ligne juste au-dessus de la plus forte fluctuation qu'il a observée : le seuil épouse le bruit de cet échantillon et non le risque que l'on veut couvrir. Le résultat est un dispositif sans fausse alerte sur le passé, qui ne garantit rien sur le futur, et dont on ne peut plus dire s'il aurait détecté une vraie dérive.

## Question 0.2 — Pourquoi sacrifier la quantization avant la surveillance ?

La quantization ne protège que du temps et du coût (un fichier plus petit, peut-être un calcul plus rapide) : sans elle, le service reste correct et seulement plus lent, alors que sans surveillance un modèle qui ne voit plus le monde sur lequel il a appris continue de répondre 200 et de manquer des lésions malignes sans que personne ne le sache.

## Décisions prises en cours de séance

| Date | Décision |
|---|---|
| 17/11/2026 à 12:40 | Mesures 3 et 4 lues. Critères du tableau appliqués sans modification. **ONNX float32** : écart max de probabilité 1,07 × 10⁻⁶ (≤ 1 × 10⁻⁵), rappel 0,9615 (≥ 0,95), 1 décision changée : plus que 0, donc ce n'est plus le modèle 2.0.0, mais moins que 40, donc la variante est acceptée sous un nouveau numéro. **ONNX int8** : rappel 0,9808 (≥ 0,95) mais 219 décisions changées (> 40) : refusée. Variante déployée : **ONNX float32, version 2.1.0**. |
