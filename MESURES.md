# MESURES — Séance 5 : optimiser et surveiller

Auteur : PELLICANO Théo — M2 Industrialisation de l'IA dans le Cloud — YNOV Campus Montpellier — 2026-2027

Toutes les valeurs ci-dessous viennent des transcripts de `hors_rendu/sorties/` (copiés ou cités dans `captures/` et `sorties/` quand ils sont demandés) ; les mesures de S4 sont recopiées de mon `MESURES.md` de S4. Poste : Windows, `amd64` natif, 12 CPU logiques, Docker Desktop 29.8.2 (12 CPU, 7,4 Gio). Dates et heures : 17/11/2026, heure de Paris.

Deux écarts avec l'énoncé, relevés en cours de séance :
- L'énoncé parle de **1 058 vignettes de validation** ; `comparer.py` lit le manifeste de S4 et en retient **4 136** (`jeu == "validation"`, dont **52 malignes**). Toutes les mesures 3 et 4 portent sur ces 4 136 vignettes.
- L'énoncé de la question 4.5 dit que le Dockerfile ne diffère de celui de S4 que par `MODEL_PATH`. Par rapport à **mon** Dockerfile de S4 (base `python:3.12.15-slim`), il diffère beaucoup plus (voir 4.5).

## Partie 1 — Point de départ

### Mesure 1 — recopiée de mon `MESURES.md` de S4

| Grandeur (S4) | Ma valeur |
|---|---|
| Empreinte de `modele.pt` identique à `sha256_poids` | **oui** (`1ab1761fb5bce32c…`, contrôlé à nouveau en début de séance) |
| Taille de l'image `dermascan-api:2.0.0` | **1 633 328 340 octets** (1,63 Go) |
| Taille du fichier de modèle | **44 780 171 octets** (44,8 Mo) |
| Démarrage jusqu'au premier `/health` à 200 | **8,83 s** (10,2 s sous `--network none`) |
| Mémoire du conteneur au repos | **290,6 MiB** |
| Latence `/predict` p50 / p95, local, 50 appels | **29,6 / 37,7 ms** avec les 12 cœurs ; **169,5 / 272,5 ms** sous `--cpus 1` |
| Mesure 19 : palier le plus chargé sur B1, CPU du plan | 40 utilisateurs : **8,43 req/s**, p50 **1 100 ms**, p95 **4 700 ms**, p99 12 000 ms, 0 échec ; CPU du plan **93 à 100 %** |
| Mesure 20 : part de `3D: white` | **52,80 %** dans le manifeste (20 294 vignettes), **52,73 %** dans l'archive ; flux de S2 (`train-metadata.csv` complet) : **28,71 %** de `white`, 71,29 % de `XP` |

### Question 1.1 — Pourquoi une comparaison faite avec tous les cœurs du portable peut conclure à l'inverse de B1

Avec 12 cœurs, un moteur qui parallélise bien ses opérateurs paraît rapide, et le classement entre deux moteurs dépend de leur facilité à se répartir ; avec un seul vCPU cet avantage disparaît et le moteur qui multiplie les fils perd. Mes propres mesures en donnent un exemple : mon service S4 passe de 29,6 ms (12 cœurs) à 169,5 ms sous `--cpus 1` (facteur 5,7), et le même ONNX Runtime passe de 16,6 ms avec 1 fil à 83,6 ms avec 8 fils sous `--cpus 1`.

### Question 1.2 — Pourquoi les 20 premiers appels sont jetés

Le premier appel paie des coûts d'initialisation qui ne reviennent plus : (1) les allocations de mémoire et la préparation des tampons et du pool de fils du moteur, qui se font à la demande au premier passage dans chaque opérateur ; (2) le chargement paresseux des pages du fichier de modèle depuis le disque et le remplissage des caches du processeur, ainsi que l'initialisation des bibliothèques d'image et de calcul. Je n'ai pas isolé la part de chacun de ces effets ; je n'ai mesuré que la latence après l'échauffement.

## Partie 2 — Export ONNX

### Mesure 2 — l'export

| Relevé | Ma valeur |
|---|---|
| Taille de `modele.onnx`, comparée à `modele.pt` | **44 786 210 octets** contre 44 780 171 octets : **+6 039 octets (+0,013 %)**. Le fichier tient en un seul bloc (`external_data=False`) |
| Empreinte de `modele.onnx` (16 premiers caractères) | **`961eea7b58c24270`** (complète : `961eea7b58c242707c3d8d8b803650fe10ae04e897e63ec3ba5cb107db2b510d`) |
| Écart maximal de logit sur le lot de contrôle (4 images) | **4,8 × 10⁻⁶** (l'assertion du script tolère 10⁻⁴) |
| Second export : même empreinte ? | **oui**, trois exports successifs (dont celui qui produit l'int8) donnent la même empreinte. Durée d'un export : 11,0 s |

Les deux blancs du script sont `input_names=["entree"]` (nom lu dans `comparer.py` et `app.py`) et `external_data=False`.

### Question 2.1 — `reseau.eval()` avant l'export

Sans `eval()`, le réseau est exporté en mode entraînement : les couches `BatchNorm2d` (vingt dans un ResNet18) normaliseraient avec les statistiques du lot présenté au lieu de leurs moyennes et variances apprises, et un `Dropout` éventuel serait actif et aléatoire (le ResNet18 de torchvision n'en a pas ; ma tête est une couche linéaire seule). Le calcul exporté dépendrait alors des autres images du lot, et l'écart de logit contre le réseau en mode évaluation serait grand, très au-dessus du 10⁻⁴ toléré par l'assertion, au lieu des 4,8 × 10⁻⁶ mesurés.

### Question 2.2 — `check_model` ne suffit pas

`onnx.checker.check_model` vérifie que le fichier est un graphe ONNX **bien formé** : schéma, types, versions d'opérateurs, nœuds reliés, formes cohérentes. Il ne dit rien du calcul. L'assertion qui le suit exécute le graphe avec ONNX Runtime et le réseau PyTorch sur les mêmes entrées, et vérifie que les logits ne s'écartent pas de plus de 10⁻⁴ : c'est elle qui prouve qu'on a figé le même calcul. Un graphe peut être parfaitement valide et calculer autre chose.

### Question 2.3 — 3 × 128 × 128 fixé

Le modèle a été entraîné sur des vignettes de 128 × 128 et le seuil est calé sur les scores obtenus à cette taille : une autre taille donnerait d'autres probabilités sans erreur visible ; fixer les trois axes permet aussi au moteur de planifier ses tampons une fois pour toutes. Une image de 300 pixels ne pose pas de problème au service : `app.py` la redimensionne en 128 × 128 (filtre bilinéaire, comme à l'entraînement) avant de la donner au graphe. Si on donnait directement un tenseur 300 × 300, ONNX Runtime refuserait l'entrée pour cause de dimensions invalides.

## Partie 3 — Quantization et trois variantes

### Mesure 3 — les trois variantes (4 136 vignettes de validation, 52 malignes, seuil 5,793547 × 10⁻⁵)

| Variante | Taille (Mo) | p50 (ms) | p95 (ms) | Proba de contrôle | Rappel au seuil | Alertes | ROC-AUC | Écart max de proba | Décisions changées |
|---|---|---|---|---|---|---|---|---|---|
| PyTorch | 44,8 | 21,79 | 25,09 | 0,319664 | 0,9423 (49/52) | 2 288 | 0,8727 | 0 | 0 |
| ONNX float32 | 44,8 | **11,95** | **12,88** | 0,319664 | **0,9615 (50/52)** | 2 289 | 0,8727 | 1,07 × 10⁻⁶ (`ISIC_7285904`) | **1** |
| ONNX int8 | **11,3** | 12,56 | 13,55 | 0,330780 | 0,9808 (51/52) | 2 377 | 0,8743 | **0,1043** (`ISIC_7156396`, bénigne : 0,0764 → 0,1808) | **219** |

Un lot de 1, 20 appels de chauffe non comptés, 300 mesurés, un seul fil pour PyTorch comme pour ONNX Runtime. Image de contrôle `ISIC_0699524` : 0,319666 dans le meta de S4.

### Mesure 4 — ce que l'int8 change

| Relevé | Ma valeur |
|---|---|
| Décisions passées en `a_examiner` / en `surveillance` | **154 / 65** (soit 89 alertes de plus : 2 377 contre 2 288) |
| Malignes parmi les décisions changées | **2** (`ISIC_5328578` et `ISIC_6095295`), toutes deux passées en `a_examiner` : aucune maligne perdue |
| Rapport des probabilités int8 / PyTorch : médiane, minimum, maximum | **1,108 / 0,1757 / 9,029** (moyenne 1,237) |

Pour ONNX float32, le même relevé donne : 1 décision changée, vers `a_examiner`, la maligne `ISIC_6095295`; rapport des probabilités ONNX / PyTorch de 0,999978 à 1,000031 (médiane 1,000001).

### Question 3.1 — Le rappel de PyTorch n'est pas celui du meta

Le meta de S4 donne un rappel de 0,9615 (50/52) ; `comparer.py` mesure **0,9423 (49/52)** pour PyTorch. La maligne responsable est `ISIC_6095295`, la plus proche du seuil : sa probabilité PyTorch vaut 5,793528 × 10⁻⁵, **sous** le seuil 5,793547 × 10⁻⁵ (écart de 1,9 × 10⁻¹⁰), alors qu'ONNX la calcule à 5,793608 × 10⁻⁵, **au-dessus**. C'est la même lésion qu'à ma question 4.1 de S4, où elle tombait exactement sur le seuil et basculait au moindre arrondi lors d'un aller-retour par un fichier texte ; ici elle bascule à cause de l'ordre des opérations flottantes entre deux moteurs (ou entre le GPU de l'entraînement et le CPU de mon poste). Conclusion : le rappel est un entier sur 52 malignes qu'une seule lésion à 10⁻¹⁰ du seuil fait varier de 0,019 ; ONNX « améliore » le rappel de PyTorch uniquement parce que cette lésion tombe de l'autre côté, et l'int8 paraît encore meilleur (0,9808) en changeant 219 décisions. Une variante se juge donc sur les **décisions changées** (leur nombre, leur sens, les malignes concernées), pas sur le rappel seul.

### Question 3.2 — ONNX float32 est-il plus rapide que PyTorch ?

Sur mon poste, oui : p50 de 11,95 ms contre 21,79 ms (facteur 0,55) et p95 de 12,88 ms contre 25,09 ms, à un fil. Cela ne prouve rien pour B1 : la vitesse d'un moteur dépend du modèle de processeur et de son jeu d'instructions vectorielles, que les noyaux d'ONNX Runtime et de PyTorch exploitent différemment ; mon poste est `amd64` natif (pas d'émulation), mais B1 est un vCPU partagé d'un processeur que je ne connais pas, et sur un poste ARM faisant tourner une image `linux/amd64` en émulation le classement pourrait s'inverser. Je ne connaîtrai la vitesse sur B1 qu'avec la mesure 11. Le gain qui reste acquis même si la latence ne bougeait pas est celui de l'installation : l'image n'embarque plus torch ni torchvision (couche `pip` de 176 Mo contre 1,02 Go en S4, image de 419 Mo contre 1,63 Go, construction de 86 s contre 2 998 s). L'int8, lui, n'apporte aucun gain de latence ici (12,56 ms contre 11,95 ms pour le float32) : seulement la taille du fichier, divisée par quatre.

### Question 3.3 — Mes critères du tableau de décisions, appliqués sans modification

- **ONNX float32** : écart de probabilité 1,07 × 10⁻⁶ (≤ 10⁻⁵) ; rappel 0,9615 (≥ 0,95) ; **1** décision changée : c'est plus que 0, donc **ce n'est plus le modèle 2.0.0**, mais c'est moins que 40 et la maligne concernée passe en `a_examiner` (aucune perdue) : la variante est **acceptée sous un nouveau numéro**.
- **ONNX int8** : rappel 0,9808 (≥ 0,95) mais écart de probabilité de 0,1043 (≫ 10⁻⁵) et **219** décisions changées (> 40) : **refusée**. Elle n'est pas non plus le modèle 2.0.0.

| Décision de déploiement | Ma valeur |
|---|---|
| Variante déployée | **ONNX float32** (`modele.onnx`, empreinte `961eea7b58c24270…`) |
| Version du modèle servi | **2.1.0** |

### Question 3.4 — Ce qu'un nouveau modèle exige avant de servir de vrais patients

Deux choses à refaire : (1) **valider le rappel au seuil et le seuil lui-même sur des données indépendantes** de celles qui ont servi à choisir ce seuil, avec le moteur réellement servi, et rédiger à nouveau la model card avec ces chiffres ; (2) **faire revue et validation cliniques** de ce que la variante change (les décisions qui basculent, une par une) et du plan de surveillance, par les personnes qui engagent leur responsabilité. Aujourd'hui j'ai fait la première **partiellement** : j'ai mesuré rappel, alertes et décisions changées avec le moteur ONNX, mais sur les 4 136 vignettes de validation qui ont déjà servi à fixer le seuil en S4, donc sans données indépendantes ; je n'ai fait aucune revue clinique.

## Partie 4 — Une image sans PyTorch et un journal d'inférence

### Question 4.1 — Les quatre points du diff de `app.py`

- **Moteur** : `torch` + `torchvision` (reconstruction du ResNet18, chargement du `state_dict`, `eval()`) deviennent une `onnxruntime.InferenceSession` sur `modele.onnx`, fournisseur CPU, `intra_op_num_threads = ORT_THREADS` (1 par défaut) ; le prétraitement passe en NumPy, avec le même redimensionnement bilinéaire en 128 × 128.
- **Covariables** : `/predict` accepte quatre champs de formulaire **facultatifs** à côté de l'image (`age_approx`, `sex`, `anatom_site_general`, `tbp_tile_type`), validés contre un domaine fermé ; l'âge n'est journalisé qu'en tranche de dix ans ; une valeur hors domaine renvoie 400 « covariable invalide ».
- **Ligne de journal** : chaque appel écrit sur la sortie standard une ligne JSON au format de S3 (`ts`, `request_id`, `model_version`, `image_digest`, `patient_ref` pseudonymisé par sha256(sel:id) tronqué à 16 caractères seulement si `INFERENCE_LOG_SALT` existe, probabilité, seuil, décision, covariables, `inference_ms`, `http_status`, `input_schema_version`, `env`) ; les refus 400 sont journalisés aussi.
- **`/model/info`** : s'ajoutent `format` (`onnx`), `fichier_servi`, `sha256_fichier_servi` (empreinte du fichier lu au démarrage) et `covariables_acceptees`.

### Question 4.2 — Pourquoi l'option 3

- **Contre l'option 1 (surveiller l'API tabulaire de S2)** : ce n'est pas le service qui répond aux patients ; ses entrées sont des colonnes tabulaires, pas les images que le CNN reçoit, et une dérive observée là ne dirait rien de ce que voit le modèle de S5.
- **Contre l'option 2 (une statistique d'image côté serveur, comme la luminance)** : elle ne dépend d'aucun client, mais c'est un substitut : une luminance moyenne peut ne pas bouger quand un centre passe en lumière polarisée, elle ajoute du calcul sur le chemin de la réponse, et il faudrait recalculer la référence sur les 16 158 images alors que les vraies covariables existent déjà dans `train-metadata.csv`.
- **Faiblesse de l'option 3** : les valeurs viennent du **poste de consultation**, pas de nous. S'il ment (il envoie toujours `3D: white`), la surveillance est aveugle sans que rien ne le montre ; s'il les omet, la covariable disparaît du calcul et seul le taux d'absence, que je dois alors surveiller aussi, le trahit.

### Question 4.3 — Modalité inconnue refusée, modalité absente acceptée

Une modalité **inconnue** (`UV`) est refusée parce qu'elle est un signal ambigu qu'on ne veut pas laisser entrer dans le journal : une faute de saisie, un nouveau dispositif ou un client cassé ; acceptée, elle créerait une modalité absente de la référence qui fausserait le PSI de la fenêtre sans que personne le voie ; refusée, l'erreur est visible à la source et le domaine du journal reste celui de `train-metadata.csv`. Une modalité **absente** est acceptée parce que ces champs sont facultatifs : refuser une prédiction à un patient parce qu'une métadonnée manque serait pire que de perdre un point de surveillance. Côté tableau de bord, l'absence apparaît comme « absent » dans la requête 2 et n'entre pas dans le PSI de la requête 3 (filtre `isnotempty`), au lieu de le polluer.

### Question 4.4 — Pourquoi ne pas ajouter le SDK Application Insights aujourd'hui

- **Tiré de 4.3** (démarrage hors ligne) : l'image doit démarrer et servir sous `--network none` ; un SDK de télémétrie cherche à joindre un point de collecte, ajoute une dépendance réseau à l'exécution, et le `print` sur la sortie standard fonctionne partout, en local comme sur Azure.
- **Tiré de la règle zéro CVE** : un SDK entraîne des dizaines de paquets Python (`azure-core`, `opentelemetry`, `requests`…) qui s'ajoutent aux 84 paquets que `docker scout` vient de passer à 0 vulnérabilité, et chaque paquet ajouté élargit la surface à scanner et à corriger.

### Question 4.5 — `chown -R` et Dockerfile

**Ce que le Dockerfile change réellement.** L'énoncé dit qu'il ne diffère de celui de S4 que par `MODEL_PATH` ; par rapport à **mon** Dockerfile de S4, le diff est bien plus grand : la base passe de `python:3.12.15-slim` à **Wolfi** (`cgr.dev/chainguard/wolfi-base`, figée par digest) avec Python installé par `apk` ; `pip` est installé puis retiré dans le même `RUN` (`apk add py3.12-pip && pip install … && apk del py3.12-pip`) ; l'utilisateur est créé par `adduser -D -u 10001 appuser` (au lieu de `useradd --create-home --shell /usr/sbin/nologin`) ; `MODEL_PATH` devient `/app/model/modele.onnx` ; `exec` disparaît du `CMD` ; et la ligne `chown -R appuser:appuser /app` n'existe plus. Ma colonne S4 n'a donc pas le même point de départ que celle de l'énoncé.

**Ce que coûterait le `chown -R` placé après `COPY model/`.** Mon image S4 le montre : dans `docker history dermascan-api:2.0.0`, la couche `useradd … && chown -R appuser:appuser /app` pèse **44,9 Mo**, soit la taille de `COPY model/` (44,8 Mo) : changer le propriétaire d'un fichier le recopie en entier dans une nouvelle couche (le système de couches ne sait pas stocker une simple métadonnée), donc le modèle est stocké **deux fois**. Pour S5 cela ajouterait environ 44,8 Mo aux 419 Mo (+ 11 %) ; la couche `adduser` de S5 pèse 41 kB.

**Pourquoi son absence ne casse rien.** L'application ne fait que **lire** `/app` (code et modèle, lisibles par tous) : elle n'y écrit rien, puisque `PYTHONDONTWRITEBYTECODE=1` supprime les fichiers `.pyc` et que les journaux vont sur la sortie standard ; gunicorn écrit ses fichiers temporaires dans `/tmp`, ouvert à tous ; et `adduser -D` crée le répertoire personnel de `appuser`. La preuve est expérimentale : le conteneur tourne sous `USER appuser` et a répondu 200 hors ligne, en local, et à 1 201 + 2 200 appels sur Azure, sans erreur.

### Question 4.6 — Pourquoi `**/*.pt` et pas `*.pt`

Dans un `.dockerignore`, `*.pt` ne vaut que pour la **racine** du contexte : `*` ne franchit pas un `/`, donc `model/modele.pt` serait envoyé au démon et copié dans l'image. `**/*.pt` vaut à tous les niveaux. Pour les lignes héritées de S2, un `*.csv` seul laisserait entrer `model/manifeste_echantillon.csv` et tout CSV rangé dans un sous-dossier, comme `train-metadata.csv`, qui contient des identifiants de patients.

### Mesure 5 — l'image S5

| Relevé | Ma valeur |
|---|---|
| Code retour de `docker scout cves --exit-code`, décompte C / H / M / L | **0** ; **0 C / 0 H / 0 M / 0 L**, 84 paquets, 125 Mo de contenu (capture `captures/01_docker_scout.txt`). La porte « zéro CVE » est franchie sans exception écrite ; en S4 : 37 CVE (0 / 1 / 7 / 29) |
| Durée de construction | **86,3 s** (`--pull --platform linux/amd64 --provenance=false`, étapes `apk` et `pip` exécutées) ; S4 : 2 998,2 s |
| `probabilite_maligne` de l'image de contrôle, servie par le conteneur | **0,319664**, `a_examiner` |
| Probabilité de contrôle de la même variante (mesure 3) et écart | 0,319664 ; **écart 0** (6 décimales). Contre le meta de S4 (0,319666) : 2 × 10⁻⁶ |
| `sha256_fichier_servi`, égal à l'empreinte de la variante | **oui** (`961eea7b…`) |
| `patient_ref` journalisé contient-il `IP_0000001` en clair | **non** : `0a3277019c29254c` ; aucun sel dans la ligne |
| Modalité inconnue `tbp_tile_type=UV` | **400** : `{"detail":"tbp_tile_type inconnu : 'UV'","error":"covariable invalide"}` |

Ligne de journal complète d'un appel avec covariables (`captures/02_predict_local.txt`) :

```json
{"ts": "2026-11-17T11:47:55.974Z", "request_id": "c6e507b0-f6e2-417e-956d-8c53ae20a222", "model_version": "2.1.0", "image_digest": null, "patient_ref": "0a3277019c29254c", "probabilite_maligne": 0.319664, "seuil_applique": 5.793547245973642e-05, "rappel_cible": 0.95, "decision": "a_examiner", "covariables": {"age_tranche": "60-69", "sex": "male", "anatom_site_general": null, "tbp_tile_type": "3D: white"}, "inference_ms": 13.0, "http_status": 200, "input_schema_version": "2.0", "env": "local"}
```

(`image_digest` est `null` en local : la variable `IMAGE_DIGEST` n'est posée que sur Azure.)

### Mesure 6 — S4 contre S5, sur mon poste

| Grandeur | S4 | S5 |
|---|---|---|
| Taille de l'image | 1 633 328 340 octets (1,63 Go) | **418 674 190 octets** (419 Mo), facteur 3,9 |
| Taille du fichier de modèle | 44 780 171 octets | 44 786 210 octets |
| Démarrage jusqu'au premier `/health` à 200 (hors ligne) | 10,2 s (8,83 s avec réseau) | **1,1 s** hors ligne (0,71 s avec réseau, sondes de 0,25 s) |
| Mémoire au repos | 290,6 MiB | **155,1 MiB** (156,8 MiB après les appels) |
| Latence `/predict` p50 / p95, 50 appels, tous les cœurs | 29,6 / 37,7 ms | **17,1 / 18,5 ms** |
| Latence p50 / p95 sous `--cpus 1`, `ORT_THREADS` par défaut (1) | — (S4 avec torch : 169,5 / 272,5 ms) | **16,6 / 17,1 ms** |
| Latence p50 / p95 sous `--cpus 1`, `ORT_THREADS=8` | — | **83,6 / 84,6 ms** |
| Architecture du poste | `amd64` natif | `amd64` natif |

Couches de l'image S5 (`docker history`) : Python installé par `apk` 53,7 Mo, couche `pip` 176 Mo, `COPY model/` 44,8 Mo.

### Question 4.7 — L'écart entre les deux dernières lignes

Sous `--cpus 1`, le conteneur dispose de l'équivalent d'un seul processeur en temps de calcul (quota de 100 ms par période de 100 ms), alors que l'ordonnanceur lui laisse voir mes 12 cœurs. Avec 8 fils, ONNX Runtime découpe chaque opérateur sur 8 fils qui se disputent ce quota : le travail utile ne diminue pas, il s'y ajoute les changements de contexte et la synchronisation, et le quota consommé en parallèle est suivi d'attentes imposées. Résultat mesuré : 83,6 ms contre 16,6 ms, soit un facteur 5. La cause exacte (attente active des fils, limitation du quota) est une explication plausible que je n'ai pas vérifiée. `app.py` fixe un fil par défaut parce que le service ne doit pas supposer le nombre de cœurs qu'on lui laisse. Ce réglage compte partout où le conteneur a un quota de CPU plus petit que le nombre de cœurs visibles : le plan App Service B1 (1 vCPU) de cette séance, et tout déploiement du projet avec une limite de CPU (conteneur à 1 vCPU, pod à `limits.cpu`) ; avec plusieurs workers `gunicorn`, c'est le produit (workers × `ORT_THREADS`) qui ne doit pas dépasser les vCPU alloués.

## Partie 5.4 — Le plan tient-il, maintenant ?

Locust 2.47.0, `locustfile_s4.py` inchangé (poids 10 / 1 / 1 pour `/predict`, `/health`, `/model/info`, `wait_time = between(1, 5)`, image de contrôle de S4), `--headless -r 5 -t 3m`, deux paliers de 10 puis 40 utilisateurs (ceux de ma mesure 19), pause de 60 s entre les deux. Palier 1 : 15:47:48 → 15:50:49 UTC ; palier 2 : 15:51:49 → 15:54:50 UTC. Contrôle préalable : 10 appels `/predict` à 200. Ligne de base réseau : `time_connect` médian **43,7 ms** (min 34,1 ; max 130,2), contre 51,2 ms en S4 et 49,0 ms en S3. Les fichiers non retouchés sont `s5_palier1_*.csv`, `s5_palier2_*.csv` et les rapports `captures/04_rapport_locust_palier1.html` et `captures/04_rapport_locust_palier2.html`.

### Mesure 11 — S4 contre S5 sur B1, mêmes paliers (lignes `Aggregated`)

| Palier | S4 : débit, p50, p95 (mesure 19) | S5 : débit, p50, p95 | S5 : CPU max du plan |
|---|---|---|---|
| 1 (10 utilisateurs) | 2,90 req/s, 170 ms, 1 900 ms | **3,26 req/s, 88 ms, 260 ms** | **25 %** (S4 : 70 %) |
| 2 (40 utilisateurs) | 8,43 req/s, 1 100 ms, 4 700 ms | **9,77 req/s, 530 ms, 3 500 ms** | **99 %** (S4 : 100 %) |

Échecs : **0** sur 582 requêtes au palier 1 et **0** sur 1 751 au palier 2. `POST /predict` seul : palier 1 : 2,75 req/s, p50 90 ms, p95 270 ms, p99 530 ms (491 requêtes) ; palier 2 : 8,19 req/s, p50 540 ms, p95 3 500 ms, p99 6 300 ms (1 467 requêtes). Pour mémoire, S3 : palier 1 : 3,19 req/s, p50 90 ms, p95 400 ms ; palier 2 : 11,46 req/s, p50 180 ms, p95 1 800 ms.

CPU du plan (`az monitor metrics list`, moyenne par minute, UTC) : palier 1 : 25, 24, 23 % (15:48 à 15:50) ; palier 2 : 31, 13, 57, 81 % (15:51 à 15:54), puis **99 %** à 15:55, la minute qui suit la fin du test. Au repos, avant le test, le plan affichait déjà 12 à 16 %. La mémoire du plan est restée entre 72 et 76 % (S4 : 80 à 82 %). Comme en S4, la métrique est en retard sur la charge (31 % puis 13 % pendant un palier en cours, puis la montée à 99 % après sa fin) : je lis « CPU saturé au palier 2 », sans pouvoir dire à la minute près à quel moment, et le maximum de 99 % est mesuré après la fin du test.

Limites : un seul passage par palier, matériel mutualisé, poste sur Wi-Fi ; le taux de montée en charge est `-r 5` aux deux paliers (en S4 : `-r 2` puis `-r 8`), ce que demande l'énoncé ; les CPU de S4 et de S5 ne se comparent pas à la minute près.

### Question 5.7 — L'engagement de S3 tient-il avec S5 ?

**Oui, à 10 utilisateurs.** L'engagement de S3 est « p95 de `POST /predict` inférieur à 450 ms jusqu'à 10 utilisateurs simultanés, moins de 1 % d'échecs » : S5 donne un p95 de **270 ms** pour `/predict` (260 ms sur l'ensemble) et **0 échec sur 491**, avec une marge de 1,7 fois ; S4 ne le tenait pas (2 000 ms). C'est mieux que le palier de S3 lui-même (p95 de 400 ms). Il ne tient **pas** au-delà : à 40 utilisateurs, le p95 est de 3 500 ms avec un CPU du plan à 99 % ; l'engagement reste « jusqu'à 10 utilisateurs », et le monter demanderait plus de vCPU, pas un autre moteur.

**Part du moteur et part de l'int8.** Tout le gain revient au moteur : l'int8 est refusée (219 décisions changées, question 3.3), donc elle ne sert pas, et même servie elle n'aurait rien apporté à la latence, puisque mesurée à un fil elle n'est pas plus rapide que le float32 (12,56 ms contre 11,95 ms, mesure 3) ; son seul gain est la taille du fichier. Le gain du moteur, avec mes chiffres de la mesure 6, est de 29,6 à 17,1 ms au p50 avec tous les cœurs, et surtout de 169,5 ms (S4, un CPU) à 16,6 ms (S5, un CPU, un fil), soit un facteur 10 pour le calcul seul. Sur B1, le p50 de bout en bout passe de 170 à 88 ms et le p95 de 1 900 à 260 ms : l'amélioration est moins grande que le facteur 10 du calcul, parce que la durée d'une requête comprend aussi le réseau (un `time_connect` de 44 ms), l'envoi de l'image et le frontal de la plateforme, que l'export n'a pas changés, et parce que sous charge la file d'attente d'un worker unique amplifie tout écart. Le p95 s'effondre davantage que le p50 (÷ 7) parce que c'est la file d'attente qui fait le p95 : moins de temps de CPU par requête, c'est une file qui se forme beaucoup moins.
