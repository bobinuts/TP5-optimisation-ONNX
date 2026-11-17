# SURVEILLANCE — Séance 5 : optimiser et surveiller

Auteur : PELLICANO Théo — M2 Industrialisation de l'IA dans le Cloud — YNOV Campus Montpellier — 2026-2027

Les décisions de ce document (fenêtre glissante de **200** inférences, seuil d'alerte du PSI **0,25**, seuil de surveillance **0,10**, leviers `eclairage` puis `age`) sont celles du tableau de `JOURNAL.md`, datées du 17/11/2026 à 12:23 et non modifiées. Le PSI est celui de `derive.py` : Σ (q − p) · ln(q / p), proportions plafonnées à 10⁻⁴ en dessous. Version servie : **2.1.0** (ONNX float32). Les « appels » sont des appels à `/predict` ; « délai d'alerte » est le nombre d'inférences entre le premier appel dérivé et la première fenêtre glissante à PSI ≥ 0,25.

## Partie 5.1 — La distribution de référence

### Mesure 7 — la référence (`reference_covariables.json`)

Source : jeu d'entraînement du manifeste de S4 (`jeu == "entrainement"`), **16 158 vignettes**.

| Covariable | Modalités et proportions | Taux d'absence |
|---|---|---|
| `tbp_tile_type` | `3D: white` 0,533853 ; `3D: XP` 0,466147 | 0 |
| `age_tranche` (les trois plus fréquentes) | 60-69 : 0,314101 ; 50-59 : 0,248489 ; 70-79 : 0,179201 (puis 40-49 : 0,134401 ; 30-39 : 0,058508 ; 80-89 : 0,04748 ; 20-29 : 0,017571 ; 10-19 : 0,000249) | 0,006746 |
| `anatom_site_general` (les deux plus fréquentes) | `posterior torso` 0,309293 ; `lower extremity` 0,266171 (puis `anterior torso` 0,211764 ; `upper extremity` 0,181188 ; `head/neck` 0,031585) | 0,018319 |
| `sex` | `male` 0,710404 ; `female` 0,289596 | 0,013739 |
| Nombre de vignettes de la référence | 16 158 | |

### Question 5.1 — Pourquoi la part de `3D: white` de la référence n'est pas celle de la mesure 20 ?

La mesure 20 (52,80 %) porte sur le manifeste **entier**, soit 20 294 vignettes : 16 158 d'entraînement à 53,39 % de `white` et 4 136 de validation à 50,51 %. Le mélange pondéré redonne 52,80 %. La référence ne prend que l'entraînement : 53,39 %. Je choisis le jeu d'entraînement parce que la dérive est un écart entre ce que le modèle a **appris** et ce qu'il reçoit : le manifeste entier mêle des vignettes qu'il n'a pas vues, qui ont servi à choisir le seuil, avec celles qu'il a vues. L'archive entière serait un mauvais repère pour une autre raison : le manifeste garde tous les positifs et un tirage de négatifs (1,45 % de positifs), l'archive n'en compte que 0,135 %, et le flux rejoué par `rejouer_flux.py` est tiré dans l'archive : prendre l'archive comme référence rendrait la phase conforme conforme par construction, ce qui est un raisonnement circulaire.

### Question 5.2 — PSI à la main pour le flux de S2

Flux = `train-metadata.csv` complet de S2 (ma mesure 20) : q(white) = 0,2871 et q(XP) = 0,7129 ; référence : p(white) = 0,533853 et p(XP) = 0,466147.

- `white` : (0,2871 − 0,533853) × ln(0,2871 / 0,533853) = (−0,246753) × (−0,62029) = **0,15306**
- `XP` : (0,7129 − 0,466147) × ln(0,7129 / 0,466147) = 0,246753 × 0,42482 = **0,10483**
- **PSI ≈ 0,2579**, juste **au-dessus** de mon seuil d'alerte de 0,25 (de 0,008).

Ce que cela m'apprend : pour une covariable à deux modalités, il faut un écart énorme pour atteindre 0,25. Pour une part de `white` qui baisse à q (le reste en `XP`), le PSI vaut : 0,50 → 0,0046 ; 0,45 → 0,0282 ; 0,40 → 0,0724 ; 0,35 → 0,1387 ; 0,30 → 0,2299. Le flux de S2 est une population très différente de la référence (la moitié de moins de `white`) et ne franchit le seuil que de justesse : une baisse de la part de `white` de 53 % à 40 % (une dérive tout à fait réelle) resterait sous 0,10. Le seuil de 0,25 est une convention de la pratique, et mon choix était prudent pour les fausses alertes, mais il laisse passer des dérives modérées ; le seuil de surveillance à 0,10 sert à les voir.

## Partie 5.2 — Répétition locale

Conteneur à journal vierge (`dermascan-api:2.1.0`, `ENVIRONMENT=local`, sel neuf), 400 appels conformes tirés uniformément dans l'archive, puis 400 appels dérivés ; les journaux bruts sont `journal_local.jsonl` (levier éclairage) et `journal_local_age.jsonl` (levier âge), les séries de PSI `sorties/psi_fenetres_local_eclairage.csv` et `sorties/psi_fenetres_local_age.csv`. Tous les appels ont répondu 200.

### Mesure 8 — la dérive provoquée en local, levier `eclairage` (fenêtre 200, seuil 0,25)

| Covariable | PSI max en flux conforme | PSI de la dernière fenêtre | Délai d'alerte (inférences après le début de la dérive) |
|---|---|---|---|
| `tbp_tile_type` | 0,0152 | **4,9885** | **86** (le PSI franchit 0,10 après 59) |
| `age_tranche` | 0,0646 | 0,0731 | aucune alerte |
| `anatom_site_general` | 0,0444 | 0,0174 | aucune alerte (maximum 0,1217 pendant la phase dérivée, 0,10 franchi après 44 appels) |
| `sex` | 0,0134 | 0,0149 | aucune alerte |
| Taux de `a_examiner` : conforme / dérivé | **53,0 %** (212/400) / **57,0 %** (228/400) | | |

### Mesure 9 — second levier : `age` (≥ 70 ans)

| Relevé | Ma valeur |
|---|---|
| Levier | `age` (tirage restreint à `age_approx` ≥ 70) |
| Covariable qui alerte, et délai | `age_tranche` : PSI de la dernière fenêtre **7,0481**, alerte après **53 appels** (0,10 franchi après 28) |
| Autre covariable qui bouge sans alerter, et son PSI | `sex` : PSI **0,1857** en fin de phase (0,10 franchi après 295 appels, jamais 0,25) : la part d'hommes passe de 71 % (référence) à 84,3 % dans les fenêtres dérivées ; `tbp_tile_type` monte à 0,0666 (maximum 0,1109), `anatom_site_general` à 0,0392 |
| Taux de `a_examiner` : conforme / dérivé | **53,0 %** (212/400) / **55,75 %** (223/400) |

### Question 5.3 — Un tableau de bord qui ne suivrait que la sortie aurait-il vu la dérive ?

Pour l'éclairage : non. Le taux de `a_examiner` passe de 53,0 % à 57,0 %, soit quatre points pour un flux qui passe de 53 % à 100 % de `3D: XP` (PSI de 4,99). Sur des fenêtres de 200 appels, le taux oscille déjà entre 0,495 et 0,585 en flux conforme et entre 0,505 et 0,605 en flux dérivé : les deux plages se recouvrent, et un taux de 57 % se voit aussi en flux conforme. Pour l'âge : non plus (53,0 % contre 55,75 % ; fenêtres de 0,495 à 0,585 contre 0,515 à 0,570). Les entrées ont pourtant franchi 0,25 dans les deux cas. Conclusion : on surveille les entrées parce qu'elles réagissent à la **cause** (une population qui change) avant que le modèle n'en montre un effet, et on surveille la sortie parce qu'elle voit l'**effet**, y compris celui d'une dérive qu'aucune covariable ne reflète (une covariable que je ne journalise pas, ou un changement de relation entre les images et la maladie) : aucune des deux ne remplace l'autre.

### Question 5.4 — Une covariable a-t-elle dépassé le seuil de surveillance en flux conforme ?

En local (400 appels conformes, 201 fenêtres), aucune : la plus haute est `age_tranche` à 0,0646, puis `anatom_site_general` à 0,0444, `tbp_tile_type` à 0,0152 et `sex` à 0,0134. Sur Azure (600 appels conformes, 401 fenêtres glissantes), **`anatom_site_general`** a atteint **0,1072**, donc au-dessus de 0,10 (12 fenêtres sur 401), sans jamais approcher 0,25. Ce n'est pas une dérive : le flux conforme est tiré uniformément dans l'archive, dont la composition est celle de la référence ; c'est du **bruit d'échantillonnage**. Le bruit moyen d'un PSI mesuré sur n appels sans dérive vaut à peu près (k − 1) / n pour k modalités : plus une covariable a de modalités, plus elle est bruyante, et l'ordre observé suit cet ordre (8 tranches d'âge > 5 sites > 2 modalités pour l'éclairage et le sexe). Avec une fenêtre de 200, 0,10 est un niveau que le bruit peut atteindre pour une covariable à cinq modalités dont l'une est rare (`head/neck`, 3,2 % de la référence, donc 6 appels attendus par fenêtre) ; c'est pour cela que 0,10 ne déclenche qu'une surveillance, pas une alerte.

### Question 5.5 — Fenêtre deux fois plus petite

**Compromis en une phrase** : une fenêtre de 100 divise le délai d'alerte par deux mais double le bruit, donc fait apparaître des fausses alertes.

**Mesuré** (mêmes journaux, fenêtre 100, seuil 0,25 ; calcul fait à part, sans écraser `sorties/psi_fenetres.csv`) : le délai d'alerte de l'éclairage passe de 86 à **40** appels ; mais en flux conforme `anatom_site_general` atteint **0,2672** (6 fenêtres sur 301 au-dessus de 0,25 : de **fausses alertes**), `age_tranche` monte à 0,1573 (97 fenêtres sur 301 au-dessus de 0,10) ; avec le levier âge, `anatom_site_general` alerte après 1 appel, ce qui est faux. Avec la fenêtre de 200, aucune fausse alerte.

**Délai théorique pour l'éclairage** : une fenêtre de taille W qui reçoit, à la fraction f de son contenu, un flux 100 % `3D: XP` (le reste étant à la composition de la référence) a q(white) = 0,533853 × (1 − f). Le PSI franchit 0,10 à f = **0,2937** (soit 58,7 appels pour W = 200) et 0,25 à f = **0,4556** (soit **91,1 appels** pour W = 200, 45,6 pour W = 100). Mesuré en local : 59 et 86 appels ; sur Azure : alerte après 94 appels. Le petit écart avec la théorie vient de la composition aléatoire de la partie conforme de la fenêtre.

## Partie 5.3 — Sur Azure

Déploiement : image `dermascan-api:2.1.0` publiée dans `acrdermascantp42` (digest `sha256:d3afac9be257e8cac4b9cc62da23769de4f493f8ec3a964d5bae0a415d759edf`, identique à l'identifiant de l'image locale), groupe `rg-dermascan-s5-tp42` (tags `projet=dermascan seance=s5 a_detruire=2026-11-17`), plan B1 `plan-dermascan-s5-tp42`, application `dermascan-api-s5-tp42`, réglages `WEBSITES_PORT=8000`, `PORT=8000`, `MODEL_VERSION=2.1.0`, `ENVIRONMENT=recette`, `IMAGE_DIGEST`, `INFERENCE_LOG_SALT` (sel neuf, non reproduit), identité managée avec `AcrPull`. Premier `/health` à 200 : **55,8 s** après le `restart`. `/model/info` en ligne : `captures/03_model_info_en_ligne.json`. Espace `law-dermascan-s5-tp42` (rétention **30 jours**, le minimum de la tarification par Go : les journaux d'une recette n'ont pas à durer), paramètre de diagnostic `diag-s5` (`AppServiceConsoleLogs` et `AppServiceHTTPLogs`). Recyclage du conteneur après le diagnostic : nouveau `hostname` vu **74 s** après sa création (16:06:30 → 16:07:44). Rejeu à 1 appel par seconde : 600 appels conformes de 16:17:04 à 16:27:08, puis 600 appels dérivés (`eclairage`) de 16:27:08 à 16:37:13 (premier appel dérivé journalisé à 15:27:12,8 UTC).

### Mesure 10 — la surveillance en ligne

| Relevé | Ma valeur |
|---|---|
| Délai entre un appel et son apparition dans Log Analytics | **40 s au minimum, 74 s en médiane, 213 s au maximum** (`ingestion_time()` moins `TimeGenerated`, sur les 1 201 lignes) |
| Lignes exportées / appels envoyés | **1 201 / 1 201** (1 appel de test, 600 conformes, 600 dérivés) ; `derive.py` en retient 1 200 (le test n'a pas de phase) |
| PSI de l'éclairage par fenêtre de 5 min, phase conforme (min – max) | **0,0037 – 0,0154** (fenêtres 15:15 et 15:20 UTC, 173 et 300 appels) ; la fenêtre 15:05 ne contient que l'appel de test (1 appel) et n'est pas interprétable |
| Première fenêtre de 5 min au-dessus du seuil d'alerte, et heure de début de la dérive | fenêtre **15:25 – 15:30 UTC** (16:25 – 16:30 à Paris) : PSI **0,4279** (127 appels conformes et 168 dérivés) ; la dérive a commencé à **15:27:12,8 UTC** ; fenêtres suivantes : 4,9885 (15:30, 15:35) |
| Délai d'alerte de `derive.py` sur le journal Azure (inférences) | **94** appels après le début de la dérive (fenêtre glissante 200, seuil 0,25) ; `age_tranche` 0,1638, `anatom_site_general` 0,0259 et `sex` 0,0036 en dernière fenêtre, sans alerte |
| Capture du tableau de bord (requêtes 1 à 4) | `captures/05_tableau_de_bord.png` |

Les PSI par fenêtre de 5 minutes ci-dessus sont ceux de la **requête 3** exécutée sur l'espace par `az monitor log-analytics query` (`sorties/psi_fenetres_5min_requete3.json` : 15:05 → 4,2293 sur 1 appel, 15:15 → 0,0154 sur 173, 15:20 → 0,0037 sur 300, 15:25 → 0,4279 sur 295, 15:30 → 4,9885 sur 300, 15:35 → 4,9885 sur 132). Je les ai recoupés avec un recalcul fait à part depuis `journal_azure.jsonl` (même formule, même référence, fenêtres rangées d'après l'horodatage de chaque ligne) : mêmes valeurs. La requête 3 s'affiche aussi sur la capture. Les PSI de `derive.py` sur le journal Azure, par covariable : `tbp_tile_type` max conforme 0,0434, dernière fenêtre 4,9885 ; `age_tranche` 0,0646 ; `anatom_site_general` 0,1072 (voir Q5.4) ; `sex` 0,0134.

Dans la fenêtre de 5 minutes en cours de remplissage, la requête 3, recalculée appel par appel à part, franchit 0,25 à **15:28:52 UTC**, soit **100 appels dérivés** (100 s) après le premier appel dérivé, un peu plus tôt que les 94 inférences de `derive.py` parce que le contenu de la fenêtre n'est pas le même (fenêtre de temps contre fenêtre d'appels).

### Question 5.6 — Délai total entre le premier patient dérivé et l'alerte vue par un humain

Premier patient dérivé : 15:27:12,8 UTC. Termes de ce délai :

1. **Accumulation** : il faut que la fenêtre contienne assez d'appels dérivés pour que le PSI franchisse 0,25 : environ 100 appels à 1 appel par seconde, donc **100 s** (15:28:52) ; c'est le délai physique d'une détection par fenêtre, d'autant plus long que le débit est faible.
2. **Ingestion** : la ligne n'est interrogeable qu'après le circuit stdout, App Service et Log Analytics : **40 à 213 s, 74 s en médiane** sur mes 1 201 lignes.
3. **Rafraîchissement du tableau de bord** : l'actualisation automatique de mon tableau de bord est **désactivée** (on le voit sur la capture) ; sans clic, la vignette garde ses anciennes valeurs.
4. **Le regard humain** : personne n'est alerté, puisque l'alerte Azure Monitor automatique n'est pas demandée ; l'alerte existe seulement quand quelqu'un ouvre le tableau de bord.

Au mieux (actualisation immédiate, ingestion médiane) : 100 + 74 ≈ **175 s**, soit environ 3 minutes (15:30:06) ; avec l'ingestion maximale, environ **5 min 15 s**, avant le premier clic de rafraîchissement. Celui que je peux réduire **sans rien changer au seuil** est le quatrième : remplacer le regard humain par une règle d'alerte de journaux planifiée sur la requête 3 (évaluation toutes les 5 minutes, notification par courriel), ou au moins activer l'actualisation automatique. L'ingestion ne dépend pas de moi, et le premier terme ne se réduit qu'en changeant la fenêtre, donc en changeant le compromis de la question 5.5.

## Partie 6 — Nettoyage

### Mesure 12 — fin de séance

| Contrôle | Mon observation |
|---|---|
| Nombre et types de ressources du groupe avant destruction | **3** dans `rg-dermascan-s5-tp42` : un plan App Service Linux B1 (`Microsoft.Web/serverFarms`), l'application (`Microsoft.Web/sites`) et l'espace Log Analytics (`Microsoft.OperationalInsights/workspaces`) ; le paramètre de diagnostic est attaché à l'application et n'est pas listé à part. L'abonnement avait alors deux groupes : le registre et celui de la séance |
| `az group exists` sur le groupe de séance | **`false`**, à la septième sonde (17:01:01), environ 1 min 10 s après le lancement de la suppression ; l'attribution `AcrPull` du registre a été retirée avant (aucune attribution n'est listée après), et l'espace Log Analytics a été supprimé avec `--force` |
| Le groupe du registre est-il toujours là ? Tags du dépôt `dermascan-api` | **oui** (`az group exists` : `true`) ; tags **`1.0.0`, `2.0.0`, `2.1.0`**, digests `sha256:9269a14b…`, `sha256:6170676b…` et `sha256:d3afac9b…` |
| Le tableau de bord partagé survit-il à la destruction du groupe ? Ses graphiques affichent-ils encore quelque chose ? | **Non observé.** Avant la destruction, `az resource list --resource-type Microsoft.Portal/dashboards` ne trouvait **aucun** tableau de bord dans l'abonnement, et aucun groupe `dashboards` n'existait : le tableau `dermascan-s5` que j'ai créé dans le portail n'est donc pas une ressource Azure partagée, mais probablement un tableau de bord privé de mon profil, ce que je n'ai pas vérifié. Je n'ai pas relevé l'état de ses tuiles après la destruction |
| Crédit restant, et écart avec la séance 4 | **Non relevé** : le portail n'a pas affiché la valeur après la séance. Repère : 86 € sur 90 € au début de la séance (12:21). Estimation, non mesurée : le plan B1 a existé environ 1 h 45 (de 15:16 à 17:01, heure de Paris) à environ 0,018 USD par heure (tarif écrit dans mes scripts), soit de l'ordre de 0,03 USD, et l'ingestion de 1 201 lignes de journal (677 ko) est négligeable |

Contrôle d'extinction : `az group exists` sur le groupe de séance renvoie `false` ; `docker ps -a` ne liste aucun conteneur ; `AcrPull` est retiré ; les images `dermascan-api:1.0.0`, `2.0.0` et `2.1.0` restent sur le poste (pas de `docker system prune -a`).

### Question 6.1 — Le tableau de bord épinglé n'est pas dans le groupe de séance : que montre-t-il après la destruction de l'espace, et que garder pour que la surveillance survive aux séances ?

Ce qui suit est un raisonnement, pas une observation (voir la mesure 12). Une tuile de tableau de bord ne contient que le texte de sa requête et l'identifiant de l'espace Log Analytics qu'elle interroge : les données ne sont pas dedans. Le tableau de bord n'étant pas détruit avec le groupe, ses tuiles restent affichées mais ne peuvent plus résoudre l'espace supprimé : elles s'ouvrent en erreur ou vides. Et comme j'ai supprimé l'espace avec `--force`, il n'y a pas de corbeille pour récupérer l'historique : les 1 201 lignes ne sont plus que dans `journal_azure.jsonl`.

Pour que la surveillance survive aux séances, il faut garder trois choses qui ne partagent pas le cycle de vie de la séance : (1) un **espace Log Analytics permanent**, dans un groupe à part (`rg-dermascan-surveillance`, par exemple) avec une rétention choisie, qui reçoit les journaux de chaque version déployée ; sans historique, il n'y a pas de comparaison dans le temps ; (2) la **définition du tableau de bord et des requêtes en dépôt** (`requetes.kql`, `reference_covariables.json` avec sa date et sa source, et le tableau de bord exporté en modèle ARM), pour le recréer d'une commande dans un groupe permanent ; (3) une **règle d'alerte** sur la requête 3, qui supprime le regard humain du délai de la question 5.6. Il faut aussi refaire la référence à chaque nouveau modèle : elle décrit les données d'entraînement d'une version donnée.
