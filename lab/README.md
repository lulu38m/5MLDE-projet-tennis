# Labo d'entraînement

Un terrain d'essai, à côté du pipeline, pour pousser la prédiction des matchs ATP aussi loin que possible. Rien ici
n'est servi par l'API ni lancé par Prefect, et rien n'est tenu d'être réintégrable : on cherche le meilleur score,
proprement mesuré.

```sh
make lab-install                    # une fois : LightGBM, XGBoost, CatBoost, Optuna (groupe `lab` de pyproject)
make lab-reference                  # la référence à battre : le modèle actuel du pipeline, sur les mêmes saisons
make lab-run MODEL=lightgbm         # un essai, tracé dans MLflow (expérience lab-atp)
make lab-tune MODEL=catboost TRIALS=100
make lab-ensemble                   # moyenne des meilleurs essais de chaque famille (FIT=1 : poids appris)
make lab-leaderboard                # le classement : log loss et accuracy
make lab-final                      # le test final sur 2026, une fois les choix arrêtés
```

Sur Mac, si LightGBM ou XGBoost refuse de se charger (`Library not loaded: libomp.dylib`) : `brew install libomp`.

Familles : `elo`, `logreg`, `hgb`, `lightgbm`, `xgboost`, `catboost`. Jeux de features : `all`, `diffs`,
`no_stats`, `elo` (`lab/config.yaml`).

## Résultats (16 novembre 2026)

Mesurés ici, sur une machine à 2 cœurs ; à refaire sur le Mac, avec plus d'essais Optuna.

| Modèle | eval 2024–2025 : log loss | accuracy | holdout 2026 : log loss | accuracy |
|---|---|---|---|---|
| Modèle actuel du pipeline (LR sur ses features) | 0,6168 | 65,3 % | 0,6100 | 66,2 % |
| Elo seul | 0,6270 | 63,4 % | | |
| Régression logistique réglée | 0,6050 | 66,3 % | 0,5997 | 66,1 % |
| LightGBM réglé | 0,6050 | 66,0 % | 0,5935 | 67,3 % |
| CatBoost réglé | 0,6068 | 66,1 % | 0,5971 | 67,3 % |
| XGBoost réglé | 0,6056 | 65,8 % | 0,5941 | 67,4 % |
| **Ensemble, poids appris** | **0,6034** | **66,4 %** | **0,5952** | **66,7 %** |

L'ensemble (logreg 0,44, XGBoost 0,25, LightGBM 0,22, CatBoost 0,06) gagne environ 0,014 de log loss sur le
modèle actuel, sur les deux saisons d'évaluation comme sur 2026, jamais regardée avant le test final. L'accuracy
gagne environ un point sur 2024–2025 ; sur 2026 (1 437 matchs, ±1,2 point de bruit), les écarts entre modèles ne
sont pas significatifs. Le gain vient surtout des features : bien réglés, les quatre modèles se valent à 0,002 près.

## Comment on mesure

**Validation « walk-forward ».** Chaque saison est prédite par un modèle entraîné sur les seules saisons d'avant
(depuis 2011 ; 2010 sert d'échauffement aux Elo et aux statistiques). Trois groupes, fixés dans `lab/config.yaml` :

| Groupe | Saisons | Sert à |
|---|---|---|
| `tune` | 2021, 2022, 2023 | Optuna choisit ses hyperparamètres ici, et seulement ici |
| `eval` | 2024, 2025 | le classement des essais |
| `holdout` | 2026 (janvier à mai) | le test final, `make lab-final` ; le regarder pour choisir en ferait une saison `eval` |

**Deux métriques à égalité** : la log loss (les probabilités sont-elles justes ?) et l'accuracy (le vainqueur
est-il le bon ?). Optuna optimise la log loss : elle est continue et stable, alors que l'accuracy avance par sauts
d'un match et ferait choisir au hasard entre des essais proches. L'accuracy est mesurée et tracée partout.

**Ordre de grandeur du bruit.** Une saison compte environ 2 900 matchs : l'accuracy d'une saison a une incertitude
d'environ ±0,9 point, la moyenne des deux saisons `eval` d'environ ±0,6 point. Un écart d'accuracy de 0,3 point
entre deux essais n'est donc pas un résultat ; un écart de log loss de 0,002, mesuré sur les mêmes matchs, l'est
davantage.

## Ce qui change par rapport au pipeline

- **Features** (`lab/features.py`, environ 150 colonnes, calculées en 20 s puis mises en cache dans
  `state/lab/`) : Elo général, par surface, mélangé et « avec marge » (pondéré par la part de jeux gagnés) ;
  points gagnés au service et en retour, corrigés de la force de l'adversaire, en moyenne exponentielle, générale
  et par surface ; **chaîne de Markov** point → jeu → set → match qui en déduit une probabilité de victoire ;
  forme (10 derniers matchs, moyenne exponentielle, résultats contre l'attendu Elo, 52 semaines) ; fatigue
  (repos, matchs et minutes sur 7 et 30 jours, minutes déjà jouées dans le tournoi) ; expérience (surface,
  niveau, bilan en 5 sets et en set décisif, passé au tournoi) ; classement (rang, points, évolution sur 3 mois,
  meilleur rang) ; profil (âge, taille, gaucher, qualifié, invité, tête de série) ; face-à-face général et par
  surface ; contexte (surface, niveau, tour, format, taille du tableau).
- **Aucune fuite du futur** : les résultats d'une journée n'entrent dans l'état des joueurs qu'à la fin de la
  journée ; un test vérifie que les features d'un match ne changent pas quand on retire tout ce qui vient après.
- **Modèles symétriques** : chaque match est appris dans les deux sens et la prédiction est la moyenne de p(A, B)
  et de 1 − p(B, A). Sans cela, un arbre donnerait une probabilité différente selon l'ordre des joueurs.
- **Boostings** : arrêt anticipé sur les 12 derniers mois du jeu d'entraînement, puis réentraînement sur tout le
  jeu avec le nombre d'arbres trouvé (les saisons les plus récentes ne sont jamais perdues).
- **Optuna** cherche aussi le jeu de features et un poids de récence (les saisons récentes peuvent peser plus).
- **Ensembles** : moyenne des logits des meilleurs essais de chaque famille, ou combinaison apprise sur les
  saisons `tune` (`FIT=1`).

## Où voir les essais

- **MLflow**, expérience `lab-atp` : celui de la stack (`make up`, http://127.0.0.1:7282) s'il répond, sinon une
  base locale `state/lab/mlflow.db` que `make lab-mlflow` ouvre sur http://127.0.0.1:7283. Chaque essai : métriques
  par saison (`s2024.log_loss`…) et par groupe (`eval.accuracy`…), paramètres, importance des features,
  prédictions. Un réglage Optuna est un run parent, un run enfant par essai.
- **`make lab-leaderboard`** : le classement local (`state/lab/leaderboard.jsonl`), qui marche même sans MLflow.

## Idées pour aller plus loin

- **Cotes des bookmakers** (tennis-data.co.uk) : la meilleure feature connue, et la référence honnête (un modèle
  public bat rarement les cotes de clôture).
- **Matchs Challenger et qualifications** (fichiers `atp_matches_qual_chall_*` de Sackmann) : bien plus d'historique
  pour les jeunes joueurs, dont l'Elo est aujourd'hui mal estimé à leurs premiers matchs ATP.
- **Données plus fraîches** pour prédire de vrais matchs à venir : les fichiers verrouillés s'arrêtent fin mai 2026.
