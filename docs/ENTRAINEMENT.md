# Entraînement, suivi MLflow et registre

Le flow Prefect `entrainement-atp` transforme les matchs **validés** (lots passés par Great Expectations) en
features, entraîne une grille de candidats en parallèle, garde le meilleur sur la validation, le juge sur le test et
l'enregistre dans le registre MLflow. Tout ce qui permet de comprendre, comparer et réentraîner un modèle est tracé.

## Lancer

```sh
make up          # mock_source, PostgreSQL, Prefect (http://127.0.0.1:4201), MLflow (http://127.0.0.1:7282)
make ingest      # il faut des matchs en base : au minimum l'amorçage 2010 → 2014
make train       # une campagne, 4 candidats à la fois (configs/train.yaml) ; `make train WORKERS=8`
make train-seq   # la même, un candidat à la fois : la référence pour mesurer le parallélisme
make mlflow      # ouvre l'interface MLflow (onglet « Model training » → Runs)
```

Le terminal affiche le bilan et le lien du run MLflow ; le run Prefect contient le même lien dans ses logs, et le
run MLflow pointe vers le run Prefect (tag `prefect.flow_run_url`).

## Déroulé : une méthode = une task Prefect

| Étape | Task Prefect | Rôle |
|---|---|---|
| 1 | `DatasetBuilder.build` | Lit les matchs validés dans PostgreSQL et calcule les features de chaque match joué |
| 2 | `TemporalSplitter.split` | Entraînement → validation (6 mois) → test (6 derniers mois) |
| 3 | `profil et dérive des données` | Jeux associés au run (`log_input`), instantané Parquet + SHA-256, profil, dérive |
| 4 | `baselines` | Elo pur (sans apprentissage) et classement seul |
| 5 | `PipelineFactory.create` × N | Prétraitement + modèle de chaque candidat |
| 6 | `Trainer.fit` × N | Ajustement sur l'entraînement seul, **en parallèle** |
| 7 | `Evaluator.evaluate` × N | Métriques sur entraînement, validation et test |
| 8 | `Evaluator.segments` × 5N | Métriques du test par surface, niveau, tour, écart de classement, Grand Chelem |
| 9 | `Evaluator.diagnostics` × N | Latence, taille, importance par permutation, pires prédictions, performance mois par mois, courbe d'apprentissage |
| 10 | `Reporter.log` × N | Le run enfant complet : métriques, courbes, graphiques, rapport HTML, modèle |
| 11 | `comparaison avec la production` | Le modèle en service (alias `production`, phase 5) sur le même test |
| 12 | `Registrar.register` | Nouvelle version du candidat retenu |

Chaque campagne repart de la base (pas de cache entre campagnes). `demo.sleep_s` (`configs/project.yaml`) ralentit
chaque étape pour la rendre lisible dans l'UI Prefect.

## Features (version 1)

Calculées par `tennis_mlops.features.FeatureBuilder`, **le même code** que l'API utilisera (phase 7) : pas
d'écart entre entraînement et service. Le builder rejoue l'historique jour par jour : les features d'un match sont
calculées avec l'état de **la veille**, puis les résultats du jour entrent dans l'état. Un match ne voit jamais son
propre résultat, ni ceux du même jour (testé en inversant tous les résultats futurs : les features ne bougent pas).

| Feature | Sens (joueur 1 − joueur 2) |
|---|---|
| `rank_log_ratio`, `points_log_ratio` | Écart de classement et de points ATP (échelle log), publiés avant le tournoi |
| `age_diff`, `height_diff` | Âge et taille (une taille hors 150-230 cm est traitée comme inconnue) |
| `elo_diff`, `surface_elo_diff` | Elo global et Elo de la surface du match, K = 250 / (matchs + 5)^0,4 |
| `elo_win_prob` | Probabilité Elo (moitié globale, moitié surface) |
| `experience_diff` | Nombre de matchs joués (log) |
| `form_diff` | Taux de victoire lissé sur les 10 derniers matchs |
| `activity_diff`, `rest_days_diff` | Matchs joués sur 365 jours, jours depuis le dernier match (plafonné) |
| `h2h_diff`, `h2h_total` | Face-à-face avant ce match |
| `best_of`, `surface`, `tourney_level`, `round`, `p1_hand`, `p2_hand` | Contexte du match |

Le label vaut 1 si le **joueur 1** gagne. Qui est joueur 1 est tiré de l'empreinte du `match_id` : la moitié du
temps, c'est le perdant. Les forfaits ne sont ni des exemples ni des mises à jour d'Elo.

## Protocole

- **Découpage temporel** à partir du dernier match en base : test = 6 derniers mois, validation = 6 mois avant,
  entraînement = de 2011 à la validation (2010 sert d'échauffement : l'Elo de tous les joueurs part de 1 500).
  Un découpage aléatoire mélangerait le futur dans l'entraînement et donnerait des scores flatteurs.
- **Sélection** sur le log loss de validation, **bilan** sur le test, qui ne sert jamais à choisir.
- **Baselines** sur les mêmes partitions : Elo pur, et régression logistique sur le seul écart de classement.
- **Refus** si une partition compte moins de 300 matchs (`split.min_rows`) : la campagne échoue clairement.

Grille par défaut (8 candidats) : régression logistique (C = 0,05 ; 1), gradient boosting (taux 0,03 / 0,1 ×
profondeur 3 / 6), forêt aléatoire (profondeur 8 / 14). Le prétraitement (imputation avec indicateurs de
valeur manquante, mise à l'échelle pour la logistique, one-hot tolérant aux modalités inconnues) est **dans** le
pipeline : MLflow le sauvegarde avec le modèle.

## Ce qui est tracé dans MLflow

### Où regarder

| Onglet MLflow | Ce qu'on y voit |
|---|---|
| **Artifacts → `rapport.html`** (candidat) | Le candidat sur une page : chiffres clés contre l'Elo pur, scores par partition et baselines, log loss mois par mois, graphiques, segments, plus grosses surprises, coûts, fiche modèle |
| **Artifacts → `rapport_campagne.html`** (campagne) | La campagne sur une page : candidats classés (le retenu surligné), baselines, le retenu dans le temps, comparaison à la production, données, dérive, durées, reproductibilité |
| **Model metrics** | Graphiques regroupés par section (voir ci-dessous) ; les séries `monthly/*` et `learning/*` s'affichent en courbes |
| **Overview** | Paramètres, tags (`best.*`, `registry.*`, `production.comparison`), jeux associés |
| Liste des runs, vue **Chart** | Comparer les candidats d'une campagne sur une métrique (`validation/log_loss`, `cost/fit_s`…) |

Les rapports HTML n'ont ni JavaScript ni ressource externe (graphiques intégrés en base64) : ils s'ouvrent aussi
hors MLflow, après téléchargement.

### Sections des métriques

MLflow range les graphiques de l'onglet « Model metrics » par préfixe (`section/nom`).

| Section | Run | Contenu |
|---|---|---|
| `train/`, `validation/`, `test/` | candidat | `log_loss`, `brier`, `accuracy`, `roc_auc`, `ece` |
| `vs_baselines/` | candidat | Écart de log loss à chaque baseline (`validation_log_loss_minus_elo` < 0 : le modèle fait mieux) |
| `segments/` | candidat | Log loss du test par segment (`segments/surface.Clay`, `segments/ecart_classement.200plus`…) |
| `cost/` | les deux | `fit_s`, `latency_1_row_p50_ms`, `latency_1_row_p95_ms`, `latency_1000_rows_ms`, `model_size_kb` ; `peak_rss_mb` sur la campagne |
| `monthly/` | candidat | **Courbes** mois par mois (validation puis test) : `log_loss`, `accuracy`, `log_loss_minus_elo`, `matches` ; un mois à moins de `min_segment_rows` matchs est écarté |
| `learning/` | candidat (gradient boosting) | **Courbes** `train_log_loss` et `validation_log_loss` à chaque itération : on y voit l'arrêt anticipé et le surapprentissage |
| `baseline_elo/`, `baseline_rang/` | campagne | Métriques des baselines sur validation et test |
| `best/` | campagne | Métriques du candidat retenu |
| `monthly_best/` | campagne | Les courbes mensuelles du retenu |
| `production/` | campagne | Le modèle en service sur le même test, et l'écart avec le retenu (à partir de la phase 5) |
| `parallel/` | campagne | `workers`, `training_wall_s`, `fit_sum_s`, `fit_sum_over_wall` |
| `duration/` | campagne | Durée de chaque étape (`dataset_s`, `training_wall_s`, `evaluation_and_logging_s`…) |
| `drift/` | campagne | `max_psi`, `features_over_alert` par rapport à la campagne précédente |

Les séries mensuelles portent la date du mois : dans « Model metrics », choisir l'axe X « Time (wall) » affiche les
mois du calendrier au lieu des numéros d'étape.

**Run parent « campagne »** (`tags.kind = campaign`)

| Catégorie | Contenu |
|---|---|
| Configuration | `config/train_resolved.yaml`, `config/project_resolved.yaml`, paramètres des features, graine, nb de workers |
| Données | 3 jeux associés (`atp-train`, `atp-validation`, `atp-test`, avec digest), `data/features.parquet` + SHA-256, `data/profile_train.json`, `data/provenance.json` (lots valides / en quarantaine, dernier lot, watermark), `data/drift_vs_previous.json` |
| Reproductibilité | sha Git (et état modifié), `uv.lock` + `pyproject.toml`, empreinte de `uv.lock`, version de Python, `dataset_version`, lien du run Prefect |
| Résultats | `rapport_campagne.html`, `candidates.csv`, `plots/candidates_validation.png`, `plots/monthly_best.png`, `evaluation/monthly_best.csv`, métriques des baselines et du retenu, comparaison à la production |
| Coûts | durée de chaque étape, durée réelle de l'entraînement vs somme des durées, mémoire maximale du processus |

**Run enfant par candidat** (`tags.kind = candidate`)

| Catégorie | Contenu |
|---|---|
| Paramètres | famille, hyperparamètres, graine, version et liste des features, bornes et tailles des partitions |
| Rapport | `rapport.html` : tout le candidat sur une page |
| Métriques | log loss, Brier, accuracy, ROC AUC, ECE sur train / validation / test ; écart aux baselines ; log loss par segment ; courbes mensuelles et d'apprentissage |
| Coûts | durée d'entraînement, latence d'une ligne (p50, p95), lot de 1 000 lignes, taille du modèle |
| Graphiques | calibration, ROC, matrice de confusion, distribution des probabilités, importance par permutation, log loss par segment, log loss mois par mois contre l'Elo, courbe d'apprentissage |
| Tables | `evaluation/` : segments, importance, calibration, performance mensuelle, courbe d'apprentissage (CSV) |
| Analyse d'erreurs | `evaluation/worst_predictions_test.csv` : les 50 plus grosses surprises |
| Fiche | `model_card.md` : usage prévu, données, métriques comparées aux baselines, limites |
| Modèle | pipeline complet, format **skops** (types scikit-learn déclarés de confiance, tout autre type refusé), signature, exemple d'entrée, dépendances exportées de `uv.lock` ; `predict` du modèle pyfunc renvoie les probabilités |

**Registre** : modèle `atp-winner`, une version par campagne, reliée à son run, avec les tags `campaign_run_id`,
`dataset_version`, `features_version`, métriques de validation et de test, `lifecycle=candidate`,
`qualification=pending`. La règle de mise en service, les alias et le rollback sont la phase 5.

Recharger une version dans un autre processus :

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:7282")
model = mlflow.sklearn.load_model("models:/atp-winner/1")  # pipeline complet
proba = model.predict_proba(features)[:, 1]  # features : sortie de FeatureBuilder
```

Vérifié : mêmes probabilités qu'en mémoire (écart 0) et même log loss de test que celui tracé.

## Parallélisme : mesures

Les candidats sont des tasks Prefect lancées avec `.map()` sur un `ThreadPoolTaskRunner`. Les calculs de
scikit-learn libèrent le verrou de Python, donc des threads suffisent ; chaque candidat a son propre run MLflow,
écrit avec son identifiant explicite (aucun run « actif » partagé entre threads).

Mesuré sur 20 783 matchs (2010 → 2016), machine à 2 cœurs :

| | `make train` (4 workers) | `make train-seq` (1 worker) |
|---|---:|---:|
| Entraînement des 8 candidats (`parallel/training_wall_s`) | 6,5 s | 8,3 s |
| Évaluation et journalisation | 56 s | 70 s |
| Campagne complète | 1 min 14 | 1 min 29 |

Le gain est limité ici par les 2 cœurs, et parce que le gradient boosting utilise déjà plusieurs cœurs à lui seul.
La « somme des durées » (`parallel/fit_sum_s`) est gonflée en parallèle (les candidats se partagent les cœurs) :
la bonne comparaison est la durée réelle des deux commandes. Sur le Mac, comparer `parallel/training_wall_s` des
deux dernières campagnes dans MLflow (ou la section « Durées » de leurs `rapport_campagne.html`).

## Résultats indicatifs

Même campagne (données jusqu'au 27/11/2016, test = juin → novembre 2016) :

| Modèle | log loss test | accuracy test | ROC AUC test |
|---|---:|---:|---:|
| Régression logistique C = 0,05 (retenue) | 0,600 | 68,2 % | 0,743 |
| Baseline Elo | 0,613 | 67,4 % | 0,730 |
| Baseline classement | 0,622 | 65,1 % | 0,717 |

Le modèle bat les deux baselines, de peu : le classement et l'Elo portent l'essentiel de l'information. Les
modèles d'arbres ne font pas mieux que la logistique sur ce volume. Mois par mois, il fait mieux que l'Elo pur sur 7
des 11 mois de validation et de test ; les mois où il fait moins bien sont des écarts de l'ordre de 0,01. Les
courbes d'apprentissage du gradient boosting montrent l'arrêt anticipé de scikit-learn : avec un taux de 0,1 et une
profondeur de 6, la perte sur la validation est minimale vers 35 itérations et l'entraînement s'arrête à 48, loin
des 300 autorisées ; avec un taux de 0,03, il va jusqu'à 150 à 210 itérations.

## Incidents

| Situation | Comportement |
|---|---|
| MLflow injoignable | Erreur explicite avant tout calcul (`serveur MLflow indisponible`), flow `Failed`, rien n'est publié |
| Trop peu de données | Flow `Failed`, run de campagne fermé en `FAILED` avec `failure.step = TemporalSplitter.split` et la cause |
| Un candidat échoue (ex. hyperparamètre invalide) | Son run est fermé en `FAILED` avec la cause, les autres candidats continuent |
| Aucun candidat n'aboutit | Flow `Failed`, tous les runs ouverts sont fermés en `FAILED` |
| Base créée en phase 3 | La colonne `tourney_id` (32 caractères, trop courte pour la Coupe Davis) est élargie automatiquement |
