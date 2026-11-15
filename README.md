# tennis-mlops — Projet final 5MLDE

Chaîne MLOps complète qui prédit la probabilité de victoire d'un joueur dans un match ATP. Les matchs historiques
sont rejoués dans le temps par une API simulée, ce qui permet de démontrer l'ingestion continue, les contrôles
qualité, le réentraînement, la mise en service et la supervision.

## Prérequis

- [uv](https://docs.astral.sh/uv/) ≥ 0.11 (installe Python 3.12 automatiquement)
- `make`, `git`
- Docker Desktop (Docker Compose v2)

## Installation

```sh
make setup      # uv sync --locked + hooks pre-commit
make check      # lint + typage + tests, comme la CI
make data       # télécharge les données à la version verrouillée (~45 Mo, sans compte ni token)
make up         # stack Docker : mock_source, PostgreSQL, Prefect, MLflow, worker, alertes (`make urls`)
make ingest     # un passage d'ingestion contrôlé par Great Expectations, suivi dans Prefect
make train      # campagne d'entraînement en parallèle, tracée dans MLflow, meilleur modèle enregistré
make demo       # la chaîne tourne seule : temps simulé, ingestion, réentraînement sur événement, alertes Teams
make predict    # une prédiction de l'API (Djokovic contre Nadal sur terre battue)
```

## Commandes

| Commande | Rôle |
|---|---|
| `make setup` | Installe l'environnement et les hooks Git |
| `make lint` | ruff (lint + format) en lecture seule |
| `make format` | Corrige automatiquement le style |
| `make typecheck` | mypy en mode strict |
| `make test` | Tests unitaires + couverture |
| `make check` | Tout ce que la CI vérifie |
| `make data` | Télécharge les données brutes à la version verrouillée et vérifie les SHA-256 |
| `make data-verify` | Revérifie les fichiers locaux contre `configs/data.lock.json` |
| `make data-profile` | Profile les données → `data/raw/profile.json` |
| `make data-lock` | (Mainteneurs) Résout les révisions actuelles des sources et régénère le verrou |
| `make up` / `make down` | Démarre (`--build --wait`) / arrête la stack Docker |
| `make logs` / `make ps` | Logs et état des services |
| `make mock` | Lance `mock_source` en local, sans Docker |
| `make clock [DATE=2019-06-01]` | Lit ou règle l'horloge simulée |
| `make advance DAYS=7` | Fait avancer le temps simulé |
| `make chaos MODE=503 FAILURES=2` | Incident contrôlé (`off`, `503`, `slow`, `corrupt`, `schema`) |
| `make urls` | Adresses des interfaces (mock_source, Prefect, MLflow, PostgreSQL, rapports GX) |
| `make ingest` | Un passage d'ingestion : mock_source → Great Expectations → PostgreSQL |
| `make lots` | Derniers lots ingérés : statut, volumes, contrôles en échec |
| `make gx-docs` | Rapports HTML de Great Expectations |
| `make db` | Console SQL sur la base du projet |
| `make train [WORKERS=8]` | Campagne d'entraînement : grille de candidats en parallèle, MLflow, registre, décision de mise en service |
| `make train-seq` | La même campagne, un candidat à la fois (référence pour mesurer le gain du parallélisme) |
| `make train-weak` | Campagne volontairement mauvaise : enregistrée puis refusée par la règle de mise en service |
| `make mlflow` | Ouvre l'interface MLflow |
| `make registry` | Versions du modèle, alias (`production`, `previous`, `staging`), décisions |
| `make promote VERSION=3 REASON="…"` | Mise en production manuelle, raison obligatoire |
| `make rollback REASON="…"` | Revient à la version `previous` |
| `make trace [VERSION=3]` | Chaîne de traçabilité d'une version : run MLflow → run Prefect → lots → source |
| `make demo` / `make demo-stop` | Lance / met en pause la simulation planifiée (un pas de 7 jours toutes les 2 min) |
| `make demo-status` | Déploiements Prefect, worker, derniers runs |
| `make simulate DAYS=14` | Un seul pas de simulation, sans planning |
| `make alerts` | Dernières alertes : envoyée, filtrée, répétée, sans webhook, en échec |
| `make alert-preview` / `make alert-test` | Cartes Teams d'exemple en JSON / UNE carte « 🧪 TEST » au webhook |
| `make predict P1=… P2=… SURFACE=…` | Une prédiction de l'API (par défaut Djokovic contre Nadal sur terre) |
| `make api-status` | API prête ou dégradée, version servie, historique, qualité en ligne |
| `make predictions` | Dernières prédictions : version, durée de la requête, vrai résultat |
| `make score` / `make feedback` | Prédire les matchs de la semaine / remonter les vrais résultats (faits aussi par `make demo`) |

## Organisation du code

| Dossier | Contenu |
|---|---|
| `src/tennis_mlops/` | Chaîne MLOps (package installable, typé) |
| `src/tennis_mlops/ingestion/` | Client HTTP de la source, lots, stockage PostgreSQL, archive brute |
| `src/tennis_mlops/validation/` | Suites Great Expectations et expectations custom |
| `src/tennis_mlops/features/` | `FeatureBuilder` : features calculées avant chaque match, partagées entraînement / API |
| `src/tennis_mlops/training/` | Jeu de données, découpage temporel, pipelines, évaluation, suivi MLflow |
| `src/tennis_mlops/registry/` | Registre : tags, règle de mise en service, promotion, rollback, journal, rechargement par alias |
| `src/tennis_mlops/flows/` | Flows Prefect (`ingest.py`, `train.py`, `simulation.py`) |
| `src/tennis_mlops/orchestration/` | Déploiements, automations, déclenchement par événement, journal `pipeline_events` |
| `src/tennis_mlops/alerts/` | Cartes Teams (Adaptive Cards) et service d'alertes `alert_sink` |
| `src/tennis_mlops/serving/`, `src/tennis_mlops/api/` | Service de prédiction (historique rejoué, modèle servi, traces) et son API FastAPI |
| `src/mock_source/` | Simulateur de source : API FastAPI de rejeu ATP avec horloge simulée et chaos |
| `configs/` | Configuration de l'expérience (YAML, versionnée) |
| `tests/` | Tests unitaires (`unit/`) puis d'intégration (`integration/`, marqueur `integration`) |
| `docs/` | `DONNEES.md` (provenance, licence), `MOCK_SOURCE.md` (rejeu), `INGESTION.md` (contrôles), `ENTRAINEMENT.md`, `TAGGING.md` (registre, mise en service, rollback), `ORCHESTRATION.md`, `ALERTES.md`, `API.md` |
| `Dockerfile`, `compose.yml` | Image unique du projet et stack locale (ports publiés sur 127.0.0.1 uniquement) |
| `.env.example` | Variables d'environnement (URLs, secrets) — copier en `.env`, jamais commité |

La configuration est séparée en deux : `configs/*.yaml` décrit l'expérience (tracée dans MLflow, clés inconnues
refusées), les variables `TENNIS_*` / `MOCK_*` décrivent l'environnement (un seul `.env` partagé par les services).

## Qualité de code

- **ruff** : lint (pycodestyle, pyflakes, bugbear, nommage, annotations, docstrings…) et formatage
- **mypy --strict** avec le plugin pydantic
- **pre-commit** : ruff, mypy, fichiers > 500 Ko et clés privées bloqués à chaque commit
- **CI GitHub Actions** : `make lint typecheck test` sur chaque push et PR
- Skill Claude Code `.claude/skills/lint` : même procédure, réutilisable

## Workflow Git

- `main` reste toujours vert ; chaque phase se développe sur une branche `feat/<phase>` puis passe par une PR.
- Messages de commit au format [Conventional Commits](https://www.conventionalcommits.org/fr/) (`feat:`, `fix:`,
  `chore:`, `docs:`, `test:`).
- Un tag annoté `v0.<n>.0` à la fin de chaque phase, `v1.0.0` pour le rendu.

## Données

Matchs, classements et joueurs ATP 2010 → mai 2026 de Jeff Sackmann
([tennis_atp](https://github.com/JeffSackmann/tennis_atp)), récupérés depuis Hugging Face
([`Aneeshers/tennis-sackmann-archive`](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive)) avec le
miroir GitHub du même dataset en secours. Licence **CC BY-NC-SA 4.0** (non commercial) : aucun fichier de données
n'est versionné, `make data` les télécharge.

- **Reproductible** : `configs/data.lock.json` fige la révision de chaque source et le SHA-256 de chaque fichier ;
  la `dataset_version` qui en découle identifie exactement les données utilisées.
- **Résilient** : source principale injoignable → bascule sur le miroir ; contenu inattendu → arrêt, sans fallback.
- **Simulation** : le dataset historique est rejoué par lots (1 lot = 1 tournoi) pour simuler l'arrivée de
  nouvelles données.

Détails, profil et décisions : [`docs/DONNEES.md`](docs/DONNEES.md).

## Simulateur de source (`mock_source`)

API FastAPI qui rejoue l'historique selon une **horloge simulée** : `make advance DAYS=7` fait passer une semaine,
de nouveaux résultats deviennent disponibles. Aucune réponse ne peut révéler le futur (testé jour par jour) :
pas de résultat ni de classement postérieur à l'horloge, et un match à venir n'est annoncé qu'une fois ses deux
joueurs qualifiés. Les incidents (`503`, `slow`, `corrupt`, `schema`) servent à démontrer retries, alertes et
contrôles qualité bloquants.

Endpoints, règles du rejeu et incidents : [`docs/MOCK_SOURCE.md`](docs/MOCK_SOURCE.md).

## Ingestion et contrôles qualité

Le flow Prefect `ingestion-atp` tire les tournois terminés depuis `mock_source` (une task par appel HTTP, retries avec
backoff), contrôle chaque lot avec Great Expectations puis stocke dans PostgreSQL ce qui passe, en une transaction.

- **Bloquant** : schéma, domaines de valeurs et 4 expectations custom (score lisible, vainqueur ≠ perdant, aucun
  match dans le futur, volume = tableau).
- **Surveillance** (non bloquant) : dérive de l'écart de classement, comparée aux tournois du même niveau.
- **Lot refusé** : quarantaine (`state/quarantine/`), flow `Failed`, rien n'est stocké pour ce lot, reprise au
  passage suivant.
- **Source en panne** : retries, puis état « Dégradé » sans perte ; les données déjà ingérées restent utilisables.

Déroulé, contrôles, incidents et démo : [`docs/INGESTION.md`](docs/INGESTION.md).

## Entraînement

Le flow Prefect `entrainement-atp` rejoue l'historique validé pour calculer les features de chaque match **avec
l'état de la veille** (classement, Elo global et par surface, forme, face-à-face…), découpe le temps en
entraînement / validation / test, puis entraîne une grille de candidats **en parallèle** (une task Prefect par
méthode : `DatasetBuilder.build` → `TemporalSplitter.split` → `PipelineFactory.create` → `Trainer.fit` →
`Evaluator.evaluate` → `Reporter.log` → `ModelRegistry.register` → règle de mise en service → `Promoter.apply`).

- **MLflow** : un run parent par campagne (données, config, baselines, comparaison), un run enfant par candidat
  (métriques globales et par segment, coûts, graphiques, fiche modèle, pipeline complet rechargeable).
- **Lecture rapide** : chaque run a son rapport HTML dans l'onglet « Artifacts » (`rapport.html`,
  `rapport_campagne.html`) ; l'onglet « Model metrics » range les métriques par section et trace la performance
  mois par mois et les courbes d'apprentissage.
- **Choix** sur la validation, **bilan** sur le test, comparé aux baselines Elo et classement.
- **Registre** : le candidat retenu devient une nouvelle version de `atp-winner`.

Features, protocole, traçabilité et mesures : [`docs/ENTRAINEMENT.md`](docs/ENTRAINEMENT.md).

## Registre et mise en service

Le candidat retenu et le modèle en production sont jugés sur le même test récent. L'alias **`production`** (lu par
l'API : `models:/atp-winner@production`) n'est déplacé que si toutes les règles de `configs/promotion.yaml`
passent : données validées, Elo pur et « mieux classé » battus, au moins aussi bon que la production, aucune surface
dégradée, pas de dérive forte. Sinon la version est refusée (`lifecycle=rejected`, raison en tag) et gardée en
observation (`staging`).

- **Rollback** en une commande, qui ne dépend que de MLflow : `make rollback REASON="…"` (`production` ←
  `previous`) ; l'API suivra l'alias sans redéploiement.
- **Chaque décision est tracée** : tags de la version, section « Décision de mise en service » du rapport de
  campagne, et un run par décision dans l'expérience `atp-winner-decisions`.
- **Traçabilité** : `make trace` remonte de la version au run MLflow, au run Prefect, aux lots validés et à la source.

Alias, tags, règle, cycle de vie, démo : [`docs/TAGGING.md`](docs/TAGGING.md).

## API de prédiction

Le service `api` (`http://127.0.0.1:8000/docs`) sert la version désignée par l'alias `production` : pour un match à
venir, la probabilité de victoire de chaque joueur, la version et un `request_id`. Les features sont calculées par le
**même** `FeatureBuilder` que l'entraînement, sur l'historique validé rejoué jusqu'à la veille : aucun match passé ne
peut être prédit avec son propre résultat.

- **Sans redémarrage** : l'alias est relu toutes les 30 s ; une promotion ou un rollback est servi tout seul.
- **MLflow arrêté** : la version en cache continue de servir, `/ready` passe en `degraded` (scénario S8).
- **Tout est tracé** dans PostgreSQL : requêtes, prédictions (version, features), erreurs (`422` détaillés), vrais
  résultats. `prediction_outcomes` donne, pour une prédiction, sa version, sa durée et son résultat réel ;
  `online_metrics` l'accuracy, la log loss et le Brier en ligne par version.
- **Boucle complète** : chaque pas de `make demo` remonte les résultats validés (`POST /feedback`) puis prédit les
  matchs de la semaine à venir (`POST /predict/batch`).

Détails : [`docs/API.md`](docs/API.md).

## Orchestration et alertes

`make demo` lance une simulation planifiée dans Prefect : toutes les 2 minutes, le temps avance d'une semaine et
l'ingestion récupère les tournois terminés. Au-delà de 200 nouveaux matchs validés, l'ingestion émet l'événement
`tennis.training.requested` et une **automation** lance une campagne d'entraînement, soumise à la même règle de mise
en service. Tout tourne dans le worker de la stack ; déploiements et automations sont créés à chaque `make up`.

- **Une alerte par incident**, en carte Teams : flow en échec après ses retries, lot refusé par Great Expectations,
  source injoignable, rollback. Un incident qui persiste n'est pas renvoyé toutes les 2 minutes.
- **Webhook secret** : `TENNIS_TEAMS_WEBHOOK_URL` dans `.env` uniquement ; sans lui, les cartes sont gardées dans
  l'historique (`make alerts`) sans être envoyées.
- **Chronologie** : chaque fait marquant (lot validé ou refusé, demande d'entraînement, décision, alerte) est inscrit
  dans la table `pipeline_events`, pour les annotations Grafana.

Détails : [`docs/ORCHESTRATION.md`](docs/ORCHESTRATION.md) et [`docs/ALERTES.md`](docs/ALERTES.md).
