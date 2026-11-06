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
make up         # stack Docker : mock_source, PostgreSQL, Prefect (`make urls` affiche les adresses)
make ingest     # un passage d'ingestion contrôlé par Great Expectations, suivi dans Prefect
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
| `make urls` | Adresses des interfaces (mock_source, Prefect, PostgreSQL, rapports GX) |
| `make ingest` | Un passage d'ingestion : mock_source → Great Expectations → PostgreSQL |
| `make lots` | Derniers lots ingérés : statut, volumes, contrôles en échec |
| `make gx-docs` | Rapports HTML de Great Expectations |
| `make db` | Console SQL sur la base du projet |

## Organisation du code

| Dossier | Contenu |
|---|---|
| `src/tennis_mlops/` | Chaîne MLOps (package installable, typé) |
| `src/tennis_mlops/ingestion/` | Client HTTP de la source, lots, stockage PostgreSQL, archive brute |
| `src/tennis_mlops/validation/` | Suites Great Expectations et expectations custom |
| `src/tennis_mlops/flows/` | Flows Prefect |
| `src/mock_source/` | Simulateur de source : API FastAPI de rejeu ATP avec horloge simulée et chaos |
| `configs/` | Configuration de l'expérience (YAML, versionnée) |
| `tests/` | Tests unitaires (`unit/`) puis d'intégration (`integration/`, marqueur `integration`) |
| `docs/` | `DONNEES.md` (provenance, licence, profil), `MOCK_SOURCE.md` (API de rejeu), `INGESTION.md` (flow, contrôles) |
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
