# tennis-mlops — Projet final 5MLDE

Chaîne MLOps complète qui prédit la probabilité de victoire d'un joueur dans un match ATP. Les matchs historiques
sont rejoués dans le temps par une API simulée, ce qui permet de démontrer l'ingestion continue, les contrôles
qualité, le réentraînement, la mise en service et la supervision.

## Prérequis

- [uv](https://docs.astral.sh/uv/) ≥ 0.11 (installe Python 3.12 automatiquement)
- `make`, `git`
- Docker Desktop (à partir de la phase 2)

## Installation

```sh
make setup      # uv sync --locked + hooks pre-commit
make check      # lint + typage + tests, comme la CI
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

## Organisation du code

| Dossier | Contenu |
|---|---|
| `src/tennis_mlops/` | Code applicatif (package installable, typé) |
| `configs/` | Configuration de l'expérience (YAML, versionnée) |
| `tests/` | Tests unitaires (`unit/`) puis d'intégration (`integration/`, marqueur `integration`) |
| `.env.example` | Variables d'environnement (URLs, secrets) — copier en `.env`, jamais commité |

La configuration est séparée en deux : `configs/*.yaml` décrit l'expérience (tracée dans MLflow), les
variables `TENNIS_*` décrivent l'environnement. Les deux refusent les clés inconnues.

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

Jeff Sackmann, [tennis_atp](https://github.com/JeffSackmann/tennis_atp), via le miroir Hugging Face
[`Aneeshers/tennis-sackmann-archive`](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive).
Licence **CC BY-NC-SA 4.0** : les fichiers ne sont pas versionnés ici ; ils sont téléchargés par script (phase 1).
