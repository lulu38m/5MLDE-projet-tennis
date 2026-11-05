---
name: lint
description: Lint, format et typage du projet tennis-mlops avec uv, ruff et mypy. À utiliser après toute modification de code Python, avant de proposer un commit, ou quand on demande de « linter », « formater » ou « vérifier les types ».
---

# Lint du projet (uv + ruff + mypy)

Toujours passer par `uv run --locked` : les versions des outils sont celles de `uv.lock`, jamais une version
installée globalement.

## Étapes

1. Corriger automatiquement ce qui peut l'être :
   ```sh
   uv run --locked ruff check --fix .
   uv run --locked ruff format .
   ```
2. Vérifier les types (mode strict, configuré dans `pyproject.toml`) :
   ```sh
   uv run --locked mypy
   ```
3. Relancer les tests unitaires :
   ```sh
   uv run --locked pytest -m "not integration"
   ```
4. Tout doit passer avec `make check` (c'est ce que la CI exécute).

## Règles

- Corriger la cause d'une erreur plutôt que l'ignorer. Un `# noqa` ou `# type: ignore[code]` n'est acceptable
  qu'avec le code d'erreur précis et une raison en commentaire.
- Toute fonction et méthode de `src/` a des annotations de type complètes et une docstring (convention Google).
- Ne jamais modifier la configuration ruff/mypy pour faire passer un fichier : en parler d'abord.
- Ne pas faire de commit : l'utilisateur fait lui-même toutes les opérations Git. Terminer en lui donnant les
  commandes `git` à lancer.
