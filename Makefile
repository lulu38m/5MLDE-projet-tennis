# Point d'entrée unique des commandes du projet. `make help` liste les cibles.
.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test check clean data data-lock data-verify data-profile

UV := uv run --locked

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Installe l'environnement (uv) et les hooks pre-commit
	uv sync --locked
	$(UV) pre-commit install

lint: ## Vérifie le style sans rien modifier (ruff)
	$(UV) ruff check .
	$(UV) ruff format --check .

format: ## Corrige ce qui peut l'être automatiquement (ruff)
	$(UV) ruff check --fix .
	$(UV) ruff format .

typecheck: ## Vérifie les types (mypy --strict)
	$(UV) mypy

test: ## Lance les tests unitaires avec couverture
	$(UV) pytest -m "not integration" --cov --cov-report=term-missing

check: lint typecheck test ## Tout ce que la CI vérifie

data: ## Télécharge les données brutes à la version verrouillée (configs/data.lock.json)
	$(UV) python -m tennis_mlops.data download

data-verify: ## Vérifie les fichiers locaux contre le verrou (SHA-256)
	$(UV) python -m tennis_mlops.data verify

data-profile: ## Profile les données brutes → data/raw/profile.json
	$(UV) python -m tennis_mlops.data profile

data-lock: ## (Mainteneurs) Résout les révisions actuelles des sources et régénère le verrou
	$(UV) python -m tennis_mlops.data lock

clean: ## Supprime les caches locaux
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
