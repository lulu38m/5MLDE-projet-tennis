# Point d'entrée unique des commandes du projet. `make help` liste les cibles.
.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test check clean \
        data data-lock data-verify data-profile \
        up down logs ps mock clock advance chaos

UV := uv run --locked
MOCK_URL ?= http://127.0.0.1:8001
DAYS ?= 7
MODE ?= off
FAILURES ?=
DATE ?=

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- qualité

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

test: ## Tests unitaires + couverture (les tests sur vraies données sont ignorés si `make data` n'a pas été lancé)
	$(UV) pytest -m "not integration" --cov --cov-report=term-missing

check: lint typecheck test ## Tout ce que la CI vérifie

# --------------------------------------------------------------------------- données

data: ## Télécharge les données brutes à la version verrouillée (configs/data.lock.json)
	$(UV) python -m tennis_mlops.data download

data-verify: ## Vérifie les fichiers locaux contre le verrou (SHA-256)
	$(UV) python -m tennis_mlops.data verify

data-profile: ## Profile les données brutes → data/raw/profile.json
	$(UV) python -m tennis_mlops.data profile

data-lock: ## (Mainteneurs) Résout les révisions actuelles des sources et régénère le verrou
	$(UV) python -m tennis_mlops.data lock

# --------------------------------------------------------------------------- stack Docker

up: data ## Construit et démarre la stack (attend que les services soient sains)
	docker compose up -d --build --wait

down: ## Arrête la stack (les volumes sont conservés)
	docker compose down

logs: ## Suit les logs de la stack
	docker compose logs -f --tail=100

ps: ## État des services
	docker compose ps

# --------------------------------------------------------------------------- mock_source

mock: ## Lance mock_source en local, sans Docker (http://127.0.0.1:8001/docs)
	MOCK_HOST=127.0.0.1 $(UV) python -m mock_source

clock: ## Affiche l'horloge simulée ; `make clock DATE=2019-06-01` la règle
	@if [ -n "$(DATE)" ]; then \
		curl -s -X POST $(MOCK_URL)/admin/clock -H 'Content-Type: application/json' -d '{"date": "$(DATE)"}'; \
	else curl -s $(MOCK_URL)/admin/clock; fi; echo

advance: ## Fait avancer le temps simulé : `make advance DAYS=7`
	@curl -s -X POST $(MOCK_URL)/admin/clock/advance -H 'Content-Type: application/json' -d '{"days": $(DAYS)}'; echo

chaos: ## Incident contrôlé : `make chaos MODE=503 FAILURES=2` (MODE=off|503|slow|corrupt|schema)
	@curl -s -X POST $(MOCK_URL)/admin/chaos -H 'Content-Type: application/json' \
		-d '{"mode": "$(MODE)"$(if $(FAILURES),$(comma) "failures": $(FAILURES))}'; echo

comma := ,

# --------------------------------------------------------------------------- divers

clean: ## Supprime les caches locaux
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
