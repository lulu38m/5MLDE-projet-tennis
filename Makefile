# Point d'entrée unique des commandes du projet. `make help` liste les cibles.
.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test check clean \
        data data-lock data-verify data-profile \
        up down logs ps urls mock clock advance chaos \
        ingest lots gx-docs db train train-seq train-weak mlflow \
        registry promote rollback trace alert-preview alert-test \
        demo demo-stop demo-status simulate alerts \
        api api-status predict predictions score feedback \
        ui ui-open ui-smoke \
        dashboards grafana grafana-check prometheus \
        load load-full load-ui \
        reset scenarios scenarios-report

UV := uv run --locked
MOCK_URL ?= http://127.0.0.1:8001
DAYS ?= 7
MODE ?= off
FAILURES ?=
DELAY ?=
DATE ?=
WORKERS ?=
VERSION ?=
EXAMPLE ?=
REASON ?=
API_URL ?= http://127.0.0.1:8000
P1 ?= 104925
P2 ?= 104745
SURFACE ?= Clay
UI_URL ?= http://127.0.0.1:8501
GRAFANA_URL ?= http://127.0.0.1:3001
PROMETHEUS_URL ?= http://127.0.0.1:9091
# Les flows lancés depuis la machine hôte s'enregistrent dans le serveur Prefect de la stack.
PREFECT_API_URL ?= http://127.0.0.1:4201/api
export PREFECT_API_URL
# Pas de « conseils pour agents » de MLflow dans les logs.
export MLFLOW_DISABLE_AGENT_HINT := 1
# Nom du poste affiché dans les cartes Teams (le service d'alertes tourne dans Docker : il ne le connaît pas).
TENNIS_ALERT_HOST ?= $(shell hostname)
export TENNIS_ALERT_HOST
# Utilisateur de la machine hôte : le worker de la stack écrit dans ./state sous la même identité (compose.yml).
HOST_UID ?= $(shell id -u)
HOST_GID ?= $(shell id -g)
export HOST_UID HOST_GID
# Révision du code gravée dans l'image : les campagnes lancées par le worker la tracent dans MLflow.
GIT_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null)
export GIT_COMMIT
GX_DOCS := state/gx/gx/uncommitted/data_docs/local_site/index.html

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2}'

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

# Le parcours correcteur est écrit en Python (src/tennis_mlops/cli.py) pour marcher aussi sous Windows sans make :
# `make up` et `uv run tennis up` font exactement la même chose (.env créé s'il manque, données, stack, 1er modèle).
up: ## Construit et démarre la stack, puis un premier modèle entraîné par la chaîne et servi par l'API
	$(UV) tennis up

down: ## Arrête la stack (les volumes sont conservés)
	docker compose down

reset: ## Remise à zéro : stack arrêtée, volumes supprimés (bases, MLflow, Grafana) et état local effacé ; garde data/raw
	$(UV) tennis reset

logs: ## Suit les logs de la stack
	docker compose logs -f --tail=100

ps: ## État des services
	docker compose ps

urls: ## Adresses des interfaces, identifiants de démo, et si elles répondent
	@$(UV) tennis urls

# --------------------------------------------------------------------------- mock_source

mock: ## Lance mock_source en local, sans Docker (http://127.0.0.1:8001/docs)
	MOCK_HOST=127.0.0.1 $(UV) python -m mock_source

clock: ## Affiche l'horloge simulée ; `make clock DATE=2019-06-01` la règle
	@if [ -n "$(DATE)" ]; then \
		curl -s -X POST $(MOCK_URL)/admin/clock -H 'Content-Type: application/json' -d '{"date": "$(DATE)"}'; \
	else curl -s $(MOCK_URL)/admin/clock; fi; echo

advance: ## Fait avancer le temps simulé : `make advance DAYS=7`
	@curl -s -X POST $(MOCK_URL)/admin/clock/advance -H 'Content-Type: application/json' -d '{"days": $(DAYS)}'; echo

chaos: ## Incident : `make chaos MODE=503 FAILURES=2` (MODE=off|503|slow|corrupt|schema, DELAY=15 pour slow)
	@curl -s -X POST $(MOCK_URL)/admin/chaos -H 'Content-Type: application/json' \
		-d '{"mode": "$(MODE)"$(if $(FAILURES),$(comma) "failures": $(FAILURES))$(if $(DELAY),$(comma) "delay_s": $(DELAY))}'; echo

# --------------------------------------------------------------------------- ingestion

ingest: ## Un passage d'ingestion : mock_source → Great Expectations → PostgreSQL (suivi dans Prefect)
	$(UV) python -m tennis_mlops.flows.ingest

lots: ## Les 15 derniers lots : statut, volumes, contrôles en échec
	@docker compose exec -T postgres psql -U tennis -d tennis -c \
		"SELECT lot_id, status, rows_received AS recus, rows_expected AS attendus, rows_inserted AS ajoutes, \
		 simulated_now, left(failed_checks, 70) AS echecs FROM ingestion_lots ORDER BY updated_at DESC LIMIT 15;"

gx-docs: ## Ouvre les rapports HTML de Great Expectations (Data Docs)
	$(UV) python -m webbrowser -t "file://$(CURDIR)/$(GX_DOCS)"

db: ## Console SQL sur la base du projet
	docker compose exec postgres psql -U tennis -d tennis

# --------------------------------------------------------------------------- entraînement

train: ## Campagne d'entraînement en parallèle (configs/train.yaml), suivie dans Prefect et MLflow ; `WORKERS=8`
	$(UV) python -m tennis_mlops.flows.train $(if $(WORKERS),--workers $(WORKERS))

train-seq: ## Même campagne, un candidat à la fois : la référence pour mesurer le gain du parallélisme
	$(UV) python -m tennis_mlops.flows.train --workers 1

train-weak: ## Campagne volontairement mauvaise (configs/train-weak.yaml) : enregistrée puis refusée par la règle
	$(UV) python -m tennis_mlops.flows.train --config configs/train-weak.yaml

mlflow: ## Ouvre l'interface MLflow
	$(UV) python -m webbrowser -t "http://127.0.0.1:7282"

# --------------------------------------------------------------------------- alertes Teams

alert-preview: ## Écrit le JSON des cartes Teams d'exemple dans state/alerts (rien n'est envoyé)
	$(UV) python -m tennis_mlops.alerts preview

alert-test: ## Envoie UNE carte « 🧪 TEST » au webhook de .env : `make alert-test EXAMPLE=echec`
	$(UV) python -m tennis_mlops.alerts test $(if $(EXAMPLE),--example $(EXAMPLE))

alerts: ## Dernières alertes et ce qu'elles sont devenues (envoyée, filtrée, sans webhook, échec)
	$(UV) python -m tennis_mlops.alerts recent

# --------------------------------------------------------------------------- orchestration (phase 6)

demo: up ## Chaîne automatique : +7 jours simulés chaque minute, ingestion, réentraînement sur événement, alertes
	$(UV) tennis demo

demo-stop: ## Met la simulation planifiée en pause (le run en cours se termine)
	$(UV) python -m tennis_mlops.orchestration stop

demo-status: ## Déploiements, worker et derniers runs
	$(UV) python -m tennis_mlops.orchestration status

simulate: ## Un seul pas de simulation, sans planning : `make simulate DAYS=14`
	$(UV) python -m tennis_mlops.flows.simulation --days $(DAYS)

# --------------------------------------------------------------------------- registre (ne dépend que de MLflow)

registry: ## État du registre : versions, alias production / previous / staging, décisions
	$(UV) python -m tennis_mlops.registry status

promote: ## Mise en production manuelle : `make promote VERSION=3 REASON="meilleur en shadow"`
	@test -n "$(VERSION)" -a -n "$(REASON)" || { echo 'usage : make promote VERSION=3 REASON="…"'; exit 2; }
	$(UV) python -m tennis_mlops.registry promote --version "$(VERSION)" --reason "$(REASON)"

rollback: ## Revient à la version `previous` : `make rollback REASON="probabilités aberrantes"`
	$(UV) python -m tennis_mlops.registry rollback --reason "$(or $(REASON),rollback manuel (make rollback))"

trace: ## Chaîne de traçabilité d'une version (défaut : production) : `make trace VERSION=3`
	$(UV) python -m tennis_mlops.registry trace $(if $(VERSION),--version $(VERSION))

comma := ,

# --------------------------------------------------------------------------- divers

clean: ## Supprime les caches locaux
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# --------------------------------------------------------------------------- API de prédiction (phase 7)

api: ## Lance l'API en local, sans Docker (le service `api` de la stack fait la même chose)
	$(UV) python -m tennis_mlops.api serve --port 8000

api-status: ## État de l'API : prête ou dégradée, version servie, historique, qualité en ligne
	@curl -sS $(API_URL)/ready | $(UV) python -m json.tool
	@curl -fsS $(API_URL)/model | $(UV) python -m json.tool

predict: ## Une prédiction : `make predict P1=104925 P2=104745 SURFACE=Clay` (Djokovic contre Nadal)
	@curl -sS -X POST $(API_URL)/predict -H 'Content-Type: application/json' -H 'X-Client: make' \
	  -d '{"surface": "$(SURFACE)", "tourney_level": "M", "round": "SF", "best_of": 3, "player1": {"id": $(P1)}, "player2": {"id": $(P2)}}' \
	  | $(UV) python -m json.tool

predictions: ## Dernières prédictions : version, durée de la requête, vrai résultat
	$(UV) python -m tennis_mlops.api outcomes

score: ## Prédit les matchs de la semaine à venir via l'API (chaque pas de `make demo` le fait aussi)
	$(UV) python -m tennis_mlops.flows.scoring

feedback: ## Remonte à l'API les vrais résultats des matchs prédits (chaque pas de `make demo` le fait aussi)
	$(UV) python -m tennis_mlops.flows.feedback

# --------------------------------------------------------------------------- interface web de test (phase 8)

ui: ## Lance l'interface en local, sans Docker (le service `ui` de la stack fait la même chose)
	$(UV) python -m tennis_mlops.ui --port 8501

ui-open: ## Ouvre l'interface de test dans le navigateur
	$(UV) python -m webbrowser -t "$(UI_URL)"

ui-smoke: ## Test de fumée sur la stack : la page Match prédit, la prédiction est tracée avec le client `ui`
	$(UV) python -m tennis_mlops.ui.smoke --ui-url $(UI_URL)

# --------------------------------------------------------------------------- monitoring (phase 9)

dashboards: ## Régénère les dashboards Grafana (infra/grafana/dashboards/*.json) depuis leur code Python
	$(UV) python -m tennis_mlops.monitoring.dashboards

grafana: ## Ouvre les dashboards Grafana (lecture sans compte)
	$(UV) python -m webbrowser -t "$(GRAFANA_URL)/dashboards/f/tennis-mlops"

grafana-check: ## Vérifie Prometheus, Grafana, et chaque requête de chaque dashboard, sur la stack qui tourne
	TENNIS_GRAFANA_UI_URL=$(GRAFANA_URL) TENNIS_PROMETHEUS_URL=$(PROMETHEUS_URL) $(UV) python -m tennis_mlops.monitoring.check

prometheus: ## Ouvre Prometheus sur l'état de ses cibles
	$(UV) python -m webbrowser -t "$(PROMETHEUS_URL)/targets"

# --------------------------------------------------------------------------- charge (phase 10)

LOAD := TENNIS_API_URL=$(API_URL) TENNIS_PROMETHEUS_URL=$(PROMETHEUS_URL) $(UV) python -m loadtest.run_campaign

load: ## Campagne de charge courte (~6 min, config B) : rapport, CSV et graphiques dans loadtest/results/
	$(LOAD) quick

load-full: ## Campagne complète (~75 min) : configs A, B, C, rampe 10 → 1 000 utilisateurs, pic, endurance
	$(LOAD) full

load-ui: ## Locust avec son interface web (http://127.0.0.1:8089), charge choisie à la main
	PYTHONPATH=. $(UV) locust -f loadtest/locustfile.py --host $(API_URL) --web-host 127.0.0.1

# --------------------------------------------------------------------------- tests d'acceptation (phase 11)

ONLY ?=
TEAMS ?=
FORCE ?=

scenarios: ## Tests d'acceptation S1 → S9 sur la stack de `make up` (~10 min) ; `ONLY=S2,S5`, `SILENCE=1`, rapport HTML
	$(UV) tennis scenarios $(if $(ONLY),--only $(ONLY)) $(if $(SILENCE),--silence) $(if $(FORCE),--force)

scenarios-report: ## Ouvre le rapport du dernier `make scenarios`
	$(UV) tennis report
