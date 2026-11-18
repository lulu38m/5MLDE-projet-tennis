# Architecture

Une chaîne MLOps complète, locale, dans Docker : des matchs ATP rejoués dans le temps par une source simulée sont
ingérés et contrôlés, servent à entraîner et mettre en service un modèle, que l'API utilise pour prédire les matchs à
venir ; les vrais résultats reviennent juger ces prédictions. Prefect orchestre, MLflow trace et versionne,
PostgreSQL garde tout, Prometheus et Grafana surveillent.

Chaque brique a sa documentation détaillée ; ce document montre comment elles s'emboîtent.

## Les services

```mermaid
flowchart TB
    classDef ext fill:#eef2f7,stroke:#8a96a8,color:#1d2330
    classDef store fill:#fff6e5,stroke:#c9a14a,color:#1d2330

    UI["Interface Streamlit :8501"] -- "X-Client: ui" --> API
    subgraph ORCH["Orchestration"]
        direction LR
        PREFECT["Prefect :4201<br/>planning · automations"] -- runs --> WORKER["worker<br/>simulation · ingestion · entraînement"]
        PREFECT -- "échec, dégradé" --> SINK["alert_sink :8090"]
    end
    MOCK["mock_source :8001<br/>rejeu ATP, horloge simulée"]
    API["API FastAPI :8000<br/>sert l'alias production"]
    MLFLOW["MLflow :7282<br/>runs · registre"]
    PG[("PostgreSQL :5433<br/>tennis · prefect · mlflow")]:::store
    TEAMS["Teams"]:::ext

    WORKER -- "horloge, tournois" --> MOCK
    WORKER -- "matchs validés, journal" --> PG
    WORKER -- "campagnes, versions" --> MLFLOW
    WORKER -- "scoring, feedback" --> API
    API -- "relit l'alias" --> MLFLOW
    API -- "historique, traces" --> PG
    SINK --> TEAMS

    subgraph OBS["Supervision"]
        direction LR
        GRAF["Grafana :3001"] --> PROM["Prometheus :9091"] --> EXP["exporteur<br/>santé · runs · conteneurs"]
    end
    PROM -- "/metrics" --> API
    GRAF -- "lecture seule" --> PG
```

| Service | Image | Rôle | Doc |
|---|---|---|---|
| `mock_source` | projet | API de rejeu : horloge simulée, aucune donnée du futur, incidents à la demande (`503`, `slow`, `corrupt`, `schema`) | [MOCK_SOURCE.md](MOCK_SOURCE.md) |
| `postgres` | `postgres:18.6` | une base par usage (`tennis`, `prefect`, `mlflow`) : Prefect et MLflow y gardent aussi leurs métadonnées ; un compte par usage | — |
| `prefect` | projet | serveur Prefect 3.8.7 (UI, planning, automations) | [ORCHESTRATION.md](ORCHESTRATION.md) |
| `orchestration-setup` | projet | ponctuel : work pool, déploiements, automations (idempotent, à chaque `up`) | [ORCHESTRATION.md](ORCHESTRATION.md) |
| `worker` | projet | exécute les runs : simulation, ingestion, entraînement | [ORCHESTRATION.md](ORCHESTRATION.md) |
| `mlflow` | projet | suivi des campagnes, artefacts, registre `atp-winner` et ses alias | [ENTRAINEMENT.md](ENTRAINEMENT.md), [TAGGING.md](TAGGING.md) |
| `alert_sink` | projet | reçoit les automations et les décisions du registre, construit la carte Teams, garde l'historique | [ALERTES.md](ALERTES.md) |
| `api` | projet | prédictions servies par la version `production`, traces, feedback, métriques | [API.md](API.md) |
| `ui` | projet | interface de test : ne parle qu'à l'API | [UI.md](UI.md) |
| `exporter` | projet | ce que l'API ne mesure pas : sonde la santé de chaque service, l'horloge simulée, les runs Prefect, le CPU et la mémoire de chaque conteneur (API Docker) | [MONITORING.md](MONITORING.md) |
| `prometheus`, `grafana` | officielles | métriques ; dashboards technique, métier, croisé, provisionnés depuis Git | [MONITORING.md](MONITORING.md) |

**Une seule image** (`Dockerfile`) pour tout le code du projet, construite depuis `uv.lock` : mêmes versions partout,
du serveur Prefect au worker. Elle tourne sous un utilisateur non-root ; seul l'exporteur est root (lecture du socket
Docker).

## Le flux des données

```mermaid
flowchart LR
    classDef store fill:#fff6e5,stroke:#c9a14a,color:#1d2330
    RAW[("data/raw<br/>CSV verrouillés<br/>SHA-256")]:::store
    MOCK["mock_source"]
    ING["ingestion-atp<br/>1 task par appel HTTP<br/>retries sur panne"]
    GX{"Great Expectations<br/>36 contrôles<br/>dont 4 custom"}
    Q[("quarantaine<br/>state/quarantine")]:::store
    M[("matches<br/>ingestion_lots")]:::store
    TR["entrainement-atp<br/>FeatureBuilder<br/>8 candidats en parallèle"]
    GATE{"règle de<br/>mise en service"}
    REG[("registre MLflow<br/>production · previous · staging")]:::store
    API["API"]
    P[("predictions<br/>api_requests · errors")]:::store
    F[("feedbacks")]:::store

    RAW --> MOCK -- "lot = 1 tournoi terminé" --> ING --> GX
    GX -- refusé --> Q
    GX -- accepté --> M
    M -- "≥ 200 nouveaux matchs<br/>événement" --> TR --> GATE
    GATE -- passée --> REG
    GATE -- refusée --> REG
    REG -- "alias production<br/>relu toutes les 30 s" --> API
    M -- "historique rejoué" --> API
    API -- "scoring des matchs à venir" --> P
    M -- "vrais résultats validés" --> F --> P
```

1. **Source.** `make data` télécharge les CSV de Jeff Sackmann à une révision figée et vérifie chaque fichier
   contre `configs/data.lock.json` ; `mock_source` les rejoue selon son horloge, sans jamais rien révéler du futur
   ([DONNEES.md](DONNEES.md)).
2. **Ingestion.** Un lot = un tournoi terminé. Great Expectations le contrôle : refusé, il part en quarantaine et le
   flow échoue (alerte) ; accepté, il est stocké en une transaction. Source injoignable : état « Dégradé », rien n'est
   perdu ([INGESTION.md](INGESTION.md)).
3. **Entraînement.** Au-delà de 200 nouveaux matchs validés, l'ingestion émet `tennis.training.requested` ; une
   automation lance la campagne : le même `FeatureBuilder` que l'API calcule 19 features avant chaque match (Elo par
   surface, forme, face-à-face…), 8 candidats s'entraînent en parallèle, le meilleur sur la validation est jugé sur
   un test jamais vu ([ENTRAINEMENT.md](ENTRAINEMENT.md)).
4. **Mise en service.** La version est enregistrée ; la règle la compare à la production et aux baselines sur le même
   test. Passée : l'alias `production` bouge (`previous` garde l'ancienne). Refusée : `staging`, rien d'autre ne
   change ([TAGGING.md](TAGGING.md)).
5. **Service.** L'API relit l'alias toutes les 30 s : une promotion ou un rollback est servi sans redémarrage. Elle
   rejoue l'historique validé pour calculer les features, trace chaque requête ([API.md](API.md)).
6. **Boucle fermée.** À chaque pas, le scoring prédit les matchs de la semaine à venir ; au pas suivant, le feedback
   remonte leurs vrais résultats (validés par l'ingestion) : accuracy, log loss et Brier en ligne, par version.

## Un pas de la chaîne, dans le temps

Ce que fait chaque minute `make demo` (un pas = une semaine simulée) :

```mermaid
sequenceDiagram
    autonumber
    participant P as Prefect
    participant W as worker
    participant S as mock_source
    participant DB as PostgreSQL
    participant ML as MLflow
    participant A as API
    participant AS as alert_sink

    P->>W: simulation-atp (planifié, toutes les 60 s)
    W->>S: POST /admin/clock/advance (+7 jours)
    W->>S: GET /tournaments, GET /tournaments/{id}/results
    W->>W: Great Expectations, lot par lot
    W->>DB: lots validés (transaction), watermark, pipeline_events
    alt lot refusé ou source injoignable
        P-->>AS: automation (Failed / Dégradé)
        AS-->>AS: carte Teams, historique
    end
    W->>A: POST /feedback (résultats des matchs prédits)
    W->>A: POST /predict/batch (matchs de la semaine à venir)
    opt ≥ 200 nouveaux matchs validés
        W-->>P: événement tennis.training.requested
        P->>W: entrainement-atp (automation)
        W->>ML: campagne, candidats, version
        W->>ML: règle passée → alias production
        A->>ML: relit l'alias (≤ 30 s) → nouvelle version servie
    end
```

## Ce qui est produit, et où le retrouver

| Artefact | Où | Comment le voir |
|---|---|---|
| Données brutes et leur verrou | `data/raw/` (hors Git), `configs/data.lock.json` | `make data-verify` |
| Lots acceptés, refusés (JSON brut + rapport de validation) | `state/lots/`, `state/quarantine/` | `make lots`, Prefect |
| Rapports Great Expectations | `state/gx/…/data_docs/` | `make gx-docs` |
| Matchs validés, lots, watermark | base `tennis` : `matches`, `ingestion_lots`, `ingestion_state` | `make db` |
| Journal du pipeline (lots, demandes d'entraînement, décisions, alertes) | `pipeline_events` | annotations Grafana, `make alerts` |
| Runs des flows, retries, durées | base `prefect` | UI Prefect |
| Campagnes, candidats, métriques, graphiques, rapports HTML, modèles (skops), fiche modèle | base `mlflow` + volume `mlflow_artifacts` | UI MLflow |
| Versions, alias, tags de décision | registre MLflow `atp-winner`, expérience `atp-winner-decisions` | `make registry`, `make trace` |
| Requêtes, prédictions, erreurs, feedbacks | `api_requests`, `predictions`, `errors`, `feedbacks`, vues `prediction_outcomes`, `online_metrics` | `make predictions`, Grafana |
| Métriques de l'API et de la chaîne | volume `prometheus_data` (15 jours) | Prometheus, Grafana |
| Tests d'acceptation | `state/acceptance/rapport.html` et `.json` | `uv run tennis report` |
| Résultats des tests de charge | `loadtest/results/<campagne>/` (dans Git, sauf le journal brut) | [CHARGE.md](CHARGE.md) |

**Traçabilité.** Toute prédiction remonte à sa source : `predictions.model_version` → version du registre (tags de
provenance) → run MLflow du candidat (paramètres, `git_sha`, empreinte de `uv.lock`) → campagne → run Prefect →
lots acceptés par Great Expectations → `dataset_version` et SHA-256 des fichiers. `make trace` affiche la chaîne.

## Configuration, secrets, réseau

- **Deux sortes de réglages.** `configs/*.yaml` décrit l'expérience (rejeu, contrôles, grille, règle de mise en
  service) : versionné, tracé dans MLflow, clés inconnues refusées. `.env` décrit l'environnement (adresses, mots de
  passe de démo, webhook Teams) : jamais commité, créé depuis `.env.example` au premier `up`.
- **Le seul secret** est l'URL du webhook Teams : facultative, lue dans `.env`, jamais écrite (ni logs, ni tags, ni
  messages d'erreur).
- **Moindre privilège.** Un compte PostgreSQL par usage : `tennis` (ingestion, entraînement), `app_writer` (l'API :
  écrit ses traces, ne fait que lire les matchs), `grafana_reader` (lecture seule), `prefect`, `mlflow`. L'interface
  ne reçoit ni `.env` ni identifiants.
- **Rien n'est exposé** : tous les ports sont publiés sur `127.0.0.1`, décalés (5433, 4201, 7282…) pour coexister avec
  la stack des TP.
- **Ressources bornées** : chaque service a une limite mémoire (`compose.yml`) ; le dimensionnement de l'API est
  mesuré dans [SYNTHESE_OPS.md](SYNTHESE_OPS.md).
