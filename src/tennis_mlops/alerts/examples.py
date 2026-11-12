"""Exemples réalistes de chaque carte : aperçu avant tout envoi (`make alert-preview`) et message de test."""

from __future__ import annotations

from datetime import UTC, datetime

from tennis_mlops.alerts.cards import Alert, Tone

HOST = "MacBook-Pro-de-Lucas.local"
AT = datetime(2026, 11, 12, 11, 6, tzinfo=UTC)
PREFECT = "http://127.0.0.1:4201/runs/flow-run/5b0e8d1c-3f2a-4f7e-9a51-2c8f0e6d4b17"
MLFLOW = "http://127.0.0.1:7282/#/experiments/1/runs/7756950cb2ca40dda03e826b824a055a"
PROJECT = ("Projet", f"tennis-mlops · Lucas · poste {HOST}")


def training_failed() -> Alert:
    """Le flow d'entraînement échoue (ici : trop peu de matchs pour découper train / validation / test)."""
    return Alert(
        tone=Tone.FAILURE,
        title="❌ Échec de l'entraînement",
        subtitle="entrainement-atp · olive-falcon (déclenché par le lot 2011-0339)",
        status="État Failed · un incident résolu par les retries ne déclenche rien",
        reason="ValueError : partition « train » trop petite (212 matchs, minimum 300). Les données ne couvrent pas "
        "encore assez de mois depuis 2011-01-01.",
        highlights=(
            ("Étape en erreur", "TemporalSplitter.split"),
            ("Données jusqu'au", "06/03/2011"),
            ("Production", "v9 inchangée"),
        ),
        facts=(
            ("Flow run", "5b0e8d1c-3f2a-4f7e-9a51-2c8f0e6d4b17"),
            ("Campagne MLflow", "7756950cb2ca…  (FAILED)"),
            PROJECT,
            ("Événement", "0b6f…e21 (prefect.flow-run.Failed), pour reconnaître un doublon"),
        ),
        next_steps="Rien n'est cassé côté service : l'API garde v9. Attendre plus de données, ou avancer "
        "`split.train_start` dans `configs/train.yaml`.",
        links=(("Ouvrir le run dans Prefect", PREFECT), ("Ouvrir la campagne dans MLflow", MLFLOW)),
        host=HOST,
        at=AT,
    )


def lot_quarantined() -> Alert:
    """Great Expectations refuse un lot : rien n'est écrit en base, le flow d'ingestion échoue."""
    return Alert(
        tone=Tone.FAILURE,
        title="🚫 Lot refusé par Great Expectations",
        subtitle="ingestion-atp · Roland Garros 2016 (lot 2016-520)",
        status="État Failed · lot archivé en quarantaine, aucun match écrit en base",
        reason_title="Contrôles en échec (suite bloquante)",
        reason="- **Score lisible** : 3 matchs, ex. `6-4 6-x`, `7-6(3) ?`\n"
        "- **Vainqueur ≠ perdant** : 1 match (joueur 104925 contre lui-même)",
        highlights=(("Lot", "2016-520"), ("Matchs reçus", "127 sur 127"), ("Suite GX", "v1.0")),
        facts=(
            ("Flow run", "9c41d2aa-7b0e-4c55-8e3f-1d2b6a90c7e4"),
            ("Quarantaine", "state/quarantine/2016-520/"),
            ("Rapport détaillé", "make gx-docs"),
            PROJECT,
            ("Événement", "51ad…9f0 (prefect.flow-run.Failed)"),
        ),
        next_steps="Le watermark n'a pas avancé : l'entraînement ne voit pas ce lot. Corriger la source, puis "
        "`make ingest` rejoue le lot ; `make lots` montre son statut.",
        links=(("Ouvrir le run dans Prefect", PREFECT),),
        host=HOST,
        at=AT,
    )


def ingestion_degraded() -> Alert:
    """La source ne répond plus après tous les essais : rien n'est faux, on attend."""
    return Alert(
        tone=Tone.WARNING,
        title="⚠️ Ingestion dégradée : source injoignable",
        subtitle="ingestion-atp · amber-otter",
        status="État Dégradé · 4 tentatives épuisées, la chaîne attend la source",
        reason="Source indisponible : ConnectError sur GET /tournaments (http://mock_source:8001).",
        highlights=(
            ("Données en base jusqu'au", "27/11/2016"),
            ("Horloge simulée", "31/12/2016"),
            ("Lots validés", "282"),
        ),
        facts=(
            ("Flow run", "e2f7a9b3-6c1d-4a8e-b0f5-3d9c2e7a1b64"),
            PROJECT,
            ("Événement", "a3c9…7d2 (prefect.flow-run.Dégradé)"),
        ),
        next_steps="Les données déjà en base restent utilisables (entraînement, API). Vérifier mock_source "
        "(`make logs`) ; la prochaine exécution planifiée reprendra au watermark.",
        links=(("Ouvrir le run dans Prefect", PREFECT),),
        host=HOST,
        at=AT,
    )


def rollback() -> Alert:
    """Une personne remet la version précédente en production."""
    return Alert(
        tone=Tone.WARNING,
        title="↩️ Rollback : production v11 → v9",
        subtitle="atp-winner · décidé à la main (make rollback)",
        status="L'API suit l'alias production : elle sert v9 sans redéploiement",
        reason_title="Raison",
        reason="Probabilités aberrantes sur la terre battue depuis la mise en service de v11.",
        highlights=(("Retirée (rolled_back)", "v11"), ("Servie", "v9"), ("Rollback suivant", "impossible")),
        facts=(("Journal", "rollback v11 → v9 (atp-winner-decisions)"), PROJECT),
        next_steps="Analyser v11 (`make trace VERSION=11`) ; elle reste dans le registre, `make promote` la "
        "remettrait en service.",
        links=(("Ouvrir le registre dans MLflow", "http://127.0.0.1:7282/#/models/atp-winner"),),
        host=HOST,
        at=AT,
    )


def model_promoted() -> Alert:
    """La règle de mise en service accepte une nouvelle version."""
    return Alert(
        tone=Tone.SUCCESS,
        title="✅ Nouveau modèle en production : atp-winner v11",
        subtitle="remplace v9 · campagne entrainement-atp du 12/11 (8 candidats)",
        status="Règle de mise en service : 7 règles sur 7",
        reason_title="Sur le même test (6 derniers mois)",
        reason="Log loss **0,6002** contre 0,6005 pour v9 et 0,6125 pour l'Elo pur ; aucune surface dégradée.",
        highlights=(("Log loss", "0,6002"), ("Accuracy", "67,1 %"), ("Modèle", "logreg C=0.05")),
        facts=(
            ("Campagne MLflow", "7756950cb2ca40dda03e826b824a055a"),
            ("Données jusqu'au", "27/11/2016"),
            PROJECT,
        ),
        links=(("Rapport de campagne", MLFLOW + "/artifacts/rapport_campagne.html"),),
        host=HOST,
        at=AT,
    )


EXAMPLES = {
    "echec": training_failed,
    "quarantaine": lot_quarantined,
    "degrade": ingestion_degraded,
    "rollback": rollback,
    "promotion": model_promoted,
}
