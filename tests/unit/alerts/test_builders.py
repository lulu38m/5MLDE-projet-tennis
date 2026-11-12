"""Des faits du pipeline aux cartes : chaque famille dit l'essentiel, avec les bons liens et sans détail inventé."""

from datetime import UTC, datetime

from tennis_mlops.alerts.builders import (
    AlertKind,
    CampaignLink,
    CardContext,
    FlowRunInfo,
    PrefectEvent,
    RegistryNotice,
    Situation,
    check_lines,
    classify,
    degraded_alert,
    failure_alert,
    launcher,
    promotion_alert,
    quarantine_alert,
    rollback_alert,
)
from tennis_mlops.alerts.cards import Tone, adaptive_card, card_texts

CTX = CardContext(
    host="mac.local",
    prefect_ui="http://127.0.0.1:4201",
    mlflow_ui="http://127.0.0.1:7282/",
    at=datetime(2026, 11, 12, 10, 0, tzinfo=UTC),
    owner="Lucas",
)
EVENT = PrefectEvent("0b6f2c1d-aaaa-bbbb-cccc-1234567890ab", "prefect.flow-run.Failed")


def run(flow: str = "entrainement-atp", state: str = "Failed", **kw: object) -> FlowRunInfo:
    return FlowRunInfo(id="run-1", name="olive-falcon", flow_name=flow, state_type=state.upper(), state_name=state,
                       **kw)  # type: ignore[arg-type]  # fmt: skip


def texts(alert: object) -> str:
    return "\n".join(card_texts(adaptive_card(alert)))  # type: ignore[arg-type]


def test_training_failure_links_prefect_and_the_mlflow_campaign() -> None:
    info = run(message="ValueError : partition trop petite", failed_task="TemporalSplitter.split",
               started_at=datetime(2026, 11, 12, 9, 58, tzinfo=UTC))  # fmt: skip
    alert = failure_alert(info, EVENT, CTX, Situation(production="9", data_end="2016-11-27"),
                          CampaignLink("7756950cb2ca40dd", "3", "FAILED"))  # fmt: skip
    assert alert.title == "❌ Échec de l'entraînement" and alert.tone is Tone.FAILURE
    assert alert.highlights == (("Étape en erreur", "TemporalSplitter.split"), ("Données jusqu'au", "27/11/2016"),
                                ("Production", "v9 inchangée"))  # fmt: skip
    assert [url for _, url in alert.links] == [
        "http://127.0.0.1:4201/runs/flow-run/run-1",
        "http://127.0.0.1:7282/#/experiments/3/runs/7756950cb2ca40dd",
    ]
    assert ("Projet", "tennis-mlops · Lucas · poste mac.local") in alert.facts
    assert "0b6f…0ab (prefect.flow-run.Failed), pour reconnaître un doublon" in dict(alert.facts)["Événement"]
    assert "démarré le 12/11/2026 à 10:58" in alert.subtitle  # heure de Paris


def test_crash_and_other_flows_say_who_launched_them() -> None:
    crashed = failure_alert(run("simulation-atp", "Crashed", tags=("auto-scheduled",)), EVENT, CTX)
    assert crashed.title == "💥 Arrêt brutal de la simulation"
    assert dict(crashed.highlights) == {"Étape en erreur": "hors task", "Lancé par": "planning", "Production": "aucune"}
    assert launcher(run(automation=True)) == "automation"
    assert launcher(run(deployment="manuelle")) == "déploiement manuelle"
    assert launcher(run()) == "commande locale"
    assert failure_alert(run(message=""), EVENT, CTX).reason.startswith("non renseignée")


def test_quarantine_card_lists_failed_checks_in_plain_words() -> None:
    lot = {
        "lot_id": "2016-520", "tourney_name": "Roland Garros", "tourney_level": "G", "rows_received": 127,
        "rows_expected": 127, "archive": "state/quarantine/2016-520/",
        "failed_checks": [
            {"expectation": "expect_tennis_score_to_be_valid", "target": "score", "unexpected_count": 3,
             "sample": ["6-4 6-x", "7-6(3) ?", "x"]},
            {"expectation": "expect_batch_volume_to_match_draw", "target": "table", "sample": ["observé : 31"]},
            {"expectation": "expect_column_values_to_be_in_set", "target": "surface", "error": "valeur illisible"},
        ],
    }  # fmt: skip
    info = run("ingestion-atp")
    assert classify(info, [lot]) is AlertKind.QUARANTINE
    alert = quarantine_alert(info, [lot, {"lot_id": "2016-521"}], EVENT, CTX, "1.0")
    assert alert.subtitle == "ingestion-atp · Roland Garros · Grand Chelem (lot 2016-520)"
    assert alert.reason.splitlines()[:3] == [
        "- **Score lisible** (score) : 3 matchs, ex. `6-4 6-x`, `7-6(3) ?`",
        "- **Volume = taille du tableau** : observé : 31",
        "- **Valeur autorisée** (surface) : valeur illisible",
    ]
    assert "Autres lots refusés pendant ce passage : 2016-521." in alert.reason
    assert dict(alert.highlights) == {"Lot": "2016-520", "Matchs reçus": "127 sur 127", "Suite GX": "v1.0"}
    shown = texts(alert)
    assert "«\N{NO-BREAK SPACE}6-4 6-x\N{NO-BREAK SPACE}»" in shown and "`" not in shown  # Teams : pas de code


def test_a_broken_schema_names_the_missing_columns_first() -> None:
    checks: list[dict[str, object]] = [
        {"expectation": "expect_column_values_to_not_be_null", "target": "match_id"},
        {"expectation": "expect_winner_and_loser_to_differ", "target": "winner_id/loser_id"},
        {"expectation": "expect_column_to_exist", "target": "match_id"},
        {"expectation": "expect_column_values_to_be_in_type_list", "target": "minutes", "unexpected_count": 127,
         "sample": ["109 min", "138 min", "90 min"]},
        {"expectation": "expect_column_to_exist", "target": "winner_id"},
        {"expectation": "expect_column_values_to_be_in_type_list", "target": "winner_id", "error": "list index"},
    ]  # fmt: skip
    assert check_lines(checks).splitlines() == [
        "- **Colonnes absentes** : match_id, winner_id",
        "- **Type attendu** (minutes) : 127 matchs, ex. `109 min`, `138 min`",
        "- 3 contrôle(s) sur ces colonnes, en échec par conséquence",
    ]
    many = [{"expectation": "expect_column_values_to_be_in_set", "target": f"c{i}", "sample": ["x"]} for i in range(7)]
    assert check_lines(many).splitlines()[-1] == "- … et 2 autre(s) contrôle(s) : détail dans `make gx-docs`"


def test_unknown_checks_and_empty_lists_still_read_well() -> None:
    assert check_lines([{"expectation": "expect_something_new", "target": "x"}]) == "- **something new** (x)"
    assert "rapport Great Expectations" in check_lines([])


def test_degraded_card_tells_what_is_still_usable() -> None:
    info = run("ingestion-atp", "Dégradé", message="Source indisponible : ConnectError")
    assert classify(info, []) is AlertKind.DEGRADED
    alert = degraded_alert(info, Situation(data_end="2016-11-27", watermark="2016-12-31", lots_valid=282), EVENT, CTX)
    assert alert.tone is Tone.WARNING and alert.title.startswith("⚠️ ")
    assert alert.highlights == (("Données en base jusqu'au", "27/11/2016"), ("Ingéré jusqu'au", "31/12/2016"),
                                ("Lots validés", "282"))  # fmt: skip
    assert classify(run("ingestion-atp"), []) is AlertKind.FAILURE


def notice(**kw: object) -> RegistryNotice:
    base: dict[str, object] = {"kind": "promotion", "model": "atp-winner", "version": "11", "production_before": "9",
                               "production_after": "11", "at": "2026-11-12T10:00:00+00:00"}  # fmt: skip
    return RegistryNotice.model_validate({**base, **kw})


def test_rollback_card_says_if_another_rollback_is_possible() -> None:
    rolled = notice(kind="rollback", version="9", production_before="11", production_after="9", reason="plaintes",
                    metrics={"previous_after": "7"})  # fmt: skip
    alert = rollback_alert(rolled, CTX)
    assert alert.title == "↩️ Rollback : production v11 → v9" and alert.reason == "plaintes"
    assert dict(alert.highlights)["Rollback suivant"] == "vers v7"
    assert alert.links == (("Ouvrir le registre dans MLflow", "http://127.0.0.1:7282/#/models/atp-winner"),)
    last = rollback_alert(notice(kind="rollback", production_before="11", production_after="9"), CTX)
    assert dict(last.highlights)["Rollback suivant"] == "impossible"


def test_promotion_card_compares_on_the_same_test() -> None:
    rules = [{"rule": "segments", "passed": True, "applicable": True}, {"rule": "elo", "passed": True}]
    metrics = {
        "log_loss": 0.60021, "accuracy": 0.6712, "candidate": "logreg C=0.05", "candidates": 8,
        "data_end": "2016-11-27", "experiment_id": "1", "elo_log_loss": 0.6125, "production_log_loss": 0.6005,
    }  # fmt: skip
    alert = promotion_alert(notice(gate={"rules": rules}, metrics=metrics, campaign_run_id="abc"), CTX)
    assert alert.tone is Tone.SUCCESS and alert.title == "✅ Nouveau modèle en production : atp-winner v11"
    assert alert.subtitle == "remplace v9 · campagne entrainement-atp du 12/11 (8 candidats)"
    assert alert.status == "Règle de mise en service : 2 règles sur 2"
    assert (
        alert.reason == "Log loss **0,6002** contre 0,6005 pour v9 et 0,6125 pour l'Elo pur ; aucune surface dégradée."
    )
    assert dict(alert.highlights) == {"Log loss": "0,6002", "Accuracy": "67,1 %", "Modèle": "logreg C=0.05"}
    report = "http://127.0.0.1:7282/#/experiments/1/runs/abc/artifacts/rapport_campagne.html"
    assert alert.links[0] == ("Rapport de campagne", report)

    manual = promotion_alert(notice(kind="promotion manuelle", production_before=None, reason="bon en shadow"), CTX)
    assert manual.subtitle == "première version servie · décidé à la main (make promote)"
    assert manual.reason_title == "Raison" and manual.reason == "bon en shadow"
    assert dict(manual.highlights)["Log loss"] == "—"
