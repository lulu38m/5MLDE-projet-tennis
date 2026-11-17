"""Contre le serveur Prefect de test : lecture d'un flow run en échec, mise en place idempotente, démarrage/arrêt."""

from uuid import uuid4

import pytest
from prefect import flow, get_client, task

from tennis_mlops.alerts.prefect_reader import PrefectFlowRunReader, clean_message, flow_run_id_of, task_label
from tennis_mlops.config import DEFAULT_CONFIG_PATH, ProjectConfig
from tennis_mlops.orchestration.__main__ import EXIT_OK, main
from tennis_mlops.orchestration.setup import ALERTS_AUTOMATION, TRAINING_AUTOMATION, setup


@task(name="étape qui casse")
def breaks() -> None:
    msg = "boum"
    raise ValueError(msg)


@task(name="étape qui marche")
def works() -> int:
    return 1


@flow(name="flow-de-test-alertes")
def failing_flow() -> None:
    works()
    breaks()


def test_reader_finds_the_failed_task_and_the_cause() -> None:
    state = failing_flow(return_state=True)
    run_id = str(state.state_details.flow_run_id)
    info = PrefectFlowRunReader().flow_run(f"prefect.flow-run.{run_id}")
    assert info is not None
    assert (info.flow_name, info.state_type, info.state_name) == ("flow-de-test-alertes", "FAILED", "Failed")
    assert info.failed_task == "étape qui casse" and info.message == "ValueError : boum"
    assert info.deployment is None and not info.automation and info.started_at is not None
    assert PrefectFlowRunReader().flow_run(str(uuid4())) is None
    assert PrefectFlowRunReader().flow_run("pas-un-uuid") is None


def test_helpers() -> None:
    assert flow_run_id_of("prefect.flow-run.abc") == "abc"
    assert clean_message("Flow run encountered an exception: DataQualityError: lot refusé") == (
        "DataQualityError : lot refusé"
    )
    assert clean_message("Source indisponible : 503") == "Source indisponible : 503"
    assert task_label("TemporalSplitter.split-0", "0") == "TemporalSplitter.split"
    assert task_label("contrôles 2016-520", "a1b2") == "contrôles 2016-520"  # nom de run choisi : intact


def test_setup_is_idempotent_and_keeps_the_demo_running(capsys: pytest.CaptureFixture[str]) -> None:
    config = ProjectConfig.from_yaml(DEFAULT_CONFIG_PATH)
    with get_client(sync_client=True) as client:
        first = setup(client, config, 4, "http://alert_sink:8090")
        assert first[0] == "work pool tennis-pool créé"
        assert "déploiement simulation-atp/planifiee (en pause, toutes les 60 s)" in first
        assert main(["start"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "simulation lancée" in out and "aucun worker actif" in out  # personne pour exécuter : on le dit

        again = setup(client, config, 4, "http://alert_sink:8090")  # `make up` en pleine démo
        assert again[0] == "work pool tennis-pool : déjà là"
        assert "déploiement simulation-atp/planifiee (actif, toutes les 60 s)" in again
        assert f"automation « {ALERTS_AUTOMATION} » mise à jour" in again

        simulation = client.read_deployment_by_name("simulation-atp/planifiee")
        assert len(client.read_deployment_schedules(simulation.id)) == 1  # pas de doublon de planning
        assert simulation.concurrency_options is not None
        assert simulation.concurrency_options.collision_strategy.value == "CANCEL_NEW"
        training = client.read_deployment_by_name("entrainement-atp/sur-evenement")
        assert training.job_variables == {"env": {"PREFECT_TASKS_RUNNER_THREAD_POOL_MAX_WORKERS": "4"}}
        assert training.entrypoint == "tennis_mlops.flows.train:training_flow"
        names = sorted(a.name for a in client.read_automations())
        assert names == sorted([ALERTS_AUTOMATION, TRAINING_AUTOMATION])
        (retrain,) = client.read_automations_by_name(TRAINING_AUTOMATION)
        assert retrain.actions[0].deployment_id == training.id  # type: ignore[union-attr]
        (alerts,) = client.read_automations_by_name(ALERTS_AUTOMATION)
        assert alerts.trigger.expect == {  # type: ignore[union-attr]
            "prefect.flow-run.Failed", "prefect.flow-run.Crashed", "prefect.flow-run.Dégradé"
        }  # fmt: skip

        assert main(["stop"]) == EXIT_OK
        assert client.read_deployment_by_name("simulation-atp/planifiee").paused
    capsys.readouterr()
    assert main(["status"]) == EXIT_OK
    status = capsys.readouterr().out
    assert "simulation-atp/planifiee" in status and "en pause" in status and "derniers runs" in status
