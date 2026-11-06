from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tennis_mlops.features import (
    FEATURE_COLUMNS,
    LABEL,
    FeatureBuilder,
    FeatureConfig,
    MatchContext,
    PlayerSnapshot,
    player1_is_winner,
)
from tests.support.synthetic import synthetic_matches


def row(match_id: str, day: date, winner: int, loser: int, **extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "match_id": match_id, "tourney_id": "T", "tourney_name": "Test Open", "tourney_level": "A",
        "surface": "Hard", "best_of": 3, "round": "R32", "match_num": int(match_id.split("-")[-1]),
        "match_date": day, "status": "completed",
        "winner_id": winner, "winner_name": f"J{winner}", "winner_hand": "R", "winner_ht": 185, "winner_age": 25.0,
        "winner_rank": 10, "winner_rank_points": 2000,
        "loser_id": loser, "loser_name": f"J{loser}", "loser_hand": "L", "loser_ht": 180, "loser_age": 28.0,
        "loser_rank": 40, "loser_rank_points": 800,
    }  # fmt: skip
    base.update(extra)
    return base


def by_id(frame: pd.DataFrame, match_id: str) -> "pd.Series[Any]":
    found = frame.set_index("match_id").loc[match_id]
    assert isinstance(found, pd.Series)
    return found


def test_orientation_is_deterministic_and_balanced() -> None:
    ids = [f"2015-{t:03d}-{n}" for t in range(200) for n in range(1, 30)]
    share = np.mean([player1_is_winner(i) for i in ids])
    assert 0.47 < share < 0.53
    assert player1_is_winner("2015-001-1") == player1_is_winner("2015-001-1")


def test_label_and_features_follow_the_orientation() -> None:
    matches = pd.DataFrame([row(f"T-{n}", date(2015, 1, 5), 1, 2) for n in range(1, 30)])
    frame = FeatureBuilder().build(matches)
    winner_first = frame["match_id"].map(player1_is_winner)
    assert (frame[LABEL] == winner_first.astype(int)).all()
    # le joueur 1 est le mieux classé (rang 10 contre 40) exactement quand il est le vainqueur
    assert ((frame["rank_log_ratio"] > 0) == winner_first).all()


def test_a_match_never_sees_its_own_result_nor_the_future() -> None:
    history = synthetic_matches(weeks=30, matches_per_week=20)
    day = sorted(history["match_date"].unique())[80]
    flipped = history.copy()
    later = flipped["match_date"] >= day  # on inverse tous les résultats à partir de ce jour
    for field in ("id", "name", "hand", "ht", "age", "rank", "rank_points"):
        w, lo = f"winner_{field}", f"loser_{field}"
        flipped.loc[later, [w, lo]] = flipped.loc[later, [lo, w]].to_numpy()
    original, altered = FeatureBuilder().build(history), FeatureBuilder().build(flipped)
    same_day = original["match_date"] == pd.Timestamp(day)
    assert same_day.sum() > 0
    a = original.loc[same_day].set_index("match_id")[["elo_diff", "form_diff", "h2h_diff", "activity_diff"]]
    b = altered.loc[same_day].set_index("match_id")[["elo_diff", "form_diff", "h2h_diff", "activity_diff"]]
    # les joueurs sont échangés (le joueur 1 reste « le vainqueur » selon le hash) : seule la valeur absolue compte
    assert np.allclose(a.abs().to_numpy(), b.abs().to_numpy())


def test_two_matches_on_the_same_day_do_not_see_each_other() -> None:
    day = date(2015, 1, 5)
    frame = FeatureBuilder().build(pd.DataFrame([row("T-1", day, 1, 2), row("T-2", day, 1, 3)]))
    assert (frame["elo_diff"] == 0).all()
    assert (frame["experience_diff"] == 0).all()


def test_elo_rewards_the_winner_from_the_next_day() -> None:
    frame = FeatureBuilder().build(
        pd.DataFrame([row("T-1", date(2015, 1, 5), 1, 2), row("T-2", date(2015, 1, 6), 1, 2)])
    )
    second = by_id(frame, "T-2")
    elo_p1_minus_p2 = second["elo_diff"] if player1_is_winner("T-2") else -second["elo_diff"]
    # K = 250 / 5^0,4 ≈ 131,3 pour un débutant, gain = K * 0,5 de chaque côté
    assert elo_p1_minus_p2 == pytest.approx(2 * 250 / 5**0.4 * 0.5)
    assert second["elo_win_prob"] != 0.5
    assert abs(second["h2h_diff"]) == 1
    assert second["h2h_total"] == 1


def test_walkovers_neither_train_nor_change_the_state() -> None:
    matches = pd.DataFrame([row("T-1", date(2015, 1, 5), 1, 2, status="walkover"), row("T-2", date(2015, 1, 6), 1, 2)])
    frame = FeatureBuilder().build(matches)
    assert list(frame["match_id"]) == ["T-2"]
    assert frame["elo_diff"].iloc[0] == 0


def test_form_activity_and_rest_days() -> None:
    config = FeatureConfig(form_window=2, activity_days=30, max_rest_days=100)
    matches = pd.DataFrame(
        [
            row("T-1", date(2015, 1, 1), 1, 9),
            row("T-2", date(2015, 1, 2), 1, 9),
            row("T-3", date(2015, 1, 3), 1, 9),
            row("T-4", date(2015, 3, 1), 1, 2),  # le joueur 2 n'a jamais joué
        ]
    )
    last = by_id(FeatureBuilder(config).build(matches), "T-4")
    sign = 1 if player1_is_winner("T-4") else -1
    assert sign * last["form_diff"] == pytest.approx(3 / 4 - 1 / 2)  # 2 victoires sur 2 (lissé) contre 0,5
    assert last["activity_diff"] == 0  # les matchs de janvier sont sortis de la fenêtre de 30 jours
    assert sign * last["rest_days_diff"] == pytest.approx((date(2015, 3, 1) - date(2015, 1, 3)).days - 100)


def test_implausible_heights_are_unknown() -> None:
    frame = FeatureBuilder().build(pd.DataFrame([row("T-1", date(2015, 1, 5), 1, 2, winner_ht=3)]))
    assert np.isnan(frame["height_diff"].iloc[0])


def test_serving_features_equal_training_features() -> None:
    history = synthetic_matches(weeks=12, matches_per_week=12)
    target = history.iloc[-1]
    full = FeatureBuilder().build(history)
    serving = FeatureBuilder()
    serving.build(history[history["match_date"] < target["match_date"]])
    winner_first = player1_is_winner(str(target["match_id"]))
    sides = ("winner", "loser") if winner_first else ("loser", "winner")
    p1, p2 = (
        PlayerSnapshot(
            player_id=int(target[f"{s}_id"]), rank=float(target[f"{s}_rank"]),
            rank_points=float(target[f"{s}_rank_points"]), age=float(target[f"{s}_age"]),
            height_cm=float(target[f"{s}_ht"]), hand=str(target[f"{s}_hand"]),
        )
        for s in sides
    )  # fmt: skip
    context = MatchContext(
        match_date=target["match_date"], surface=target["surface"], tourney_level=target["tourney_level"],
        round=target["round"], best_of=int(target["best_of"]),
    )  # fmt: skip
    served = serving.features_for(context, p1, p2)
    expected = full.set_index("match_id").loc[[target["match_id"]], list(FEATURE_COLUMNS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(served, expected)


def test_history_must_move_forward() -> None:
    builder = FeatureBuilder()
    builder.build(pd.DataFrame([row("T-1", date(2015, 1, 5), 1, 2)]))
    assert builder.last_day == date(2015, 1, 5)
    assert builder.players == 2
    with pytest.raises(ValueError, match="ordre rompu"):
        builder.build(pd.DataFrame([row("T-2", date(2015, 1, 5), 1, 3)]))
    with pytest.raises(ValueError, match="pas postérieur"):
        builder.features_for(MatchContext(match_date=date(2015, 1, 4)), PlayerSnapshot(1), PlayerSnapshot(2))


def test_columns_have_stable_types() -> None:
    frame = FeatureBuilder().build(synthetic_matches(weeks=3, matches_per_week=8))
    assert list(frame.columns[: len(FEATURE_COLUMNS)]) == list(FEATURE_COLUMNS)
    assert frame["best_of"].dtype == "float64"
    assert frame["surface"].dtype == object
