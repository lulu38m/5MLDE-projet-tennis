from datetime import date

import pytest

from mock_source.clock import ClockOutOfRangeError, SimulatedClock

from .conftest import FakeTime

LOWER, UPPER = date(2015, 1, 1), date(2015, 12, 31)


def make(speed: float = 0.0) -> tuple[SimulatedClock, FakeTime]:
    fake = FakeTime()
    return SimulatedClock(date(2015, 3, 1), LOWER, UPPER, speed=speed, time_source=fake), fake


def test_frozen_by_default() -> None:
    clock, fake = make()
    fake.seconds = 10_000
    assert clock.now() == date(2015, 3, 1)


def test_speed_is_in_simulated_days_per_real_minute() -> None:
    clock, fake = make(speed=10)
    fake.seconds = 30  # une demi-minute → 5 jours
    assert clock.now() == date(2015, 3, 6)


def test_speed_is_clamped_to_the_upper_bound() -> None:
    clock, fake = make(speed=1000)
    fake.seconds = 3600
    assert clock.now() == UPPER


def test_set_date_and_speed() -> None:
    clock, fake = make()
    assert clock.set(date(2015, 6, 1)).now == date(2015, 6, 1)
    snapshot = clock.set(speed=2)
    assert (snapshot.now, snapshot.speed) == (date(2015, 6, 1), 2)
    fake.seconds = 60
    assert clock.now() == date(2015, 6, 3)


def test_advance_returns_previous_and_new_date() -> None:
    clock, _ = make()
    assert clock.advance(7) == (date(2015, 3, 1), date(2015, 3, 8))


@pytest.mark.parametrize("when", [date(2014, 12, 31), date(2016, 1, 1)])
def test_out_of_range_dates_are_rejected(when: date) -> None:
    clock, _ = make()
    with pytest.raises(ClockOutOfRangeError):
        clock.set(when)


def test_invalid_arguments() -> None:
    clock, _ = make()
    with pytest.raises(ValueError, match="days"):
        clock.advance(0)
    with pytest.raises(ValueError, match="speed"):
        clock.set(speed=-1)
    with pytest.raises(ClockOutOfRangeError):
        clock.advance(400)
    with pytest.raises(ValueError, match="lower"):
        SimulatedClock(date(2015, 1, 1), UPPER, LOWER)
