import logging
import warnings

from tennis_mlops.log import quiet_training_noise


def test_training_noise_is_silenced_and_nothing_else() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet_training_noise()
        warnings.warn_explicit("Unknown solver options: iprint", UserWarning, "_logistic.py", 451,
                               module="sklearn.linear_model._logistic")  # fmt: skip
        warnings.warn_explicit("Unknown solver options: iprint", UserWarning, "ours.py", 1, module="tennis_mlops.x")
    assert [str(w.message) for w in caught] == ["Unknown solver options: iprint"]  # seule la nôtre passe
    assert logging.getLogger("mlflow.utils.environment").getEffectiveLevel() == logging.ERROR
