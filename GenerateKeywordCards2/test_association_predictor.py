from pathlib import Path

import numpy as np
import pytest

from .association_predictor import RegressionAssociationPredictor
from .embeddings import WordEmbeddings


def _fitted_predictor() -> RegressionAssociationPredictor:
    embeddings = WordEmbeddings(["word", "associated"], np.array([[1.0], [1.0]]))
    return RegressionAssociationPredictor(
        embeddings, np.array([2.0]), intercept=0.0, feature_scale=1.0
    )


def test_regression_predictor_threshold_can_be_tuned() -> None:
    predictor = _fitted_predictor()

    assert predictor("word", "associated", 0.5) is True
    assert predictor("word", "associated", 0.9) is False


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.1])
def test_regression_predictor_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        _fitted_predictor()("word", "associated", threshold)


def test_regression_predictor_model_round_trip(tmp_path: Path) -> None:
    predictor = _fitted_predictor()
    model_path = tmp_path / "model.json"
    predictor.save_model(str(model_path))

    loaded_predictor = RegressionAssociationPredictor.load_model(
        predictor.embeddings, str(model_path)
    )

    assert loaded_predictor("word", "associated", 0.5) is True
    assert loaded_predictor("word", "associated", 0.9) is False
