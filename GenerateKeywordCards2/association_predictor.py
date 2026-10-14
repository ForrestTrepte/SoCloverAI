import gc
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from GenerateKeywordCards2.embeddings import WordEmbeddings
from GenerateKeywordCards2.swow import SWOWAssociations


def _read_int(path: str) -> int | None:
    try:
        with open(path) as f:
            text = f.read().strip()
        return None if text == "max" else int(text)
    except (OSError, ValueError):
        return None


def _log_free_memory(label: str) -> None:
    """Prints available memory, both host-wide and within the container's cgroup limit."""
    available = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
    except OSError:
        pass
    limit = _read_int("/sys/fs/cgroup/memory.max")
    used = _read_int("/sys/fs/cgroup/memory.current")
    gb = 1024**3
    parts = []
    if available is not None:
        parts.append(f"host available {available / gb:.2f} GB")
    if limit is not None and used is not None:
        parts.append(f"cgroup free {(limit - used) / gb:.2f} GB of {limit / gb:.2f} GB")
    print(f"[memory] {label}: " + ", ".join(parts))


def get_distance_threshold_association_predictor(
    embeddings: WordEmbeddings, distance_threshold: float
) -> Callable[[str, str], bool]:
    def predictor(word: str, associated_word: str) -> bool:
        return embeddings.distance(word, associated_word) <= distance_threshold

    return predictor


@dataclass
class PredictorTrainingDataRow:
    word_a: str
    embedding_a: np.ndarray
    word_b: str
    embedding_b: np.ndarray
    is_association: bool


class PredictorTrainingData:
    def __init__(self, swow: SWOWAssociations, embeddings: WordEmbeddings):
        self.embeddings = embeddings
        association_rows = []
        for word_a, associations in swow.associations.items():
            word_a_associations = swow.associations[word_a].get_associations(True, True)
            for word_b in word_a_associations:
                association_rows.append(
                    PredictorTrainingDataRow(
                        word_a,
                        embeddings.get_embedding(word_a),
                        word_b,
                        embeddings.get_embedding(word_b),
                        is_association=True,
                    )
                )

        nonassociation_rows: list[PredictorTrainingDataRow] = []
        seeded_random = random.Random(123)
        swow_words = list(swow.associations.keys())
        while len(nonassociation_rows) < len(association_rows):
            word_a = seeded_random.choice(swow_words)
            word_b = seeded_random.choice(swow_words)
            if word_a == word_b:
                continue
            if word_b in swow.associations[word_a].forward:
                continue
            if word_b in swow.associations[word_a].backward:
                continue
            assert word_a not in swow.associations[word_b].forward
            assert word_a not in swow.associations[word_b].backward

            nonassociation_rows.append(
                PredictorTrainingDataRow(
                    word_a,
                    embeddings.get_embedding(word_a),
                    word_b,
                    embeddings.get_embedding(word_b),
                    is_association=False,
                )
            )

        held_out_percentage = 0.2
        held_out_count = round(len(association_rows) * held_out_percentage)

        seeded_random.shuffle(association_rows)
        held_out_association_rows = association_rows[:held_out_count]
        association_rows = association_rows[held_out_count:]

        seeded_random.shuffle(nonassociation_rows)
        held_out_nonassociation_rows = nonassociation_rows[:held_out_count]
        nonassociation_rows = nonassociation_rows[held_out_count:]

        self.training_rows = association_rows + nonassociation_rows
        seeded_random.shuffle(self.training_rows)
        self.held_out_rows = held_out_association_rows + held_out_nonassociation_rows
        seeded_random.shuffle(self.held_out_rows)

    def get_regression_association_predictor(
        self, probability_threshold: float, max_training_rows: int, max_iter: int = 1000
    ) -> Callable[[str, str], bool]:
        """
        Fits a logistic regression on the element-wise product of the two words' embeddings.
        Since embeddings are L2-normalized, summing those products gives cosine similarity,
        so equal weights reproduce the distance-threshold predictor; learned weights let the
        model emphasize the dimensions that matter most for human associations.
        """
        max_training_rows = min(max_training_rows, len(self.training_rows))
        training_rows = self.training_rows[:max_training_rows]
        # Products of unit-norm vectors are ~1/dim in magnitude, which L2 regularization
        # would flatten to zero weights (constant predictions); rescale to O(1).
        dim = len(training_rows[0].embedding_a)
        feature_scale = float(dim)
        _log_free_memory(f"before building features ({max_training_rows} rows)")
        features = np.empty((len(training_rows), dim), dtype=np.float32)
        for i, row in enumerate(training_rows):
            np.multiply(row.embedding_a, row.embedding_b, out=features[i])
        features *= feature_scale
        labels = np.array([row.is_association for row in training_rows], dtype=np.float64)
        del training_rows
        _log_free_memory(
            f"after building features ({features.nbytes / 1024**3:.2f} GB array)"
        )

        # Same objective as LogisticRegression(C=1): sum of log-loss + 0.5*|w|^2, with an
        # unpenalized intercept. sklearn's lbfgs would copy the whole matrix to float64
        # (doubling memory), so evaluate loss and gradient over float64 chunks instead.
        chunk_size = 8192

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            w, b = params[:-1], params[-1]
            loss = 0.5 * float(w @ w)
            grad = np.zeros_like(params)
            grad[:-1] = w
            for start in range(0, len(features), chunk_size):
                x = features[start : start + chunk_size].astype(np.float64)
                y = labels[start : start + chunk_size]
                z = x @ w + b
                loss += float(np.sum(np.logaddexp(0.0, z) - y * z))
                residual = 1.0 / (1.0 + np.exp(-z)) - y
                grad[:-1] += x.T @ residual
                grad[-1] += residual.sum()
            return loss, grad

        _log_free_memory("before fit")
        result = minimize(
            objective,
            np.zeros(dim + 1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter},
        )
        print(f"  L-BFGS: {result.nit} iterations, {result.message}")
        _log_free_memory("after fit")
        del features
        gc.collect()
        weights = result.x[:-1]
        intercept = result.x[-1]

        # Evaluate the fitted model directly rather than via predict_proba, which has a lot
        # of per-call overhead for single rows: probability >= p  <=>  logit >= log(p/(1-p)).
        embeddings = self.embeddings
        logit_threshold = np.log(probability_threshold / (1 - probability_threshold))

        def predictor(word: str, associated_word: str) -> bool:
            feature = embeddings.get_embedding(word) * embeddings.get_embedding(
                associated_word
            )
            return bool(
                feature * feature_scale @ weights + intercept >= logit_threshold
            )

        return predictor
