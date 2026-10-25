import gc
import json
import random
from collections.abc import Callable, Collection
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
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


# A Predictor type. Takes two words and a threshold, returns whether they are associated.
PredictorType = Callable[[str, str, float], bool]


def get_distance_threshold_association_predictor(
    embeddings: WordEmbeddings,
) -> PredictorType:
    def predictor(word: str, associated_word: str, threshold: float) -> bool:
        return embeddings.distance(word, associated_word) <= threshold

    return predictor


class RegressionAssociationPredictor:
    """Callable fitted logistic-regression model for association prediction."""

    def __init__(
        self,
        embeddings: WordEmbeddings,
        weights: np.ndarray,
        intercept: float,
        feature_scale: float,
    ):
        self.embeddings = embeddings
        self.weights = weights
        self.intercept = intercept
        self.feature_scale = feature_scale

    def __call__(self, word: str, associated_word: str, threshold: float) -> bool:
        """Predict using a probability cutoff; higher cutoffs generally favor precision."""
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        logit_threshold = np.log(threshold / (1 - threshold))
        feature = self.embeddings.get_embedding(word) * self.embeddings.get_embedding(
            associated_word
        )
        return bool(
            feature * self.feature_scale @ self.weights + self.intercept
            >= logit_threshold
        )

    def save_model(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            json.dump(
                {
                    "weights": self.weights.tolist(),
                    "intercept": float(self.intercept),
                    "feature_scale": self.feature_scale,
                },
                f,
            )

    @classmethod
    def load_model(
        cls, embeddings: WordEmbeddings, file_path: str
    ) -> "RegressionAssociationPredictor":
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(
            embeddings,
            np.array(data["weights"]),
            float(data["intercept"]),
            float(data["feature_scale"]),
        )


@dataclass
class PredictorTrainingDataRow:
    word_a: str
    embedding_index_a: int
    word_b: str
    embedding_index_b: int
    is_association: bool


class PredictorTrainingData:
    def __init__(
        self,
        swow: SWOWAssociations,
        embeddings: WordEmbeddings,
        negative_weight: float | None,
    ):
        """
        Training data has equal numbers of association and sampled non-association rows.
        Each non-association row stands in for many real ones, so it is weighted by
        negative_weight when training. That makes the fit reflect the true (rare)
        association rate. None estimates it as (non-associated pairs) / (associated pairs);
        1.0 gives the unweighted, balanced behavior.
        """
        self.embeddings = embeddings
        association_rows = []
        for word_a in swow.associations:
            word_a_associations = swow.associations[word_a].get_associations(True, True)
            for word_b in word_a_associations:
                association_rows.append(
                    PredictorTrainingDataRow(
                        word_a,
                        embeddings.get_embedding_index(word_a),
                        word_b,
                        embeddings.get_embedding_index(word_b),
                        is_association=True,
                    )
                )

        if negative_weight is None:
            word_count = len(swow.associations)
            all_pair_count = word_count * (word_count - 1)
            negative_weight = (all_pair_count - len(association_rows)) / len(
                association_rows
            )
        self.negative_weight = negative_weight

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
                    embeddings.get_embedding_index(word_a),
                    word_b,
                    embeddings.get_embedding_index(word_b),
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

    def train_regression_association_predictor(
        self,
        max_training_rows: int,
        max_iter: int,
    ) -> RegressionAssociationPredictor:
        """
        Fits logistic regression on element-wise embedding products, gathering features in
        chunks during each objective evaluation to avoid storing the full feature matrix.
        Since embeddings are L2-normalized, summing those products gives cosine similarity,
        so equal weights reproduce the distance-threshold predictor; learned weights let the
        model emphasize the dimensions that matter most for human associations.
        """
        max_training_rows = min(max_training_rows, len(self.training_rows))
        training_rows = self.training_rows[:max_training_rows]
        # Products of unit-norm vectors are ~1/dim in magnitude, which L2 regularization
        # would flatten to zero weights (constant predictions); rescale to O(1).
        dim = self.embeddings.embeddings.shape[1]
        feature_scale = float(dim)
        idx_a = np.fromiter(
            (row.embedding_index_a for row in training_rows),
            dtype=np.intp,
            count=len(training_rows),
        )
        idx_b = np.fromiter(
            (row.embedding_index_b for row in training_rows),
            dtype=np.intp,
            count=len(training_rows),
        )
        labels = np.array(
            [row.is_association for row in training_rows], dtype=np.float64
        )
        # Normalize to mean 1 so the L2 penalty keeps the same relative strength as unweighted.
        sample_weights = np.where(labels > 0, 1.0, self.negative_weight)
        sample_weights *= len(sample_weights) / sample_weights.sum()
        del training_rows
        embedding_matrix = self.embeddings.embeddings.astype(np.float32, copy=False)
        index_bytes = idx_a.nbytes + idx_b.nbytes
        _log_free_memory(
            f"before fit ({len(idx_a):,} rows, {index_bytes:,} bytes of indices)"
        )

        # Same objective as LogisticRegression(C=1) with sample weights: weighted sum of log-loss + 0.5*|w|^2, with an
        # unpenalized intercept. Rebuild features by chunk so memory stays bounded.
        chunk_size = 8192
        last_objective_params: np.ndarray | None = None
        last_objective_loss: float | None = None

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal last_objective_params, last_objective_loss
            w, b = params[:-1], params[-1]
            w_float32 = w.astype(np.float32)
            b_float32 = np.float32(b)
            loss = 0.5 * float(w @ w)
            grad = np.zeros_like(params)
            grad[:-1] = w
            for start in range(0, len(idx_a), chunk_size):
                end = start + chunk_size
                x = embedding_matrix[idx_a[start:end]]
                x *= embedding_matrix[idx_b[start:end]]
                x *= np.float32(feature_scale)
                y = labels[start:end]
                sw = sample_weights[start:end]
                z = x @ w_float32 + b_float32
                z_float64 = z.astype(np.float64)
                loss += float(
                    np.sum(sw * (np.logaddexp(0.0, z_float64) - y * z_float64))
                )
                residual = (
                    np.float32(1.0) / (np.float32(1.0) + np.exp(-z))
                    - y.astype(np.float32)
                ) * sw.astype(np.float32)
                grad[:-1] += x.T @ residual
                grad[-1] += residual.sum(dtype=np.float64)
            last_objective_params = params.copy()
            last_objective_loss = loss
            return loss, grad

        iteration = 0

        def show_progress(xk: np.ndarray) -> None:
            nonlocal iteration
            iteration += 1
            assert last_objective_loss is not None
            print(f"Iteration {iteration}/{max_iter}: loss={last_objective_loss:.6f}")

        result = minimize(
            objective,
            np.zeros(dim + 1),
            jac=True,
            method="L-BFGS-B",
            callback=show_progress,
            options={"maxiter": max_iter},
        )

        print(f"  L-BFGS: {result.nit} iterations, {result.message}")
        _log_free_memory("after fit")
        gc.collect()
        return RegressionAssociationPredictor(
            self.embeddings, result.x[:-1], float(result.x[-1]), feature_scale
        )


def plot_precision_recall_curve(
    name: str,
    predictor: PredictorType,
    threshold_range: tuple[float, float],
    association_pairs: Collection[tuple[str, str]],
    nonassociation_pairs: Collection[tuple[str, str]],
) -> float:
    thresholds = []
    accuracies = []
    precisions = []
    recalls = []
    peak_accuracy = 0.0
    peak_accuracy_threshold = 0.0
    for threshold in np.arange(threshold_range[0], threshold_range[1], 0.01):
        true_positives = 0
        false_negatives = 0
        for word_a, word_b in association_pairs:
            prediction = predictor(word_a, word_b, threshold)
            if prediction:
                true_positives += 1
            else:
                false_negatives += 1

        true_negatives = 0
        false_positives = 0
        for word_a, word_b in nonassociation_pairs:
            prediction = predictor(word_a, word_b, threshold)
            if prediction:
                false_positives += 1
            else:
                true_negatives += 1

        thresholds.append(threshold)
        accuracy = (true_positives + true_negatives) / (
            len(association_pairs) + len(nonassociation_pairs)
        )
        if accuracy > peak_accuracy:
            peak_accuracy = accuracy
            peak_accuracy_threshold = threshold
        accuracies.append(accuracy)
        precision = true_positives / (true_positives + false_positives)
        precisions.append(precision)
        recall = true_positives / len(association_pairs)
        recalls.append(recall)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label="Accuracy")
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Performance of {name} with varying thresholds")
    plt.legend()
    plt.show()

    print(
        f"Peak accuracy for {name}: {peak_accuracy:.4f} at threshold {peak_accuracy_threshold:.2f}"
    )
    return peak_accuracy
