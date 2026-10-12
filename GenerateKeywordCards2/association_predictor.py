import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from GenerateKeywordCards2.embeddings import WordEmbeddings
from GenerateKeywordCards2.swow import SWOWAssociations


def get_distance_threshold_association_predictor(
    embeddings: WordEmbeddings, distance_threshold: float
) -> Callable[[str, str], bool]:
    def predictor(word: str, associated_word: str) -> bool:
        return embeddings.distance(word, associated_word) <= distance_threshold

    return predictor


@dataclass
class PredictorTrainingDataRow:
    embedding_a: np.ndarray
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
                        embedding_a=embeddings.get_embedding(word_a),
                        embedding_b=embeddings.get_embedding(word_b),
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
                    embedding_a=embeddings.get_embedding(word_a),
                    embedding_b=embeddings.get_embedding(word_b),
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
        self, probability_threshold: float = 0.5
    ) -> Callable[[str, str], bool]:
        """
        Fits a logistic regression on the element-wise product of the two words' embeddings.
        Since embeddings are L2-normalized, summing those products gives cosine similarity,
        so equal weights reproduce the distance-threshold predictor; learned weights let the
        model emphasize the dimensions that matter most for human associations.
        """
        features = np.stack(
            [row.embedding_a * row.embedding_b for row in self.training_rows]
        ).astype(np.float32)
        labels = np.array([row.is_association for row in self.training_rows])
        model = LogisticRegression(max_iter=1000)
        model.fit(features, labels)

        embeddings = self.embeddings

        def predictor(word: str, associated_word: str) -> bool:
            feature = embeddings.get_embedding(word) * embeddings.get_embedding(associated_word)
            probability = model.predict_proba(feature.reshape(1, -1))[0, 1]
            return bool(probability >= probability_threshold)

        return predictor
