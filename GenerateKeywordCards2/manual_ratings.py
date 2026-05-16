import csv
from pathlib import Path

from GenerateKeywordCards2.get_root_directory import get_root_directory
from GenerateKeywordCards2.rate_words import RatingsByWord


def get_manual_ratings() -> RatingsByWord:
    ratings_path = get_root_directory() / "manual_ratings.csv"
    assert ratings_path.exists()

    ratings_by_word: RatingsByWord = {}
    with ratings_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        assert reader.fieldnames

        unrated_words = set()
        for row in reader:
            word = row.get("Word")
            if not word:
                continue

            ratings = {}
            for field in reader.fieldnames:
                if field == "Word":
                    continue
                rating_str = row.get(field)
                if not rating_str:
                    continue
                try:
                    rating = float(rating_str)
                except ValueError:
                    print(
                        f"Warning: Invalid rating value '{rating_str}' for word {word} in column {field}"
                    )
                    continue
                ratings[field] = rating

            if len(ratings) == len(reader.fieldnames) - 1:
                ratings_by_word[word] = ratings
            else:
                unrated_words.add(word)

    if unrated_words:
        raise ValueError(
            f"The ratings in manual_ratings.csv are incomplete for the following {len(unrated_words)} words: {', '.join(sorted(unrated_words))}"
        )

    return ratings_by_word


def add_manual_ratings_words(words: list[str]) -> None:
    ratings_path = get_root_directory() / "manual_ratings.csv"
    assert ratings_path.exists()

    existing_words: set[str] = set()
    if ratings_path.exists():
        with ratings_path.open("r", newline="", encoding="utf-8") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                word = row.get("Word")
                if word:
                    existing_words.add(word)

    with ratings_path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["Word"])
        for word in sorted(words):
            if word not in existing_words:
                writer.writerow({"Word": word})
