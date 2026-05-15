import csv
from pathlib import Path

from GenerateKeywordCards2.get_root_directory import get_root_directory


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
        for word in words:
            if word not in existing_words:
                writer.writerow({"Word": word})
