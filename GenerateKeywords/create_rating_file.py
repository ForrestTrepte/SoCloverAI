"""
Create or update candidates_rating.csv for human review.

On first run: creates candidates_rating.csv from candidates.csv with a blank
human_rating column (1–5 scale, 1=veto, 5=must-include).

On subsequent runs: adds any new words from candidates.csv that are not yet
in candidates_rating.csv, leaving their human_rating blank.

Usage:
    cd GenerateKeywords
    python create_rating_file.py
"""

import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CANDIDATES_CSV = SCRIPT_DIR / "candidates.csv"
RATING_CSV = SCRIPT_DIR / "candidates_rating.csv"

FIELDNAMES = ["word", "category", "source", "notes", "human_rating"]


def load_candidates() -> list[dict[str, str]]:
    with open(CANDIDATES_CSV) as f:
        return list(csv.DictReader(f))


def load_existing_ratings() -> dict[str, dict[str, str]]:
    if not RATING_CSV.exists():
        return {}
    with open(RATING_CSV) as f:
        return {row["word"].lower(): row for row in csv.DictReader(f)}


def main() -> None:
    candidates = load_candidates()
    existing = load_existing_ratings()

    new_words: list[dict[str, str]] = []
    all_rows: list[dict[str, str]] = []

    for row in candidates:
        key = row["word"].lower()
        if key in existing:
            all_rows.append(existing[key])
        else:
            new_row = {
                "word": row["word"],
                "category": row.get("category", ""),
                "source": row.get("source", ""),
                "notes": row.get("notes", ""),
                "human_rating": "",
            }
            all_rows.append(new_row)
            new_words.append(new_row)

    with open(RATING_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    if not existing:
        print(f"Created {RATING_CSV} with {len(all_rows)} candidates.")
    elif new_words:
        print(f"Added {len(new_words)} new candidates to {RATING_CSV}:")
        for row in new_words:
            print(f"  {row['word']} ({row['category']})")
    else:
        print(f"{RATING_CSV} is already up to date ({len(all_rows)} candidates, no new words).")

    print(f"\nRate each word in the 'human_rating' column:")
    print("  1 = veto (definitely exclude)")
    print("  2 = weak")
    print("  3 = ok")
    print("  4 = good")
    print("  5 = must include")


if __name__ == "__main__":
    main()
