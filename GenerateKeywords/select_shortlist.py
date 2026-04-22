"""
Select a shortlist of keyword candidates for So Clover! expansion cards.

Reads candidates_llm_ratings.csv and produces candidates_shortlist.csv by:
  1. Vetoing words rated 1 by the human
  2. Force-including words rated 5 by the human
  3. Scoring the rest by weighted average (human weighted more by default)
  4. Enforcing a minimum number of words per content category
  5. Filling remaining slots by score until target-n is reached

The shortlist is then passed to cull_similar.py which removes near-duplicates
using embeddings to produce the true final set (candidates_final.csv).

Usage:
    cd GenerateKeywords
    python select_shortlist.py

    # Tune target size and weights:
    python select_shortlist.py --target-n 100 --human-weight 3 --llm-weight 1

    # Adjust category minimum:
    python select_shortlist.py --min-per-category 2

    # Preview without writing output file:
    python select_shortlist.py --dry-run
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LLM_RATING_CSV = SCRIPT_DIR / "candidates_llm_ratings.csv"
SHORTLIST_CSV = SCRIPT_DIR / "candidates_shortlist.csv"

DEFAULT_TARGET_N = 110
DEFAULT_HUMAN_WEIGHT = 2.0
DEFAULT_LLM_WEIGHT = 1.0
DEFAULT_MIN_PER_CATEGORY = 3

# Category aliases: normalize plural/singular variants to a canonical name
CATEGORY_ALIASES: dict[str, str] = {
    "nouns": "noun",
    "verbs": "verb",
    "adjectives": "adjective",
    "animals": "animal",
}

# Categories that are generation artifacts rather than semantic categories —
# no minimum enforced for these
NO_MIN_CATEGORIES = {"embeddings"}

FINAL_FIELDNAMES = ["word", "category", "source", "notes", "human_rating", "llm_rating", "llm_notes"]


def normalize_category(cat: str) -> str:
    return CATEGORY_ALIASES.get(cat.lower(), cat.lower())


def weighted_score(row: dict[str, str], human_weight: float, llm_weight: float) -> float:
    h = float(row["human_rating"]) if row.get("human_rating", "").strip() else 0.0
    l = float(row["llm_rating"]) if row.get("llm_rating", "").strip() else 0.0
    return (human_weight * h + llm_weight * l) / (human_weight + llm_weight)


def load_candidates() -> list[dict[str, str]]:
    with open(LLM_RATING_CSV) as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select final So Clover! keyword set")
    parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET_N,
                        help=f"Target number of words to select (default: {DEFAULT_TARGET_N})")
    parser.add_argument("--human-weight", type=float, default=DEFAULT_HUMAN_WEIGHT,
                        help=f"Weight for human rating (default: {DEFAULT_HUMAN_WEIGHT})")
    parser.add_argument("--llm-weight", type=float, default=DEFAULT_LLM_WEIGHT,
                        help=f"Weight for LLM rating (default: {DEFAULT_LLM_WEIGHT})")
    parser.add_argument("--min-per-category", type=int, default=DEFAULT_MIN_PER_CATEGORY,
                        help=f"Minimum words per content category (default: {DEFAULT_MIN_PER_CATEGORY})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing candidates_final.csv")
    args = parser.parse_args()

    all_rows = load_candidates()
    print(f"Loaded {len(all_rows)} candidates\n")

    # Partition into vetoed / force-include / scored pool
    vetoed: list[dict[str, str]] = []
    force_include: list[dict[str, str]] = []
    pool: list[dict[str, str]] = []

    for row in all_rows:
        h = row.get("human_rating", "").strip()
        if h == "1":
            vetoed.append(row)
        elif h == "5":
            force_include.append(row)
        else:
            pool.append(row)

    print(f"Vetoed (human=1):        {len(vetoed):>4}  {[r['word'] for r in vetoed]}")
    print(f"Force-include (human=5): {len(force_include):>4}  {[r['word'] for r in force_include]}")
    print(f"Scored pool:             {len(pool):>4}")

    # Score and sort pool
    pool.sort(key=lambda r: -weighted_score(r, args.human_weight, args.llm_weight))

    # Build selection: start with force-includes
    selected: list[dict[str, str]] = list(force_include)
    selected_words: set[str] = {r["word"].lower() for r in selected}
    remaining_pool = [r for r in pool if r["word"].lower() not in selected_words]

    # Enforce category minimums — pull from pool before filling slots
    cat_counts: dict[str, int] = defaultdict(int)
    for r in selected:
        cat_counts[normalize_category(r.get("category", ""))] += 1

    added_for_min: list[str] = []
    for row in list(remaining_pool):  # iterate a snapshot; mutate remaining_pool below
        cat = normalize_category(row.get("category", ""))
        if cat in NO_MIN_CATEGORIES:
            continue
        if cat_counts[cat] < args.min_per_category:
            selected.append(row)
            selected_words.add(row["word"].lower())
            cat_counts[cat] += 1
            added_for_min.append(row["word"])

    remaining_pool = [r for r in pool if r["word"].lower() not in selected_words]

    # Fill remaining slots by score
    slots_left = args.target_n - len(selected)
    fill_words = remaining_pool[:max(slots_left, 0)]
    selected.extend(fill_words)
    selected_words.update(r["word"].lower() for r in fill_words)

    if len(selected) > args.target_n:
        print(f"\nNote: {len(selected)} words selected (target was {args.target_n}) "
              f"because force-includes + category minimums exceed target.")

    # Sort output by category then score descending
    selected.sort(key=lambda r: (
        normalize_category(r.get("category", "")),
        -weighted_score(r, args.human_weight, args.llm_weight),
    ))

    # Summary
    print(f"\nSelected {len(selected)} words (target: {args.target_n})\n")

    by_cat: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in selected:
        by_cat[normalize_category(r.get("category", ""))].append(r)

    print(f"{'Category':<20} {'Count':>6}  Words")
    print("-" * 80)
    for cat in sorted(by_cat):
        words = by_cat[cat]
        word_str = ", ".join(r["word"] for r in words)
        print(f"  {cat:<18} {len(words):>6}  {word_str}")

    print(f"\nScore breakdown (weighted {args.human_weight}×human + {args.llm_weight}×LLM):")
    print(f"  {'Word':<22} {'Cat':<18} {'Human':>6} {'LLM':>6} {'Score':>7}")
    print("  " + "-" * 65)
    for row in sorted(selected, key=lambda r: -weighted_score(r, args.human_weight, args.llm_weight)):
        score = weighted_score(row, args.human_weight, args.llm_weight)
        cat = normalize_category(row.get("category", ""))
        print(f"  {row['word']:<22} {cat:<18} "
              f"{row.get('human_rating',''):>6} {row.get('llm_rating',''):>6} {score:>7.2f}")

    if not args.dry_run:
        with open(SHORTLIST_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FINAL_FIELDNAMES)
            writer.writeheader()
            writer.writerows(selected)
        print(f"\nWrote {len(selected)} words to {SHORTLIST_CSV}")
        print("Next: run cull_similar.py to remove near-duplicates and produce candidates_final.csv")
    else:
        print(f"\n(Dry run — nothing written)")


if __name__ == "__main__":
    main()
