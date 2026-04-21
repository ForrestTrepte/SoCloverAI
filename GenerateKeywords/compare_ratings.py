"""
Compare human vs LLM ratings for So Clover! keyword candidates.

Reads candidates_llm_ratings.csv and prints:
  - Correlation between human and LLM ratings
  - Biggest agreements (both high or both low)
  - Biggest divergences (human high / LLM low, and vice versa)
  - Rating distribution comparison

Usage:
    cd GenerateKeywords
    python compare_ratings.py

    # Show more divergence examples:
    python compare_ratings.py --top-n 20
"""

import argparse
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LLM_RATING_CSV = SCRIPT_DIR / "candidates_llm_ratings.csv"


def load_rated() -> list[dict[str, str]]:
    rows = []
    with open(LLM_RATING_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("human_rating", "").strip() and row.get("llm_rating", "").strip():
                rows.append(row)
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / denom if denom else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare human vs LLM keyword ratings")
    parser.add_argument("--top-n", type=int, default=15,
                        help="Number of examples to show per section (default: 15)")
    args = parser.parse_args()

    rows = load_rated()
    print(f"Loaded {len(rows)} words with both ratings\n")

    human = [float(r["human_rating"]) for r in rows]
    llm = [float(r["llm_rating"]) for r in rows]
    diff = [h - l for h, l in zip(human, llm)]

    r = pearson(human, llm)
    print(f"Pearson correlation (human vs LLM): {r:.3f}")
    mae = sum(abs(d) for d in diff) / len(diff)
    print(f"Mean absolute error: {mae:.2f}")

    exact = sum(1 for d in diff if d == 0)
    within1 = sum(1 for d in diff if abs(d) <= 1)
    print(f"Exact agreement: {exact}/{len(diff)} ({100*exact/len(diff):.0f}%)")
    print(f"Within 1 point:  {within1}/{len(diff)} ({100*within1/len(diff):.0f}%)")

    # Distribution comparison
    print("\nRating distribution:")
    print(f"  {'Rating':<8} {'Human':>8} {'LLM':>8}")
    for rating in range(1, 6):
        h_count = sum(1 for x in human if x == rating)
        l_count = sum(1 for x in llm if x == rating)
        print(f"  {rating:<8} {h_count:>8} {l_count:>8}")

    # Sort by divergence
    indexed = sorted(enumerate(rows), key=lambda x: diff[x[0]])

    print(f"\n--- Human >> LLM (human liked more) — top {args.top_n} ---")
    print(f"  {'Word':<22} {'Cat':<18} {'Human':>6} {'LLM':>6}  LLM notes")
    print("  " + "-" * 80)
    for i, row in reversed(indexed[-args.top_n:]):
        if diff[i] <= 0:
            break
        print(f"  {row['word']:<22} {row['category']:<18} {row['human_rating']:>6} {row['llm_rating']:>6}  {row.get('llm_notes', '')}")

    print(f"\n--- LLM >> Human (LLM liked more) — top {args.top_n} ---")
    print(f"  {'Word':<22} {'Cat':<18} {'Human':>6} {'LLM':>6}  LLM notes")
    print("  " + "-" * 80)
    for i, row in indexed[:args.top_n]:
        if diff[i] >= 0:
            break
        print(f"  {row['word']:<22} {row['category']:<18} {row['human_rating']:>6} {row['llm_rating']:>6}  {row.get('llm_notes', '')}")

    print(f"\n--- Strong agreements (both rated 4–5) ---")
    agreed_high = [(i, r) for i, r in enumerate(rows) if human[i] >= 4 and llm[i] >= 4]
    agreed_high.sort(key=lambda x: -(human[x[0]] + llm[x[0]]))
    print(f"  {'Word':<22} {'Cat':<18} {'Human':>6} {'LLM':>6}  LLM notes")
    print("  " + "-" * 80)
    for i, row in agreed_high[:args.top_n]:
        print(f"  {row['word']:<22} {row['category']:<18} {row['human_rating']:>6} {row['llm_rating']:>6}  {row.get('llm_notes', '')}")

    print(f"\n--- Strong agreements (both rated 1–2) ---")
    agreed_low = [(i, r) for i, r in enumerate(rows) if human[i] <= 2 and llm[i] <= 2]
    agreed_low.sort(key=lambda x: (human[x[0]] + llm[x[0]]))
    print(f"  {'Word':<22} {'Cat':<18} {'Human':>6} {'LLM':>6}  LLM notes")
    print("  " + "-" * 80)
    for i, row in agreed_low[:args.top_n]:
        print(f"  {row['word']:<22} {row['category']:<18} {row['human_rating']:>6} {row['llm_rating']:>6}  {row.get('llm_notes', '')}")


if __name__ == "__main__":
    main()
