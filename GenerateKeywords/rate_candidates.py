"""
Rate So Clover! keyword candidates using Claude.

Reads candidates_rating.csv and asks Claude to rate each word 1–5 on
fun/versatility as a So Clover! keyword. Writes results to candidates_llm_ratings.csv.

If candidates_llm_ratings.csv already exists, only rates words not yet rated
(safe to re-run after interruption or after adding new candidates).

Usage:
    cd GenerateKeywords
    python rate_candidates.py

    # Rate only specific categories:
    python rate_candidates.py --categories nouns verbs

    # Use a smaller batch size (default 30):
    python rate_candidates.py --batch-size 20
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR.parent / ".env")

RATING_CSV = SCRIPT_DIR / "candidates_rating.csv"
LLM_RATING_CSV = SCRIPT_DIR / "candidates_llm_ratings.csv"

MODEL = "claude-sonnet-4-6"

FIELDNAMES = ["word", "category", "source", "notes", "human_rating", "llm_rating", "llm_notes"]

SYSTEM_PROMPT = """You are an expert at the party game So Clover!

In So Clover!, players receive a card with 4 keyword words on the sides of a square.
Each player writes one clue word in each corner that connects the two adjacent keywords.
Other players then try to reconstruct the original arrangement from the clues.

Great keywords are:
- VERSATILE: the word has many possible associations across different domains
- ENTERTAINING: leads to fun, creative, humorous, or surprising clue connections
- ACCESSIBLE: well-known enough that most players have associations with it
- Multi-meaning bonus: words spanning categories (BOLT = lightning/door/fabric/sprint) are especially rich

You will be given a list of candidate keywords and must rate each one 1–5:
  1 = veto — boring, too narrow, obscure, or otherwise bad for the game
  2 = weak — limited associations or not very fun
  3 = ok — works fine but not exciting
  4 = good — versatile and fun, would make a solid keyword
  5 = excellent — exceptionally versatile, entertaining, or multi-meaning; a must-have

Respond with one line per word in this exact format:
  WORD: <rating> | <brief reason (one short phrase)>

Example:
  bolt: 5 | lightning/door/fabric/sprint — rich multi-meaning
  elegant: 3 | mostly one register, limited cross-domain connections
  enshittification: 2 | too niche and single-domain"""


def load_candidates() -> list[dict[str, str]]:
    with open(RATING_CSV) as f:
        return list(csv.DictReader(f))


def load_existing_llm_ratings() -> dict[str, dict[str, str]]:
    if not LLM_RATING_CSV.exists():
        return {}
    with open(LLM_RATING_CSV) as f:
        return {row["word"].lower(): row for row in csv.DictReader(f)}


def parse_ratings(text: str, words: list[str]) -> dict[str, tuple[str, str]]:
    """Parse LLM response into {word_lower: (rating, notes)} dict."""
    result: dict[str, tuple[str, str]] = {}
    word_set = {w.lower() for w in words}

    for line in text.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        word_part, _, rest = line.partition(":")
        word = word_part.strip().lower()
        if word not in word_set:
            continue
        rest = rest.strip()
        if "|" in rest:
            rating_str, _, notes = rest.partition("|")
        else:
            rating_str = rest
            notes = ""
        rating_str = rating_str.strip()
        if rating_str and rating_str[0].isdigit():
            result[word] = (rating_str[0], notes.strip())

    return result


def rate_batch(
    client: anthropic.Anthropic,
    batch: list[dict[str, str]],
) -> dict[str, tuple[str, str]]:
    lines = []
    for row in batch:
        notes = f" ({row['notes']})" if row.get("notes") else ""
        lines.append(f"{row['word']}{notes}")

    user_message = (
        f"Rate these {len(batch)} So Clover! keyword candidates (1–5):\n\n"
        + "\n".join(lines)
    )

    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    response_text = message.content[0].text  # type: ignore[union-attr]
    words = [row["word"] for row in batch]
    return parse_ratings(response_text, words)


def write_all(rows: list[dict[str, str]]) -> None:
    with open(LLM_RATING_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rate So Clover! keyword candidates with Claude")
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Only rate words in these categories (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Words per API call (default: 30)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    candidates = load_candidates()
    existing = load_existing_llm_ratings()

    if args.categories:
        candidates = [r for r in candidates if r.get("category", "").lower() in args.categories]
        print(f"Filtering to categories: {args.categories} ({len(candidates)} words)")

    to_rate = [r for r in candidates if not existing.get(r["word"].lower(), {}).get("llm_rating", "").strip()]
    already_rated = len(candidates) - len(to_rate)
    print(f"Total candidates: {len(candidates)}, already rated: {already_rated}, to rate: {len(to_rate)}")

    if not to_rate:
        print("Nothing to rate.")
        return

    # Build output rows: start from all candidates, overlay existing ratings
    all_rows: list[dict[str, str]] = []
    for row in load_candidates():  # reload full file for output
        key = row["word"].lower()
        if key in existing:
            all_rows.append(existing[key])
        else:
            all_rows.append({
                "word": row["word"],
                "category": row.get("category", ""),
                "source": row.get("source", ""),
                "notes": row.get("notes", ""),
                "human_rating": row.get("human_rating", ""),
                "llm_rating": "",
                "llm_notes": "",
            })

    rated_count = 0
    for batch_start in range(0, len(to_rate), args.batch_size):
        batch = to_rate[batch_start:batch_start + args.batch_size]
        print(f"\nRating batch {batch_start // args.batch_size + 1} "
              f"({batch_start + 1}–{min(batch_start + len(batch), len(to_rate))} of {len(to_rate)})...")

        ratings = rate_batch(client, batch)

        missing = [r["word"] for r in batch if r["word"].lower() not in ratings]
        if missing:
            print(f"  Warning: no rating returned for: {', '.join(missing)}")

        # Update output rows
        for out_row in all_rows:
            key = out_row["word"].lower()
            if key in ratings and not out_row.get("llm_rating"):
                rating, notes = ratings[key]
                out_row["llm_rating"] = rating
                out_row["llm_notes"] = notes
                rated_count += 1

        write_all(all_rows)
        print(f"  Rated {len(ratings)} words. Progress saved.")

        if batch_start + args.batch_size < len(to_rate):
            time.sleep(0.5)

    print(f"\nDone. Rated {rated_count} new words. Results in {LLM_RATING_CSV}")


if __name__ == "__main__":
    main()
