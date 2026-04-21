"""
Generate keyword candidates for So Clover! expansion using Claude.

Prompts Claude in batches by category and appends new words to candidates.csv,
deduplicating against both the existing 880-word game list and current candidates.

Usage:
    cd GenerateWords
    ANTHROPIC_API_KEY=<key> python generate_candidates.py

    # Target specific categories only:
    ANTHROPIC_API_KEY=<key> python generate_candidates.py --categories verbs adjectives

    # Generate more words per batch:
    ANTHROPIC_API_KEY=<key> python generate_candidates.py --per-batch 40
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR.parent / ".env")
CANDIDATES_CSV = SCRIPT_DIR / "candidates.csv"
EXISTING_KEYWORDS_CSV = SCRIPT_DIR / "CloverExistingKeywords.csv"

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are helping design keyword cards for an expansion of the party game So Clover!

In So Clover!, players are given a card with 4 keyword words arranged on the sides of a square.
A player writes one clue word in each corner that connects the two adjacent keywords.
Other players then try to reconstruct the original arrangement.

Your job is to suggest KEYWORD words — the words that go ON the cards.

Criteria for great keywords:
- VERSATILE: the word has many possible associations, enabling lots of different clue connections
- ENTERTAINING: the word leads to fun, creative, humorous, or surprising clues
- ACCESSIBLE: well-known enough that most players will have associations with it
- Multi-meaning bonus: words that span categories (e.g. BOLT = lightning/door/fabric/sprint) are especially rich
- Mix of timeless classics and contemporary/pop culture references is welcome
- Non-nouns (verbs, adjectives) that are versatile are encouraged
- Whimsical or evocative words that are fun to say or think about are welcome
- Mildly risqué words are OK if they have multiple meanings (so players can lean into or away from the connotation)

Avoid:
- Obscure words most people won't know
- Words so narrow they only connect to one thing
- Proper names that aren't very prominent (e.g. 'Kubernetes' is too niche; 'Google' is fine)
- Anything clearly offensive"""

CATEGORY_PROMPTS: dict[str, str] = {
    "nouns": (
        "Suggest versatile English nouns for So Clover! keywords. "
        "Focus on concrete everyday nouns with multiple meanings or associations. "
        "Examples of the kind of quality we want: BOLT, ECHO, SPARK, VAULT, DRAFT, TIDE, FORGE, VOID. "
        "Avoid words already in the base game (listed below)."
    ),
    "verbs": (
        "Suggest versatile English verbs for So Clover! keywords. "
        "Single-word infinitive form (e.g. LURK, BREW, DRIFT, SURGE, HAUNT, GRIND, WEAVE). "
        "Prefer verbs that are evocative and have multiple contexts. "
        "Avoid words already in the base game (listed below)."
    ),
    "adjectives": (
        "Suggest versatile English adjectives for So Clover! keywords. "
        "Examples of quality: HOLLOW, NEON, FUZZY, CRISPY, EERIE. "
        "Prefer adjectives that evoke a strong image or feeling and work across many contexts. "
        "Avoid words already in the base game (listed below)."
    ),
    "tech_and_culture": (
        "Suggest tech/internet/pop culture proper nouns and terms for So Clover! keywords. "
        "These should be VERY prominent references nearly everyone knows — like GOOGLE, NETFLIX, TIKTOK, PIXEL, MEME, VIRAL. "
        "Can include company names, product names, internet slang, or cultural phenomena. "
        "Avoid words already in the base game (listed below)."
    ),
    "whimsical": (
        "Suggest whimsical, playful, or unusual words for So Clover! keywords. "
        "These should be fun to say, evocative, or delightfully weird. "
        "Examples: BAMBOOZLE, KERFUFFLE, GOBLIN, GREMLIN, FRENZY, RUCKUS, QUIRK, GLITTER, ZAP. "
        "Can include fantasy creatures, onomatopoeia, silly-sounding words. "
        "Avoid words already in the base game (listed below)."
    ),
    "animals": (
        "Suggest animal names for So Clover! keywords. "
        "Prefer animals with rich associations beyond just the animal itself (e.g. SLOTH = laziness/sin/animal; NARWHAL = unicorn/horn/ocean). "
        "Quirky or beloved internet animals are especially welcome. "
        "Avoid words already in the base game (listed below)."
    ),
    "food": (
        "Suggest food-related words for So Clover! keywords. "
        "Prefer words with meanings beyond just the food (e.g. WAFFLE = food/to ramble; PRETZEL = food/twisted/shape). "
        "Avoid words already in the base game (listed below)."
    ),
    "abstract": (
        "Suggest abstract concepts or emotion words for So Clover! keywords. "
        "These are intentionally experimental — we want a few words that feel different from typical concrete nouns. "
        "Examples might include: CHAOS, ECHO, DRIFT, VOID, LORE, VIBE. "
        "Avoid words already in the base game (listed below)."
    ),
}


def load_exclusion_set() -> set[str]:
    exclusions: set[str] = set()
    with open(EXISTING_KEYWORDS_CSV) as f:
        for line in f:
            word = line.strip()
            if word:
                exclusions.add(word.lower())
    return exclusions


def load_current_candidates() -> set[str]:
    if not CANDIDATES_CSV.exists():
        return set()
    candidates: set[str] = set()
    with open(CANDIDATES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.add(row["word"].lower())
    return candidates


def append_candidates(new_words: list[dict[str, Any]]) -> None:
    with open(CANDIDATES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "category", "source", "notes"])
        for row in new_words:
            writer.writerow(row)


def parse_word_list(text: str) -> list[str]:
    words = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip list markers like "1.", "-", "*", "•"
        for prefix in ["•", "-", "*"]:
            if line.startswith(prefix):
                line = line[1:].strip()
                break
        if line[0].isdigit() and "." in line[:3]:
            line = line.split(".", 1)[1].strip()
        # Take only the first word (ignore inline explanations like "BOLT - lightning/door")
        word = line.split()[0].strip(".,;:'\"").strip()
        if word:
            words.append(word)
    return words


def generate_batch(
    client: anthropic.Anthropic,
    category: str,
    exclusions: set[str],
    per_batch: int,
) -> list[str]:
    existing_sample = sorted(exclusions)[:50]
    exclusion_note = (
        f"Words already in the base game (DO NOT repeat these or close variants): "
        f"{', '.join(existing_sample)}, ... (and {len(exclusions) - 50} more)"
    )

    user_message = (
        f"{CATEGORY_PROMPTS[category]}\n\n"
        f"{exclusion_note}\n\n"
        f"Return exactly {per_batch} words, one per line. "
        f"No explanations, just the words. Single words only (no phrases)."
    )

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return parse_word_list(message.content[0].text)  # type: ignore[union-attr]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate So Clover! keyword candidates")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(CATEGORY_PROMPTS.keys()),
        default=list(CATEGORY_PROMPTS.keys()),
        help="Categories to generate (default: all)",
    )
    parser.add_argument(
        "--per-batch",
        type=int,
        default=25,
        help="Words to request per category batch (default: 25)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    exclusions = load_exclusion_set()
    current_candidates = load_current_candidates()
    all_excluded = exclusions | current_candidates

    total_added = 0
    for category in args.categories:
        print(f"\n--- Generating {category} ---")
        raw_words = generate_batch(client, category, exclusions, args.per_batch)

        new_rows = []
        skipped = []
        for word in raw_words:
            if word.lower() in all_excluded:
                skipped.append(word)
                continue
            new_rows.append({"word": word, "category": category, "source": "llm", "notes": ""})
            all_excluded.add(word.lower())

        append_candidates(new_rows)
        total_added += len(new_rows)
        print(f"  Added {len(new_rows)} words, skipped {len(skipped)} duplicates")
        if new_rows:
            print(f"  New: {', '.join(r['word'] for r in new_rows)}")
        if skipped:
            print(f"  Skipped: {', '.join(skipped)}")

    print(f"\nDone. Total new candidates added: {total_added}")


if __name__ == "__main__":
    main()
