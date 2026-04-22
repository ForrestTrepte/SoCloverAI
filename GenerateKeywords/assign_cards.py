"""
Assign final keywords to So Clover! cards using embeddings-based diversity optimization.

Each card holds 4 words. Words on the same card should be as dissimilar as possible
so players have rich connection options for clues.

Algorithm:
  1. Load optional manually-specified cards from cards_manual.csv (excluded from pool)
  2. Greedy initialization: build each card by picking words maximally distant
     from one another (farthest-point selection)
  3. Local search: repeatedly swap words between cards when it reduces total
     intra-card similarity, until no improving swap exists

Usage:
    cd GenerateKeywords
    python assign_cards.py

    # Reproducible run with specific seed:
    python assign_cards.py --seed 42

    # Preview without writing:
    python assign_cards.py --dry-run

Manual cards:
    Create cards_manual.csv with one row per card, four word columns:
        word1,word2,word3,word4
        Michael,Peter,Forrest,Brian
    These cards are taken as-is and their words excluded from the pool.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR.parent / ".env")

GENERATE_CLUES_DIR = SCRIPT_DIR.parent / "GenerateClues"
FINAL_CSV = SCRIPT_DIR / "candidates_final.csv"
MANUAL_CSV = SCRIPT_DIR / "cards_manual.csv"
CARDS_CSV = SCRIPT_DIR / "cards.csv"
EMBEDDINGS_NPZ = GENERATE_CLUES_DIR / "words_by_frequency_embeddings.npz"
WORDS_JSON = GENERATE_CLUES_DIR / "words_by_frequency.json"

DEFAULT_SEED = 0


def load_final_words() -> list[str]:
    with open(FINAL_CSV) as f:
        return [row["word"] for row in csv.DictReader(f)]


def load_manual_cards() -> list[list[str]]:
    if not MANUAL_CSV.exists():
        return []
    cards = []
    with open(MANUAL_CSV) as f:
        for row in csv.DictReader(f):
            card = [row[f"word{i}"] for i in range(1, 5) if row.get(f"word{i}", "").strip()]
            if card:
                cards.append(card)
    return cards


def load_embeddings(words: list[str]) -> np.ndarray:
    """Return (N, D) embedding matrix; zero vector for words not in vocab."""
    data = np.load(EMBEDDINGS_NPZ)
    all_embeddings: np.ndarray = data["embeddings"]
    with open(WORDS_JSON) as f:
        all_words: list[str] = json.load(f)
    n_embeddings = len(all_embeddings)
    word_to_idx = {w.lower(): i for i, w in enumerate(all_words[:n_embeddings])}

    dim = all_embeddings.shape[1]
    missing = []
    result = np.zeros((len(words), dim), dtype=np.float32)
    for i, word in enumerate(words):
        idx = word_to_idx.get(word.lower())
        if idx is not None:
            result[i] = all_embeddings[idx]
        else:
            missing.append(word)

    if missing:
        print(f"Warning: {len(missing)} words not in embedding vocab "
              f"(treated as neutral similarity): {', '.join(missing)}")
    return result


def card_cost(card: list[int], sim: np.ndarray) -> float:
    """Sum of pairwise cosine similarities within a card (lower = more diverse)."""
    total = 0.0
    n = len(card)
    for i in range(n):
        for j in range(i + 1, n):
            total += sim[card[i], card[j]]
    return total


def swap_delta(card_i: list[int], card_j: list[int], wi: int, wj: int, sim: np.ndarray) -> float:
    """Cost change from swapping card_i[wi] with card_j[wj]. Negative = improvement."""
    w_out = card_i[wi]
    w_in = card_j[wj]
    others_i = [card_i[k] for k in range(len(card_i)) if k != wi]
    others_j = [card_j[k] for k in range(len(card_j)) if k != wj]
    delta = 0.0
    for x in others_i:
        delta += sim[w_in, x] - sim[w_out, x]
    for x in others_j:
        delta += sim[w_out, x] - sim[w_in, x]
    return delta


def greedy_init(n: int, n_cards: int, sim: np.ndarray, rng: np.random.Generator) -> list[list[int]]:
    """
    Build initial card assignment greedily.
    Each card: pick a seed word, then add the 3 words most distant from all
    words already on the card (farthest-point selection).
    """
    order = rng.permutation(n).tolist()
    cards: list[list[int]] = []

    for _ in range(n_cards):
        card = [order.pop(0)]
        for _ in range(3):
            best_w, best_min_dist = -1, -1.0
            for w in order:
                min_dist = min(1.0 - float(sim[w, c]) for c in card)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_w = w
            card.append(best_w)
            order.remove(best_w)
        cards.append(card)

    return cards


def local_search(cards: list[list[int]], sim: np.ndarray, max_passes: int = 500) -> list[list[int]]:
    """
    Improve card assignment by swapping words between cards.
    Continues until no improving swap is found or max_passes is exhausted.
    """
    n_cards = len(cards)
    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n_cards):
            for j in range(i + 1, n_cards):
                for wi in range(len(cards[i])):
                    for wj in range(len(cards[j])):
                        delta = swap_delta(cards[i], cards[j], wi, wj, sim)
                        if delta < -1e-9:
                            # Perform the swap
                            cards[i][wi], cards[j][wj] = cards[j][wj], cards[i][wi]
                            improved = True

    return cards


def total_cost(cards: list[list[int]], sim: np.ndarray) -> float:
    return sum(card_cost(c, sim) for c in cards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign keywords to So Clover! cards")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing cards.csv")
    args = parser.parse_args()

    all_words = load_final_words()
    manual_cards = load_manual_cards()

    if manual_cards:
        print(f"Loaded {len(manual_cards)} manual card(s):")
        for i, card in enumerate(manual_cards):
            print(f"  Manual {i+1}: {', '.join(card)}")
        print()

    # Remove manual card words from pool
    manual_words = {w.lower() for card in manual_cards for w in card}
    unknown_manual = manual_words - {w.lower() for w in all_words}
    if unknown_manual:
        print(f"Warning: manual card words not found in candidates_final.csv: "
              f"{', '.join(sorted(unknown_manual))}")

    pool = [w for w in all_words if w.lower() not in manual_words]
    n = len(pool)
    n_cards = n // 4
    leftover = n % 4

    print(f"Pool: {n} words → {n_cards} cards of 4", end="")
    if leftover:
        print(f" + {leftover} leftover word(s) (not assigned to a card)")
    else:
        print()

    if n_cards == 0:
        print("Error: not enough words to form any cards.", file=sys.stderr)
        sys.exit(1)

    # Trim leftover words (lowest priority: just drop from the end for now)
    if leftover:
        leftover_words = pool[-leftover:]
        pool = pool[:-leftover]
        print(f"Leftover (unassigned): {', '.join(leftover_words)}")
    print()

    if not EMBEDDINGS_NPZ.exists():
        print(f"Error: embeddings not found at {EMBEDDINGS_NPZ}", file=sys.stderr)
        sys.exit(1)

    print("Loading embeddings...")
    emb = load_embeddings(pool)

    print("Computing similarity matrix...")
    sim = (emb @ emb.T).astype(np.float64)

    rng = np.random.default_rng(args.seed)

    print(f"Greedy initialization (seed={args.seed})...")
    cards = greedy_init(n, n_cards, sim, rng)
    cost_before = total_cost(cards, sim)
    avg_before = cost_before / (n_cards * 6)
    print(f"  Avg intra-card similarity: {avg_before:.4f}")

    print("Running local search...")
    cards = local_search(cards, sim)
    cost_after = total_cost(cards, sim)
    avg_after = cost_after / (n_cards * 6)
    print(f"  Avg intra-card similarity: {avg_after:.4f}  "
          f"(improvement: {avg_before - avg_after:.4f})\n")

    # Map indices back to words
    word_cards: list[list[str]] = [[pool[i] for i in card] for card in cards]

    # Prepend manual cards
    all_cards = manual_cards + word_cards

    # Print results
    print(f"{'#':<5} {'Word 1':<20} {'Word 2':<20} {'Word 3':<20} {'Word 4':<20} {'Sim':>6}")
    print("-" * 95)
    for idx, card in enumerate(all_cards):
        label = f"M{idx+1}" if idx < len(manual_cards) else str(idx + 1)
        # Compute cost for display (manual cards may lack embeddings; skip cost for them)
        if idx >= len(manual_cards):
            pool_indices = [pool.index(w) for w in card]
            c = card_cost(pool_indices, sim)
            sim_str = f"{c/6:.4f}"
        else:
            sim_str = "  —"
        print(f"{label:<5} {card[0]:<20} {card[1]:<20} {card[2]:<20} {card[3]:<20} {sim_str:>6}")

    if not args.dry_run:
        with open(CARDS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["card", "word1", "word2", "word3", "word4"])
            for idx, card in enumerate(all_cards):
                label = f"M{idx+1}" if idx < len(manual_cards) else str(idx + 1)
                writer.writerow([label] + card)
        print(f"\nWrote {len(all_cards)} cards to {CARDS_CSV}")
    else:
        print(f"\n(Dry run — nothing written)")


if __name__ == "__main__":
    main()
