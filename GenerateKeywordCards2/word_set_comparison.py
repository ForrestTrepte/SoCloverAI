from dataclasses import dataclass


@dataclass(frozen=True)
class WordSetComparison:
    """Compares two lists of words that relate to the same word."""

    word: str
    words_a: list[str]
    words_b: list[str]

    @property
    def overlap(self) -> list[str]:
        return [w for w in self.words_a if w in self.words_b]

    @property
    def only_a(self) -> list[str]:
        return [w for w in self.words_a if w not in self.words_b]

    @property
    def only_b(self) -> list[str]:
        return [w for w in self.words_b if w not in self.words_a]

    @property
    def jaccard(self) -> float:
        union_size = len(set(self.words_a) | set(self.words_b))
        if union_size == 0:
            return 0.0
        return len(self.overlap) / union_size

    @classmethod
    def summarize(
        cls,
        comparisons: list["WordSetComparison"],
        name_a: str,
        name_b: str,
    ) -> None:
        for comparison in comparisons:
            print(f"{comparison.word}:")
            print(
                f"  agreement     = {len(comparison.overlap)}: {comparison.overlap}"
            )
            print(
                f"  {name_b}-only = {len(comparison.only_b)}: {sorted(comparison.only_b)}"
            )
            print(
                f"  {name_a}-only = {len(comparison.only_a)}: {sorted(comparison.only_a)}"
            )
            print(f"  jaccard       = {comparison.jaccard:.2f}")

        print()

        mean_overlap_count = sum(len(c.overlap) for c in comparisons) / len(comparisons)
        print(f"Mean overlap count: {mean_overlap_count}")

        mean_jaccard = sum(c.jaccard for c in comparisons) / len(comparisons)
        print(f"Mean jaccard: {mean_jaccard}")
