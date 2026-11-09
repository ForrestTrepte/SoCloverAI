import pytest

from .word_set_comparison import WordSetComparison


def test_comparison_sets() -> None:
    comparison = WordSetComparison("cue", ["a", "b", "c"], ["b", "c", "d"])

    assert comparison.overlap == ["b", "c"]
    assert comparison.only_a == ["a"]
    assert comparison.only_b == ["d"]
    assert comparison.jaccard == pytest.approx(0.5)


def test_jaccard_of_empty_sets_is_zero() -> None:
    assert WordSetComparison("cue", [], []).jaccard == 0.0


def test_summarize_prints_means(capsys: pytest.CaptureFixture[str]) -> None:
    WordSetComparison.summarize(
        [
            WordSetComparison("x", ["a", "b"], ["a", "b"]),
            WordSetComparison("y", ["a"], ["b"]),
        ],
        name_a="predicted",
        name_b="swow",
    )

    output = capsys.readouterr().out
    assert "Mean overlap count: 1.0" in output
    assert "Mean jaccard: 0.5" in output
