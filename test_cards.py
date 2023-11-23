import pytest
from cards import Card


def test_card():
    card = Card("abc", "def", "ghi", "jkl")
    assert card[0] == "abc"
    assert card[1] == "def"
    assert card[2] == "ghi"
    assert card[3] == "jkl"
