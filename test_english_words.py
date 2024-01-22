from english_words import remove_word_forms_of


def test_remove_word_forms_of():
    removed0 = remove_word_forms_of(
        "apple", ["apple", "apples", "applesauce", "banana", "bananas"]
    )
    assert removed0 == ["banana", "bananas"]

    removed1 = remove_word_forms_of(
        "presidential",
        [
            "president",
            "prestige",
            "presidencies",
            "present",
            "presidential",
            "pretext",
            "preside",
            "banana",
            "presidentially",
        ],
    )
    assert removed1 == ["prestige", "present", "pretext", "banana"]
