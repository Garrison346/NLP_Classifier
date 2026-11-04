import pytest

from nlp_classifier.preprocessing import preprocess_text


def test_preprocess_text_returns_tokens():
    result = preprocess_text(
        "The Transformer architecture is fascinating!"
    )

    assert isinstance(result, list)
    assert "transformer" in result
    assert "architecture" in result


def test_preprocess_text_removes_stopwords():
    result = preprocess_text(
        "This is a simple test",
        remove_stop_words=True,
    )

    assert "is" not in result
    assert "a" not in result


def test_preprocess_text_keeps_stopwords_by_default():
    result = preprocess_text("What can you do?")

    assert result == ["what", "can", "you", "do"]


def test_preprocess_text_converts_to_lowercase():
    result = preprocess_text("TRANSFORMER ARCHITECTURE")

    assert result == ["transformer", "architecture"]


def test_preprocess_text_removes_punctuation():
    result = preprocess_text("Hello, world!")

    assert result == ["hello", "world"]


def test_preprocess_text_handles_empty_string():
    assert preprocess_text("") == []


def test_preprocess_text_handles_whitespace():
    assert preprocess_text("   \t\n  ") == []


def test_preprocess_text_separates_hyphenated_words():
    result = preprocess_text("machine-learning")

    assert result == ["machine", "learning"]


def test_preprocess_text_preserves_numbers():
    text = "The company reported 150 employees in 2024."

    result = preprocess_text(text)

    assert "150" in result
    assert "2024" in result


def test_preprocess_text_handles_stopword_only_input():
    assert preprocess_text("the is a", remove_stop_words=True) == []


def test_preprocess_text_rejects_non_string_input():
    with pytest.raises(TypeError):
        preprocess_text(123)


def test_preprocess_text_preserves_international_characters():
    result = preprocess_text("Café München naïve")

    assert result == ["café", "münchen", "naïve"]


def test_preprocess_text_preserves_negation():
    result = preprocess_text("Revenue did not increase")

    assert "not" in result


@pytest.mark.parametrize(
    "word",
    [
        "hasn't",
        "haven't",
        "hadn't",
        "mustn't",
        "needn't",
        "mightn't",
        "shan't",
    ],
)
def test_preprocess_text_preserves_contracted_negation_when_removing_stopwords(
    word,
):
    assert preprocess_text(word, remove_stop_words=True) == [word]


def test_preprocess_text_preserves_decimal():
    result = preprocess_text("Revenue increased by 2.5 million")

    assert "2.5" in result


def test_preprocess_text_preserves_percentage():
    result = preprocess_text("Revenue increased by 24%")

    assert "24%" in result


def test_preprocess_text_preserves_currency():
    result = preprocess_text("The project cost $3,100")

    assert "$3,100" in result


def test_preprocess_text_preserves_iso_date():
    result = preprocess_text("The report was published on 2026-11-04")
    assert "2026-11-04" in result


def test_preprocess_text_preserves_slash_date():
    result = preprocess_text("The meeting is scheduled for 11/04/2026")
    assert "11/04/2026" in result


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pages 3-5", ["pages", "3", "5"]),
        ("covid-19", ["covid", "19"]),
        ("2020-2021", ["2020", "2021"]),
        ("5+3", ["5", "3"]),
    ],
)
def test_preprocess_text_does_not_treat_joined_hyphen_as_sign(text, expected):
    assert preprocess_text(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("-5 degrees", ["-5", "degrees"]),
        ("a change of +3", ["a", "change", "of", "+3"]),
        ("(-5)", ["-5"]),
        ("-$5", ["-$5"]),
    ],
)
def test_preprocess_text_preserves_standalone_signs(text, expected):
    assert preprocess_text(text) == expected


def test_preprocess_text_normalizes_curly_apostrophes():
    assert preprocess_text("don’t") == preprocess_text("don't") == ["don't"]


def test_preprocess_text_removes_stopwords_with_curly_apostrophes():
    assert preprocess_text("it’s", remove_stop_words=True) == []