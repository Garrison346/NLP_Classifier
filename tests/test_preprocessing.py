from nlp_classifier.preprocessing import preprocess_text


def test_preprocess_text_returns_tokens():
    result = preprocess_text(
        "The Transformer architecture is fascinating!"
    )

    assert isinstance(result, list)
    assert "transformer" in result
    assert "architecture" in result


def test_preprocess_text_removes_stopwords():
    result = preprocess_text("This is a simple test")

    assert "is" not in result
    assert "a" not in result