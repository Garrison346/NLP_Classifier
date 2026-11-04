import json

import pytest

from nlp_classifier.intent_classifier import (
    INTENTS_PATH,
    TEST_INTENTS_PATH,
    UNKNOWN_INTENT,
    evaluate_model,
    evaluate_on_test_set,
    load_intents,
    load_model,
    predict_intent,
    save_model,
    train_model,
)
from nlp_classifier.preprocessing import preprocess_text


# A tiny dataset keeps these tests fast and independent of intents.json.
TEXTS = [
    "hello there",
    "hello friend",
    "hi hello",
    "hello again",
    "hello everyone",
    "goodbye now",
    "goodbye friend",
    "see you goodbye",
    "goodbye again",
    "goodbye everyone",
]
LABELS = ["greeting"] * 5 + ["goodbye"] * 5


@pytest.fixture(scope="module")
def model():
    return train_model(TEXTS, LABELS)


def test_load_intents_returns_aligned_examples_and_labels(tmp_path):
    path = tmp_path / "intents.json"
    path.write_text(
        json.dumps(
            {
                "intents": [
                    {"tag": "greeting", "patterns": ["Hi", "Hello"]},
                    {"tag": "goodbye", "patterns": ["Bye"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    texts, labels = load_intents(path)

    assert texts == ["Hi", "Hello", "Bye"]
    assert labels == ["greeting", "greeting", "goodbye"]


def test_predict_intent_returns_matching_intent(model):
    result = predict_intent(model, "hello", threshold=0.0)

    assert result.intent == "greeting"
    assert 0.5 < result.confidence <= 1.0


def test_predict_intent_returns_unknown_below_threshold(model):
    result = predict_intent(model, "hello", threshold=1.0)

    assert result.intent == UNKNOWN_INTENT
    assert result.confidence < 1.0


def test_predict_intent_accepts_confidence_equal_to_threshold(model):
    confidence = predict_intent(model, "hello", threshold=0.0).confidence

    result = predict_intent(model, "hello", threshold=confidence)

    assert result.intent == "greeting"


@pytest.mark.parametrize("text", ["", "   ", "?!", "xyzzy plugh"])
def test_predict_intent_returns_unknown_without_known_words(model, text):
    # Two balanced intents give an unknown message a confidence near 0.5.
    assert predict_intent(model, text, threshold=0.6).intent == UNKNOWN_INTENT


def test_predict_intent_rejects_non_string_input(model):
    with pytest.raises((TypeError, ValueError)):
        predict_intent(model, 123)


def test_saved_model_makes_same_prediction_after_loading(model, tmp_path):
    path = tmp_path / "nested" / "model.joblib"

    save_model(model, path)
    loaded = load_model(path)

    assert predict_intent(loaded, "goodbye") == predict_intent(model, "goodbye")


def test_evaluate_model_counts_every_example():
    _, matrix = evaluate_model(TEXTS, LABELS)

    assert list(matrix.index) == ["goodbye", "greeting"]
    assert matrix.to_numpy().sum() == len(TEXTS)


def test_evaluate_on_test_set_scores_known_results(model, tmp_path):
    path = tmp_path / "test.json"
    path.write_text(
        json.dumps(
            {
                "intents": [
                    {"tag": "greeting", "patterns": ["hello", "goodbye"]},
                ],
                "out_of_scope": ["xyzzy plugh"],
            }
        ),
        encoding="utf-8",
    )

    scores = evaluate_on_test_set(model, path, threshold=0.5)

    assert scores["accuracy_without_threshold"] == 0.5
    assert scores["answered"] == 1.0
    assert scores["correct_when_answered"] == 0.5
    assert scores["correct_overall"] == 0.5
    assert scores["out_of_scope_rejected"] == 0.0


def test_test_set_does_not_overlap_training_data():
    train_texts, train_labels = load_intents(INTENTS_PATH)
    test_texts, test_labels = load_intents(TEST_INTENTS_PATH)
    train_tokens = {tuple(preprocess_text(text)) for text in train_texts}

    assert set(test_labels) == set(train_labels)
    assert not train_tokens & {tuple(preprocess_text(t)) for t in test_texts}


def test_project_dataset_is_balanced_and_has_no_duplicates():
    texts, labels = load_intents(INTENTS_PATH)
    counts = {label: labels.count(label) for label in set(labels)}
    token_lists = [tuple(preprocess_text(text)) for text in texts]

    assert len(counts) == 6
    assert min(counts.values()) >= 20
    assert max(counts.values()) - min(counts.values()) <= 5
    assert len(token_lists) == len(set(token_lists))
    assert all(token_lists)
