import json

import pytest

from nlp_classifier import chatbot as chatbot_module
from nlp_classifier.chatbot import (
    ACTION_REPLIES,
    UNKNOWN_REPLY,
    Chatbot,
    load_responses,
)
from nlp_classifier.intent_classifier import (
    INTENTS_PATH,
    UNKNOWN_INTENT,
    IntentPrediction,
    load_intents,
)


RESPONSES = {"greeting": ["Hello!", "Hi!"]}


def make_chatbot(monkeypatch, intent, confidence=0.9):
    """Create a chatbot whose model always predicts the given intent."""
    monkeypatch.setattr(
        chatbot_module,
        "predict_intent",
        lambda model, message: IntentPrediction(intent, confidence, intent),
    )

    return Chatbot(model=None, responses=RESPONSES)


def test_load_responses_skips_intents_without_responses(tmp_path):
    path = tmp_path / "intents.json"
    path.write_text(
        json.dumps(
            {
                "intents": [
                    {"tag": "greeting", "patterns": ["Hi"], "responses": ["Hello!"]},
                    {"tag": "classify_document", "patterns": ["Classify this"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_responses(path) == {"greeting": ["Hello!"]}


def test_respond_uses_a_fixed_response(monkeypatch):
    reply = make_chatbot(monkeypatch, "greeting").respond("hello")

    assert reply.text in RESPONSES["greeting"]


def test_respond_includes_the_prediction(monkeypatch):
    reply = make_chatbot(monkeypatch, "greeting", 0.85).respond("hello")

    assert reply.prediction == IntentPrediction("greeting", 0.85, "greeting")


def test_respond_handles_unknown_intent(monkeypatch):
    reply = make_chatbot(monkeypatch, UNKNOWN_INTENT).respond("xyzzy")

    assert reply.text == UNKNOWN_REPLY


@pytest.mark.parametrize(
    "intent",
    ["classify_document", "ask_document_question", "summarize_document"],
)
def test_respond_says_document_features_are_not_ready(monkeypatch, intent):
    reply = make_chatbot(monkeypatch, intent).respond("document request")

    assert reply.text == ACTION_REPLIES[intent]
    assert "yet" in reply.text


def test_respond_rejects_intent_without_a_reply(monkeypatch):
    with pytest.raises(ValueError):
        make_chatbot(monkeypatch, "new_intent").respond("hello")


def test_every_project_intent_has_a_reply():
    _, labels = load_intents(INTENTS_PATH)
    responses = load_responses(INTENTS_PATH)

    assert set(labels) == set(responses) | set(ACTION_REPLIES)
