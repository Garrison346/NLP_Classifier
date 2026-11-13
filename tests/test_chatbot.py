import json

import pytest

from nlp_classifier import chatbot as chatbot_module
from nlp_classifier.chatbot import (
    ACTION_REPLIES,
    MISSING_NLTK_DATA_REPLY,
    NO_DOCUMENT_REPLY,
    NOT_FOUND_REPLY,
    NO_SUMMARY_REPLY,
    QUESTION_INTENT,
    SUMMARY_INTENT,
    UNKNOWN_REPLY,
    Chatbot,
    load_responses,
)
from nlp_classifier.document_processor import DocumentError
from test_document_qa import make_document
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


def test_respond_says_document_classification_is_not_ready(monkeypatch):
    intent = "classify_document"
    reply = make_chatbot(monkeypatch, intent).respond("document request")

    assert reply.text == ACTION_REPLIES[intent]
    assert "yet" in reply.text


def test_respond_rejects_intent_without_a_reply(monkeypatch):
    with pytest.raises(ValueError):
        make_chatbot(monkeypatch, "new_intent").respond("hello")


def test_every_project_intent_has_a_reply():
    _, labels = load_intents(INTENTS_PATH)
    responses = load_responses(INTENTS_PATH)

    handled = set(responses) | set(ACTION_REPLIES) | {QUESTION_INTENT, SUMMARY_INTENT}

    assert set(labels) == handled


# Document questions

ORCA_DOCUMENT = make_document(
    (1, ["The orca is also called the killer whale.", "Orcas live in every ocean."]),
    (2, ["A thick layer of blubber keeps orcas warm in cold water."]),
)


def make_document_chatbot(monkeypatch, intent):
    """Create a chatbot with ORCA_DOCUMENT loaded."""
    monkeypatch.setattr(chatbot_module, "process_document", lambda path: ORCA_DOCUMENT)
    bot = make_chatbot(monkeypatch, intent)
    bot.load_document("orcas.pdf")
    return bot


def test_load_document_reports_success(monkeypatch):
    monkeypatch.setattr(chatbot_module, "process_document", lambda path: ORCA_DOCUMENT)
    bot = make_chatbot(monkeypatch, QUESTION_INTENT)

    message = bot.load_document("orcas.pdf")

    assert message.startswith("Loaded notes.pdf (2 of 2 pages have text).")
    assert bot.document_name == "notes.pdf"


def test_load_document_reports_errors_and_keeps_previous_document(monkeypatch):
    bot = make_document_chatbot(monkeypatch, QUESTION_INTENT)

    def fail(path):
        raise DocumentError("Could not find the file: missing.pdf")

    monkeypatch.setattr(chatbot_module, "process_document", fail)

    message = bot.load_document("missing.pdf")

    assert message == "I couldn't load that document. Could not find the file: missing.pdf"
    assert bot.document_name == "notes.pdf"
    assert bot.respond("What keeps orcas warm?").answer is not None


def test_load_document_explains_missing_nltk_data(monkeypatch):
    def fail(document):
        raise LookupError("Resource stopwords not found.")

    monkeypatch.setattr(chatbot_module, "process_document", lambda path: ORCA_DOCUMENT)
    monkeypatch.setattr(chatbot_module, "DocumentIndex", fail)
    bot = make_chatbot(monkeypatch, QUESTION_INTENT)

    assert bot.load_document("orcas.pdf") == MISSING_NLTK_DATA_REPLY
    assert bot.document_name is None


def test_respond_asks_for_document_when_none_is_loaded(monkeypatch):
    reply = make_chatbot(monkeypatch, QUESTION_INTENT).respond("What keeps orcas warm?")

    assert reply.text == NO_DOCUMENT_REPLY
    assert reply.answer is None


def test_respond_answers_question_with_page_number(monkeypatch):
    reply = make_document_chatbot(monkeypatch, QUESTION_INTENT).respond(
        "What keeps orcas warm?"
    )

    assert reply.text == "A thick layer of blubber keeps orcas warm in cold water. (page 2)"
    assert reply.answer.page_number == 2


def test_respond_says_when_answer_is_not_found(monkeypatch):
    reply = make_document_chatbot(monkeypatch, QUESTION_INTENT).respond(
        "Who won the football championship?"
    )

    assert reply.text == NOT_FOUND_REPLY.format(name="notes.pdf")
    assert reply.answer is None


def test_respond_tries_unknown_message_as_question_when_document_is_loaded(monkeypatch):
    reply = make_document_chatbot(monkeypatch, UNKNOWN_INTENT).respond(
        "What keeps orcas warm?"
    )

    assert reply.answer is not None
    assert reply.text.startswith("A thick layer of blubber")


def test_respond_gives_unknown_reply_when_fallback_finds_nothing(monkeypatch):
    reply = make_document_chatbot(monkeypatch, UNKNOWN_INTENT).respond(
        "Who won the football championship?"
    )

    assert reply.text == UNKNOWN_REPLY
    assert reply.answer is None


def test_respond_keeps_fixed_replies_when_document_is_loaded(monkeypatch):
    reply = make_document_chatbot(monkeypatch, "greeting").respond("hello")

    assert reply.text in RESPONSES["greeting"]
    assert reply.answer is None


# Summaries


def test_respond_asks_for_document_before_summarizing(monkeypatch):
    reply = make_chatbot(monkeypatch, SUMMARY_INTENT).respond("Summarize this")

    assert reply.text == NO_DOCUMENT_REPLY


def test_respond_summarizes_loaded_document(monkeypatch):
    document = make_document(
        (1, [
            "The orca is a large and highly social member of the dolphin family.",
            "Orcas hunt seals, fish, and squid in groups across every ocean.",
            "Orca families travel together and share hunting skills over generations.",
        ]),
    )
    monkeypatch.setattr(chatbot_module, "process_document", lambda path: document)
    bot = make_chatbot(monkeypatch, SUMMARY_INTENT)
    bot.load_document("orcas.pdf")

    reply = bot.respond("Summarize this")

    lines = reply.text.split("\n")
    assert lines[0] == "Summary of notes.pdf:"
    assert len(lines) == 4
    assert lines[1] == (
        "- The orca is a large and highly social member of the dolphin family. (page 1)"
    )


def test_respond_explains_when_document_is_too_short_to_summarize(monkeypatch):
    document = make_document((1, ["Too short.", "Also short."]))
    monkeypatch.setattr(chatbot_module, "process_document", lambda path: document)
    bot = make_chatbot(monkeypatch, SUMMARY_INTENT)
    bot.load_document("short.pdf")

    reply = bot.respond("Summarize this")

    assert reply.text == NO_SUMMARY_REPLY.format(name="notes.pdf")
