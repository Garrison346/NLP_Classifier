import pytest

from nlp_classifier import main as main_module
from nlp_classifier.chatbot import ChatReply
from nlp_classifier.intent_classifier import IntentPrediction
from nlp_classifier.main import format_reply, main, run_chat


class FakeChatbot:
    """A chatbot that records messages and gives a fixed reply."""

    def __init__(self):
        self.messages = []

    def respond(self, message):
        self.messages.append(message)
        return ChatReply("Fake reply", IntentPrediction("greeting", 0.91, "greeting"))


def type_lines(monkeypatch, *lines):
    """Make input() return each line in turn, or raise a provided exception."""
    remaining = iter(lines)

    def fake_input(prompt):
        line = next(remaining)
        if isinstance(line, BaseException):
            raise line
        return line

    monkeypatch.setattr("builtins.input", fake_input)


def test_format_reply_hides_prediction_by_default():
    reply = ChatReply("Hello!", IntentPrediction("greeting", 0.91, "greeting"))

    assert format_reply(reply) == "Bot: Hello!"


def test_format_reply_shows_prediction_in_debug_mode():
    reply = ChatReply("Hello!", IntentPrediction("greeting", 0.91, "greeting"))

    assert "[greeting, confidence 0.91]" in format_reply(reply, debug=True)


def test_format_reply_shows_top_guess_for_unknown_intent():
    prediction = IntentPrediction("unknown", 0.65, "classify_document")
    reply = ChatReply("Sorry.", prediction)

    text = format_reply(reply, debug=True)

    assert "[unknown, top guess classify_document, confidence 0.65]" in text


def test_run_chat_replies_until_exit(monkeypatch, capsys):
    bot = FakeChatbot()
    type_lines(monkeypatch, "hello", "exit", "never read")

    run_chat(bot)

    assert bot.messages == ["hello"]
    assert "Bot: Fake reply" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["exit", "quit", "EXIT", "  Quit  "])
def test_run_chat_recognizes_exit_commands(monkeypatch, command):
    bot = FakeChatbot()
    type_lines(monkeypatch, command)

    run_chat(bot)

    assert bot.messages == []


def test_run_chat_ignores_blank_messages(monkeypatch):
    bot = FakeChatbot()
    type_lines(monkeypatch, "", "   ", "hello", "exit")

    run_chat(bot)

    assert bot.messages == ["hello"]


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_run_chat_exits_cleanly_when_input_ends(monkeypatch, capsys, error):
    type_lines(monkeypatch, error)

    run_chat(FakeChatbot())

    assert "Goodbye!" in capsys.readouterr().out


def test_main_explains_how_to_train_when_model_is_missing(monkeypatch, capsys):
    def missing_model():
        raise FileNotFoundError

    monkeypatch.setattr(main_module, "load_model", missing_model)

    assert main([]) == 1
    assert main_module.TRAINING_COMMAND in capsys.readouterr().err


def test_main_runs_chat_with_debug_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(main_module, "load_model", lambda: None)
    monkeypatch.setattr(
        main_module,
        "run_chat",
        lambda chatbot, debug: calls.append(debug),
    )

    assert main(["--debug"]) == 0
    assert calls == [True]
