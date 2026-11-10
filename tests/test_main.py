import pytest

from nlp_classifier import main as main_module
from nlp_classifier.chatbot import HELP_TEXT, ChatReply
from nlp_classifier.document_qa import Answer
from nlp_classifier.intent_classifier import IntentPrediction
from nlp_classifier.main import format_reply, main, parse_load_command, run_chat


class FakeChatbot:
    """A chatbot that records messages and gives a fixed reply."""

    def __init__(self):
        self.messages = []
        self.loaded = []

    def respond(self, message):
        self.messages.append(message)
        return ChatReply("Fake reply", IntentPrediction("greeting", 0.91, "greeting"))

    def load_document(self, path):
        self.loaded.append(path)
        return f"Loaded {path}"


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


def test_format_reply_shows_answer_score_in_debug_mode():
    answer = Answer("Orcas are dolphins.", "Orcas are dolphins.", 2, 0.42)
    reply = ChatReply("Orcas are dolphins. (page 2)", IntentPrediction(
        "ask_document_question", 0.8, "ask_document_question"), answer)

    assert "answer score 0.42" in format_reply(reply, debug=True)
    assert "score" not in format_reply(reply)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("load notes.pdf", "notes.pdf"),
        ("LOAD notes.pdf", "notes.pdf"),
        ('load "my notes.pdf"', "my notes.pdf"),
        ("load   data/report.pdf  ", "data/report.pdf"),
        ("load", ""),
        ("loader notes.pdf", None),
        ("Can you load this?", None),
    ],
)
def test_parse_load_command(message, expected):
    assert parse_load_command(message) == expected


def test_run_chat_loads_document_without_sending_it_as_a_message(monkeypatch, capsys):
    bot = FakeChatbot()
    type_lines(monkeypatch, 'load "my notes.pdf"', "exit")

    run_chat(bot)

    assert bot.loaded == ["my notes.pdf"]
    assert bot.messages == []
    assert "Bot: Loaded my notes.pdf" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["help", "HELP", "  Help  "])
def test_run_chat_shows_help_without_sending_it_as_a_message(monkeypatch, capsys, command):
    bot = FakeChatbot()
    type_lines(monkeypatch, command, "exit")

    run_chat(bot)

    assert bot.messages == []
    assert f"Bot: {HELP_TEXT}" in capsys.readouterr().out


def test_run_chat_sends_longer_help_requests_to_the_chatbot(monkeypatch):
    bot = FakeChatbot()
    type_lines(monkeypatch, "help me understand this", "exit")

    run_chat(bot)

    assert bot.messages == ["help me understand this"]


def test_run_chat_explains_load_command_without_path(monkeypatch, capsys):
    bot = FakeChatbot()
    type_lines(monkeypatch, "load", "exit")

    run_chat(bot)

    assert bot.loaded == []
    assert main_module.LOAD_USAGE in capsys.readouterr().out


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


def test_main_loads_document_from_option(monkeypatch, capsys):
    loaded = []
    monkeypatch.setattr(main_module, "load_model", lambda: None)
    monkeypatch.setattr(main_module, "run_chat", lambda chatbot, debug: None)
    monkeypatch.setattr(
        main_module.Chatbot,
        "load_document",
        lambda self, path: loaded.append(path) or f"Loaded {path}",
    )

    assert main(["--document", "notes.pdf"]) == 0
    assert loaded == ["notes.pdf"]
    assert "Bot: Loaded notes.pdf" in capsys.readouterr().out
