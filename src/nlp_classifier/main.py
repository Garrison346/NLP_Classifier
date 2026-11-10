"""Command-line entry point for the chatbot."""

import argparse
import sys

from nlp_classifier.chatbot import HELP_TEXT, ChatReply, Chatbot, load_responses
from nlp_classifier.intent_classifier import MODEL_PATH, load_model


EXIT_COMMANDS = {"exit", "quit"}
HELP_COMMAND = "help"
LOAD_COMMAND = "load"
LOAD_USAGE = "Type \"load\" followed by a PDF's path, for example: load data/documents/report.pdf"
TRAINING_COMMAND = "uv run python -m nlp_classifier.intent_classifier"


def format_reply(reply: ChatReply, debug: bool = False) -> str:
    """
    Format a chatbot reply for the terminal.

    Args:
        reply: The chatbot's reply.
        debug: Whether to include the predicted intent, confidence, and the
            document answer's score.

    Returns:
        The text to display.
    """
    text = f"Bot: {reply.text}"

    if debug:
        prediction = reply.prediction
        details = [prediction.intent]

        if prediction.intent != prediction.top_guess:
            details.append(f"top guess {prediction.top_guess}")

        details.append(f"confidence {prediction.confidence:.2f}")

        if reply.answer is not None:
            details.append(f"answer score {reply.answer.score:.2f}")

        text += f"\n     [{', '.join(details)}]"

    return text


def parse_load_command(message: str) -> str | None:
    """
    Get the path from a load command.

    Args:
        message: The user's message.

    Returns:
        The path, an empty string if no path was given, or None if the
        message is not a load command.
    """
    command, _, path = message.partition(" ")

    if command.lower() != LOAD_COMMAND:
        return None

    # Allow quotes around paths that contain spaces.
    return path.strip().strip("\"'")


def run_chat(chatbot: Chatbot, debug: bool = False) -> None:
    """
    Chat in the terminal until the user exits.

    Args:
        chatbot: The chatbot to talk to.
        debug: Whether to show how each reply was chosen.
    """
    print('Chatbot ready. Type "load" and a PDF path to ask about a document,')
    print('or "exit" to leave.')

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C or Ctrl+D leaves the prompt on the current line.
            print()
            break

        if not message:
            continue

        if message.lower() in EXIT_COMMANDS:
            break

        if message.lower() == HELP_COMMAND:
            print(f"Bot: {HELP_TEXT}")
            continue

        path = parse_load_command(message)
        if path is not None:
            print(f"Bot: {chatbot.load_document(path) if path else LOAD_USAGE}")
            continue

        print(format_reply(chatbot.respond(message), debug))

    print("Goodbye!")


def main(argv: list[str] | None = None) -> int:
    """
    Run the command-line chatbot.

    Args:
        argv: Command-line arguments. Defaults to those given to the program.

    Returns:
        A process exit code, where 0 means success.
    """
    parser = argparse.ArgumentParser(description="Chat with the NLP chatbot.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show how each reply was chosen",
    )
    parser.add_argument(
        "--document",
        help="path to a PDF to load at the start",
    )
    args = parser.parse_args(argv)

    # PDF text can contain characters that some terminals cannot display.
    sys.stdout.reconfigure(errors="replace")

    try:
        model = load_model()
    except FileNotFoundError:
        print(
            f"No trained model found at {MODEL_PATH}.\n"
            f"Train one first with: {TRAINING_COMMAND}",
            file=sys.stderr,
        )
        return 1

    chatbot = Chatbot(model, load_responses())

    if args.document:
        print(f"Bot: {chatbot.load_document(args.document)}")

    run_chat(chatbot, debug=args.debug)

    return 0


if __name__ == "__main__":
    sys.exit(main())
