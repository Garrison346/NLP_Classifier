"""Command-line entry point for the chatbot."""

import argparse
import sys

from nlp_classifier.chatbot import ChatReply, Chatbot, load_responses
from nlp_classifier.intent_classifier import MODEL_PATH, load_model


EXIT_COMMANDS = {"exit", "quit"}
TRAINING_COMMAND = "uv run python -m nlp_classifier.intent_classifier"


def format_reply(reply: ChatReply, debug: bool = False) -> str:
    """
    Format a chatbot reply for the terminal.

    Args:
        reply: The chatbot's reply.
        debug: Whether to include the predicted intent and confidence.

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
        text += f"\n     [{', '.join(details)}]"

    return text


def run_chat(chatbot: Chatbot, debug: bool = False) -> None:
    """
    Chat in the terminal until the user exits.

    Args:
        chatbot: The chatbot to talk to.
        debug: Whether to show each predicted intent and confidence.
    """
    print('Chatbot ready. Type "exit" or "quit" to leave.')

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
        help="show the predicted intent and confidence for each message",
    )
    args = parser.parse_args(argv)

    try:
        model = load_model()
    except FileNotFoundError:
        print(
            f"No trained model found at {MODEL_PATH}.\n"
            f"Train one first with: {TRAINING_COMMAND}",
            file=sys.stderr,
        )
        return 1

    run_chat(Chatbot(model, load_responses()), debug=args.debug)

    return 0


if __name__ == "__main__":
    sys.exit(main())
