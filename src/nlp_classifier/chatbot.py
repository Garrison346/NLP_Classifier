"""Turn user messages into chatbot replies.

This module does not read from or print to the terminal, so the command line
and a future user interface can share the same logic.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from sklearn.pipeline import Pipeline

from nlp_classifier.intent_classifier import (
    INTENTS_PATH,
    UNKNOWN_INTENT,
    IntentPrediction,
    predict_intent,
)


UNKNOWN_REPLY = "Sorry, I didn't understand that. Type \"help\" to see what I can do."
CLASSIFY_DOCUMENT_REPLY = (
    "It sounds like you want to classify a document, "
    "but I can't do that yet. Document support is coming soon."
)
DOCUMENT_QUESTION_REPLY = (
    "It sounds like you have a question about a document, "
    "but I can't read documents yet. Document support is coming soon."
)
SUMMARIZE_DOCUMENT_REPLY = (
    "It sounds like you want a document summary, "
    "but I can't do that yet. Document support is coming soon."
)

# Replies for intents that will run code instead of using a fixed response.
ACTION_REPLIES = {
    "classify_document": CLASSIFY_DOCUMENT_REPLY,
    "ask_document_question": DOCUMENT_QUESTION_REPLY,
    "summarize_document": SUMMARIZE_DOCUMENT_REPLY,
}


@dataclass(frozen=True)
class ChatReply:
    """The chatbot's reply and the prediction that led to it."""

    text: str
    prediction: IntentPrediction


def load_responses(path: Path = INTENTS_PATH) -> dict[str, list[str]]:
    """
    Load fixed responses from an intents JSON file.

    Args:
        path: Path to the intents file.

    Returns:
        A mapping of intent tags to their responses. Intents without
        responses are left out.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return {
        intent["tag"]: intent["responses"]
        for intent in data["intents"]
        if intent.get("responses")
    }


class Chatbot:
    """Choose a reply for each user message."""

    def __init__(self, model: Pipeline, responses: dict[str, list[str]]) -> None:
        """
        Create a chatbot.

        Args:
            model: A trained intent classifier.
            responses: Fixed responses for each intent.
        """
        self.model = model
        self.responses = responses

    def respond(self, message: str) -> ChatReply:
        """
        Reply to a user message.

        Args:
            message: The user's message.

        Returns:
            The reply text and the prediction behind it.
        """
        prediction = predict_intent(self.model, message)

        return ChatReply(self._reply_for(prediction.intent), prediction)

    def _reply_for(self, intent: str) -> str:
        """Choose the reply for a predicted intent."""
        if intent in ACTION_REPLIES:
            return ACTION_REPLIES[intent]

        if intent in self.responses:
            return random.choice(self.responses[intent])

        if intent == UNKNOWN_INTENT:
            return UNKNOWN_REPLY

        raise ValueError(f"No reply is available for intent: {intent}")
