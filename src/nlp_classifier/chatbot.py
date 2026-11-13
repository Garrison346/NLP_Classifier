"""Turn user messages into chatbot replies.

This module does not read from or print to the terminal, so the command line
and a future user interface can share the same logic.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from sklearn.pipeline import Pipeline

from nlp_classifier.document_processor import Document, DocumentError, process_document
from nlp_classifier.document_qa import Answer, DocumentIndex
from nlp_classifier.summarizer import summarize
from nlp_classifier.intent_classifier import (
    INTENTS_PATH,
    UNKNOWN_INTENT,
    IntentPrediction,
    predict_intent,
)


QUESTION_INTENT = "ask_document_question"
SUMMARY_INTENT = "summarize_document"

UNKNOWN_REPLY = "Sorry, I didn't understand that. Type \"help\" to see what I can do."

# Shown for the "help" command, which does not use the intent classifier.
HELP_TEXT = (
    "I can answer questions about a PDF and summarize it. Type \"load\" "
    "followed by its path, then ask me about it or ask for a summary. Type "
    "\"exit\" to leave. Identifying a document's theme is coming soon."
)
NO_DOCUMENT_REPLY = (
    "Load a PDF first by typing \"load\" followed by its path, "
    "for example: load data/documents/report.pdf"
)
NOT_FOUND_REPLY = "I couldn't find the answer to that in {name}."
MISSING_NLTK_DATA_REPLY = (
    "Answering questions needs NLTK's stopword data. "
    "Download it with: uv run python -m nltk.downloader stopwords"
)
CLASSIFY_DOCUMENT_REPLY = (
    "It sounds like you want to classify a document, "
    "but I can't do that yet. Document classification is coming soon."
)
NO_SUMMARY_REPLY = "{name} has no sentences long enough to summarize."

# Replies for document features that are not available yet.
ACTION_REPLIES = {
    "classify_document": CLASSIFY_DOCUMENT_REPLY,
}


@dataclass(frozen=True)
class ChatReply:
    """
    The chatbot's reply and how it was chosen.

    Attributes:
        text: The reply to show the user.
        prediction: The intent prediction for the message.
        answer: The document answer behind the reply, if there was one.
    """

    text: str
    prediction: IntentPrediction
    answer: Answer | None = None


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


def format_answer(answer: Answer) -> str:
    """Format a document answer with the page it came from."""
    return f"{answer.sentence} (page {answer.page_number})"


class Chatbot:
    """Choose a reply for each user message."""

    def __init__(self, model: Pipeline, responses: dict[str, list[str]]) -> None:
        """
        Create a chatbot with no document loaded.

        Args:
            model: A trained intent classifier.
            responses: Fixed responses for each intent.
        """
        self.model = model
        self.responses = responses
        self.document: Document | None = None
        self.document_name: str | None = None
        self.index: DocumentIndex | None = None

    def load_document(self, path: str | Path) -> str:
        """
        Load a PDF so that questions can be answered from it.

        The previous document stays loaded if the new one cannot be read.

        Args:
            path: Path to the PDF.

        Returns:
            A message saying whether the document was loaded.
        """
        try:
            document = process_document(path)
            index = DocumentIndex(document)
        except DocumentError as error:
            return f"I couldn't load that document. {error}"
        except LookupError:
            return MISSING_NLTK_DATA_REPLY

        self.document = document
        self.document_name = document.name
        self.index = index

        return (
            f"Loaded {document.name} ({len(document.pages)} of "
            f"{document.page_count} pages have text). Ask me a question about it, or ask for a summary."
        )

    def respond(self, message: str) -> ChatReply:
        """
        Reply to a user message.

        When a document is loaded, a message the classifier is unsure about is
        also tried as a question, because many real questions are classified
        as unknown. It is only answered if a good match is found.

        Args:
            message: The user's message.

        Returns:
            The reply, the prediction behind it, and any document answer.
        """
        prediction = predict_intent(self.model, message)
        intent = prediction.intent

        if intent == QUESTION_INTENT and self.index is None:
            return ChatReply(NO_DOCUMENT_REPLY, prediction)

        if intent == QUESTION_INTENT or (
            intent == UNKNOWN_INTENT and self.index is not None
        ):
            answer = self.index.answer(message)

            if answer is not None:
                return ChatReply(format_answer(answer), prediction, answer)

            if intent == QUESTION_INTENT:
                return ChatReply(
                    NOT_FOUND_REPLY.format(name=self.document_name),
                    prediction,
                )

        if intent == SUMMARY_INTENT:
            return ChatReply(self._summary_reply(), prediction)

        return ChatReply(self._reply_for(intent), prediction)

    def _summary_reply(self) -> str:
        """Summarize the loaded document, or explain why it cannot be."""
        if self.document is None:
            return NO_DOCUMENT_REPLY

        summary = summarize(self.document)
        if not summary:
            return NO_SUMMARY_REPLY.format(name=self.document_name)

        lines = [f"Summary of {self.document_name}:"]
        lines += [
            f"- {sentence.text} (page {sentence.page_number})"
            for sentence in summary
        ]

        return "\n".join(lines)

    def _reply_for(self, intent: str) -> str:
        """Choose the reply for an intent that does not search the document."""
        if intent in ACTION_REPLIES:
            return ACTION_REPLIES[intent]

        if intent in self.responses:
            return random.choice(self.responses[intent])

        if intent == UNKNOWN_INTENT:
            return UNKNOWN_REPLY

        raise ValueError(f"No reply is available for intent: {intent}")
