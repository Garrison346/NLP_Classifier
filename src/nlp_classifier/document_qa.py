"""Answer questions about a processed document using TF-IDF retrieval.

The answer is the document sentence that best matches the question, found in
two steps: rank passages by cosine similarity to the question, then choose
the best sentence within the top passages.

Run this module to ask questions about a PDF:

    uv run python -m nlp_classifier.document_qa path/to/file.pdf
"""

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_classifier.document_processor import (
    Document,
    DocumentError,
    process_document,
    split_sentences,
)
from nlp_classifier.preprocessing import preprocess_text


# Answers scoring below this are treated as not found.
MIN_ANSWER_SCORE = 0.2

# How many of the best-matching passages to search for an answer sentence.
PASSAGES_TO_SEARCH = 3

# How much a sentence's own match counts, compared with its passage's match.
SENTENCE_WEIGHT = 0.5

# Shorter sentences are usually headings, such as "Dividends.", which match a
# single word strongly but do not answer anything.
MIN_ANSWER_WORDS = 4

STEMMER = SnowballStemmer("english")


def tokenize_for_search(text: str) -> list[str]:
    """
    Turn text into search terms.

    Stopwords are removed so that words such as "what" and "is" do not drive
    the match, and words are stemmed so that "scaling" matches "scaled".

    Args:
        text: A question or document text.

    Returns:
        A list of search terms.
    """
    return [
        STEMMER.stem(token)
        for token in preprocess_text(text, remove_stop_words=True)
    ]


@dataclass(frozen=True)
class Answer:
    """
    A sentence that answers a question.

    Attributes:
        sentence: The best-matching sentence.
        passage: The passage containing the sentence, for more context.
        page_number: The page where the passage starts.
        score: How well the answer matches, from 0 to 1.
    """

    sentence: str
    passage: str
    page_number: int
    score: float


class DocumentIndex:
    """Search a single document for answers to questions."""

    def __init__(
        self,
        document: Document,
        tokenizer: Callable[[str], list[str]] = tokenize_for_search,
        min_score: float = MIN_ANSWER_SCORE,
        passages_to_search: int = PASSAGES_TO_SEARCH,
        sentence_weight: float = SENTENCE_WEIGHT,
        min_answer_words: int = MIN_ANSWER_WORDS,
    ) -> None:
        """
        Build a search index for a document.

        Word importance (IDF) is learned from this document's passages, so a
        word that appears everywhere in the document counts for little.

        Args:
            document: A processed document.
            tokenizer: Turns text into search terms.
            min_score: Answers scoring below this are treated as not found.
            passages_to_search: How many top passages to search for sentences.
            sentence_weight: How much a sentence's own match counts, from 0 to
                1. The rest of the score comes from its passage's match.
            min_answer_words: Shorter sentences are never given as answers.
        """
        self.document = document
        self.min_score = min_score
        self.passages_to_search = passages_to_search
        self.sentence_weight = sentence_weight

        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            lowercase=False,  # The tokenizer already lowercases.
            token_pattern=None,  # Unused because a tokenizer is provided.
            sublinear_tf=True,  # Repeating a word adds less and less weight.
        )

        passage_texts = [passage.text for passage in document.passages]

        # Use each passage's original sentences when available, because
        # splitting the joined text again loses heading boundaries.
        self.passage_sentences = [
            [
                sentence
                for sentence in passage.sentences or split_sentences(passage.text)
                if len(sentence.split()) >= min_answer_words
            ]
            for passage in document.passages
        ]

        try:
            self.passage_vectors = self.vectorizer.fit_transform(passage_texts)
        except ValueError:
            # The document has no searchable words, such as only stopwords.
            self.passage_vectors = None

    def answer(self, question: str) -> Answer | None:
        """
        Find the sentence that best answers a question.

        Args:
            question: The user's question.

        Returns:
            The best answer, or None if nothing matches well enough.
        """
        if self.passage_vectors is None:
            return None

        question_vector = self.vectorizer.transform([question])
        if question_vector.nnz == 0:
            return None

        # TF-IDF vectors have length 1, so a dot product is cosine similarity.
        passage_scores = (self.passage_vectors @ question_vector.T).toarray().ravel()
        top_passages = np.argsort(passage_scores)[::-1][: self.passages_to_search]

        best: Answer | None = None
        for index in top_passages:
            sentences = self.passage_sentences[index]
            if not sentences:
                continue

            sentence_vectors = self.vectorizer.transform(sentences)
            sentence_scores = (sentence_vectors @ question_vector.T).toarray().ravel()

            for sentence, sentence_score in zip(sentences, sentence_scores):
                score = (
                    self.sentence_weight * sentence_score
                    + (1 - self.sentence_weight) * passage_scores[index]
                )
                if best is None or score > best.score:
                    passage = self.document.passages[index]
                    best = Answer(sentence, passage.text, passage.page_number, float(score))

        if best is None or best.score < self.min_score:
            return None

        return best


def main(argv: list[str] | None = None) -> int:
    """
    Ask questions about a PDF in the terminal.

    Args:
        argv: Command-line arguments. Defaults to those given to the program.

    Returns:
        A process exit code, where 0 means success.
    """
    parser = argparse.ArgumentParser(description="Ask questions about a PDF.")
    parser.add_argument("path", help="path to a PDF file")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(errors="replace")

    try:
        index = DocumentIndex(process_document(args.path))
    except DocumentError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print('Ask a question, or type "exit" to leave.')
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        answer = index.answer(question)
        if answer is None:
            print("Answer: I couldn't find that in the document.")
        else:
            print(f"Answer: {answer.sentence}")
            print(f"        [page {answer.page_number}, score {answer.score:.2f}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
