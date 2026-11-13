"""Summarize a processed document by selecting its most important sentences.

This is extractive summarization: every summary sentence is quoted from the
document and shown in its original order. Three methods are available:

- lead: the first sentences, a simple baseline that real methods should beat.
- tfidf: sentences containing the document's most important words.
- textrank: sentences most similar to many other sentences, ranked with the
  PageRank algorithm on a graph of sentence similarities.

The tfidf and textrank methods also avoid choosing sentences that repeat each
other, using Maximal Marginal Relevance (MMR).

Run this module to summarize a PDF:

    uv run python -m nlp_classifier.summarizer path/to/file.pdf
"""

import argparse
import re
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_classifier.document_processor import (
    Document,
    DocumentError,
    Sentence,
    process_document,
    split_sentences,
)
from nlp_classifier.document_qa import tokenize_for_search


METHODS = ("lead", "tfidf", "textrank")
DEFAULT_METHOD = "textrank"

# Summaries have between 3 and 6 sentences, about one for every
# SENTENCES_PER_SUMMARY_SENTENCE sentences in the document.
MIN_SUMMARY_SENTENCES = 3
MAX_SUMMARY_SENTENCES = 6
SENTENCES_PER_SUMMARY_SENTENCE = 10

# Shorter sentences are usually headings or labels, such as "Dividends."
MIN_SENTENCE_WORDS = 6

# Sentences where more than this share of words contain digits are usually
# table rows, such as "Wholesale revenue 31,480 28,960".
MAX_NUMBER_SHARE = 0.4

# Sentences with web addresses are usually citations or source lists.
WEB_ADDRESS = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|org|gov|edu|net)\b", re.IGNORECASE)

# PageRank settings: the chance of following a link rather than jumping to a
# random sentence, and when to stop iterating.
DAMPING = 0.85
TOLERANCE = 1e-6
MAX_ITERATIONS = 100

# Maximal Marginal Relevance (MMR): how strongly a candidate is penalized for
# resembling sentences already chosen, from 0 (no penalty) to 1. It applies to
# the tfidf and textrank methods, but not the lead baseline.
REDUNDANCY_PENALTY = 0.3


@dataclass(frozen=True)
class SummarySentence:
    """
    A sentence chosen for a summary.

    Attributes:
        text: The sentence.
        page_number: The page where it starts.
        score: How important the method judged it. Scores from different
            methods are not comparable.
    """

    text: str
    page_number: int
    score: float


def summary_length(sentence_count: int) -> int:
    """
    Choose how many sentences a summary should have.

    Args:
        sentence_count: The number of sentences that could be chosen.

    Returns:
        A length from MIN_SUMMARY_SENTENCES to MAX_SUMMARY_SENTENCES, or fewer
        if the document has fewer sentences.
    """
    length = round(sentence_count / SENTENCES_PER_SUMMARY_SENTENCE)
    length = max(MIN_SUMMARY_SENTENCES, min(MAX_SUMMARY_SENTENCES, length))

    return min(length, sentence_count)


def _is_table_row(text: str) -> bool:
    """Check whether most words in a sentence contain digits."""
    words = text.split()
    numbers = sum(bool(re.search(r"\d", word)) for word in words)

    return numbers / len(words) > MAX_NUMBER_SHARE


def candidate_sentences(document: Document) -> list[Sentence]:
    """
    Choose the sentences that could appear in a summary.

    Short sentences, table rows, citations with web addresses, and repeated
    sentences are left out.

    Args:
        document: A processed document.

    Returns:
        The candidate sentences in reading order.
    """
    sentences = document.sentences or [
        Sentence(text, page.number)
        for page in document.pages
        for text in split_sentences(page.text)
    ]

    seen: set[str] = set()
    candidates = []

    for sentence in sentences:
        if (
            len(sentence.text.split()) >= MIN_SENTENCE_WORDS
            and not _is_table_row(sentence.text)
            and not WEB_ADDRESS.search(sentence.text)
            and sentence.text not in seen
        ):
            seen.add(sentence.text)
            candidates.append(sentence)

    return candidates


def score_lead(texts: list[str]) -> np.ndarray:
    """Score sentences so that earlier ones score higher."""
    return np.linspace(1, 0, num=len(texts), endpoint=False)


def score_tfidf(texts: list[str]) -> np.ndarray:
    """
    Score sentences by how important their words are to the document.

    A word's importance is its total TF-IDF weight across all sentences: it is
    high for words used often in the document but in few sentences. Each
    sentence scores the average importance of its distinct words, so long
    sentences are not favored just for being long.

    Args:
        texts: The candidate sentences.

    Returns:
        One score per sentence.
    """
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_for_search,
        lowercase=False,
        token_pattern=None,
        norm=None,  # Keep raw weights so they can be added across sentences.
    )
    matrix = vectorizer.fit_transform(texts)
    importance = np.asarray(matrix.sum(axis=0)).ravel()

    present = matrix > 0
    word_counts = np.asarray(present.sum(axis=1)).ravel()
    totals = np.asarray(present @ importance).ravel()

    return np.divide(totals, word_counts, out=np.zeros(len(texts)), where=word_counts > 0)


def sentence_similarity(texts: list[str]) -> np.ndarray:
    """
    Compare every pair of sentences.

    Args:
        texts: The sentences.

    Returns:
        A square matrix of TF-IDF cosine similarities, from 0 to 1.

    Raises:
        ValueError: If no sentence has a searchable word.
    """
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_for_search,
        lowercase=False,
        token_pattern=None,
    )
    vectors = vectorizer.fit_transform(texts)

    return (vectors @ vectors.T).toarray()


def select_diverse(
    scores: np.ndarray,
    similarity: np.ndarray,
    count: int,
    penalty: float = REDUNDANCY_PENALTY,
) -> list[int]:
    """
    Choose high-scoring sentences that do not repeat each other.

    This is Maximal Marginal Relevance (MMR). Sentences are chosen one at a
    time. Each candidate's score is scaled to between 0 and 1, then reduced by
    its similarity to the most similar sentence already chosen:

        (1 - penalty) * score - penalty * highest similarity

    Args:
        scores: Each sentence's importance.
        similarity: Similarities between sentences, from sentence_similarity.
        count: How many sentences to choose.
        penalty: How much similarity to chosen sentences counts, from 0 to 1.
            At 0, sentences are chosen by score alone.

    Returns:
        The positions of the chosen sentences, in the order they were chosen.
    """
    spread = scores.max() - scores.min()
    scaled = (scores - scores.min()) / spread if spread > 0 else np.ones_like(scores)

    chosen: list[int] = []
    remaining = list(range(len(scores)))

    while remaining and len(chosen) < count:
        def value(index: int) -> float:
            overlap = max((similarity[index, j] for j in chosen), default=0.0)
            return (1 - penalty) * scaled[index] - penalty * overlap

        # Earlier sentences win ties.
        best = max(remaining, key=lambda index: (value(index), -index))
        chosen.append(best)
        remaining.remove(best)

    return chosen


def score_textrank(texts: list[str]) -> np.ndarray:
    """
    Score sentences with TextRank.

    Sentences are nodes in a graph, linked by the cosine similarity of their
    TF-IDF vectors. PageRank then gives high scores to sentences that are
    similar to many other well-connected sentences, which tend to express the
    document's central ideas.

    Args:
        texts: The candidate sentences.

    Returns:
        One score per sentence, adding up to 1.
    """
    similarity = sentence_similarity(texts)
    np.fill_diagonal(similarity, 0)

    # Each sentence shares its score among the sentences it links to, in
    # proportion to similarity. A sentence with no links shares it equally.
    count = len(texts)
    totals = similarity.sum(axis=1, keepdims=True)
    transition = np.divide(
        similarity,
        totals,
        out=np.full_like(similarity, 1 / count),
        where=totals > 0,
    )

    scores = np.full(count, 1 / count)
    for _ in range(MAX_ITERATIONS):
        updated = (1 - DAMPING) / count + DAMPING * (transition.T @ scores)
        if np.abs(updated - scores).sum() < TOLERANCE:
            return updated
        scores = updated

    return scores


SCORERS = {
    "lead": score_lead,
    "tfidf": score_tfidf,
    "textrank": score_textrank,
}


def summarize(
    document: Document,
    method: str = DEFAULT_METHOD,
    max_sentences: int | None = None,
    redundancy_penalty: float = REDUNDANCY_PENALTY,
) -> list[SummarySentence]:
    """
    Summarize a document.

    Args:
        document: A processed document.
        method: "lead", "tfidf", or "textrank".
        max_sentences: How many sentences to choose. Defaults to a length
            based on the document's size.
        redundancy_penalty: How strongly to avoid sentences that repeat ones
            already chosen, from 0 to 1. Not used by the lead method.

    Returns:
        The chosen sentences in their original order, or an empty list if no
        sentence is long enough to use.

    Raises:
        ValueError: If the method is not recognized.
    """
    if method not in SCORERS:
        raise ValueError(f"Unknown method {method!r}. Choose from: {', '.join(METHODS)}")

    candidates = candidate_sentences(document)
    if not candidates:
        return []

    length = summary_length(len(candidates)) if max_sentences is None else max_sentences
    texts = [sentence.text for sentence in candidates]

    try:
        scores = SCORERS[method](texts)
        similarity = sentence_similarity(texts)
    except ValueError:
        # No sentence has a searchable word, so fall back to the opening.
        method = "lead"
        scores = score_lead(texts)

    if method == "lead" or redundancy_penalty == 0:
        # Sort by score, using position to break ties.
        ranked = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
        picked = ranked[:length]
    else:
        picked = select_diverse(scores, similarity, length, redundancy_penalty)

    # Show the chosen sentences in reading order.
    chosen = sorted(picked)

    return [
        SummarySentence(candidates[index].text, candidates[index].page_number, float(scores[index]))
        for index in chosen
    ]


def main(argv: list[str] | None = None) -> int:
    """
    Summarize a PDF in the terminal.

    Args:
        argv: Command-line arguments. Defaults to those given to the program.

    Returns:
        A process exit code, where 0 means success.
    """
    parser = argparse.ArgumentParser(description="Summarize a PDF.")
    parser.add_argument("path", help="path to a PDF file")
    parser.add_argument(
        "--method",
        choices=METHODS,
        default=DEFAULT_METHOD,
        help=f"summarization method (default: {DEFAULT_METHOD})",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        help="number of sentences (default: based on document length)",
    )
    parser.add_argument(
        "--redundancy-penalty",
        type=float,
        default=REDUNDANCY_PENALTY,
        help=f"how strongly to avoid repeated ideas, from 0 to 1 (default: {REDUNDANCY_PENALTY})",
    )
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(errors="replace")

    try:
        document = process_document(args.path)
    except DocumentError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    summary = summarize(document, args.method, args.sentences, args.redundancy_penalty)
    if not summary:
        print("This document has no sentences long enough to summarize.")
        return 0

    print(f"Summary of {document.name} ({args.method}):")
    for sentence in summary:
        print(f"- {sentence.text} (page {sentence.page_number})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
