"""Score document Q&A on a file of questions with known answers.

The questions file is JSON, normally kept in data/documents/ so it is not
published with the documents it quotes:

    {"questions": [
        {"document": "report.pdf", "question": "...", "answer": "a phrase the
         right sentence contains, or null if the document has no answer",
         "split": "tune" or "test"}
    ]}

Choose settings using the "tune" questions, then check them once on the
"test" questions, so the final score is not inflated by tuning:

    uv run python -m nlp_classifier.evaluate_qa data/documents/qa_eval.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from nlp_classifier.document_processor import DocumentError, process_document
from nlp_classifier.document_qa import MIN_ANSWER_SCORE, DocumentIndex
from nlp_classifier.intent_classifier import PROJECT_ROOT


DOCUMENTS_FOLDER = PROJECT_ROOT / "data" / "documents"


@dataclass(frozen=True)
class EvaluationResult:
    """
    How well Q&A did on a set of questions.

    Attributes:
        correct: Answerable questions given a sentence containing the answer.
        answerable: The number of answerable questions.
        wrong: Answerable questions given a sentence without the answer.
        rejected: Unanswerable questions correctly given no answer.
        unanswerable: The number of unanswerable questions.
    """

    correct: int
    answerable: int
    wrong: int
    rejected: int
    unanswerable: int


def load_questions(path: str | Path, split: str = "all") -> list[dict]:
    """
    Load evaluation questions.

    Args:
        path: Path to the questions file.
        split: "tune", "test", or "all".

    Returns:
        The questions in the chosen split.
    """
    questions = json.loads(Path(path).read_text(encoding="utf-8"))["questions"]

    return [q for q in questions if split == "all" or q["split"] == split]


def evaluate(
    questions: list[dict],
    documents_folder: str | Path = DOCUMENTS_FOLDER,
    min_score: float = MIN_ANSWER_SCORE,
) -> EvaluationResult:
    """
    Score Q&A on a list of questions.

    Each document is processed once, no matter how many questions use it.

    Args:
        questions: Questions from load_questions.
        documents_folder: The folder containing the documents.
        min_score: Answers scoring below this are treated as not found.

    Returns:
        The evaluation result.

    Raises:
        DocumentError: If a document cannot be processed.
    """
    indexes: dict[str, DocumentIndex] = {}
    correct = wrong = rejected = answerable = unanswerable = 0

    for item in questions:
        name = item["document"]
        if name not in indexes:
            document = process_document(Path(documents_folder) / name)
            indexes[name] = DocumentIndex(document, min_score=min_score)

        answer = indexes[name].answer(item["question"])
        expected = item["answer"]

        if expected is None:
            unanswerable += 1
            rejected += answer is None
        else:
            answerable += 1
            if answer is not None and expected.lower() in answer.sentence.lower():
                correct += 1
            elif answer is not None:
                wrong += 1

    return EvaluationResult(correct, answerable, wrong, rejected, unanswerable)


def main(argv: list[str] | None = None) -> int:
    """
    Score Q&A from the command line.

    Args:
        argv: Command-line arguments. Defaults to those given to the program.

    Returns:
        A process exit code, where 0 means success.
    """
    parser = argparse.ArgumentParser(description="Score document Q&A.")
    parser.add_argument("questions", help="path to a questions JSON file")
    parser.add_argument(
        "--split",
        choices=["tune", "test", "all"],
        default="tune",
        help='questions to use (default: "tune")',
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_ANSWER_SCORE,
        help=f"answer threshold (default: {MIN_ANSWER_SCORE})",
    )
    parser.add_argument(
        "--documents",
        default=DOCUMENTS_FOLDER,
        help="folder containing the documents (default: data/documents)",
    )
    args = parser.parse_args(argv)

    try:
        result = evaluate(
            load_questions(args.questions, args.split),
            documents_folder=args.documents,
            min_score=args.min_score,
        )
    except (OSError, KeyError, ValueError, DocumentError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Split: {args.split}, minimum score {args.min_score}")
    print(f"Correct answers: {result.correct} of {result.answerable}")
    print(f"Wrong answers shown: {result.wrong}")
    print(
        f"Unanswerable questions given no answer: "
        f"{result.rejected} of {result.unanswerable}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
