"""Score summaries by how many of a document's key points they cover.

The key points file is JSON, normally kept in data/documents/ so it is not
published with the documents it describes. Each key point lists phrases, and
the point counts as covered if any summary sentence contains any of them,
ignoring case:

    {"report.pdf": [
        {"point": "Revenue grew", "phrases": ["revenue rose", "$42,250,000"]}
    ]}

Run this module to compare the summarization methods:

    uv run python -m nlp_classifier.evaluate_summary data/documents/summary_key_points.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from nlp_classifier.document_processor import DocumentError, process_document
from nlp_classifier.evaluate_qa import DOCUMENTS_FOLDER
from nlp_classifier.summarizer import METHODS, SummarySentence, summarize


@dataclass(frozen=True)
class Coverage:
    """
    How many key points a summary covered.

    Attributes:
        covered: The key points the summary covered.
        missed: The key points the summary did not cover.
    """

    covered: list[str]
    missed: list[str]


def load_key_points(path: str | Path) -> dict[str, list[dict]]:
    """
    Load key points for each document.

    A key point may give a single "phrase" instead of a "phrases" list.

    Args:
        path: Path to the key points file.

    Returns:
        A mapping of document file names to their key points, each with a
        "point" description and a list of "phrases".
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return {
        name: [
            {
                "point": item["point"],
                "phrases": item.get("phrases") or [item["phrase"]],
            }
            for item in points
        ]
        for name, points in data.items()
    }


def score_coverage(summary: list[SummarySentence], key_points: list[dict]) -> Coverage:
    """
    Check which key points a summary covers.

    Args:
        summary: The summary sentences.
        key_points: Key points from load_key_points.

    Returns:
        The covered and missed key points.
    """
    text = " ".join(sentence.text for sentence in summary).lower()
    covered, missed = [], []

    for item in key_points:
        found = any(phrase.lower() in text for phrase in item["phrases"])
        (covered if found else missed).append(item["point"])

    return Coverage(covered, missed)


def main(argv: list[str] | None = None) -> int:
    """
    Compare summarization methods from the command line.

    Args:
        argv: Command-line arguments. Defaults to those given to the program.

    Returns:
        A process exit code, where 0 means success.
    """
    parser = argparse.ArgumentParser(description="Compare summarization methods.")
    parser.add_argument("key_points", help="path to a key points JSON file")
    parser.add_argument(
        "--documents",
        default=DOCUMENTS_FOLDER,
        help="folder containing the documents (default: data/documents)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="print each summary and the key points it missed",
    )
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(errors="replace")

    try:
        key_points = load_key_points(args.key_points)
        documents = {
            name: process_document(Path(args.documents) / name)
            for name in key_points
        }
    except (OSError, KeyError, ValueError, DocumentError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    totals = {method: 0 for method in METHODS}
    point_count = sum(len(points) for points in key_points.values())

    for name, document in documents.items():
        print(f"\n{name}")
        for method in METHODS:
            summary = summarize(document, method)
            coverage = score_coverage(summary, key_points[name])
            totals[method] += len(coverage.covered)
            print(
                f"  {method:9} {len(coverage.covered)} of {len(key_points[name])} "
                f"key points in {len(summary)} sentences"
            )

            if args.show:
                for sentence in summary:
                    print(f"      - {sentence.text} (page {sentence.page_number})")
                for point in coverage.missed:
                    print(f"      missed: {point}")

    print(f"\nTotal across {len(documents)} documents:")
    for method in METHODS:
        print(f"  {method:9} {totals[method]} of {point_count} key points")

    return 0


if __name__ == "__main__":
    sys.exit(main())
