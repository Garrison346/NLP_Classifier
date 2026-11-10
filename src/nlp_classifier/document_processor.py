"""Extract, clean, and split the text of PDF documents.

Run this module to see how a PDF is processed:

    uv run python -m nlp_classifier.document_processor path/to/file.pdf
"""

import argparse
import math
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
from pypdf import PdfReader
from pypdf.errors import DependencyError


MAX_PASSAGE_WORDS = 120
PASSAGE_OVERLAP_SENTENCES = 1

# Some PDFs, such as those from Google Docs, can extract with one word per
# line. Text is treated this way if it has at least this many blocks, and at
# least this share of them are a single word.
WORD_PER_LINE_MIN_BLOCKS = 5
WORD_PER_LINE_MIN_SHARE = 0.7

# One or more blank lines, including lines containing only spaces.
BLANK_LINE_GAP = re.compile(r"\n(?:[^\S\n]*\n)+")

# Lines this close to a page's top or bottom can be headers or footers. PDFs may
# list a header, footer, and page number together, in any order. A line counts
# as repeated if it is on at least this share of pages, and long lines are
# never treated as headers or footers.
EDGE_LINES = 3
REPEATED_LINE_MIN_PAGE_SHARE = 0.6
MAX_REPEATED_LINE_WORDS = 15

# A heading must be short compared with the page's full lines of text. This
# avoids mistaking a line of wrapped text for a heading.
MAX_HEADING_WORDS = 8
MAX_HEADING_LENGTH_SHARE = 0.6
NUMBERED_LINE = re.compile(r"\d+\.\s+\S")
TABLE_NUMBER_LINE = re.compile(r"[$(\-]*\d[\d,.]*[%)]*|-")
SENTENCE_END_CHARACTERS = ".!?:;\"”')]"

# Periods after these words usually do not end a sentence.
ABBREVIATIONS = {
    "al",  # As in "et al."
    "approx",
    "corp",
    "dept",
    "dr",
    "e.g",
    "fig",
    "figs",
    "i.e",
    "inc",
    "jr",
    "ltd",
    "mr",
    "mrs",
    "ms",
    "p",
    "pg",
    "pp",
    "prof",
    "rev",
    "sr",
    "u.k",
    "u.s",
    "vs",
}

# Single characters that PDFs often use in place of common letter pairs.
LIGATURES = str.maketrans(
    {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
)

# Build the tokenizer once. This one needs no separately downloaded NLTK data.
_parameters = PunktParameters()
_parameters.abbrev_types = ABBREVIATIONS
SENTENCE_TOKENIZER = PunktSentenceTokenizer(_parameters)


class DocumentError(Exception):
    """A document could not be processed. The message is safe to show users."""


class DocumentNotFoundError(DocumentError):
    """The file does not exist."""


class UnsupportedDocumentError(DocumentError):
    """The file is not a supported type."""


class DocumentReadError(DocumentError):
    """The file could not be read, for example because it is corrupt."""


class PasswordProtectedError(DocumentError):
    """The file requires a password to open."""


class NoExtractableTextError(DocumentError):
    """The file has no text to extract, as with most scanned PDFs."""


@dataclass(frozen=True)
class Page:
    """The cleaned text of one page. Numbers start at 1."""

    number: int
    text: str


@dataclass(frozen=True)
class Passage:
    """
    A group of consecutive sentences.

    Attributes:
        text: The passage text.
        page_number: The page where the passage starts. Its last sentence may
            finish on the next page.
        sentences: The sentences that make up the passage. Splitting the text
            again would lose the boundaries between headings and sentences.
    """

    text: str
    page_number: int
    sentences: tuple[str, ...] = ()


@dataclass(frozen=True)
class Document:
    """
    A processed document.

    Attributes:
        name: The file name.
        page_count: Pages in the file, including those without text.
        pages: Only the pages that contain text.
        passages: Text split into overlapping passages.
    """

    name: str
    page_count: int
    pages: list[Page]
    passages: list[Passage]


def clean_text(text: str) -> str:
    """
    Fix common problems in text extracted from a PDF.

    A line ending in a hyphen is rejoined when the next line starts in
    lowercase. This repairs "inter-\\nnational" but also turns "well-\\nknown"
    into "wellknown", because the two cases cannot be told apart reliably.

    Args:
        text: Raw extracted text.

    Returns:
        Text with paragraphs separated by a blank line.
    """
    text = _repair_word_per_line(_normalize_characters(text))

    # Rejoin words split across lines.
    text = re.sub(r"(?<=[^\W\d_])- ?\n ?(?=[a-zß-ÿ])", "", text)

    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [" ".join(paragraph.split()) for paragraph in paragraphs]

    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _normalize_characters(text: str) -> str:
    """Standardize characters, line endings, and spacing."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(LIGATURES)
    text = text.replace("­", "")  # Invisible soft hyphens.
    text = text.replace(" ", " ")  # Non-breaking spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
    text = re.sub(r"[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]", "", text)

    return re.sub(r"[^\S\n]+", " ", text)


def _repair_word_per_line(text: str) -> str:
    """
    Rejoin text that was extracted with one word per line.

    In this kind of text, one blank line separates words, and a larger gap
    separates paragraphs. Other text is returned unchanged.
    """
    blocks = [block for block in BLANK_LINE_GAP.split(text) if block.strip()]

    if len(blocks) < WORD_PER_LINE_MIN_BLOCKS:
        return text

    single_words = sum(len(block.split()) == 1 for block in blocks)
    if single_words / len(blocks) < WORD_PER_LINE_MIN_SHARE:
        return text

    return BLANK_LINE_GAP.sub(
        lambda gap: "\n\n" if gap.group().count("\n") >= 3 else " ",
        text,
    )


def _normalize_repeated_line(line: str) -> str:
    """Make a line comparable across pages, where only its numbers may differ."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _edge_line_indexes(lines: list[str]) -> set[int]:
    """Find the positions of the non-blank lines nearest a page's edges."""
    indexes = [index for index, line in enumerate(lines) if line.strip()]

    return set(indexes[:EDGE_LINES] + indexes[-EDGE_LINES:])


def _find_repeated_lines(edge_lines: list[set[str]], min_share: float) -> set[str]:
    """Find lines that appear on at least a share of pages, and at least twice."""
    counts = Counter(line for lines in edge_lines for line in lines)
    min_pages = max(2, math.ceil(len(edge_lines) * min_share))

    return {line for line, count in counts.items() if count >= min_pages}


def _remove_repeated_lines(pages: list[str]) -> list[str]:
    """
    Remove headers, footers, and page numbers that repeat across pages.

    Only short lines near the top or bottom of a page can be removed, and only
    when the same line, ignoring numbers, is near an edge of most pages. Odd and
    even pages are also checked separately, because they can have different
    headers. Word-per-line text must be repaired first, or its repeated words
    would look like headers.

    The first page must match more strictly, because a cover page's title often
    matches the header on later pages. A page with nothing but repeated lines
    becomes empty, unless every page would, which suggests they are real text.

    Args:
        pages: The text of each page, in order.

    Returns:
        The text of each page without its repeated lines.
    """
    if len(pages) < 2:
        return pages

    page_lines = [page.split("\n") for page in pages]
    edge_lines = [
        {
            _normalize_repeated_line(lines[index])
            for index in _edge_line_indexes(lines)
            if len(lines[index].split()) <= MAX_REPEATED_LINE_WORDS
        }
        for lines in page_lines
    ]
    odd_pages, even_pages = edge_lines[0::2], edge_lines[1::2]
    share = REPEATED_LINE_MIN_PAGE_SHARE

    repeated = (
        _find_repeated_lines(edge_lines, share)
        | _find_repeated_lines(odd_pages, share)
        | _find_repeated_lines(even_pages, share)
    )
    repeated_on_first_page = (
        _find_repeated_lines(edge_lines, 1.0) | _find_repeated_lines(odd_pages, 1.0)
    )

    result = []
    for page_index, lines in enumerate(page_lines):
        edges = _edge_line_indexes(lines)
        removable = repeated_on_first_page if page_index == 0 else repeated
        kept = [
            line
            for index, line in enumerate(lines)
            if index not in edges or _normalize_repeated_line(line) not in removable
        ]
        result.append("\n".join(kept))

    if not any(page.strip() for page in result):
        return pages

    return result


def _is_heading(line: str, previous: str, following: str, longest: int) -> bool:
    """Check whether a line looks like a heading, given the lines around it."""
    words = line.split()

    if not following or len(words) > MAX_HEADING_WORDS:
        return False

    # Only a finished sentence or a table row can come before a heading.
    if (
        previous
        and previous[-1] not in SENTENCE_END_CHARACTERS
        and not TABLE_NUMBER_LINE.fullmatch(previous)
    ):
        return False

    if NUMBERED_LINE.match(line):
        return following[0].isupper() or following[0].isdigit()

    # A plain number on the next line suggests a table row, not a heading.
    return (
        (following[0].isupper() or NUMBERED_LINE.match(following) is not None)
        and line[0].isupper()
        and line[-1] not in ".!?,:;"
        and len(line) <= longest * MAX_HEADING_LENGTH_SHARE
    )


def _separate_headings(text: str) -> str:
    """
    Put likely headings in their own paragraphs.

    This is a best guess based on layout, so it can miss a heading or wrongly
    separate a short line.

    Args:
        text: Text that still has its original line breaks.

    Returns:
        Text with a blank line before and after each likely heading.
    """
    lines = [line.strip() for line in text.split("\n")]
    longest = max((len(line) for line in lines), default=0)
    result: list[str] = []
    previous = ""

    for index, line in enumerate(lines):
        if not line:
            result.append("")
            previous = ""
            continue

        following = next((later for later in lines[index + 1:] if later), "")
        at_paragraph_end = index + 1 < len(lines) and not lines[index + 1]

        if not at_paragraph_end and _is_heading(line, previous, following, longest):
            result.extend(["", line, ""])
            previous = ""
            continue

        # A full line of text after a table row starts a new paragraph.
        if (
            TABLE_NUMBER_LINE.fullmatch(previous)
            and len(line) > longest * MAX_HEADING_LENGTH_SHARE
        ):
            result.append("")

        result.append(line)
        previous = line

    return "\n".join(result)


def split_sentences(text: str) -> list[str]:
    """
    Split cleaned text into sentences.

    Paragraphs are split separately so that a heading without punctuation
    does not merge with the sentence after it. A number such as "1." stays
    with the text that follows it, and "No." stays with a number after it.

    Args:
        text: Cleaned text.

    Returns:
        A list of sentences.
    """
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph_sentences: list[str] = []
        pending_number = ""

        for sentence in SENTENCE_TOKENIZER.tokenize(paragraph):
            if re.fullmatch(r"\d+\.", sentence):
                pending_number = f"{pending_number}{sentence} "
            elif (
                paragraph_sentences
                and sentence[0].isdigit()
                and re.search(r"\bNo\.$", paragraph_sentences[-1])
            ):
                paragraph_sentences[-1] += f" {sentence}"
            else:
                paragraph_sentences.append(pending_number + sentence)
                pending_number = ""

        if pending_number:
            paragraph_sentences.append(pending_number.strip())

        sentences.extend(paragraph_sentences)

    return sentences


def split_passages(
    sentences: Iterable[str],
    max_words: int = MAX_PASSAGE_WORDS,
    overlap_sentences: int = PASSAGE_OVERLAP_SENTENCES,
) -> list[str]:
    """
    Group sentences into passages of limited length.

    Consecutive passages share sentences so that an idea spanning a boundary
    still appears whole in one passage.

    Args:
        sentences: Sentences in reading order.
        max_words: Maximum number of words in a passage.
        overlap_sentences: Sentences to repeat at the start of a new passage.

    Returns:
        A list of passages.

    Raises:
        ValueError: If max_words is not positive or the overlap is negative.
    """
    groups = _group_sentences(sentences, max_words, overlap_sentences)

    return [" ".join(group) for group in groups]


def _group_sentences(
    sentences: Iterable[str],
    max_words: int,
    overlap_sentences: int,
) -> list[list[str]]:
    """Group sentences as split_passages does, keeping each sentence separate."""
    if max_words < 1:
        raise ValueError("max_words must be at least 1.")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative.")

    groups = []
    current: list[str] = []

    for sentence in _limit_sentence_lengths(sentences, max_words):
        if current and _word_count(current + [sentence]) > max_words:
            groups.append(current)
            current = current[-overlap_sentences:] if overlap_sentences else []

            # Drop the overlap if it leaves no room for the next sentence.
            if _word_count(current + [sentence]) > max_words:
                current = []

        current = current + [sentence]

    if current:
        groups.append(current)

    return groups


def _word_count(sentences: list[str]) -> int:
    """Count the words in a list of sentences."""
    return sum(len(sentence.split()) for sentence in sentences)


def _limit_sentence_lengths(
    sentences: Iterable[str],
    max_words: int,
) -> Iterable[str]:
    """Break up overly long sentences, which are often tables or lists."""
    for sentence in sentences:
        words = sentence.split()
        for start in range(0, len(words), max_words):
            yield " ".join(words[start:start + max_words])


def extract_pages(path: str | Path) -> tuple[list[Page], int]:
    """
    Extract cleaned text from each page of a PDF.

    Args:
        path: Path to the PDF.

    Returns:
        The pages containing text, and the file's total page count.

    Raises:
        DocumentNotFoundError: If the file does not exist.
        UnsupportedDocumentError: If the file is not a PDF.
        PasswordProtectedError: If the file requires a password.
        DocumentReadError: If the file cannot be read.
        NoExtractableTextError: If no page contains text.
    """
    path = Path(path)

    if not path.is_file():
        raise DocumentNotFoundError(f"Could not find the file: {path}")

    if path.suffix.lower() != ".pdf":
        raise UnsupportedDocumentError(
            f"Only PDF files are supported, but got: {path.name}"
        )

    try:
        reader = PdfReader(path)

        # Many protected PDFs open without a password, using an empty one.
        if reader.is_encrypted and not reader.decrypt(""):
            raise PasswordProtectedError(f"{path.name} requires a password.")

        raw_pages = [page.extract_text() for page in reader.pages]
    except DocumentError:
        raise
    except DependencyError as error:
        raise DocumentReadError(
            f"{path.name} uses encryption that this program cannot open."
        ) from error
    except Exception as error:
        # Damaged PDFs can raise many exception types from inside pypdf.
        raise DocumentReadError(
            f"{path.name} could not be read. It may be damaged."
        ) from error

    prepared_pages = _remove_repeated_lines(
        [_repair_word_per_line(_normalize_characters(raw)) for raw in raw_pages]
    )

    pages = []
    for number, prepared in enumerate(prepared_pages, start=1):
        text = clean_text(_separate_headings(prepared))
        if text:
            pages.append(Page(number, text))

    if not pages:
        raise NoExtractableTextError(
            f"{path.name} has no text to extract. It may be a scanned document."
        )

    return pages, len(raw_pages)


def _continues_sentence(previous: str, following: str) -> bool:
    """Check whether a sentence appears to continue into the following text."""
    return previous[-1] not in SENTENCE_END_CHARACTERS and following[0].islower()


def _split_page_sentences(pages: list[Page]) -> list[tuple[int, list[str]]]:
    """
    Split pages into sentences, rejoining those divided by a page break.

    A rejoined sentence stays with the page where it starts.

    Args:
        pages: Pages containing text.

    Returns:
        Page numbers and their sentences, leaving out any page whose only
        sentence moved to the previous page.
    """
    result: list[tuple[int, list[str]]] = []
    previous_number = None

    for page in pages:
        sentences = split_sentences(page.text)

        if (
            result
            and page.number == previous_number + 1
            and _continues_sentence(result[-1][1][-1], sentences[0])
        ):
            start = result[-1][1][-1]
            separator = "" if start.endswith("-") else " "
            result[-1][1][-1] = start.removesuffix("-") + separator + sentences.pop(0)

        previous_number = page.number

        if sentences:
            result.append((page.number, sentences))

    return result


def process_document(
    path: str | Path,
    max_words: int = MAX_PASSAGE_WORDS,
    overlap_sentences: int = PASSAGE_OVERLAP_SENTENCES,
) -> Document:
    """
    Extract a PDF's text and split it into passages.

    Passages are built separately for each page so that each one has a single
    starting page number. Only a sentence divided by a page break crosses over.

    Args:
        path: Path to the PDF.
        max_words: Maximum number of words in a passage.
        overlap_sentences: Sentences repeated between consecutive passages.

    Returns:
        The processed document.

    Raises:
        DocumentError: If the document cannot be processed.
    """
    pages, page_count = extract_pages(path)

    passages = [
        Passage(" ".join(group), page_number, tuple(group))
        for page_number, sentences in _split_page_sentences(pages)
        for group in _group_sentences(sentences, max_words, overlap_sentences)
    ]

    return Document(Path(path).name, page_count, pages, passages)


def main(argv: list[str] | None = None) -> int:
    """
    Show how a PDF is processed.

    Args:
        argv: Command-line arguments. Defaults to those given to the program.

    Returns:
        A process exit code, where 0 means success.
    """
    parser = argparse.ArgumentParser(description="Show how a PDF is processed.")
    parser.add_argument("path", help="path to a PDF file")
    parser.add_argument(
        "--show",
        type=int,
        default=3,
        help="number of passages to display (default: 3)",
    )
    args = parser.parse_args(argv)

    # PDFs can contain characters that some terminals cannot display.
    sys.stdout.reconfigure(errors="replace")

    try:
        document = process_document(args.path)
    except DocumentError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"File: {document.name}")
    print(f"Pages with text: {len(document.pages)} of {document.page_count}")
    print(f"Passages: {len(document.passages)}")

    for passage in document.passages[:args.show]:
        print(f"\n[Page {passage.page_number}] {passage.text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
