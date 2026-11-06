import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.errors import DependencyError

from nlp_classifier import document_processor
from nlp_classifier.document_processor import (
    DocumentNotFoundError,
    DocumentReadError,
    NoExtractableTextError,
    PasswordProtectedError,
    UnsupportedDocumentError,
    _remove_repeated_lines,
    _separate_headings,
    clean_text,
    extract_pages,
    main,
    process_document,
    split_passages,
    split_sentences,
)


def write_pdf(path, pages):
    """Write a simple PDF where each page is a list of text lines."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        None,  # The page tree is added once the page numbers are known.
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_ids = []

    for lines in pages:
        escaped = [
            line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            for line in lines
        ]
        commands = " T* ".join(f"({line}) Tj" for line in escaped)
        stream = f"BT /F1 12 Tf 72 720 Td 14 TL {commands} ET".encode("latin-1")
        if not lines:
            stream = b""

        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        content_id = len(objects)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % content_id
        )
        page_ids.append(len(objects))

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids).encode()
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids))

    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += b"%d 0 obj\n%s\nendobj\n" % (number, body)

    xref_offset = len(output)
    output += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        output += b"%010d 00000 n \n" % offset
    output += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    output += b"startxref\n%d\n%%%%EOF\n" % xref_offset

    path.write_bytes(bytes(output))
    return path


def write_encrypted_pdf(path, source, password):
    """Write an encrypted copy of a PDF."""
    writer = PdfWriter(clone_from=PdfReader(source))
    writer.encrypt(user_password=password, algorithm="RC4-128")
    writer.write(path)
    return path


@pytest.fixture
def pdf(tmp_path):
    return write_pdf(
        tmp_path / "report.pdf",
        [
            ["The first page has a sentence. It also has another one."],
            [],
            ["Revenue grew in the third quarter."],
        ],
    )


# clean_text


def test_clean_text_rejoins_word_split_across_lines():
    assert clean_text("an inter-\nnational deal") == "an international deal"


def test_clean_text_rejoins_word_when_hyphen_is_followed_by_space():
    assert clean_text("an inter- \nnational deal") == "an international deal"


def test_clean_text_keeps_hyphen_before_capitalized_line():
    assert clean_text("North-\nAmerica") == "North- America"


def test_clean_text_keeps_hyphen_within_a_line():
    assert clean_text("a well-known fact") == "a well-known fact"


def test_clean_text_joins_lines_within_a_paragraph():
    assert clean_text("one line\nnext line") == "one line next line"


def test_clean_text_keeps_paragraph_breaks():
    assert clean_text("first\n \n\nsecond") == "first\n\nsecond"


def test_clean_text_replaces_ligatures():
    assert clean_text("ﬁnal ﬂow oﬃce") == "final flow office"


def test_clean_text_removes_soft_hyphens_and_nonbreaking_spaces():
    assert clean_text("co­operate now") == "cooperate now"


def test_clean_text_removes_control_characters():
    assert clean_text("\x7f Prey\x00 shortages\x9f here") == "Prey shortages here"


def test_clean_text_normalizes_line_endings_and_page_breaks():
    assert clean_text("one\r\ntwo\fthree") == "one two\n\nthree"


def word_per_line(*paragraphs):
    """Format text the way pypdf can extract Google Docs PDFs."""
    formatted = ["\n \n".join(paragraph.split()) for paragraph in paragraphs]
    return " \n" + "\n \n \n".join(formatted) + "\n \n"


def test_clean_text_rejoins_word_per_line_text():
    text = word_per_line("The quick brown fox jumps over the dog.")

    assert clean_text(text) == "The quick brown fox jumps over the dog."


def test_clean_text_keeps_paragraphs_in_word_per_line_text():
    text = word_per_line("A Short Title Here", "The first paragraph starts here.")

    assert clean_text(text) == "A Short Title Here\n\nThe first paragraph starts here."


def test_clean_text_keeps_heading_in_normal_text():
    text = "Short\n\nHeading\n\nA normal paragraph of many words sits here."

    assert clean_text(text) == text


def test_clean_text_keeps_paragraphs_when_few_blocks_are_single_words():
    text = "\n\n".join(
        [
            "Heading",
            "A paragraph of several words.",
            "Another paragraph of several words.",
            "Conclusion",
            "A final paragraph of several words.",
        ]
    )

    assert clean_text(text) == text


def test_clean_text_needs_five_blocks_to_detect_word_per_line_text():
    assert clean_text(word_per_line("Thanks for reading this.")) == (
        "Thanks\n\nfor\n\nreading\n\nthis."
    )
    assert clean_text(word_per_line("Thanks for reading this now.")) == (
        "Thanks for reading this now."
    )


@pytest.mark.parametrize("text", ["", "  \n\t\n "])
def test_clean_text_handles_text_without_content(text):
    assert clean_text(text) == ""


# _remove_repeated_lines


def test_remove_repeated_lines_removes_header_and_page_number():
    pages = [
        "THE GUIDE\n1\nFirst body line.",
        "THE GUIDE\n2\nSecond body line.",
        "THE GUIDE\n3\nThird body line.",
    ]

    assert _remove_repeated_lines(pages) == [
        "First body line.",
        "Second body line.",
        "Third body line.",
    ]


def test_remove_repeated_lines_removes_numbered_footer():
    pages = [
        "Body line one.\nMore text.\nPage 1 of 2",
        "Body line two.\nOther text.\nPage 2 of 2",
    ]

    assert _remove_repeated_lines(pages) == [
        "Body line one.\nMore text.",
        "Body line two.\nOther text.",
    ]


def test_remove_repeated_lines_keeps_header_missing_from_most_pages():
    pages = [f"Body {word} line.\nMore {word}.\nAnd {word}.\nEnd {word}.\nDone {word}."
             for word in ["one", "two", "three", "four"]]
    pages[0] = "Rare line\n" + pages[0]

    assert _remove_repeated_lines(pages) == pages


def test_remove_repeated_lines_keeps_repeated_line_away_from_page_edges():
    pages = [
        f"Top {word}.\nNext {word}.\nA repeated line.\nLast {word}.\nEnd {word}."
        for word in ["one", "two", "three"]
    ]

    assert _remove_repeated_lines(pages) == pages


def test_remove_repeated_lines_keeps_line_on_a_single_page_document():
    assert _remove_repeated_lines(["Title\nBody."]) == ["Title\nBody."]


def test_remove_repeated_lines_keeps_page_that_would_be_emptied():
    pages = ["The same text.", "The same text.", "The same text."]

    assert _remove_repeated_lines(pages) == pages


def test_extract_pages_keeps_word_per_line_words_that_repeat_across_pages(tmp_path):
    texts = [
        "Civil rights were discussed here today.",
        "Civil rights were debated there yesterday.",
        "Civil rights remain important to everyone.",
    ]
    pages_of_lines = [
        [line for word in text.split() for line in (word, " ")]
        for text in texts
    ]
    path = write_pdf(tmp_path / "docs.pdf", pages_of_lines)

    pages, _ = extract_pages(path)

    assert [page.text for page in pages] == texts


def test_extract_pages_removes_repeated_header_from_pdf(tmp_path):
    path = write_pdf(
        tmp_path / "header.pdf",
        [
            ["THE GUIDE | NOTES", "1", "The first page has text."],
            ["THE GUIDE | NOTES", "2", "The second page has text."],
        ],
    )

    pages, _ = extract_pages(path)

    assert [page.text for page in pages] == [
        "The first page has text.",
        "The second page has text.",
    ]


# _separate_headings

WRAPPED_LINE = "This line of body text is long enough to fill the page width and"


def test_separate_headings_separates_short_line_after_finished_sentence():
    text = f"{WRAPPED_LINE}\nfinishes here.\nAppearance and adaptations\nOrcas have fins."

    assert _separate_headings(text).split("\n") == [
        WRAPPED_LINE,
        "finishes here.",
        "",
        "Appearance and adaptations",
        "",
        "Orcas have fins.",
    ]


def test_separate_headings_separates_numbered_heading_ending_in_question_mark():
    text = f"{WRAPPED_LINE}\nfinishes here.\n1. What is an orca?\nThe orca swims."

    assert "\n\n1. What is an orca?\n\n" in _separate_headings(text)


def test_separate_headings_keeps_wrapped_line_after_unfinished_sentence():
    text = f"{WRAPPED_LINE}\nShort line here\nBob finishes it."

    assert "\n\n" not in _separate_headings(text)


def test_separate_headings_keeps_full_width_line_after_finished_sentence():
    text = f"Done.\n{WRAPPED_LINE}\nBob finishes it."

    assert "\n\n" not in _separate_headings(text)


def test_separate_headings_keeps_line_followed_by_lowercase():
    text = f"{WRAPPED_LINE} done.\nShort line here\nand it continues."

    assert "\n\n" not in _separate_headings(text)


def test_separate_headings_keeps_last_line_of_page():
    text = f"{WRAPPED_LINE} done.\nDespite this being a peaceful"

    assert "\n\n" not in _separate_headings(text)


def test_separate_headings_keeps_short_paragraph_after_blank_line():
    text = f"{WRAPPED_LINE} done.\n\nCivil Rights Movement\n\nThe text starts here."

    assert _separate_headings(text) == text


def test_extract_pages_separates_heading_in_pdf(tmp_path):
    path = write_pdf(
        tmp_path / "heading.pdf",
        [[WRAPPED_LINE, "finishes here.", "Appearance and adaptations", "Orcas have fins."]],
    )

    pages, _ = extract_pages(path)

    assert "\n\nAppearance and adaptations\n\nOrcas have fins." in pages[0].text


# split_sentences


def test_split_sentences_keeps_section_number_with_its_heading():
    assert split_sentences("1. What is an orca?\n\nThe orca swims.") == [
        "1. What is an orca?",
        "The orca swims.",
    ]


def test_split_sentences_keeps_number_at_end_of_text():
    assert split_sentences("See item 4.\n\n5.") == ["See item 4.", "5."]


def test_split_sentences_splits_on_end_punctuation():
    assert split_sentences("One here. Two here! Three here?") == [
        "One here.",
        "Two here!",
        "Three here?",
    ]


@pytest.mark.parametrize(
    "sentence",
    [
        "Dr. Smith wrote the report.",
        "Revenue rose 2.5% this year.",
        "The price was $3.50 per unit.",
        "See Fig. 3 for details.",
        "The U.S. Senate voted.",
        "Smith et al. found a link.",
        "Martin Luther King Jr. led the march.",
        "It was organized by Rev. Hosea Williams.",
        "The quote appears on (pg. 705) of the book.",
    ],
)
def test_split_sentences_does_not_split_inside_a_sentence(sentence):
    assert split_sentences(sentence) == [sentence]


def test_split_sentences_separates_heading_from_next_paragraph():
    assert split_sentences("Introduction\n\nThis is the text.") == [
        "Introduction",
        "This is the text.",
    ]


def test_split_sentences_handles_empty_text():
    assert split_sentences("") == []


# split_passages


def test_split_passages_groups_sentences_up_to_limit():
    result = split_passages(["a b", "c d", "e f"], max_words=4, overlap_sentences=0)

    assert result == ["a b c d", "e f"]


def test_split_passages_repeats_overlap_sentences():
    result = split_passages(["a b", "c d", "e f"], max_words=4, overlap_sentences=1)

    assert result == ["a b c d", "c d e f"]


def test_split_passages_drops_overlap_when_next_sentence_would_not_fit():
    result = split_passages(["a b", "c d e f"], max_words=4, overlap_sentences=1)

    assert result == ["a b", "c d e f"]


def test_split_passages_breaks_up_overly_long_sentence():
    result = split_passages(["a b c d e"], max_words=2, overlap_sentences=0)

    assert result == ["a b", "c d", "e"]


def test_split_passages_keeps_every_word_and_respects_limit():
    sentences = [f"s{i} " + "word " * (i % 9 + 1) for i in range(100)]

    result = split_passages(sentences, max_words=20, overlap_sentences=1)

    assert all(len(passage.split()) <= 20 for passage in result)
    for i in range(100):
        assert any(f"s{i} " in passage for passage in result)


def test_split_passages_handles_no_sentences():
    assert split_passages([]) == []


@pytest.mark.parametrize(
    "kwargs",
    [{"max_words": 0}, {"overlap_sentences": -1}],
)
def test_split_passages_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        split_passages(["a b"], **kwargs)


# extract_pages


def test_extract_pages_keeps_real_page_numbers_and_skips_empty_pages(pdf):
    pages, page_count = extract_pages(pdf)

    assert page_count == 3
    assert [page.number for page in pages] == [1, 3]
    assert "Revenue grew" in pages[1].text


def test_extract_pages_accepts_string_path_and_uppercase_extension(tmp_path):
    path = write_pdf(tmp_path / "REPORT.PDF", [["Some text here."]])

    pages, _ = extract_pages(str(path))

    assert pages[0].text == "Some text here."


def test_extract_pages_rejoins_word_split_across_lines_in_a_real_pdf(tmp_path):
    path = write_pdf(tmp_path / "split.pdf", [["An inter-", "national deal."]])

    pages, _ = extract_pages(path)

    assert pages[0].text == "An international deal."


def test_extract_pages_raises_for_missing_file(tmp_path):
    with pytest.raises(DocumentNotFoundError):
        extract_pages(tmp_path / "missing.pdf")


def test_extract_pages_raises_for_directory(tmp_path):
    with pytest.raises(DocumentNotFoundError):
        extract_pages(tmp_path)


def test_extract_pages_raises_for_other_file_types(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Some text", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentError):
        extract_pages(path)


@pytest.mark.parametrize("content", [b"", b"this is not a pdf"])
def test_extract_pages_raises_for_unreadable_file(tmp_path, content):
    path = tmp_path / "broken.pdf"
    path.write_bytes(content)

    with pytest.raises(DocumentReadError):
        extract_pages(path)


def test_extract_pages_raises_when_no_page_has_text(tmp_path):
    path = write_pdf(tmp_path / "scanned.pdf", [[], []])

    with pytest.raises(NoExtractableTextError):
        extract_pages(path)


def test_extract_pages_raises_for_password_protected_file(tmp_path, pdf):
    path = write_encrypted_pdf(tmp_path / "locked.pdf", pdf, "secret")

    with pytest.raises(PasswordProtectedError):
        extract_pages(path)


def test_extract_pages_opens_file_encrypted_with_empty_password(tmp_path, pdf):
    path = write_encrypted_pdf(tmp_path / "open.pdf", pdf, "")

    pages, _ = extract_pages(path)

    assert "Revenue grew" in pages[1].text


def test_extract_pages_explains_unsupported_encryption(monkeypatch, pdf):
    def raise_dependency_error(path):
        raise DependencyError("cryptography is required")

    monkeypatch.setattr(document_processor, "PdfReader", raise_dependency_error)

    with pytest.raises(DocumentReadError, match="encryption"):
        extract_pages(pdf)


# process_document


def test_process_document_returns_passages_with_page_numbers(pdf):
    document = process_document(pdf)

    assert document.name == "report.pdf"
    assert document.page_count == 3
    assert document.passages[0].page_number == 1
    assert document.passages[0].text == (
        "The first page has a sentence. It also has another one."
    )
    assert document.passages[1].page_number == 3


def test_process_document_does_not_mix_pages_in_a_passage(tmp_path):
    path = write_pdf(tmp_path / "two.pdf", [["Page one text."], ["Page two text."]])

    document = process_document(path)

    assert [passage.text for passage in document.passages] == [
        "Page one text.",
        "Page two text.",
    ]


def test_process_document_rejoins_sentence_divided_by_page_break(tmp_path):
    path = write_pdf(
        tmp_path / "split.pdf",
        [
            ["The first sentence. This was a peaceful act of"],
            ["protest in the city. The next sentence."],
        ],
    )

    document = process_document(path)

    assert [(p.page_number, p.text) for p in document.passages] == [
        (1, "The first sentence. This was a peaceful act of protest in the city."),
        (2, "The next sentence."),
    ]


def test_process_document_rejoins_word_divided_by_page_break(tmp_path):
    path = write_pdf(tmp_path / "word.pdf", [["An inter-"], ["national deal."]])

    document = process_document(path)

    assert [p.text for p in document.passages] == ["An international deal."]


def test_process_document_handles_sentence_spanning_three_pages(tmp_path):
    path = write_pdf(
        tmp_path / "three.pdf",
        [["The sentence starts"], ["and continues"], ["and ends here."]],
    )

    document = process_document(path)

    assert [(p.page_number, p.text) for p in document.passages] == [
        (1, "The sentence starts and continues and ends here."),
    ]


@pytest.mark.parametrize(
    "pages",
    [
        [["The first page ends."], ["then a lowercase start."]],
        [["A heading without punctuation"], ["The Next Page Is Capitalized."]],
        [["No ending here"], [], ["after a blank page."]],
    ],
)
def test_process_document_keeps_separate_text_on_separate_pages(tmp_path, pages):
    path = write_pdf(tmp_path / "separate.pdf", pages)

    document = process_document(path)

    assert len(document.passages) == 2
    assert document.passages[0].text == pages[0][0]


def test_process_document_handles_word_per_line_pdf(tmp_path):
    words = "This sentence is one. This sentence is two.".split()
    lines = [line for word in words for line in (word, " ")]
    path = write_pdf(tmp_path / "docs.pdf", [lines])

    document = process_document(path, overlap_sentences=0)

    assert document.passages[0].text == "This sentence is one. This sentence is two."


# main


def test_main_prints_summary(pdf, capsys):
    assert main([str(pdf)]) == 0

    output = capsys.readouterr().out
    assert "Pages with text: 2 of 3" in output
    assert "[Page 3] Revenue grew" in output


def test_main_limits_displayed_passages(pdf, capsys):
    assert main([str(pdf), "--show", "1"]) == 0

    assert "Revenue grew" not in capsys.readouterr().out


def test_main_reports_errors(tmp_path, capsys):
    assert main([str(tmp_path / "missing.pdf")]) == 1

    assert "Could not find the file" in capsys.readouterr().err
