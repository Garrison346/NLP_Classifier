import pytest

from nlp_classifier import document_qa
from nlp_classifier.document_processor import Document, DocumentError, Page, Passage
from nlp_classifier.document_qa import DocumentIndex, main, tokenize_for_search


def make_document(*passages):
    """Create a document from (page number, sentences) pairs."""
    built = [
        Passage(" ".join(sentences), page_number, tuple(sentences))
        for page_number, sentences in passages
    ]
    pages = [Page(p.page_number, p.text) for p in built]

    return Document("notes.pdf", max(p.page_number for p in built), pages, built)


ORCA_DOCUMENT = make_document(
    (
        1,
        [
            "The orca is also called the killer whale.",
            "Its scientific name is Orcinus orca.",
            "Orcas live in every ocean on Earth.",
        ],
    ),
    (
        2,
        [
            "A thick layer of blubber keeps orcas warm in cold water.",
            "Females can live for ninety years or more.",
            "Orcas hunt seals, fish, and sometimes other whales.",
        ],
    ),
    (
        3,
        [
            "Boat noise can interfere with how orcas communicate.",
            "Pollution can build up in the bodies of top predators.",
            "People can help by following whale-watching rules.",
        ],
    ),
)


@pytest.fixture(scope="module")
def index():
    return DocumentIndex(ORCA_DOCUMENT)


# tokenize_for_search


def test_tokenize_for_search_removes_stopwords_and_stems():
    assert tokenize_for_search("What is scaling the data?") == ["scale", "data"]


def test_tokenize_for_search_matches_word_forms():
    assert tokenize_for_search("scaled") == tokenize_for_search("scaling")


# DocumentIndex.answer


def test_answer_returns_best_sentence_and_page(index):
    answer = index.answer("What keeps orcas warm?")

    assert answer.sentence == "A thick layer of blubber keeps orcas warm in cold water."
    assert answer.page_number == 2
    assert "Females can live" in answer.passage
    assert 0 < answer.score <= 1


def test_answer_finds_sentence_using_different_word_forms(index):
    answer = index.answer("How does noise from boats affect communication?")

    assert answer.sentence.startswith("Boat noise")


def test_answer_returns_none_for_unrelated_question(index):
    assert index.answer("Who won the football championship?") is None


@pytest.mark.parametrize("question", ["", "   ", "What is it?", "??"])
def test_answer_returns_none_without_searchable_words(index, question):
    assert index.answer(question) is None


def test_answer_returns_none_below_minimum_score():
    index = DocumentIndex(ORCA_DOCUMENT, min_score=1.0)

    assert index.answer("What keeps orcas warm?") is None


def test_answer_skips_short_heading_sentences():
    document = make_document(
        (1, ["Dividends.", "The board paid dividends of five dollars per share."]),
    )

    answer = DocumentIndex(document).answer("How much were the dividends?")

    assert answer.sentence == "The board paid dividends of five dollars per share."


def test_answer_uses_passage_sentences_instead_of_splitting_text_again():
    passage = Passage(
        "How people can help Support marine conservation efforts today.",
        1,
        ("How people can help", "Support marine conservation efforts today."),
    )
    document = Document("guide.pdf", 1, [Page(1, passage.text)], [passage])

    answer = DocumentIndex(document).answer("How can people support conservation?")

    assert answer.sentence == "Support marine conservation efforts today."


def test_answer_splits_passage_text_when_sentences_are_missing():
    passage = Passage("Cats sleep a lot every day. Dogs bark at the mail carrier.", 1)
    document = Document("pets.pdf", 1, [Page(1, passage.text)], [passage])

    answer = DocumentIndex(document).answer("Why do dogs bark?")

    assert answer.sentence == "Dogs bark at the mail carrier."


def test_answer_returns_none_for_document_without_searchable_words():
    document = make_document((1, ["It is what it is.", "This is that."]))

    assert DocumentIndex(document).answer("What is it?") is None


# main


def test_main_answers_questions_until_exit(monkeypatch, capsys):
    monkeypatch.setattr(document_qa, "process_document", lambda path: ORCA_DOCUMENT)
    lines = iter(["What keeps orcas warm?", "", "Who won the football championship?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))

    assert main(["notes.pdf"]) == 0

    output = capsys.readouterr().out
    assert "Answer: A thick layer of blubber keeps orcas warm" in output
    assert "[page 2, score" in output
    assert "I couldn't find that in the document." in output


def test_main_reports_document_errors(monkeypatch, capsys):
    def fail(path):
        raise DocumentError("Could not find the file: missing.pdf")

    monkeypatch.setattr(document_qa, "process_document", fail)

    assert main(["missing.pdf"]) == 1
    assert "Could not find the file" in capsys.readouterr().err
