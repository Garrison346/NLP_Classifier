import numpy as np
import pytest

from nlp_classifier import summarizer
from nlp_classifier.document_processor import Document, DocumentError, Page, Sentence
from nlp_classifier.summarizer import (
    MAX_SUMMARY_SENTENCES,
    MIN_SUMMARY_SENTENCES,
    candidate_sentences,
    main,
    score_lead,
    score_textrank,
    score_tfidf,
    select_diverse,
    summarize,
    summary_length,
)


def make_document(*sentences):
    """Create a document from (page number, sentence) pairs."""
    built = [Sentence(text, page) for page, text in sentences]
    pages = [
        Page(page, " ".join(s.text for s in built if s.page_number == page))
        for page in sorted({s.page_number for s in built})
    ]

    return Document("guide.pdf", max(s.page_number for s in built), pages, [], built)


# Three sentences share a topic, and three unrelated ones share no words.
ORCA_DOCUMENT = make_document(
    (1, "Orcas are highly social dolphins that live in close family groups."),
    (1, "The weather in the harbor was cloudy that morning."),
    (1, "Orca family groups share hunting skills between generations of dolphins."),
    (2, "A new engine was installed during the previous winter."),
    (2, "Social dolphins such as orcas learn hunting skills from their family groups."),
    (2, "Lunch for everyone was served at noon upstairs."),
)


# summary_length


@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, 1), (2, 2), (10, 3), (34, 3), (45, 4), (52, 5), (60, 6), (500, 6)],
)
def test_summary_length_scales_with_document_size(count, expected):
    assert summary_length(count) == expected


def test_summary_length_stays_within_limits():
    lengths = {summary_length(count) for count in range(3, 1000)}

    assert min(lengths) == MIN_SUMMARY_SENTENCES
    assert max(lengths) == MAX_SUMMARY_SENTENCES


# candidate_sentences


def test_candidate_sentences_skip_short_tables_citations_and_repeats():
    document = make_document(
        (1, "Dividends."),
        (1, "Wholesale revenue 31,480 28,960 Retail revenue 10,770 9,740"),
        (1, "Read more about orca biology at fisheries.noaa.gov/species/killer-whale today."),
        (1, "Orcas are highly social dolphins that live in close family groups."),
        (2, "Orcas are highly social dolphins that live in close family groups."),
        (2, "Revenue rose 9.2% to $42,250,000 compared with the year before."),
    )

    assert [s.text for s in candidate_sentences(document)] == [
        "Orcas are highly social dolphins that live in close family groups.",
        "Revenue rose 9.2% to $42,250,000 compared with the year before.",
    ]


def test_candidate_sentences_split_pages_when_sentence_list_is_missing():
    page = Page(1, "Orcas are highly social dolphins in family groups. Too short.")
    document = Document("guide.pdf", 1, [page], [])

    assert [s.text for s in candidate_sentences(document)] == [
        "Orcas are highly social dolphins in family groups.",
    ]


# Scoring methods


def test_score_lead_favors_earlier_sentences():
    scores = score_lead(["a", "b", "c"])

    assert scores[0] > scores[1] > scores[2]


def test_score_tfidf_favors_sentences_with_the_documents_important_words():
    texts = [s.text for s in ORCA_DOCUMENT.sentences]

    scores = score_tfidf(texts)

    on_topic = [scores[0], scores[2], scores[4]]
    off_topic = [scores[1], scores[3], scores[5]]
    assert min(on_topic) > max(off_topic)


def test_score_textrank_favors_sentences_similar_to_many_others():
    texts = [s.text for s in ORCA_DOCUMENT.sentences]

    scores = score_textrank(texts)

    assert scores.sum() == pytest.approx(1)
    on_topic = [scores[0], scores[2], scores[4]]
    off_topic = [scores[1], scores[3], scores[5]]
    assert min(on_topic) > max(off_topic)


def test_score_textrank_gives_equal_scores_to_unrelated_sentences():
    scores = score_textrank(["Orcas swim fast.", "Taxes rose sharply.", "Engines need oil."])

    assert np.allclose(scores, 1 / 3)


# summarize


@pytest.mark.parametrize("method", ["tfidf", "textrank"])
def test_summarize_picks_central_sentences_in_reading_order(method):
    summary = summarize(ORCA_DOCUMENT, method, max_sentences=3)

    assert [s.text for s in summary] == [
        ORCA_DOCUMENT.sentences[0].text,
        ORCA_DOCUMENT.sentences[2].text,
        ORCA_DOCUMENT.sentences[4].text,
    ]
    assert [s.page_number for s in summary] == [1, 1, 2]


def test_summarize_lead_takes_the_first_sentences():
    summary = summarize(ORCA_DOCUMENT, "lead", max_sentences=2)

    assert [s.text for s in summary] == [s.text for s in ORCA_DOCUMENT.sentences[:2]]


def test_summarize_uses_default_length():
    assert len(summarize(ORCA_DOCUMENT)) == MIN_SUMMARY_SENTENCES


def test_summarize_returns_empty_list_without_candidates():
    document = make_document((1, "Too short."), (1, "Also short."))

    assert summarize(document) == []


def test_summarize_falls_back_to_lead_without_searchable_words():
    document = make_document(
        (1, "It is what it is and that is that."),
        (1, "This is what it was and what it will be."),
    )

    summary = summarize(document, "textrank", max_sentences=1)

    assert [s.text for s in summary] == ["It is what it is and that is that."]


# select_diverse

# Sentences 0 and 1 repeat each other. The others are unrelated.
SIMILARITY = np.array(
    [
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.1],
        [0.1, 0.1, 0.1, 1.0],
    ]
)
SCORES = np.array([1.0, 0.9, 0.8, 0.0])


def test_select_diverse_without_penalty_takes_highest_scores():
    assert select_diverse(SCORES, SIMILARITY, 2, penalty=0) == [0, 1]


def test_select_diverse_skips_sentence_that_repeats_a_chosen_one():
    # Sentence 1: 0.7 * 0.9 - 0.3 * 0.9 = 0.36. Sentence 2: 0.7 * 0.8 - 0.3 * 0.1 = 0.53.
    assert select_diverse(SCORES, SIMILARITY, 2, penalty=0.3) == [0, 2]


def test_select_diverse_returns_all_when_count_is_large():
    chosen = select_diverse(SCORES, SIMILARITY, 10, penalty=0.3)

    assert sorted(chosen) == [0, 1, 2, 3]


def test_select_diverse_prefers_unrelated_sentence_when_scores_are_equal():
    chosen = select_diverse(np.full(4, 0.5), SIMILARITY, 2, penalty=0.3)

    assert chosen == [0, 2]


def test_summarize_avoids_repeated_sentences():
    document = make_document(
        (1, "Orcas are highly social dolphins that live in close family groups."),
        (1, "Orcas are highly social dolphins living in close family groups."),
        (2, "Orca family groups share hunting skills between generations."),
        (2, "Social orcas hunt seals together and share hunting skills."),
        (2, "The weather in the harbor was cloudy that morning."),
    )

    repeated = summarize(document, "textrank", max_sentences=2, redundancy_penalty=0)
    varied = summarize(document, "textrank", max_sentences=2, redundancy_penalty=0.3)

    assert [s.text for s in repeated] == [s.text for s in document.sentences[:2]]
    assert [s.text for s in varied] == [
        document.sentences[0].text,
        document.sentences[2].text,
    ]


def test_summarize_lead_ignores_redundancy_penalty():
    document = make_document(
        (1, "Orcas are highly social dolphins that live in close family groups."),
        (1, "Orcas are highly social dolphins living in close family groups."),
        (2, "Orca family groups share hunting skills between generations."),
    )

    summary = summarize(document, "lead", max_sentences=2, redundancy_penalty=0.9)

    assert [s.text for s in summary] == [s.text for s in document.sentences[:2]]


def test_summarize_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        summarize(ORCA_DOCUMENT, "magic")


# main


def test_main_prints_summary(monkeypatch, capsys):
    monkeypatch.setattr(summarizer, "process_document", lambda path: ORCA_DOCUMENT)

    assert main(["guide.pdf", "--method", "lead", "--sentences", "1"]) == 0

    output = capsys.readouterr().out
    assert "Summary of guide.pdf (lead):" in output
    assert f"- {ORCA_DOCUMENT.sentences[0].text} (page 1)" in output


def test_main_explains_documents_too_short_to_summarize(monkeypatch, capsys):
    document = make_document((1, "Too short."))
    monkeypatch.setattr(summarizer, "process_document", lambda path: document)

    assert main(["short.pdf"]) == 0
    assert "no sentences long enough" in capsys.readouterr().out


def test_main_reports_document_errors(monkeypatch, capsys):
    def fail(path):
        raise DocumentError("Could not find the file: missing.pdf")

    monkeypatch.setattr(summarizer, "process_document", fail)

    assert main(["missing.pdf"]) == 1
    assert "Could not find the file" in capsys.readouterr().err
