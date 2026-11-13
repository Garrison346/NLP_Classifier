import json

import pytest

from nlp_classifier import evaluate_summary
from nlp_classifier.document_processor import DocumentError
from nlp_classifier.evaluate_summary import (
    Coverage,
    load_key_points,
    main,
    score_coverage,
)
from nlp_classifier.summarizer import SummarySentence
from test_summarizer import ORCA_DOCUMENT


KEY_POINTS = {
    "guide.pdf": [
        {"point": "Orcas are social", "phrases": ["highly social", "pod"]},
        {"point": "Skills are shared", "phrase": "hunting skills"},
        {"point": "The crew ate lunch", "phrases": ["lunch"]},
    ]
}


@pytest.fixture
def key_points_file(tmp_path):
    path = tmp_path / "key_points.json"
    path.write_text(json.dumps(KEY_POINTS), encoding="utf-8")
    return path


def test_load_key_points_accepts_single_phrase(key_points_file):
    points = load_key_points(key_points_file)

    assert points["guide.pdf"][1] == {"point": "Skills are shared", "phrases": ["hunting skills"]}


def test_score_coverage_matches_any_phrase_ignoring_case(key_points_file):
    summary = [
        SummarySentence("Orcas are HIGHLY SOCIAL animals.", 1, 0.5),
        SummarySentence("They pass on Hunting Skills to their young.", 2, 0.4),
    ]

    coverage = score_coverage(summary, load_key_points(key_points_file)["guide.pdf"])

    assert coverage == Coverage(
        covered=["Orcas are social", "Skills are shared"],
        missed=["The crew ate lunch"],
    )


def test_score_coverage_of_empty_summary_misses_everything(key_points_file):
    points = load_key_points(key_points_file)["guide.pdf"]

    assert score_coverage([], points).covered == []


def test_main_compares_methods(monkeypatch, key_points_file, capsys):
    monkeypatch.setattr(evaluate_summary, "process_document", lambda path: ORCA_DOCUMENT)

    assert main([str(key_points_file), "--documents", "docs", "--show"]) == 0

    output = capsys.readouterr().out
    assert "textrank  2 of 3 key points in 3 sentences" in output
    assert "missed: The crew ate lunch" in output
    assert "Total across 1 documents:" in output


def test_main_reports_missing_key_points_file(tmp_path, capsys):
    assert main([str(tmp_path / "missing.json")]) == 1

    assert "Error:" in capsys.readouterr().err


def test_main_reports_document_errors(monkeypatch, key_points_file, capsys):
    def fail(path):
        raise DocumentError("Could not find the file: guide.pdf")

    monkeypatch.setattr(evaluate_summary, "process_document", fail)

    assert main([str(key_points_file)]) == 1
    assert "Could not find the file" in capsys.readouterr().err
