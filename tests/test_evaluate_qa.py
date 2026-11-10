import json

import pytest

from nlp_classifier import evaluate_qa
from nlp_classifier.document_processor import DocumentError
from nlp_classifier.evaluate_qa import (
    EvaluationResult,
    evaluate,
    load_questions,
    main,
)
from test_document_qa import ORCA_DOCUMENT


QUESTIONS = [
    {"document": "orcas.pdf", "question": "What keeps orcas warm?",
     "answer": "blubber", "split": "tune"},
    {"document": "orcas.pdf", "question": "How long can females live?",
     "answer": "ninety years", "split": "tune"},
    {"document": "orcas.pdf", "question": "What do orcas hunt?",
     "answer": "squid", "split": "test"},
    {"document": "orcas.pdf", "question": "Who won the football championship?",
     "answer": None, "split": "test"},
]


@pytest.fixture
def questions_file(tmp_path):
    path = tmp_path / "qa_eval.json"
    path.write_text(json.dumps({"questions": QUESTIONS}), encoding="utf-8")
    return path


@pytest.fixture
def processed(monkeypatch):
    """Use ORCA_DOCUMENT for every path and record which paths were processed."""
    paths = []

    def fake_process(path):
        paths.append(path)
        return ORCA_DOCUMENT

    monkeypatch.setattr(evaluate_qa, "process_document", fake_process)
    return paths


@pytest.mark.parametrize(("split", "count"), [("all", 4), ("tune", 2), ("test", 2)])
def test_load_questions_filters_by_split(questions_file, split, count):
    assert len(load_questions(questions_file, split)) == count


def test_evaluate_counts_correct_wrong_and_rejected(processed):
    result = evaluate(QUESTIONS, documents_folder="docs")

    assert result == EvaluationResult(
        correct=2, answerable=3, wrong=1, rejected=1, unanswerable=1
    )


def test_evaluate_processes_each_document_once(processed, tmp_path):
    evaluate(QUESTIONS, documents_folder=tmp_path)

    assert processed == [tmp_path / "orcas.pdf"]


def test_evaluate_counts_no_answer_as_neither_correct_nor_wrong(processed):
    result = evaluate(QUESTIONS[:1], min_score=1.0)

    assert result == EvaluationResult(
        correct=0, answerable=1, wrong=0, rejected=0, unanswerable=0
    )


def test_main_prints_results_for_chosen_split(processed, questions_file, capsys):
    assert main([str(questions_file), "--split", "test", "--documents", "docs"]) == 0

    output = capsys.readouterr().out
    assert "Split: test" in output
    assert "Correct answers: 0 of 1" in output
    assert "Unanswerable questions given no answer: 1 of 1" in output


def test_main_reports_missing_questions_file(tmp_path, capsys):
    assert main([str(tmp_path / "missing.json")]) == 1

    assert "Error:" in capsys.readouterr().err


def test_main_reports_document_errors(monkeypatch, questions_file, capsys):
    def fail(path):
        raise DocumentError("Could not find the file: orcas.pdf")

    monkeypatch.setattr(evaluate_qa, "process_document", fail)

    assert main([str(questions_file)]) == 1
    assert "Could not find the file" in capsys.readouterr().err
