"""Train, evaluate, save, and use the chatbot's intent classifier.

Run this module to compare models and train the final classifier:

    uv run python -m nlp_classifier.intent_classifier
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from nlp_classifier.preprocessing import normalize_text, preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTENTS_PATH = PROJECT_ROOT / "data" / "intents.json"
TEST_INTENTS_PATH = PROJECT_ROOT / "data" / "intents_test.json"
MODEL_PATH = PROJECT_ROOT / "models" / "intent_classifier.joblib"

# Returned when the model is not confident enough in any known intent.
UNKNOWN_INTENT = "unknown"
CONFIDENCE_THRESHOLD = 0.7

CV_FOLDS = 5
RANDOM_STATE = 42

# A larger C applies weaker regularization, which gave better separated
# confidence scores on this small dataset.
LOGISTIC_REGRESSION_C = 10


@dataclass(frozen=True)
class IntentPrediction:
    """A predicted intent and the model's confidence in its top choice."""

    intent: str
    confidence: float


def load_intents(path: Path = INTENTS_PATH) -> tuple[list[str], list[str]]:
    """
    Load training examples from an intents JSON file.

    Args:
        path: Path to the intents file.

    Returns:
        Parallel lists of example messages and their intent labels.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    texts = []
    labels = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(pattern)
            labels.append(intent["tag"])

    return texts, labels


def _logistic_regression() -> LogisticRegression:
    """Create the classifier used for the final model."""
    return LogisticRegression(C=LOGISTIC_REGRESSION_C, max_iter=5000)


def build_pipeline(classifier: BaseEstimator | None = None) -> Pipeline:
    """
    Build a TF-IDF and classifier pipeline.

    Whole-word features capture meaning, while character features help with
    short messages, slang, and misspellings that share parts of known words.

    Args:
        classifier: The classifier to use. Defaults to Logistic Regression,
            which provides the probabilities used for confidence checks.

    Returns:
        An untrained scikit-learn pipeline.
    """
    if classifier is None:
        classifier = _logistic_regression()

    word_features = TfidfVectorizer(
        tokenizer=preprocess_text,
        lowercase=False,  # preprocess_text already lowercases.
        token_pattern=None,  # Unused because a tokenizer is provided.
    )
    character_features = TfidfVectorizer(
        analyzer="char_wb",
        preprocessor=normalize_text,
        ngram_range=(2, 4),
    )
    features = FeatureUnion(
        [("words", word_features), ("characters", character_features)]
    )

    return Pipeline([("features", features), ("classifier", classifier)])


def _cross_validator() -> StratifiedKFold:
    """Create folds that keep each intent's share of examples balanced."""
    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def compare_models(texts: list[str], labels: list[str]) -> pd.DataFrame:
    """
    Compare a baseline and candidate models using cross-validation.

    Args:
        texts: Example messages.
        labels: Intent label for each message.

    Returns:
        Macro-F1 mean and standard deviation per model, best first.
    """
    candidates = {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": _logistic_regression(),
        "Linear SVM": LinearSVC(),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    rows = []
    for name, classifier in candidates.items():
        scores = cross_val_score(
            build_pipeline(classifier),
            texts,
            labels,
            cv=_cross_validator(),
            scoring="f1_macro",
        )
        rows.append(
            {
                "model": name,
                "macro_f1_mean": scores.mean(),
                "macro_f1_std": scores.std(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_model(texts: list[str], labels: list[str]) -> tuple[str, pd.DataFrame]:
    """
    Evaluate the default pipeline on held-out folds.

    Every example is predicted by a model that did not train on it.

    Args:
        texts: Example messages.
        labels: Intent label for each message.

    Returns:
        A per-intent text report and a confusion matrix, where rows are the
        true intents and columns are the predicted intents.
    """
    predictions = cross_val_predict(
        build_pipeline(),
        texts,
        labels,
        cv=_cross_validator(),
    )
    intents = sorted(set(labels))

    report = classification_report(labels, predictions, zero_division=0)
    matrix = pd.DataFrame(
        confusion_matrix(labels, predictions, labels=intents),
        index=intents,
        columns=intents,
    )

    return report, matrix


def train_model(texts: list[str], labels: list[str]) -> Pipeline:
    """Train the default pipeline on all of the provided examples."""
    return build_pipeline().fit(texts, labels)


def predict_intent(
    model: Pipeline,
    text: str,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> IntentPrediction:
    """
    Predict the intent of a message.

    Args:
        model: A trained pipeline that supports predict_proba.
        text: The user's message.
        threshold: Minimum confidence required to accept a prediction.

    Returns:
        The predicted intent, or UNKNOWN_INTENT if confidence is below
        the threshold, along with the top confidence score.
    """
    probabilities = model.predict_proba([text])[0]
    best_index = probabilities.argmax()
    confidence = float(probabilities[best_index])

    if confidence < threshold:
        return IntentPrediction(UNKNOWN_INTENT, confidence)

    return IntentPrediction(str(model.classes_[best_index]), confidence)


def evaluate_on_test_set(
    model: Pipeline,
    path: Path = TEST_INTENTS_PATH,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, float]:
    """
    Score a trained model on a held-out test file.

    Never train on the test file or tune against its individual errors,
    because that would make these scores look better than real use.

    Args:
        model: A trained pipeline that supports predict_proba.
        path: Path to a test file. It uses the same format as the intents
            file, plus an "out_of_scope" list of messages to reject.
        threshold: Minimum confidence required to accept a prediction.

    Returns:
        Scores between 0 and 1:
        - accuracy_without_threshold: correct if every message is answered.
        - answered: in-scope messages that passed the threshold.
        - correct_when_answered: answered messages with the right intent.
        - correct_overall: in-scope messages answered with the right intent.
        - out_of_scope_rejected: out-of-scope messages returned as unknown.
    """
    texts, labels = load_intents(path)
    out_of_scope = json.loads(Path(path).read_text(encoding="utf-8"))["out_of_scope"]

    best_guesses = [predict_intent(model, text, threshold=0.0) for text in texts]
    correct = [
        guess.intent == label for guess, label in zip(best_guesses, labels)
    ]
    answered = [guess.confidence >= threshold for guess in best_guesses]
    correct_and_answered = sum(c and a for c, a in zip(correct, answered))
    rejected = sum(
        predict_intent(model, text, threshold).intent == UNKNOWN_INTENT
        for text in out_of_scope
    )

    return {
        "accuracy_without_threshold": sum(correct) / len(texts),
        "answered": sum(answered) / len(texts),
        "correct_when_answered": correct_and_answered / max(sum(answered), 1),
        "correct_overall": correct_and_answered / len(texts),
        "out_of_scope_rejected": rejected / len(out_of_scope),
    }


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    """Save a trained model, creating its folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load a trained model. Only load model files that you trust."""
    return joblib.load(path)


def main() -> None:
    """Compare models, report results, and save the final classifier."""
    texts, labels = load_intents()

    print(f"Loaded {len(texts)} examples across {len(set(labels))} intents.\n")

    print(f"Model comparison ({CV_FOLDS}-fold cross-validation):")
    print(compare_models(texts, labels).round(3).to_string(index=False))

    report, matrix = evaluate_model(texts, labels)
    print("\nLogistic Regression on held-out folds:")
    print(report)
    print("Confusion matrix (rows are true, columns are predicted):")
    print(matrix.to_string())

    model = train_model(texts, labels)

    print(f"\nHeld-out test set (threshold {CONFIDENCE_THRESHOLD}):")
    for name, score in evaluate_on_test_set(model).items():
        print(f"  {name}: {score:.1%}")

    save_model(model)
    print(f"\nSaved final model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
