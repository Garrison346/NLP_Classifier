import re
import unicodedata

from nltk.corpus import stopwords


# Protect words and contractions that express negation.
NEGATION_WORDS = {
    "no",
    "not",
    "nor",
    "never",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
    "couldn't",
    "shouldn't",
    "can't",
    "cannot",
    "hasn't",
    "haven't",
    "hadn't",
    "mustn't",
    "needn't",
    "mightn't",
    "shan't",
}

# Load English stopwords once when the module is imported.
STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS


# Recognize dates before general numbers so dates remain single tokens.
DATE_PATTERN = r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"

# Recognize signed numbers, thousands separators, decimals,
# currency symbols, and percentages. A sign only counts when it does not
# directly follow a word or number, so "3-5" and "covid-19" stay unsigned.
NUMBER_PATTERN = (
    r"(?:(?<!\w)[-+])?"
    r"(?:[$€£¥])?"
    r"\d+"
    r"(?:,\d{3})*"
    r"(?:\.\d+)?"
    r"%?"
)

# Recognize Unicode words and contractions.
WORD_PATTERN = r"[^\W\d_]+(?:'[^\W\d_]+)*"

TOKEN_PATTERN = re.compile(
    rf"{DATE_PATTERN}|{NUMBER_PATTERN}|{WORD_PATTERN}"
)


def normalize_text(text: str) -> str:
    """
    Normalize text without discarding meaningful information.

    Steps:
    1. Validate input.
    2. Normalize Unicode characters.
    3. Convert curly apostrophes to straight apostrophes.
    4. Convert text to lowercase.
    5. Normalize whitespace.

    Args:
        text: Raw input text.

    Returns:
        Normalized text.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text while preserving numerical expressions
    and international characters.

    Hyphenated words are separated into individual tokens.

    Args:
        text: Raw or normalized text.

    Returns:
        A list of tokens.
    """
    normalized_text = normalize_text(text)

    return TOKEN_PATTERN.findall(normalized_text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove English stopwords while retaining negation words.

    Args:
        tokens: A list of tokens.

    Returns:
        A list with stopwords removed.
    """
    return [
        token for token in tokens
        if token not in STOP_WORDS
    ]


def preprocess_text(
    text: str,
    remove_stop_words: bool = False,
) -> list[str]:
    """
    Normalize and tokenize text, optionally removing stopwords.

    Stopwords are kept by default because the NLTK list includes words such
    as "what", "how", and "can", which carry meaning in intents and questions.

    Args:
        text: Raw input text.
        remove_stop_words: Whether to remove English stopwords.

    Returns:
        A list of processed tokens.

    Raises:
        TypeError: If text is not a string.
    """
    tokens = tokenize_text(text)

    if remove_stop_words:
        tokens = remove_stopwords(tokens)

    return tokens