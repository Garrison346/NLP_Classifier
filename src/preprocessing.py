import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")


def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return tokens


if __name__ == "__main__":

    sample = "The Transformer architecture is revolutionizing Natural Language Processing!"

    processed = preprocess_text(sample)

    print("Original:", sample)
    print("Processed:", processed)