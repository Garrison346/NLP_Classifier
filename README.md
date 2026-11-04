# NLP Classifier & Chatbot

## Overview

This project focuses on developing an NLP-based chatbot using statistical and traditional machine-learning techniques. The goal is to build a system capable of understanding user input, classifying document themes, summarizing documents, and answering questions about them.

Rather than relying entirely on pretrained language models or LLMs, this project emphasizes implementing NLP preprocessing techniques, feature extraction, and training machine-learning classifiers.

## Project Goals

- Develop an intent-classification model to identify user requests.
- Implement document classification to determine the primary theme of uploaded documents.
- Develop document processing capabilities for extracting and analyzing text.
- Implement document-based question answering using information retrieval techniques.
- Implement extractive document summarization.
- Explore and compare different statistical and machine-learning NLP approaches.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| Scikit-learn | Machine-learning models and classification |
| NLTK | Natural language processing and preprocessing |
| NumPy | Numerical operations |
| Pandas | Dataset management and manipulation |
| Joblib | Saving and loading trained models |
| PyPDF | PDF text extraction |
| Streamlit | Future chatbot user interface |
| uv | Python package and project management |

## Setup

Install the project and its dependencies:

```bash
uv sync
```

Download the NLTK stopword data, which is required for stopword removal and its tests:

```bash
uv run python -m nltk.downloader stopwords
```

Run the tests:

```bash
uv run pytest
```

Train and evaluate the intent classifier. Trained models are not stored in Git, so run this after cloning:

```bash
uv run python -m nlp_classifier.intent_classifier
```

Start the command-line chatbot. Type `exit` or `quit` to leave:

```bash
uv run nlp-classifier
```

Add `--debug` to see the predicted intent and confidence for each message, plus the model's top guess when a message is rejected:

```bash
uv run nlp-classifier --debug
```

## Project Structure

```text
NLP_Classifier/
│
├── data/
│   ├── documents/              # Local documents, not stored in Git
│   ├── intents.json            # Intent training examples
│   ├── intents_test.json       # Held-out test messages, never used for training
│   └── intents_user_test.json  # Messages from real chatbot use, never used for training
│
├── models/                     # Trained models, created by training, not stored in Git
├── notebooks/                  # Local notebooks, none committed yet
│
├── src/
│   └── nlp_classifier/
│       ├── __init__.py
│       ├── main.py                 # Command-line chat
│       ├── chatbot.py              # Reply logic
│       ├── preprocessing.py        # Text normalization and tokenization
│       ├── intent_classifier.py    # Intent training, evaluation, and prediction
│       ├── document_processor.py   # Placeholder, not implemented yet
│       └── document_classifier.py  # Placeholder, not implemented yet
│
├── tests/
│   ├── test_chatbot.py
│   ├── test_intent_classifier.py
│   ├── test_main.py
│   └── test_preprocessing.py
│
├── .python-version
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Current Development Progress

### 1. Project Setup (Complete)
- Initialized Python project using uv.
- Created an isolated Python virtual environment.
- Installed required dependencies.
- Established the project directory structure.

### 2. NLP Preprocessing (Complete)

Preprocessing is implemented in `preprocessing.py` and covered by tests:
- Unicode, whitespace, apostrophe, and lowercase normalization.
- Tokenization that removes punctuation while keeping dates, numbers, currency, percentages, contractions, and international characters.
- Optional stop-word removal that keeps negation words.

Stop-words are kept by default because the standard list includes words such as "what", "how", and "can". Removing them lowered intent classification scores.

Stemming and lemmatization are not implemented. They may be tested later as experiments.

### 3. Intent Classification (Complete, Initial Version)

`intent_classifier.py` trains a model to recognize seven intents: `greeting`, `goodbye`, `thanks`, `help`, `classify_document`, `ask_document_question`, and `summarize_document`.

- **Data:** 380 training examples, with 65 each for `classify_document` and `ask_document_question`, and 50 for each other intent.
- **Features:** TF-IDF on whole words and on character sequences.
- **Model:** Logistic Regression, compared against Linear SVM, Multinomial Naive Bayes, and a baseline.
- **Unclear messages:** predictions below a 0.7 confidence threshold return `unknown`.

Results on a separate test set of 105 on-topic and 20 off-topic messages:

| Measure | Result |
|---|---|
| Accuracy without the threshold | 91.4% |
| On-topic messages answered at the threshold | 67.6% |
| Answered messages with the correct intent | 97.2% |
| Off-topic messages rejected | 70.0% |

Repeated 5-fold cross-validation on the training data gives a macro-F1 of about 0.83, compared with about 0.04 for the baseline.

**Known limitations**
- The training and test messages were written by the same author, not collected from real users, so real-world scores will likely be lower.
- The test set is small, and it was also used to compare versions of the dataset.
- `intents_user_test.json` is a small set of real messages that exposed weak document handling. The current model handles all five, but new training examples were written after seeing them, so this is not an independent score.
- Off-topic detection is weak. Adding an off-topic training intent rejected more of these messages, but it also rejected too many valid ones, so it was not adopted.

### 4. Command-Line Chatbot (Complete, Initial Version)

`chatbot.py` turns each message into a reply without using the terminal, so a future interface can reuse it. `main.py` runs the chat in the terminal.

The chatbot responds to greetings, goodbyes, thanks, and requests for help. It recognizes document requests, but explains that those features are not available yet. Messages below the confidence threshold get a reply saying it did not understand.

### 5. Planned Components

The remaining work, in planned order:

1. Document processing.
2. Document question answering.
3. Document summarization.
4. Document theme classification.
5. Streamlit user interface.

**Document Processing**
- PDF text extraction that keeps track of page numbers.
- Cleanup of common PDF problems, such as words split across lines.
- Splitting text into sentences and passages.
- Clear errors for missing files, corrupt PDFs, and scanned PDFs with no extractable text.

**Document Question Answering**
- Document text extraction.
- Text segmentation.
- TF-IDF-based document retrieval.
- Cosine similarity for identifying relevant passages.
- Extractive question answering.

**Document Summarization**
- Sentence segmentation.
- TF-IDF-based sentence scoring.
- Extractive summaries built from the highest-scoring sentences.

**Document Theme Classification**
- Document text extraction and preprocessing.
- TF-IDF feature extraction.
- Supervised document classification.
- Model evaluation and hyperparameter tuning.

## Development Philosophy

This project emphasizes understanding and implementing the underlying concepts of statistical NLP and machine learning.

The primary focus is on learning how text is transformed into numerical representations, how machine-learning algorithms identify patterns in language, and how those models can be applied to practical NLP applications.
