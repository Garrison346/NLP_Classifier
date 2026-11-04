# NLP Classifier & Chatbot

## Overview

This project focuses on developing an NLP-based chatbot using statistical and traditional machine-learning techniques. The goal is to build a system capable of understanding user input, classifying document themes, and answering questions about uploaded documents.

Rather than relying entirely on pretrained language models or LLMs, this project emphasizes implementing NLP preprocessing techniques, feature extraction, and training machine-learning classifiers.

## Project Goals

- Develop an intent-classification model to identify user requests.
- Implement document classification to determine the primary theme of uploaded documents.
- Develop document processing capabilities for extracting and analyzing text.
- Implement document-based question answering using information retrieval techniques.
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

## Project Structure

```text
NLP_Classifier/
│
├── data/ 
│   └── intents.json
│
├── src/
│   └── nlp_classifier/
│       ├── __init__.py
│       ├── main.py
│       ├── preprocessing.py
│       ├── intent_classifier.py
│       ├── document_classifier.py
│       ├── document_processor.py
│       └── chatbot.py
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_intent_classifier.py
│
├── .python-version
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Current Development Progress

### 1. Project Setup
- Initialized Python project using uv.
- Created an isolated Python virtual environment.
- Installed required dependencies.
- Established the initial project directory structure.

### 2. NLP Preprocessing

Initial text preprocessing functionality is being developed using NLTK.

Planned preprocessing operations include:
- Lowercasing text.
- Removing punctuation and special characters.
- Tokenization.
- Stop-word removal.
- Stemming and lemmatization.

### 3. Planned Machine-Learning Components

**Intent Classification**
- JSON-based intent dataset.
- TF-IDF feature extraction.
- Logistic Regression classification.
- Comparison of classification algorithms.

**Document Theme Classification**
- Document text extraction and preprocessing.
- TF-IDF feature extraction.
- Supervised document classification.
- Model evaluation and hyperparameter tuning.

**Document Question Answering**
- Document text extraction.
- Text segmentation.
- TF-IDF-based document retrieval.
- Cosine similarity for identifying relevant passages.
- Extractive question answering.

## Development Philosophy

This project emphasizes understanding and implementing the underlying concepts of statistical NLP and machine learning.

The primary focus is on learning how text is transformed into numerical representations, how machine-learning algorithms identify patterns in language, and how those models can be applied to practical NLP applications.
