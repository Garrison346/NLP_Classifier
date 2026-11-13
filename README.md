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

Download the NLTK stopword data, which is required for document question answering, stopword removal, and their tests:

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

Start the command-line chatbot. Type `help` for instructions, or `exit` or `quit` to leave:

```bash
uv run nlp-classifier
```

To ask questions about a PDF, type `load` followed by its path in the chat, such as `load data/documents/report.pdf`. Use quotes around a path that contains spaces. You can also load a PDF when the chatbot starts:

```bash
uv run nlp-classifier --document path/to/file.pdf
```

Add `--debug` to see the predicted intent and confidence for each message, the model's top guess when a message is rejected, and the score of each document answer:

```bash
uv run nlp-classifier --debug
```

See how a PDF is processed:

```bash
uv run python -m nlp_classifier.document_processor path/to/file.pdf
```

Ask questions about a PDF without the intent classifier:

```bash
uv run python -m nlp_classifier.document_qa path/to/file.pdf
```

Score Q&A on a file of questions with known answers. The file format is described in `evaluate_qa.py`. Use `--split test` only for a final check after choosing settings:

```bash
uv run python -m nlp_classifier.evaluate_qa data/documents/qa_eval.json
```

Summarize a PDF without the intent classifier. Add `--method lead`, `tfidf`, or `textrank` to choose the method:

```bash
uv run python -m nlp_classifier.summarizer path/to/file.pdf
```

Compare the summarization methods on a file of key points. The file format is described in `evaluate_summary.py`:

```bash
uv run python -m nlp_classifier.evaluate_summary data/documents/summary_key_points.json
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
│       ├── document_processor.py   # PDF text extraction, cleanup, and splitting
│       ├── document_qa.py          # Question answering with TF-IDF retrieval
│       ├── evaluate_qa.py          # Scores Q&A on questions with known answers
│       ├── summarizer.py           # Extractive summaries with lead, TF-IDF, and TextRank
│       ├── evaluate_summary.py     # Scores summaries by key-point coverage
│       └── document_classifier.py  # Placeholder, not implemented yet
│
├── tests/
│   ├── test_chatbot.py
│   ├── test_document_processor.py
│   ├── test_document_qa.py
│   ├── test_evaluate_qa.py
│   ├── test_evaluate_summary.py
│   ├── test_intent_classifier.py
│   ├── test_main.py
│   ├── test_preprocessing.py
│   └── test_summarizer.py
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

`intent_classifier.py` trains a model to recognize six intents: `greeting`, `goodbye`, `thanks`, `classify_document`, `ask_document_question`, and `summarize_document`.

- **Data:** 330 training examples, with 65 each for `classify_document` and `ask_document_question`, and 50 for each other intent.
- **Features:** TF-IDF on whole words and on character sequences.
- **Model:** Logistic Regression, compared against Linear SVM, Multinomial Naive Bayes, and a baseline.
- **Unclear messages:** predictions below a 0.7 confidence threshold return `unknown`.

Results on a separate test set of 90 on-topic and 20 off-topic messages:

| Measure | Result |
|---|---|
| Accuracy without the threshold | 93.3% |
| On-topic messages answered at the threshold | 70.0% |
| Answered messages with the correct intent | 98.4% |
| Off-topic messages rejected | 80.0% |

Repeated 5-fold cross-validation on the training data gives a macro-F1 of about 0.84, compared with about 0.06 for the baseline.

A `help` intent was removed because the chatbot is a class project rather than a public tool. It sent some document questions, such as "How do orcas sleep?", to the help reply instead. Typing `help` still shows instructions, as a fixed command that does not use the classifier.

**Known limitations**
- The training and test messages were written by the same author, not collected from real users, so real-world scores will likely be lower.
- The test set is small, and it was also used to compare versions of the dataset.
- `intents_user_test.json` is a small set of real messages that exposed weak document handling. The current model handles all five, but new training examples were written after seeing them, so this is not an independent score.
- Off-topic detection is weak. Adding an off-topic training intent rejected more of these messages, but it also rejected too many valid ones, so it was not adopted.

### 4. Command-Line Chatbot (Complete, Initial Version)

`chatbot.py` turns each message into a reply without using the terminal, so a future interface can reuse it. `main.py` runs the chat in the terminal.

The chatbot responds to greetings, goodbyes, and thanks, and answers questions about and summarizes a loaded PDF. Typing `help` shows instructions. It recognizes requests to classify a document, but explains that this feature is not available yet. Messages below the confidence threshold get a reply saying it did not understand.

### 5. Document Processing (Complete, Initial Version)

`document_processor.py` prepares PDFs for the document features.

- **Extraction:** reads text with PyPDF and keeps each page's number.
- **Cleanup:** repairs text extracted with one word per line, as can happen with Google Docs PDFs. It also removes many repeated headers, footers, and page numbers, separates likely headings from sentences, rejoins words split across lines, and fixes ligatures, soft hyphens, control characters, and irregular whitespace.
- **Splitting:** divides text into sentences using NLTK, then into overlapping passages of up to 120 words. A sentence divided by a page break is rejoined and stays with the page where it starts.
- **Errors:** gives clear messages for missing files, other file types, damaged PDFs, password-protected PDFs, and PDFs with no extractable text.

**Known limitations**
- Automated tests use simple generated PDFs. Manual testing has covered four PDFs, two from Google Docs and two from ReportLab. PDFs exported directly from Word have not been tested, so other PDFs may reveal more problems.
- Some Google Docs PDFs extract without any paragraph gaps, so headings and tables can merge with nearby sentences, and the first number in a list can attach to the sentence before it.
- Formulas can gain extra spaces, as in `x ₁`.
- Scanned pages are skipped because there is no text recognition (OCR).
- Two-column text was read in the correct order in one generated test, but other multi-column layouts may be read in the wrong order.
- Table rows usually stay in order, but column headings are not linked to their values, so a number's year or category can be unclear.
- Headers and footers are only removed when they are short and repeat near the page edges on at least 60% of all pages, odd pages, or even pages. Others can remain, and a real line that repeats this way can be removed.
- Heading detection is a best guess based on layout, so it can miss a heading or separate a short line by mistake.
- Sentence splitting relies on a short list of abbreviations. Words not on the list, such as `Ore.`, can end a sentence too early, and a sentence that really ends in a listed word, such as `Inc.`, can merge with the next one.
- A table can be treated as one long sentence, and it may be cut at a word limit rather than between rows.
- PDFs that use AES encryption may not open. This has not been tested with a real AES-encrypted file.
- A hyphenated word broken across lines loses its hyphen, so "well-known" can become "wellknown".
- Footnote markers can stay attached to text, as in `value.1`.
- Citations after a quotation, such as `(pg. 705).`, can become separate sentences.
- Page breaks are only rejoined when the next page starts in lowercase.

### 6. Document Question Answering (Complete, Initial Version)

`document_qa.py` answers questions with a sentence from the loaded document and the page it came from.

- **Search terms:** stopwords are removed and words are stemmed, so "scaling" matches "scaled".
- **Retrieval:** TF-IDF is fitted on the document's passages. The three passages most similar to the question, by cosine similarity, are searched for the best sentence.
- **Scoring:** a sentence's score combines its own similarity with its passage's similarity. Sentences under four words, which are usually headings, are never given as answers.
- **Not found:** answers scoring below 0.2 get a reply saying the answer was not found.

In the chatbot, a message classified as a document question is answered from the loaded document. When a document is loaded, a message the intent classifier is unsure about is also tried as a question, and is only answered if a match passes the threshold.

Results on 55 answerable and 12 unanswerable questions about four PDFs:

| Measure | Result |
|---|---|
| Correct answers using Q&A alone, without the threshold | 36 of 55 (65%) |
| Correct answers through the chatbot | 34 of 55 (62%) |
| Unanswerable questions given no answer through the chatbot | 7 of 12 (58%) |

These settings were chosen using the same questions, which were written by the same author as the code after reading the documents, so real-world results will likely be lower. The questions are kept in `data/documents/`, so they are not published, and are now split into 46 for choosing settings and 21 held out for final checks. On the held-out questions, Q&A alone answers 11 of 17 correctly and gives no answer to 3 of 4 unanswerable questions.

**Experiment: averaged word vectors**

Blending TF-IDF with averaged GloVe word vectors (100 dimensions) was tested to help match words with similar meanings, such as "another name" and "also called". It made results worse. On the held-out questions, the best blend chosen on the tuning questions answered 9 of 17 correctly and rejected 1 of 4 unanswerable questions, so it was not adopted. Averaging made most sentences about the same topic look alike, related words with opposite meanings looked similar (such as "warm" and "colder"), and higher scores overall made unanswerable questions harder to reject. Sentence embeddings, which are trained to compare whole sentences, have not been tested.

**Known limitations**
- Answers come from matching words, not meaning, so a question that uses different words from the document, such as "another name" for "also called", often finds the wrong sentence.
- When an answer is not in the document, a sentence that shares the question's topic words can still be returned with a high score.
- Tables have little structure after extraction, so questions about table values are unreliable.
- Merged headings can appear at the start of an answer.
- Only one document can be loaded at a time.

### 7. Document Summarization (Complete, Initial Version)

`summarizer.py` builds extractive summaries: it selects the document's most important sentences and shows them in their original order, with their page numbers. Summaries have 3 to 6 sentences, depending on the document's length.

Short sentences, table rows, citations with web addresses, and repeated sentences are not used. Three methods are available:

- **Lead (baseline):** the first sentences. Real methods should beat it.
- **TF-IDF scoring:** sentences whose words are most important to the document, using the average importance of each sentence's words.
- **TextRank (default):** sentences linked by TF-IDF cosine similarity form a graph, and the PageRank algorithm ranks the sentences most similar to many others.

TF-IDF scoring and TextRank choose sentences with Maximal Marginal Relevance (MMR), which avoids repetition. Sentences are chosen one at a time, and each candidate's score is reduced by its similarity to the sentences already chosen, with a penalty of 0.3. The summarizer command's `--redundancy-penalty` option changes it, and 0 turns it off.

Methods are compared by key-point coverage: each document has 4 to 6 key points, each with phrases that a sentence covering it should contain. The key points are kept in `data/documents/`, so they are not published.

Results on four documents, with 6 key points each:

| Method | Orca guide | Financial report | Civil Rights essay | K-means guide | Total |
|---|---|---|---|---|---|
| Lead | 1 | 2 | 0 | 1 | 4 of 24 |
| TF-IDF scoring, no MMR | 2 | 1 | 2 | 1 | 6 of 24 |
| TF-IDF scoring, MMR | 2 | 0 | 2 | 2 | 6 of 24 |
| TextRank, no MMR | 4 | 2 | 1 | 3 | 10 of 24 |
| **TextRank, MMR (default)** | **5** | **2** | **2** | **3** | **12 of 24** |

TextRank with MMR covers the most key points, so it is the default. MMR removed a near-duplicate sentence from the essay's summary, where the introduction and conclusion list the same events, and from the orca guide's. Penalties of 0.2 and 0.3 both gave 12 of 24 for TextRank, while 0.7 dropped it to 7. Even the default covers only half of the key points. The key points were written after seeing the summaries, which may favor the methods' choices, and a key point only counts when a sentence covers it directly, not when an event is just named in a list.

**Experiment: clustering sentences with K-means**

Grouping sentences into clusters with K-means, then taking one sentence from each cluster, was tested as a way to cover more of a document's topics. The best version, which took the sentence with the highest TextRank score from each cluster, averaged 8.4 of 24 key points over five random seeds, with results from 6 to 10 depending on the seed, compared with 12 for TextRank with MMR. In a blind comparison of the two methods' summaries, the project author preferred TextRank with MMR for three of four documents, and clustering for the orca guide. The default was shown first in every pair, which may have favored it. Clustering was not adopted, because it scored lower and its summaries change with the random seed. The comparison also showed that exact-phrase key points can miss good summaries, such as clustering's orca summary, so reading the summaries remains important.

**Known limitations**
- MMR only catches repetition in wording, so two sentences that say the same thing in different words can both be chosen.
- TextRank favors long sentences, which link to more sentences, so it can choose a long but unimportant sentence.
- Merged headings and table rows can still appear when they are joined to a normal sentence.
- The evaluation covers only a few documents, so only large differences between methods are meaningful. The MMR penalty was also chosen using these key points.

### 8. Planned Components

The remaining work, in planned order:

1. Document theme classification.
2. Streamlit user interface.

**Document Theme Classification**
- Document text extraction and preprocessing.
- TF-IDF feature extraction.
- Supervised document classification.
- Model evaluation and hyperparameter tuning.

## Development Philosophy

This project emphasizes understanding and implementing the underlying concepts of statistical NLP and machine learning.

The primary focus is on learning how text is transformed into numerical representations, how machine-learning algorithms identify patterns in language, and how those models can be applied to practical NLP applications.
