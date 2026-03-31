# 🚗 Sentiment Analysis on Car Reviews
### Unstructured Data Course — NLP Pipeline

## Overview
End-to-end NLP pipeline for sentiment classification of car reviews.
Reviews are classified into **Positive**, **Neutral**, and **Negative** sentiment
using a full pipeline from classical ML to fine-tuned transformers and LLM comparison.

---

## Dataset
- **Source:** [florentgbelidji/car-reviews](https://huggingface.co/datasets/florentgbelidji/car-reviews) (Hugging Face)
- **Origin:** Edmunds.com user car reviews
- **Size:** 36,984 reviews
- **Features used:** `Review` (free text), `Rating` (1–5, converted to sentiment label)
- **Label mapping:**
  - Rating 1–2 → Negative
  - Rating 3 → Neutral
  - Rating 4–5 → Positive
- **Class distribution:** 85% Positive · 8% Neutral · 7% Negative (heavily imbalanced)

---

## Project Structure
```
TEXT-UnstructuredData/
├── sentiment_analysis_car_reviews.ipynb  # Main notebook
├── pyproject.toml                        # Dependencies (uv)
├── uv.lock                               # Locked dependency versions
├── .env.example                          # API keys template
├── .python-version                       # Python version pin
├── .gitignore
└── README.md
```
> ⚠️ BERT checkpoints (`bert_epoch_*.pt`) are not tracked in git due to file size (~400MB each).

---

## Setup

### Requirements
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- A [Groq](https://console.groq.com) API key (free, no credit card needed)

### Installation
```bash
# Clone the repo
git clone git@github.com:rauljimenorivera/TEXT-UnstructuredData.git
cd TEXT-UnstructuredData

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

### API Keys
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com).

---

## How to Run
Open the notebook in VS Code or Jupyter and run all cells in order:
```bash
# Open in VS Code
code .
```

Make sure the Python interpreter is set to `.venv` before running.
The notebook is self-contained — it downloads the dataset automatically via Hugging Face.

> ⚠️ The BERT fine-tuning section (section 3.2) takes ~2 hours on CPU.
> Checkpoints are saved after each epoch so progress is not lost if interrupted.

---

## Pipeline
| Step | Description |
|------|-------------|
| 1. EDA | Data exploration, class distribution, text length analysis, word frequency, WordClouds |
| 2. Preprocessing | Stopword removal, lemmatization, TF-IDF feature extraction |
| 3. ML Baseline | Logistic Regression, Naive Bayes, Linear SVM, Random Forest |
| 4. Deep Learning (scratch) | Feedforward Neural Network with TF-IDF features (PyTorch) |
| 5. Pretrained Model | BERT fine-tuning (`bert-base-uncased`) on balanced subset of 9,000 reviews |
| 6. LLM Comparison | Zero-shot classification with LLaMA 3.1 8B via Groq API |

---

## Results

| Model                    | F1 Macro | F1 Negative | F1 Neutral | F1 Positive | Accuracy |
|--------------------------|----------|-------------|------------|-------------|----------|
| Logistic Regression      | 0.58     | 0.48        | 0.34       | 0.92        | 0.83     |
| Naive Bayes              | 0.42     | 0.26        | 0.06       | 0.93        | 0.87     |
| Linear SVM               | 0.56     | 0.44        | 0.29       | 0.94        | 0.85     |
| Random Forest            | 0.36     | 0.14        | 0.00       | 0.93        | 0.86     |
| Feedforward NN (scratch) | 0.55     | 0.42        | 0.29       | 0.94        | 0.86     |
| **BERT Fine-tuned**      | **0.77** | **0.76**    | **0.68**   | **0.86**    | **0.76** |
| LLaMA 3.1 8B (zero-shot) | 0.50     | 0.62        | 0.10       | 0.79        | 0.56     |

### Key Findings
- **Class imbalance is the main challenge** — accuracy alone is misleading, F1 Macro is the right metric
- **Logistic Regression is the best classical ML model** — fast (13s) and hard to beat without embeddings
- **FFNN from scratch adds no value** — overfits severely when using TF-IDF features
- **BERT fine-tuning is the clear winner** — F1 Macro jumps from 0.58 → 0.77 (+33%)
- **Fine-tuned BERT beats zero-shot LLaMA 3.1 8B** — 110M fine-tuned params outperform 8B zero-shot params, proving that domain-specific fine-tuning matters more than raw model size

---

## Dependencies
Managed with [uv](https://github.com/astral-sh/uv). Main libraries:
`transformers` · `torch` · `scikit-learn` · `datasets` · `nltk` · `groq` · `pandas` · `matplotlib` · `wordcloud`

---

## Author
Raul Jimeno Rivera & Ruben Navarro Tudury