# 🔎 SmartReview — Explainable Review Sentiment Analyzer

A credible, honest, and polished AI/ML portfolio project that analyzes customer review sentiment using a scikit-learn pipeline (TF-IDF + Logistic Regression) with local feature explainability and an optional external Google Gemini AI Studio integration.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │                 USER INPUT                   │
                      │  - Single Review Text Entry                  │
                      │  - CSV File Upload (up to 5,000 rows / 10MB) │
                      └──────────────────────┬───────────────────────┘
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
             ┌─────────────────────────┐             ┌─────────────────────────┐
             │    Local ML Pipeline    │             │   Optional Gemini API   │
             │   (Default / Offline)   │             │ (₹0 Google AI Studio)   │
             └───────────┬─────────────┘             └───────────┬─────────────┘
                         │                                       │
                         ▼                                       │
             ┌─────────────────────────┐                         │
             │   Text Preprocessing    │                         │
             │ - Lowercasing           │                         │
             │ - Contraction expansion │                         │
             │ - Negation preservation │                         │
             │ - URL / HTML stripping  │                         │
             └───────────┬─────────────┘                         │
                         │                                       │
                         ▼                                       │
             ┌─────────────────────────┐                         │
             │    TF-IDF Vectorizer    │                         │
             │ - Unigrams + Bigrams    │                         │
             │ - sublinear_tf=True     │                         │
             │ - min_df=2              │                         │
             └───────────┬─────────────┘                         │
                         │                                       │
                         ▼                                       │
             ┌─────────────────────────┐                         │
             │   Logistic Regression   │                         │
             │ - solver='lbfgs'        │                         │
             │ - class_weight='balance'│                         │
             │ - max_iter=1000, C=1.0  │                         │
             └───────────┬─────────────┘                         │
                         │                                       │
             ┌───────────┴─────────────┐                         │
             ▼                         ▼                         │
   ┌───────────────────┐     ┌───────────────────┐               │
   │Class Probabilities│     │Feature Attribution│               │
   │ P(Pos), P(Neg)    │     │ w_j * x_j signals │               │
   └─────────┬─────────┘     └─────────┬─────────┘               │
             │                         │                         │
             └────────────┬────────────┘                         │
                          │                                      │
                          ▼                                      ▼
            ┌──────────────────────────────────────────────────────────┐
            │                  STREAMLIT USER INTERFACE                │
            │  - 🔍 Single Review Analysis with Token Explainability   │
            │  - 📂 Batch CSV Processing & Enriched Export             │
            │  - 📊 Analytics Dashboard (Plots, Histograms, Top Words) │
            │  - ℹ️ Model Information & Baseline Comparison Cards     │
            └──────────────────────────────────────────────────────────┘
```

---

## Core Capabilities

| Feature | Description |
|---------|-------------|
| **Single Review Analysis** | Instant sentiment classification with confidence score and class probability distributions. |
| **White-Box Explainability** | Real-time feature attribution displaying the exact words/n-grams driving each prediction based on model coefficients. |
| **Batch CSV Analysis** | Upload CSV datasets (up to 5,000 rows); preview results, filter by sentiment, and export enriched CSVs. |
| **Analytics Dashboard** | Sentiment distributions (pie charts), source breakdowns, confidence histograms, and top word frequencies per class. |
| **Rigorous Evaluation** | Measured metrics on held-out test data, comparison against a dummy majority-class baseline, and Stratified 5-Fold Cross-Validation. |
| **Dual-Engine Architecture** | 100% offline local model by default; seamless fallback from optional Google Gemini Developer API (free tier at ₹0). |
| **Defensive Engineering** | Strict input validation (empty text, whitespace, null/NaN string placeholders, length limits), column collision prevention, and automated tests. |

---

## Technology Stack

- **Language:** Python 3.11+
- **Application Framework:** Streamlit (v1.66+)
- **Machine Learning:** scikit-learn (TF-IDF Vectorizer, Logistic Regression, Pipeline, DummyClassifier, StratifiedKFold)
- **Data Manipulation:** pandas, NumPy
- **Visualizations:** matplotlib, seaborn
- **Model Persistence:** joblib
- **API Integration:** official `google-genai` SDK (Google Gemini Developer API, ₹0 free tier)
- **Automated Testing:** pytest (119 automated unit, integration, and UI smoke tests)

---

## Project Structure

```
SmartReview/
├── app.py                          # Streamlit application entry point
├── train_pipeline.py               # Standalone training and evaluation script
├── verify_predictions.py           # Quick functional verification script
├── requirements.txt                # Pinned dependency specifications
├── pytest.ini                      # Pytest runner configuration
├── README.md                       # Architecture, evaluation, and documentation
├── .gitignore                      # Git exclusions (models, secrets, caches)
├── .env.example                    # Environment variable template
├── data/
│   └── sample_reviews.csv          # Bundled demonstration dataset (87 rows)
├── models/
│   ├── .gitkeep
│   ├── sentiment_pipeline.joblib   # Serialized scikit-learn Pipeline
│   ├── training_metadata.json      # Dataset and split metadata
│   └── evaluation_results.json     # Measured test, baseline, and CV metrics
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # CSV ingestion, schema validation, stratified split
│   ├── preprocessing.py            # Contraction expansion, negation preservation
│   ├── train_model.py              # Pipeline definition, serialization, lazy training
│   ├── predictor.py                # Single/batch inference, feature attribution (explainability)
│   ├── api_client.py               # Robust Google Gemini API client with granular error handling
│   └── evaluation.py              # Metrics, baseline evaluation, cross-validation, confusion matrix
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py       # Text cleaning, contraction, negation tests
    ├── test_data_loader.py         # CSV schema, duplicate handling, stratification tests
    ├── test_predictor.py           # Input validation, placeholder rejection, explainability tests
    ├── test_api_client.py          # Mocked API client call handling and JSON parsing tests
    ├── test_training.py            # Pipeline structure, serialization, probability checks
    ├── test_evaluation.py          # Baseline comparison and cross-validation tests
    └── test_app_smoke.py           # Streamlit AppTest framework UI smoke tests
```

---

## Machine Learning Methodology

### 1. Data Leakage Prevention
Feature extraction is encapsulated inside a scikit-learn `Pipeline([('tfidf', vectorizer), ('clf', classifier)])`.
The vectorizer's vocabulary and IDF statistics are fitted **strictly on the training split**, ensuring zero information from the held-out test split leaks into feature scaling or vocabulary generation.

### 2. Preprocessing & Negation Preservation
Standard NLP stopword removal collapses "not good" into "good", reversing the review's polarity.
SmartReview's preprocessing pipeline:
- Expands negating contractions (`don't` → `do not`, `won't` → `will not`, `can't` → `cannot`).
- Retains critical negation markers (`not`, `never`, `neither`, `nor`, `no`).
- Strips URLs, HTML markup, and non-alphanumeric punctuation while maintaining whitespace consistency.
- Extracts unigrams and bigrams (`ngram_range=(1, 2)`) to capture compound sentiment signals (`not bad`, `waste of money`).

### 3. Local Model Interpretability
Because Logistic Regression is a generalized linear model, each prediction is directly interpretable. For any input review:
$$\text{Contribution}_j = x_j \times w_{c, j}$$
where $x_j$ is the TF-IDF weight of token $j$ and $w_{c, j}$ is the learned model coefficient for predicted class $c$. The application surfaces the top contributing terms as interactive visual signals.

---

## Measured Evaluation & Benchmark Results

Evaluation was conducted on the held-out test set (20% split, $n = 18$) and through Stratified 5-Fold Cross-Validation on the training set ($n = 69$).

### Held-Out Test Set Performance ($n = 18$)

| Metric | Measured Value | Notes |
|--------|----------------|-------|
| **Accuracy** | **72.22%** | Overall correct classifications (13/18) |
| **Weighted F1** | **72.93%** | Weighted average across class support |
| **Macro F1** | **67.27%** | Unweighted average across classes; critical for imbalanced data |
| **Weighted Precision** | **74.07%** | Precision weighted by class prevalence |
| **Weighted Recall** | **72.22%** | Recall weighted by class prevalence |

### Per-Class Classification Report

```text
              precision    recall  f1-score   support

    negative       0.83      0.77      0.80        13
    positive       0.50      0.60      0.55         5

    accuracy                           0.72        18
   macro avg       0.67      0.68      0.67        18
weighted avg       0.74      0.72      0.73        18
```

### Baseline Model Comparison (Dummy Classifier)

In imbalanced classification, accuracy is often a misleading metric. We benchmark the model against a naive majority-class dummy classifier (`DummyClassifier(strategy="most_frequent")`):

| Model | Accuracy | Macro F1 | Weighted F1 | Positive F1 |
|-------|----------|----------|-------------|-------------|
| **Dummy Majority Predictor** | 72.22% | 41.94% | 60.57% | 0.00% |
| **SmartReview (TF-IDF + LR)** | **72.22%** | **67.27%** | **72.93%** | **55.00%** |
| **Net Difference** | **+0.00%** | **+25.33%** | **+12.36%** | **+55.00%** |

> **Critical ML Finding:** Because 72.2% (13/18) of the test set is negative, a naive dummy model that predicts "negative" for every review achieves 72.22% raw accuracy. However, its Macro F1 is only 41.94% because it has 0% recall and 0% precision on the positive class. SmartReview's Logistic Regression model achieves **67.27% Macro F1 (+25.33% lift)** and a **55.0% F1 score on the minority positive class**, proving genuine sentiment learning.

### Stratified 5-Fold Cross-Validation (Training Split, $n = 69$)

To eliminate single-split evaluation variance on small sample sizes, we evaluated the pipeline across 5 stratified folds:

- **CV Accuracy:** $81.10\% \pm 7.44\%$
- **CV Macro F1:** $76.41\% \pm 9.49\%$
- **CV Weighted F1:** $80.75\% \pm 7.62\%$

---

## Dataset Provenance & Limitations

- **Source:** Hand-curated demonstration dataset (`data/sample_reviews.csv`).
- **Sample Count:** 87 validated reviews.
- **Class Balance:** 62 negative (71.3%), 25 positive (28.7%).
- **Neutral Class:** The demonstration dataset is binary. In real-world customer reviews, neutral or mixed feedback represents a significant proportion. In the demo dataset, lukewarm reviews were mapped to negative.
- **Sample Size:** 87 rows is suitable for local portfolio demonstration, pipeline validation, and unit testing, but **not for production benchmark deployment**.
- **Recommended Production Datasets:**
  - [Stanford SST-2](https://huggingface.co/datasets/sst2) (67,000 sentences)
  - [Amazon US Reviews](https://huggingface.co/datasets/amazon_us_reviews)
  - [Yelp Reviews Full](https://huggingface.co/datasets/yelp_review_full)

---

## Local Setup & Installation

### Windows (PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/your-username/SmartReview.git
cd SmartReview

# 2. Create and activate virtual environment (optional if Python environment is ready)
python -m venv venv
venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### macOS / Linux

```bash
git clone https://github.com/your-username/SmartReview.git
cd SmartReview
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

### 1. Run Automated Test Suite (119 tests)

```powershell
python -m pytest
```

### 2. Retrain the Pipeline & Inspect Evaluation

```powershell
python train_pipeline.py
```

Optional custom dataset training:
```powershell
python train_pipeline.py --data path/to/dataset.csv --text-col review_text --label-col sentiment
```

### 3. Run Functional Prediction Verification

```powershell
python verify_predictions.py
```

### 4. Launch the Streamlit Web Application

```powershell
python -m streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## Optional: External Gemini API Setup (₹0 Free Tier)

SmartReview is **offline-first**: the local ML pipeline requires no API keys or internet connection.
If you wish to test LLM-based sentiment analysis, you can enable Google Gemini Developer API on the free tier (no credit card required):

1. Create a free API key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-3.8-flash
   ```
3. Restart Streamlit (`python -m streamlit run app.py`). The sidebar will automatically reveal the "External API (Google Gemini)" mode.

---

## Interview Talking Points: Key Engineering Decisions

If asked about this project in a machine-learning or software-engineering interview, here are the top 5 architectural decisions to highlight:

1. **Why TF-IDF + Logistic Regression over DistilBERT for local deployment?**
   - *Rationale:* Zero inference latency, sub-10ms predictions on commodity CPUs, zero GPU memory requirements, and 100% deterministic local inference without downloading multi-hundred-megabyte weights.
2. **Why is accuracy an inadequate evaluation metric here?**
   - *Rationale:* With 71.6% negative reviews, a dummy majority classifier achieves 72.22% test accuracy by doing nothing. Evaluating unweighted **Macro F1 (67.27% vs 41.94% baseline)** and minority class metrics exposes whether the model actually learned.
3. **How does the scikit-learn Pipeline prevent data leakage?**
   - *Rationale:* Fitting vectorizers before splitting causes vocabulary and inverse-document-frequency values from test data to contaminate the training set. Wrapping vectorizer and classifier in `Pipeline` guarantees strictly separated train/test transformations.
4. **How was small-sample variance addressed?**
   - *Rationale:* An 80/20 split on 87 samples leaves only 18 test reviews (where 1 sample shifts accuracy by 5.5%). We supplemented the held-out test split with **Stratified 5-Fold Cross-Validation** ($81.10\% \pm 7.44\%$) to measure consistency across subsets.
5. **How is local explainability achieved without heavy frameworks?**
   - *Rationale:* Rather than adding heavy external dependencies like SHAP or LIME, we exploited the mathematical properties of the linear model ($w_j \cdot x_j$) to extract exact n-gram contributions directly from the pipeline.

---

## License

This project is licensed under the MIT License.
