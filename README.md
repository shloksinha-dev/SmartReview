# 🔎 SmartReview — AI-Powered Review Sentiment Analyzer

A portfolio-quality web application that analyses customer reviews and classifies their sentiment using a machine-learning pipeline and an optional external AI API.

---

## Features

| Feature | Description |
|---------|-------------|
| **Single Review Analysis** | Paste any review and get an instant positive / negative prediction with confidence scores |
| **Batch CSV Analysis** | Upload a CSV file with hundreds of reviews; download enriched results |
| **Analytics Dashboard** | Sentiment distribution charts, word-frequency breakdown, confidence histogram |
| **ML Pipeline** | TF-IDF vectorizer + Logistic Regression, trained and evaluated transparently |
| **Optional API Mode** | Google Gemini (gemini-3.8-flash) integration for LLM-powered sentiment (free tier at ₹0 via Google AI Studio) |
| **Model Info Page** | Evaluation metrics, confusion matrix, architecture details, and honest limitations |
| **Professional UI** | Dark-slate theme, sidebar navigation, styled metric cards |

---

## Screenshots

> *(Replace these placeholders with real screenshots once the app is running.)*

| Page | Preview |
|------|---------|
| Analyze Review | `screenshots/analyze.png` |
| Batch Analysis | `screenshots/batch.png` |
| Dashboard | `screenshots/dashboard.png` |
| Model Info | `screenshots/model_info.png` |

---

## Technology Stack

- **Python 3.11+**
- **Streamlit** — web framework
- **scikit-learn** — TF-IDF + Logistic Regression pipeline
- **pandas / NumPy** — data handling
- **matplotlib / seaborn** — visualisations
- **joblib** — model serialisation
- **python-dotenv** — environment variable loading
- **google-genai** — optional Google Gemini API integration (₹0 free tier)
- **pytest** — automated testing

---

## Project Structure

```
SmartReview/
├── app.py                          # Streamlit application entry point
├── train_pipeline.py               # Standalone training script
├── requirements.txt
├── pytest.ini
├── README.md
├── .gitignore
├── .env.example                    # Environment variable template
├── data/
│   └── sample_reviews.csv          # Bundled demonstration dataset
├── models/
│   ├── .gitkeep
│   ├── sentiment_pipeline.joblib   # Generated after training
│   ├── training_metadata.json      # Generated after training
│   └── evaluation_results.json     # Generated after training
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Dataset loading and validation
│   ├── preprocessing.py            # Text cleaning (negation-aware)
│   ├── train_model.py              # Pipeline definition, save/load
│   ├── predictor.py                # Single and batch prediction
│   ├── api_client.py               # Google Gemini API integration
│   └── evaluation.py              # Metrics, confusion matrix, reporting
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_data_loader.py
│   ├── test_predictor.py
│   ├── test_api_client.py
│   └── test_training.py
└── .streamlit/
    ├── config.toml                 # Dark theme configuration
    └── secrets.toml.example       # Streamlit Cloud secrets template
```

---

## Local Installation

### Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/your-username/SmartReview.git
cd SmartReview

# 2. Create a virtual environment
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

## Dataset

**Source:** Bundled demonstration dataset (`data/sample_reviews.csv`).

This is a **small, hand-curated sample** (~85 rows) intended solely for portfolio demonstration. It is NOT a production benchmark.

**Limitations:**
- Too small for reliable real-world generalisation.
- Vocabulary is narrow (general product reviews only).
- Binary labels only (positive / negative).

**For production use**, replace with:
- [Stanford SST-2](https://huggingface.co/datasets/sst2) — 67K movie reviews
- [Amazon US Reviews](https://huggingface.co/datasets/amazon_us_reviews)
- [Yelp Review Full](https://huggingface.co/datasets/yelp_review_full)

---

## Model Training

Run the standalone training script from the project root:

```powershell
# Windows
python train_pipeline.py

# With a custom dataset
python train_pipeline.py --data path/to/your/reviews.csv --text-col review --label-col label
```

This will:
1. Load and validate the dataset.
2. Split into 80% train / 20% test (stratified where possible).
3. Fit the TF-IDF + Logistic Regression pipeline on training data **only**.
4. Evaluate on the held-out test set.
5. Save `models/sentiment_pipeline.joblib` and `models/evaluation_results.json`.
6. Print a classification report to the console.

The model is also trained automatically on first app launch if no saved model is found.

---

## Model Evaluation

Evaluation is performed on a **held-out test set** (never seen during training) using:
- Accuracy
- Precision, Recall, F1 (weighted average)
- Per-class classification report
- Confusion matrix

> ⚠ **Important:** Metrics reflect the bundled sample dataset only.
> Do not interpret these numbers as production-level performance claims.

---

## Running the Application

```powershell
# Windows — from the project root
python -m streamlit run app.py

# If streamlit is on your PATH
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## CSV Upload

1. Go to **Batch Analysis** in the sidebar.
2. Download the CSV template if you don't have a file ready.
3. Upload your CSV (max 10 MB, up to 5,000 rows).
4. Select the column containing review text.
5. Click **Run Batch Analysis**.
6. Filter, preview, and download the enriched results CSV.

**Required CSV format:**
```csv
review_text
"This product is fantastic!"
"Terrible quality, broke immediately."
```
*(The column name can be anything — you select it in the UI.)*

---

## Optional API Configuration (₹0 Free Tier)

The Google Gemini API integration is entirely optional. The local ML model is always available by default and runs 100% locally on your machine.

Google provides a generous free tier via Google AI Studio that requires **no credit card, no billing, and ₹0 cost**.

```powershell
# 1. Copy the example environment file
cp .env.example .env

# 2. Get a free API key at: https://aistudio.google.com/apikey

# 3. Edit .env and paste your key
#    GEMINI_API_KEY=your-gemini-api-key-here
#    GEMINI_MODEL=gemini-3.8-flash   (optional, this is the default)

# 4. Restart the app
python -m streamlit run app.py
```

When `GEMINI_API_KEY` is detected, an **External API (Google Gemini)** option appears in the sidebar. If the key is absent, the app seamlessly defaults to the local ML model.

**Privacy notice:** In Gemini API mode, review text is transmitted to Google's servers under Google AI Studio terms. Do not submit sensitive or personal information.

---

## Privacy Considerations

| Mode | Data stays local? | Data sent externally? | Cost |
|------|-------------------|----------------------|------|
| Local ML (Default) | ✅ Yes | ❌ No | ₹0 (Offline) |
| External API (Google Gemini) | ❌ No | ✅ Yes — sent to Google | ₹0 (Free tier) |

---

## Known Limitations

1. Small training set (~85 reviews) — curated for portfolio demonstration, not production benchmarks.
2. Binary classification only (positive / negative) with current local dataset.
3. English language only.
4. No context: each review is processed independently.
5. High model confidence does not guarantee factual correctness.
6. External API mode depends on Google AI Studio rate limits and quotas.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'streamlit'`**
→ Activate your virtual environment and run `pip install -r requirements.txt`.

**`streamlit: command not found`**
→ Use `python -m streamlit run app.py` instead.

**App shows "No trained model found"**
→ Run `python train_pipeline.py` once from the project root.

**API mode says "not configured"**
→ Ensure `.env` exists, contains `GEMINI_API_KEY=...`, and the app was restarted.

**CSV upload fails with encoding error**
→ Save your CSV as UTF-8 from Excel (File → Save As → CSV UTF-8).

---

## Deployment — Streamlit Community Cloud

1. Push the repository to GitHub (ensure `.env` and `secrets.toml` remain excluded in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository, branch `main`, and entry file `app.py`.
4. In **Advanced settings → Secrets**, add (optional):
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key-here"
   GEMINI_MODEL = "gemini-3.8-flash"
   ```
5. Click **Deploy**.

> ⚠ The trained model files (`models/*.joblib`) must either be committed to the repository or the app will auto-train on first launch (slower cold start).

---

## Running Tests

```powershell
python -m pytest
```

Expected output: all tests pass.

To run a specific file:
```powershell
python -m pytest tests/test_preprocessing.py -v
```

---

## Future Improvements

1. **Larger dataset integration** — add a one-click download of SST-2 or Amazon Reviews for better generalisation.
2. **BERT/transformer fine-tuning** — replace Logistic Regression with a fine-tuned DistilBERT model for significantly higher accuracy.
3. **Neutral class support** — collect or source a 3-class labeled dataset and retrain.
4. **Aspect-based sentiment** — identify which product aspects (shipping, quality, price) drive positive/negative sentiment.
5. **Export to PDF** — generate a formatted report from the dashboard.

---

## License

MIT License. See `LICENSE` for details.

---

*Built as a portfolio project demonstrating ML pipeline construction, Streamlit UI design, API integration, and software testing.*
