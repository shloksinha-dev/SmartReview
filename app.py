"""
SmartReview — AI-Powered Review Sentiment Analyzer
Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import io
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend required for Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="SmartReview — Sentiment Analyzer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "SmartReview v1.0 — AI-Powered Review Sentiment Analyzer",
    },
)

# ── Imports from src ─────────────────────────────────────────────────────────
from src.predictor import predict_single, predict_batch, get_pipeline, reset_pipeline
from src.api_client import APIClient
from src.evaluation import load_evaluation_results, format_confusion_matrix
from src.train_model import load_metadata, model_exists
from src.preprocessing import extract_top_words
from src.data_loader import load_sample_dataset

logging.basicConfig(level=logging.WARNING)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_CSV_MB = 10
MAX_CSV_ROWS = 5_000
EXAMPLE_REVIEWS = [
    "This product is absolutely fantastic! Best purchase I've made in years.",
    "Terrible quality — it broke after two days and the seller never responded.",
    "The product works okay but it's a bit overpriced for what you get.",
    "I'm very impressed with the build quality and how fast it arrived.",
    "Do not buy this. It looked nothing like the photos and stopped working immediately.",
]

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#f59e0b",
}
SENTIMENT_EMOJIS = {
    "positive": "😊",
    "negative": "😞",
    "neutral":  "😐",
}


# ════════════════════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.metric-card .metric-label {
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
}

/* Sentiment result box */
.result-box {
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
    border: 2px solid;
}
.result-positive { background: rgba(34,197,94,0.08); border-color: #22c55e; }
.result-negative { background: rgba(239,68,68,0.08); border-color: #ef4444; }
.result-neutral  { background: rgba(245,158,11,0.08); border-color: #f59e0b; }

.result-label {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.result-confidence {
    font-size: 0.9rem;
    color: #94a3b8;
}
.result-disclaimer {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.6rem;
    font-style: italic;
}

/* Section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.2rem;
}
.section-sub {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 1.2rem;
}

/* Probability bar */
.prob-bar-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.3rem 0;
    font-size: 0.88rem;
}
.prob-label { width: 80px; color: #94a3b8; text-transform: capitalize; }
.prob-bar-bg {
    flex: 1;
    background: #1e293b;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 6px; }
.prob-value { width: 45px; text-align: right; color: #e2e8f0; }

/* API badge */
.api-badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-api  { background: #6366f1; color: #fff; }
.badge-local { background: #334155; color: #94a3b8; }

/* Warning banner */
.privacy-notice {
    background: rgba(245,158,11,0.1);
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  Session state helpers
# ════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "batch_results": None,
        "single_result": None,
        "single_text": "",
        "api_client": None,
        "pipeline_loaded": False,
        "session_analyses": [],   # [{text, label, confidence, source}]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ════════════════════════════════════════════════════════════════════════════
#  Model / API initialisation (cached)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading sentiment model…")
def load_ml_pipeline():
    """Load or auto-train the ML pipeline once per session."""
    return get_pipeline()

@st.cache_resource
def get_api_client():
    return APIClient()


# ════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔎 SmartReview")
        st.markdown("*AI-Powered Sentiment Analyzer*")
        st.divider()

        page = st.radio(
            "Navigation",
            ["🔍 Analyze Review", "📂 Batch Analysis", "📊 Dashboard", "ℹ️ Model Info"],
            label_visibility="collapsed",
        )

        st.divider()

        # Analysis mode
        api_client = get_api_client()
        st.markdown("**Analysis Mode**")
        if api_client.is_configured():
            mode = st.radio(
                "mode",
                ["Local ML Model", "External API (Google Gemini)"],
                label_visibility="collapsed",
            )
        else:
            mode = "Local ML Model"
            st.caption("🔒 Local model only  \n*(Set `GEMINI_API_KEY` to enable Gemini API)*")

        st.divider()

        # Session summary
        analyses = st.session_state.session_analyses
        if analyses:
            st.markdown("**This session**")
            counts = {}
            for a in analyses:
                lbl = a.get("label") or "unknown"
                counts[lbl] = counts.get(lbl, 0) + 1
            for lbl, n in sorted(counts.items()):
                color = SENTIMENT_COLORS.get(lbl, "#94a3b8")
                emoji = SENTIMENT_EMOJIS.get(lbl, "❓")
                st.markdown(
                    f"<span style='color:{color}'>{emoji} {lbl.capitalize()}: **{n}**</span>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.caption("SmartReview v1.0 · Portfolio demo  \nModel: TF-IDF + Logistic Regression")

    return page, mode


# ════════════════════════════════════════════════════════════════════════════
#  Helper renderers
# ════════════════════════════════════════════════════════════════════════════
def render_sentiment_result(result: dict, source: str = "local"):
    """Render a styled sentiment result card."""
    error = result.get("error")
    label = result.get("label")
    confidence = result.get("confidence")
    probabilities = result.get("probabilities", {})
    explanation = result.get("explanation")   # API mode
    if error:
        st.error(f"❌ {error}")
        if source == "api":
            st.info("💡 **Local ML Model available:** The local model runs 100% offline without external API dependencies. You can switch to 'Local ML Model' in the sidebar or analyze with the local model below.")
        return

    if label is None:
        st.warning("Could not determine sentiment.")
        return

    color = SENTIMENT_COLORS.get(label, "#94a3b8")
    emoji = SENTIMENT_EMOJIS.get(label, "❓")
    css_class = f"result-{label}"
    source_badge = (
        '<span class="api-badge badge-api">API</span>'
        if source == "api"
        else '<span class="api-badge badge-local">Local ML</span>'
    )

    conf_str = f"{confidence*100:.1f}%" if confidence is not None else "N/A"
    conf_note = (
        f"Model confidence: <b>{conf_str}</b>"
        if source == "local"
        else "Confidence: not available from API"
    )

    explanation_html = ""
    if explanation:
        if isinstance(explanation, dict):
            summary = explanation.get("summary", "")
            signals = explanation.get("signals", [])
            chips_html = ""
            if signals:
                chips = [
                    f"<span style='display:inline-block; margin:3px 6px 3px 0; padding:2px 8px; border-radius:10px; background:#1e293b; border:1px solid #334155; font-size:0.78rem; color:#f1f5f9; font-family:monospace;'>{s['term']} <b style='color:{color};'>+{s['score']}</b></span>"
                    for s in signals
                ]
                chips_html = "<div style='margin-top:0.4rem;'>" + "".join(chips) + "</div>"
            if summary:
                explanation_html = f"<div style='margin-top:0.6rem; font-size:0.88rem; color:#cbd5e1;'>💡 <b>Key Signals:</b> {summary}{chips_html}</div>"
        elif isinstance(explanation, str):
            explanation_html = f"<div style='margin-top:0.6rem; font-size:0.88rem; color:#cbd5e1;'>💡 <b>Explanation:</b> {explanation}</div>"

    st.markdown(
        f"""
        <div class="result-box {css_class}">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div class="result-label" style="color:{color};">
                    {emoji} {label.capitalize()}
                </div>
                {source_badge}
            </div>
            <div class="result-confidence">{conf_note}</div>
            {explanation_html}
            <div class="result-disclaimer">
                ⚠ This is a model prediction, not a verified ground-truth label.
                Confidence reflects the model's certainty, not factual accuracy.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability bars (local model only)
    if probabilities:
        st.markdown("**Class probabilities**")
        for cls in sorted(probabilities.keys()):
            prob = probabilities[cls]
            bar_color = SENTIMENT_COLORS.get(cls, "#94a3b8")
            pct = int(prob * 100)
            st.markdown(
                f"""
                <div class="prob-bar-row">
                    <span class="prob-label">{cls}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{pct}%; background:{bar_color};"></div>
                    </div>
                    <span class="prob-value">{prob:.2%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_pie_chart(counts: dict, title: str = "Sentiment Distribution"):
    """Render a matplotlib pie chart and return the figure."""
    labels = [k.capitalize() for k in counts.keys()]
    values = list(counts.values())
    colors = [SENTIMENT_COLORS.get(k, "#94a3b8") for k in counts.keys()]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="none")
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "#0f172a", "linewidth": 2},
    )
    for t in texts:
        t.set_color("#e2e8f0")
        t.set_fontsize(10)
    for at in autotexts:
        at.set_color("#0f172a")
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax.set_title(title, color="#f8fafc", fontsize=12, pad=10)
    fig.patch.set_alpha(0)
    return fig


def render_bar_chart(word_counts: list, title: str, color: str):
    """Render a horizontal bar chart of word frequencies."""
    if not word_counts:
        return None
    words, freqs = zip(*word_counts[:15])
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="none")
    y_pos = range(len(words))
    ax.barh(list(y_pos), list(freqs), color=color, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(list(words), color="#e2e8f0", fontsize=9)
    ax.set_xlabel("Frequency", color="#94a3b8", fontsize=9)
    ax.set_title(title, color="#f8fafc", fontsize=11)
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  Page: Analyze Review
# ════════════════════════════════════════════════════════════════════════════
def page_analyze(mode: str):
    st.markdown('<div class="section-header">🔍 Analyze a Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter a customer review and receive an instant sentiment prediction.</div>', unsafe_allow_html=True)

    api_client = get_api_client()

    # Privacy notice for API mode
    if mode == "External API (Google Gemini)":
        st.markdown(
            '<div class="privacy-notice">⚠ <b>Privacy notice:</b> In Gemini API mode, your review text '
            "is transmitted to Google's servers for processing. Do not submit sensitive or personal information.</div>",
            unsafe_allow_html=True,
        )

    # Example reviews
    with st.expander("📌 Load an example review"):
        for i, ex in enumerate(EXAMPLE_REVIEWS):
            if st.button(f"Example {i+1}", key=f"ex_{i}"):
                st.session_state.single_text = ex

    # Text area
    review_text = st.text_area(
        "Customer review",
        value=st.session_state.single_text,
        height=140,
        max_chars=10_000,
        placeholder="Type or paste a customer review here…",
        key="review_input",
    )

    col_analyze, col_clear = st.columns([2, 1])
    with col_analyze:
        analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.single_text = ""
            st.session_state.single_result = None
            st.rerun()

    # Run analysis
    if analyze_clicked:
        text = review_text.strip()
        if not text:
            st.warning("Please enter a review before clicking Analyze.")
        elif len(text) < 3:
            st.warning("Review is too short. Please enter at least 3 characters.")
        else:
            with st.spinner("Analysing…"):
                if mode == "External API (Google Gemini)":
                    result = api_client.analyze(text)
                    src = "api"
                else:
                    result = predict_single(text)
                    src = "local"

                st.session_state.single_result = (result, src)
                st.session_state.single_text = review_text

                # Record in session history
                if not result.get("error"):
                    st.session_state.session_analyses.append({
                        "text": text[:80] + ("…" if len(text) > 80 else ""),
                        "label": result.get("label"),
                        "confidence": result.get("confidence"),
                        "source": src,
                    })

    # Display result
    if st.session_state.single_result:
        result, src = st.session_state.single_result
        st.divider()
        render_sentiment_result(result, source=src)
        if src == "api" and result.get("error") and review_text.strip():
            if st.button("⚡ Analyze with Local ML Model instead", use_container_width=True):
                local_res = predict_single(review_text.strip())
                st.session_state.single_result = (local_res, "local")
                if not local_res.get("error"):
                    st.session_state.session_analyses.append({
                        "text": review_text.strip()[:80] + ("…" if len(review_text.strip()) > 80 else ""),
                        "label": local_res.get("label"),
                        "confidence": local_res.get("confidence"),
                        "source": "local",
                    })
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  Page: Batch Analysis
# ════════════════════════════════════════════════════════════════════════════
def page_batch():
    st.markdown('<div class="section-header">📂 Batch CSV Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a CSV file containing customer reviews for bulk sentiment analysis.</div>', unsafe_allow_html=True)

    # Template download
    sample_df = pd.DataFrame({
        "review_text": [
            "This product is great!",
            "Terrible experience, never buying again.",
            "It is okay, nothing special.",
        ]
    })
    csv_template = sample_df.to_csv(index=False)
    st.download_button(
        "⬇ Download CSV template",
        data=csv_template,
        file_name="review_template.csv",
        mime="text/csv",
    )

    st.divider()

    uploaded = st.file_uploader(
        f"Upload CSV (max {MAX_CSV_MB} MB, up to {MAX_CSV_ROWS:,} rows)",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.info("Upload a CSV file to begin batch analysis.")
        return

    # Size check
    size_mb = uploaded.size / 1_048_576
    if size_mb > MAX_CSV_MB:
        st.error(f"File is {size_mb:.1f} MB — maximum allowed is {MAX_CSV_MB} MB.")
        return

    # Read CSV
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    st.caption(f"File loaded: **{len(df_raw):,} rows**, **{len(df_raw.columns)} columns**")

    # Column selection
    text_col = st.selectbox(
        "Select the column containing review text",
        options=list(df_raw.columns),
        index=0,
    )

    if text_col not in df_raw.columns:
        st.error("Selected column not found.")
        return

    if st.button("▶ Run Batch Analysis", type="primary"):
        if len(df_raw) > MAX_CSV_ROWS:
            st.warning(f"File has {len(df_raw):,} rows. Only the first {MAX_CSV_ROWS:,} will be processed.")
            df_work = df_raw.head(MAX_CSV_ROWS).copy()
        else:
            df_work = df_raw.copy()

        # Sanitize missing or NaN values to empty strings so validator catches them
        texts = df_work[text_col].fillna("").astype(str).tolist()

        with st.spinner(f"Analysing {len(texts):,} reviews…"):
            results_df = predict_batch(texts)

        # Merge results back without duplicate column collisions
        df_work = df_work.reset_index(drop=True)
        results_df = results_df.reset_index(drop=True)
        cols_to_drop = [
            c for c in results_df.columns
            if c in df_work.columns and c != "review_text"
        ]
        if cols_to_drop:
            df_work = df_work.drop(columns=cols_to_drop)
        output_df = pd.concat([df_work, results_df.drop(columns=["review_text"])], axis=1)

        valid_mask = output_df["is_valid"] == True
        skipped = int((~valid_mask).sum())

        st.session_state.batch_results = {
            "df": output_df,
            "skipped": skipped,
            "total": len(output_df),
        }

    if st.session_state.batch_results:
        batch = st.session_state.batch_results
        output_df = batch["df"]
        skipped = batch["skipped"]
        total = batch["total"]

        st.divider()

        # Summary metrics
        valid_df = output_df[output_df["is_valid"] == True]
        dist = valid_df["predicted_sentiment"].value_counts().to_dict()

        cols = st.columns(4)
        metrics = [
            ("Total Rows", total, "#6366f1"),
            ("Analysed", len(valid_df), "#22c55e"),
            ("Skipped / Invalid", skipped, "#ef4444"),
        ]
        for col, (lbl, val, color) in zip(cols, metrics):
            with col:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value" style="color:{color};">{val}</div>
                        <div class="metric-label">{lbl}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        with cols[3]:
            pos_pct = dist.get("positive", 0) / max(len(valid_df), 1) * 100
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value" style="color:#22c55e;">{pos_pct:.0f}%</div>
                    <div class="metric-label">Positive Rate</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Pie chart + distribution table
        col_chart, col_table = st.columns([1, 1])
        with col_chart:
            if dist:
                fig = render_pie_chart(dist)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        with col_table:
            st.markdown("**Sentiment breakdown**")
            breakdown = pd.DataFrame([
                {"Sentiment": k.capitalize(), "Count": v, "Percentage": f"{v/len(valid_df)*100:.1f}%"}
                for k, v in sorted(dist.items())
            ])
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

        # Filters
        st.divider()
        st.markdown("**Filter results**")
        filter_opts = ["All"] + [s.capitalize() for s in sorted(dist.keys())]
        selected_filter = st.selectbox("Sentiment filter", filter_opts)

        display_df = output_df.copy()
        if selected_filter != "All":
            display_df = display_df[
                display_df["predicted_sentiment"].str.capitalize() == selected_filter
            ]

        # Preview
        st.markdown(f"**Results preview** ({len(display_df):,} rows)")
        show_cols = [text_col, "predicted_sentiment", "confidence"]
        avail_cols = [c for c in show_cols if c in display_df.columns]
        st.dataframe(
            display_df[avail_cols].rename(columns={
                text_col: "Review",
                "predicted_sentiment": "Sentiment",
                "confidence": "Confidence",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Download
        csv_out = output_df.to_csv(index=False)
        st.download_button(
            "⬇ Download full results CSV",
            data=csv_out,
            file_name="smartreview_results.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════════════════
#  Page: Dashboard
# ════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

    # Collect data from session + batch results
    session_data = st.session_state.session_analyses
    batch_data = st.session_state.batch_results

    has_session = len(session_data) > 0
    has_batch = batch_data is not None

    if not has_session and not has_batch:
        st.info(
            "No analysis data yet. Go to **Analyze Review** or **Batch Analysis** to generate results."
        )
        return

    # Build unified DataFrame
    frames = []

    if has_session:
        sess_df = pd.DataFrame(session_data)
        sess_df["source"] = "single"
        frames.append(sess_df)

    if has_batch:
        bd = batch_data["df"]
        valid = bd[bd["is_valid"] == True][["review_text", "predicted_sentiment", "confidence"]].copy()
        valid.columns = ["text", "label", "confidence"]
        valid["source"] = "batch"
        frames.append(valid)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["label"])

    if len(all_df) == 0:
        st.info("No valid predictions to display.")
        return

    dist = all_df["label"].value_counts().to_dict()
    total = len(all_df)

    # ── Top-level metrics ────────────────────────────────────────────────────
    cols = st.columns(len(dist) + 1)
    metric_data = [("Total Reviews", total, "#6366f1")] + [
        (f"{k.capitalize()} Reviews", v, SENTIMENT_COLORS.get(k, "#94a3b8"))
        for k, v in sorted(dist.items())
    ]
    for col, (lbl, val, color) in zip(cols, metric_data):
        with col:
            pct = f"({val/total*100:.0f}%)" if lbl != "Total Reviews" else ""
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value" style="color:{color};">{val} {pct}</div>
                    <div class="metric-label">{lbl}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    col_pie, col_bar = st.columns([1, 1])

    with col_pie:
        st.markdown("**Sentiment Distribution**")
        fig = render_pie_chart(dist)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_bar:
        st.markdown("**Review Count by Source**")
        source_dist = all_df["source"].value_counts().to_dict()
        fig2, ax2 = plt.subplots(figsize=(4, 3), facecolor="none")
        ax2.bar(
            [s.capitalize() for s in source_dist.keys()],
            list(source_dist.values()),
            color=["#6366f1", "#22c55e"],
        )
        ax2.set_ylabel("Count", color="#94a3b8")
        ax2.tick_params(colors="#94a3b8")
        ax2.spines[:].set_color("#334155")
        ax2.set_facecolor("none")
        fig2.patch.set_alpha(0)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── Confidence distribution ──────────────────────────────────────────────
    if "confidence" in all_df.columns:
        conf_data = all_df["confidence"].dropna()
        if len(conf_data) > 0:
            st.divider()
            st.markdown("**Model Confidence Distribution**")
            fig3, ax3 = plt.subplots(figsize=(8, 2.5), facecolor="none")
            ax3.hist(
                conf_data.astype(float),
                bins=20,
                color="#6366f1",
                alpha=0.8,
                edgecolor="#0f172a",
            )
            ax3.set_xlabel("Confidence", color="#94a3b8")
            ax3.set_ylabel("Count", color="#94a3b8")
            ax3.tick_params(colors="#94a3b8")
            ax3.spines[:].set_color("#334155")
            ax3.set_facecolor("none")
            fig3.patch.set_alpha(0)
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

    # ── Top words per sentiment ──────────────────────────────────────────────
    st.divider()
    st.markdown("**Most Frequent Words by Sentiment**")
    word_cols = st.columns(len(dist))
    for col, (sentiment, _) in zip(word_cols, sorted(dist.items())):
        with col:
            subset = all_df[all_df["label"] == sentiment]["text"].dropna().tolist()
            top_words = extract_top_words(subset, top_n=12)
            color = SENTIMENT_COLORS.get(sentiment, "#94a3b8")
            if top_words:
                fig_w = render_bar_chart(
                    top_words,
                    title=f"{sentiment.capitalize()} Reviews",
                    color=color,
                )
                if fig_w:
                    st.pyplot(fig_w, use_container_width=True)
                    plt.close(fig_w)
            else:
                st.caption(f"No word data for {sentiment}.")

    # ── Data table ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**All Analysed Reviews**")

    filter_col, _ = st.columns([1, 2])
    with filter_col:
        filter_sent = st.selectbox(
            "Filter by sentiment",
            ["All"] + [s.capitalize() for s in sorted(dist.keys())],
            key="dash_filter",
        )

    display = all_df.copy()
    if filter_sent != "All":
        display = display[display["label"].str.capitalize() == filter_sent]

    st.dataframe(
        display[["text", "label", "confidence", "source"]].rename(columns={
            "text": "Review",
            "label": "Sentiment",
            "confidence": "Confidence",
            "source": "Source",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════════════
#  Page: Model Info
# ════════════════════════════════════════════════════════════════════════════
def page_model_info():
    st.markdown('<div class="section-header">ℹ️ Model Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Technical details, evaluation results, and known limitations.</div>', unsafe_allow_html=True)

    # Model architecture
    with st.expander("🧠 Model Architecture", expanded=True):
        st.markdown("""
**Pipeline:** TF-IDF Vectorizer → Logistic Regression

| Component | Setting | Reason |
|-----------|---------|--------|
| `TfidfVectorizer` | `ngram_range=(1,2)` | Captures bigrams like *not good* |
| `TfidfVectorizer` | `sublinear_tf=True` | Dampens effect of very frequent terms |
| `TfidfVectorizer` | `max_features=50,000` | Caps memory usage |
| `TfidfVectorizer` | `min_df=2` | Ignores very rare terms |
| `LogisticRegression` | `class_weight='balanced'` | Handles class imbalance |
| `LogisticRegression` | `max_iter=1000` | Avoids convergence warnings |
| `LogisticRegression` | `C=1.0` | Default regularisation |

Preprocessing inside the pipeline: lowercasing, contraction expansion,
URL/HTML removal, punctuation stripping.
        """)

    # Training metadata
    metadata = load_metadata()
    if metadata:
        with st.expander("📋 Training Metadata"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Samples", metadata.get("training_samples", "—"))
                st.metric("Classes", ", ".join(metadata.get("classes", [])))
            with col2:
                st.metric("Random Seed", metadata.get("random_state", "—"))
                st.metric("Dataset", metadata.get("dataset", "—"))

    # Evaluation results
    eval_results = load_evaluation_results()
    if eval_results:
        with st.expander("📈 Evaluation Results (held-out test set)", expanded=True):
            st.info(
                "⚠ These metrics reflect performance on the **bundled sample demonstration dataset** (88 rows). "
                "Because of class imbalance (71% negative), **Macro F1** is the most honest metric to evaluate."
            )
            cols = st.columns(4)
            with cols[0]:
                st.metric("Test Accuracy", f"{eval_results.get('accuracy', 0):.4f}")
            with cols[1]:
                st.metric("Weighted F1", f"{eval_results.get('f1', 0):.4f}")
            with cols[2]:
                macro_f1 = eval_results.get('macro_f1')
                st.metric("Macro F1", f"{macro_f1:.4f}" if macro_f1 is not None else "—")
            with cols[3]:
                st.metric("Test Samples", eval_results.get("test_samples", "—"))

            # Baseline Comparison
            baseline = eval_results.get("baseline")
            if baseline:
                st.markdown("#### ⚖️ Baseline Model Comparison (Dummy Classifier)")
                st.caption(
                    "Comparing against a naive majority-class classifier (`DummyClassifier(strategy='most_frequent')`). "
                    "In an imbalanced dataset, raw accuracy can be deeply misleading."
                )
                b_cols = st.columns(3)
                with b_cols[0]:
                    st.metric(
                        "Baseline Accuracy",
                        f"{baseline.get('accuracy', 0):.4f}",
                        help="Always predicting the majority class (negative)",
                    )
                with b_cols[1]:
                    st.metric(
                        "Baseline Macro F1",
                        f"{baseline.get('macro_f1', 0):.4f}",
                        help="Unweighted F1 of majority guesser",
                    )
                with b_cols[2]:
                    f1_diff = (eval_results.get('macro_f1', 0) - baseline.get('macro_f1', 0))
                    st.metric(
                        "Model Macro F1 Gain",
                        f"{eval_results.get('macro_f1', 0):.4f}",
                        delta=f"{f1_diff:+.4f} vs Baseline",
                    )
                st.markdown(
                    "<small style='color:#94a3b8;'>Notice: Because 72% of the test set is negative, a dummy predictor also achieves 72.22% accuracy. "
                    "However, its Macro F1 is only 41.94% (0% on positive). SmartReview's Logistic Regression achieves 67.27% Macro F1, "
                    "demonstrating genuine sentiment learning across both classes.</small>",
                    unsafe_allow_html=True,
                )

            # Cross-Validation
            cv = eval_results.get("cross_validation")
            if cv:
                st.markdown(f"#### 🔄 Stratified {cv.get('folds', 5)}-Fold Cross-Validation (Training Data)")
                st.caption("Guards against high variance from a single 80/20 train/test split on a small dataset.")
                c_cols = st.columns(3)
                with c_cols[0]:
                    st.metric("CV Accuracy", f"{cv.get('accuracy_mean', 0):.4f} ± {cv.get('accuracy_std', 0):.4f}")
                with c_cols[1]:
                    st.metric("CV Macro F1", f"{cv.get('f1_macro_mean', 0):.4f} ± {cv.get('f1_macro_std', 0):.4f}")
                with c_cols[2]:
                    st.metric("CV Weighted F1", f"{cv.get('f1_weighted_mean', 0):.4f} ± {cv.get('f1_weighted_std', 0):.4f}")

            st.markdown("#### 📋 Detailed Classification Report")
            report_text = eval_results.get("classification_report_text", "Not available.")
            st.code(report_text, language="text")

            # Confusion matrix
            cm_data = eval_results.get("confusion_matrix")
            cm_labels = eval_results.get("confusion_matrix_labels")
            if cm_data and cm_labels:
                st.markdown("#### 🎯 Confusion Matrix")
                cm_df = pd.DataFrame(
                    cm_data,
                    index=[f"Actual: {l}" for l in cm_labels],
                    columns=[f"Predicted: {l}" for l in cm_labels],
                )
                st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("No evaluation results found. Run `python train_pipeline.py` to generate them.")

    # Model Explainability & Interpretability
    with st.expander("🔍 Model Explainability (Feature Attribution)"):
        st.markdown("""
**How Local Interpretability Works in SmartReview:**

SmartReview uses a transparent, white-box linear architecture:
1. **TF-IDF Vectorization with Negation Bigrams:** The review is tokenized and transformed into TF-IDF term weights (unigrams and bigrams like `not good`).
2. **Logistic Regression Weights:** For any input review, we compute the linear contribution of each active n-gram:
   $$\\text{Contribution}_j = x_j \\times w_{c, j}$$
   where $x_j$ is the TF-IDF feature weight and $w_{c, j}$ is the learned model coefficient for the predicted class $c$.
3. **Interactive Attribution:** The top contributing terms are displayed in real-time in the **Analyze Review** tab, providing verifiable rationale for every classification.
        """)

    # Dataset info
    with st.expander("📦 Dataset Information"):
        st.markdown("""
**Source:** Bundled demonstration dataset (`data/sample_reviews.csv`)

**Important limitations:**
- This is a **small, hand-curated sample** dataset created for portfolio demonstration.
- It contains approximately 85 reviews — this is far too small for production use.
- The vocabulary is limited and the model may not generalise to real-world reviews.
- No external benchmark dataset is used by default (avoids download requirements).
- Binary classification only (positive/negative). Neutral reviews use negative class data for demonstration.

**For production use**, consider replacing with:
- [Stanford SST-2](https://huggingface.co/datasets/sst2) (67K movie reviews)
- [Amazon Product Reviews](https://huggingface.co/datasets/amazon_us_reviews)
- [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full)
        """)

    # Limitations
    with st.expander("⚠️ Known Limitations"):
        st.markdown("""
1. **Small training set:** The bundled dataset (~85 reviews) is a demonstration only.
2. **Binary focus:** The model is trained on positive/negative labels. Neutral detection is not supported by the current dataset.
3. **Domain sensitivity:** Trained on general product reviews; may perform poorly on domain-specific text (medical, legal, technical).
4. **No multilingual support:** Only English reviews are handled.
5. **No context window:** Each review is processed independently.
6. **Confidence ≠ correctness:** High confidence does not guarantee the prediction is correct.
7. **No retraining mechanism in the UI:** Retrain via `python train_pipeline.py`.
        """)

    # API info
    api_client = get_api_client()
    with st.expander("🔌 External API Integration (Google Gemini)"):
        if api_client.is_configured():
            st.success("✅ Google Gemini API is configured and ready.")
        else:
            st.info("ℹ️ Google Gemini API key not detected.")
        st.markdown("""
**Provider:** Google Gemini Developer API (free tier at ₹0)
**Default Model:** `gemini-3.8-flash`

**₹0 Free-Tier Setup (No Credit Card / Billing Required):**
1. Visit [Google AI Studio](https://aistudio.google.com/apikey).
2. Sign in with your Google account and click **Create API Key**.
3. Copy the key and add it to `.env` in the project root:
   ```
   GEMINI_API_KEY=your-gemini-api-key-here
   GEMINI_MODEL=gemini-3.8-flash
   ```
4. Restart Streamlit or reload the page.

**Privacy & Security:**
- **Local ML Model:** Runs 100% locally on your machine. No review text ever leaves your device.
- **Gemini API Mode:** Review text is transmitted to Google's servers under Google AI Studio terms. Do not submit sensitive or private personal information.
- **Key Safety:** Your API key is loaded strictly from environment variables or Streamlit secrets and is never exposed in the interface or committed to version control.
        """)


# ════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ════════════════════════════════════════════════════════════════════════════
def main():
    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    # Ensure the pipeline is warm on first load
    try:
        load_ml_pipeline()
        st.session_state.pipeline_loaded = True
    except Exception as exc:
        st.error(f"Failed to load ML model: {exc}")

    page, mode = render_sidebar()

    if page == "🔍 Analyze Review":
        page_analyze(mode)
    elif page == "📂 Batch Analysis":
        page_batch()
    elif page == "📊 Dashboard":
        page_dashboard()
    elif page == "ℹ️ Model Info":
        page_model_info()


if __name__ == "__main__":
    main()
