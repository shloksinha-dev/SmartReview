"""
Text preprocessing utilities for SmartReview.

Design decisions:
- Preserve negation words (not, no, never, don't, won't, etc.) so that
  "not good" is not collapsed to "good".
- Lowercase everything for consistency.
- Remove punctuation except apostrophes (needed for contractions).
- Strip excess whitespace.
- Do NOT use stemming/lemmatization on the first pass to avoid losing
  sentiment-bearing suffixes ("amazing" vs "amaze").
"""

import re
import string
from typing import List


# Negation words that MUST be preserved
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot",
    "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "mightn't", "mustn't", "needn't",
}

# Very common functional words to exclude from word-frequency reports
# (only used in dashboard word-cloud; NOT removed during ML feature extraction)
DASHBOARD_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "this", "that", "i", "my", "me",
    "was", "with", "very", "so", "be", "have", "had", "has",
    "its", "are", "as", "by", "up", "if", "he", "she", "they",
    "we", "you", "all", "about", "would", "could", "from", "than",
    "more", "been", "also", "just", "even", "after", "before",
    "out", "into", "about", "over", "there", "here", "what", "when",
    "which", "who", "how", "will", "can", "do", "did", "get", "got",
}


def clean_text(text: str) -> str:
    """
    Clean a single review string while preserving negation signals.

    Steps:
    1. Lowercase.
    2. Expand common contractions that contain negations.
    3. Remove URLs.
    4. Remove punctuation except apostrophes.
    5. Collapse whitespace.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Cleaned text ready for TF-IDF vectorization.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Expand contractions to preserve negation tokens
    contraction_map = {
        r"won't": "will not",
        r"can't": "cannot",
        r"n't": " not",          # covers don't, doesn't, didn't, etc.
        r"'re": " are",
        r"'ve": " have",
        r"'ll": " will",
        r"'d": " would",
        r"'m": " am",
    }
    for pattern, replacement in contraction_map.items():
        text = re.sub(pattern, replacement, text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags if any
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove punctuation (keep letters, digits, spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_batch(texts: List[str]) -> List[str]:
    """Apply clean_text to a list of strings."""
    return [clean_text(t) for t in texts]


def extract_top_words(
    texts: List[str],
    top_n: int = 20,
    exclude_stopwords: bool = True,
) -> List[tuple]:
    """
    Count word frequencies in a list of review strings.

    Used only for dashboard visualizations, NOT for ML feature extraction.

    Parameters
    ----------
    texts : list of str
    top_n : int
    exclude_stopwords : bool
        If True, removes common function words from counts.

    Returns
    -------
    list of (word, count) tuples sorted descending by count.
    """
    from collections import Counter

    counter: Counter = Counter()
    for text in texts:
        cleaned = clean_text(text)
        words = cleaned.split()
        for word in words:
            if len(word) < 2:
                continue
            if exclude_stopwords and word in DASHBOARD_STOPWORDS:
                continue
            counter[word] += 1

    return counter.most_common(top_n)
