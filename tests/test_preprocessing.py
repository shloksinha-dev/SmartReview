"""
Tests for src/preprocessing.py
"""

import pytest
from src.preprocessing import clean_text, preprocess_batch, extract_top_words


class TestCleanText:
    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_preserves_negation_not(self):
        result = clean_text("This is not good at all.")
        assert "not" in result

    def test_preserves_negation_never(self):
        result = clean_text("I never liked this product.")
        assert "never" in result

    def test_expands_dont_contraction(self):
        result = clean_text("I don't like it.")
        assert "not" in result

    def test_expands_wont_contraction(self):
        result = clean_text("I won't buy again.")
        assert "not" in result or "will" in result

    def test_expands_cant_contraction(self):
        result = clean_text("I can't believe it.")
        assert "not" in result or "cannot" in result

    def test_removes_url(self):
        result = clean_text("Visit https://example.com for more.")
        assert "https" not in result
        assert "example" not in result

    def test_removes_html_tags(self):
        result = clean_text("<b>Great</b> product!")
        assert "<b>" not in result
        assert "great" in result

    def test_removes_punctuation(self):
        result = clean_text("Wow!!! Amazing... product??")
        assert "!" not in result
        assert "?" not in result
        assert "." not in result

    def test_collapses_whitespace(self):
        result = clean_text("  too   many   spaces  ")
        assert "  " not in result
        assert result == result.strip()

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_numeric_string(self):
        result = clean_text("Rating: 4/5 stars")
        assert isinstance(result, str)

    def test_whitespace_only(self):
        result = clean_text("   ")
        assert result == ""

    def test_real_positive_review(self):
        text = "This product is absolutely amazing! Best purchase I've made."
        result = clean_text(text)
        assert "amazing" in result
        assert "best" in result

    def test_real_negative_review_preserves_not_good(self):
        text = "This is not good at all, I don't recommend it."
        result = clean_text(text)
        assert "not" in result


class TestPreprocessBatch:
    def test_returns_list(self):
        result = preprocess_batch(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_empty_list(self):
        result = preprocess_batch([])
        assert result == []

    def test_handles_none_in_list(self):
        result = preprocess_batch([None, "hello"])
        assert result[0] == ""
        assert "hello" in result[1]

    def test_all_cleaned(self):
        texts = ["HELLO WORLD", "I don't like it"]
        result = preprocess_batch(texts)
        assert result[0] == "hello world"
        assert "not" in result[1]


class TestExtractTopWords:
    def test_returns_list_of_tuples(self):
        result = extract_top_words(["great product amazing"])
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], tuple)
            assert len(result[0]) == 2

    def test_empty_list(self):
        result = extract_top_words([])
        assert result == []

    def test_top_n_respected(self):
        texts = ["apple banana cherry date elderberry fig grape"] * 5
        result = extract_top_words(texts, top_n=3)
        assert len(result) <= 3

    def test_stopwords_excluded_by_default(self):
        texts = ["the product is great the product is amazing"]
        result = extract_top_words(texts)
        words = [w for w, _ in result]
        assert "the" not in words
        assert "is" not in words

    def test_sorted_by_frequency(self):
        texts = ["great great great good good bad"]
        result = extract_top_words(texts)
        if len(result) >= 2:
            assert result[0][1] >= result[1][1]
