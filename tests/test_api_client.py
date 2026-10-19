"""
Tests for src/api_client.py (Google Gemini API integration).

All tests use mocks — no real API calls or keys are made.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.api_client import APIClient


class TestAPIClientConfiguration:
    def test_not_configured_without_key(self):
        """Client should not be configured when GEMINI_API_KEY is absent."""
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("GEMINI_API_KEY", None)
            client = APIClient()
            assert client.is_configured() is False

    def test_configured_with_env_key(self):
        """Client should be configured when GEMINI_API_KEY is set."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "AIzaSyFakeTestKey12345"}):
            with patch("src.api_client.APIClient._init_client"):
                client = APIClient()
                client._api_key = "AIzaSyFakeTestKey12345"
                client._client = MagicMock()
                assert client.is_configured() is True

    def test_analyze_returns_error_when_not_configured(self):
        """analyze() should return an error dict when no API key is present."""
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("GEMINI_API_KEY", None)
            client = APIClient()
            result = client.analyze("Test review text")
            assert result["error"] is not None
            assert "GEMINI_API_KEY" in result["error"]
            assert result["label"] is None

    def test_analyze_result_schema(self):
        """analyze() should always return a dict with required keys."""
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("GEMINI_API_KEY", None)
            client = APIClient()
            result = client.analyze("Test review")
            for key in ["label", "explanation", "source", "model", "confidence", "error"]:
                assert key in result, f"Missing key: {key}"

    def test_source_is_api(self):
        """source field should always be 'api'."""
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("GEMINI_API_KEY", None)
            client = APIClient()
            result = client.analyze("Test")
            assert result["source"] == "api"


class TestAPIClientResponseParsing:
    """Test the response parsing logic with mocked API calls."""

    def _make_configured_client(self):
        """Helper: create a client with a mocked API key and client."""
        client = APIClient()
        client._api_key = "AIzaSyMockKey"
        client._client = MagicMock()
        return client

    def test_parses_positive_label(self):
        client = self._make_configured_client()
        mock_response = json.dumps({"label": "positive", "explanation": "Customer expresses satisfaction."})
        with patch.object(client, "_call_api", return_value={"raw": mock_response, "error": None}):
            result = client.analyze("This is wonderful!")
            assert result["label"] == "positive"
            assert result["explanation"] == "Customer expresses satisfaction."
            assert result["error"] is None

    def test_parses_negative_label(self):
        client = self._make_configured_client()
        mock_response = json.dumps({"label": "negative", "explanation": "Product failed to function."})
        with patch.object(client, "_call_api", return_value={"raw": mock_response, "error": None}):
            result = client.analyze("This is terrible!")
            assert result["label"] == "negative"
            assert result["error"] is None

    def test_parses_neutral_label(self):
        client = self._make_configured_client()
        mock_response = json.dumps({"label": "neutral", "explanation": "Average feedback without strong polarity."})
        with patch.object(client, "_call_api", return_value={"raw": mock_response, "error": None}):
            result = client.analyze("It is okay I guess.")
            assert result["label"] == "neutral"
            assert result["error"] is None

    def test_handles_api_error(self):
        client = self._make_configured_client()
        with patch.object(client, "_call_api", return_value={"raw": None, "error": "Network or timeout error"}):
            result = client.analyze("Test review")
            assert result["error"] is not None
            assert result["label"] is None

    def test_handles_invalid_json_response(self):
        client = self._make_configured_client()
        with patch.object(client, "_call_api", return_value={"raw": "Not a valid json response", "error": None}):
            result = client.analyze("Test review")
            assert result["error"] is not None
            assert "Could not parse" in result["error"]

    def test_handles_markdown_wrapped_json(self):
        """Test markdown code fences like ```json ... ``` are cleanly parsed."""
        client = self._make_configured_client()
        mock_response = "```json\n" + json.dumps({"label": "positive", "explanation": "Great item."}) + "\n```"
        with patch.object(client, "_call_api", return_value={"raw": mock_response, "error": None}):
            result = client.analyze("Love it!")
            assert result["label"] == "positive"
            assert result["error"] is None

    def test_handles_unknown_label(self):
        client = self._make_configured_client()
        mock_response = json.dumps({"label": "ecstatic_joy", "explanation": "Very happy."})
        with patch.object(client, "_call_api", return_value={"raw": mock_response, "error": None}):
            result = client.analyze("Test")
            assert result["error"] is not None
            assert "Unrecognised label" in result["error"]
            assert result["label"] is None

    def test_label_normalisation(self):
        """'pos' should be normalised to 'positive'."""
        client = self._make_configured_client()
        mock_response = json.dumps({"label": "pos", "explanation": "Good."})
        with patch.object(client, "_call_api", return_value={"raw": mock_response, "error": None}):
            result = client.analyze("Test")
            assert result["label"] == "positive"

    def test_confidence_is_none_for_api(self):
        """API results never fabricate a confidence score."""
        client = self._make_configured_client()
        mock_response = json.dumps({"label": "positive", "explanation": "Good."})
        with patch.object(client, "_call_api", return_value={"raw": mock_response, "error": None}):
            result = client.analyze("Great!")
            assert result["confidence"] is None


class TestAPIClientCallHandling:
    """Test _call_api error handling for Google Gemini error types."""

    def test_default_model_is_gemini_3_8_flash(self):
        """Default model should be gemini-3.8-flash."""
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("GEMINI_MODEL", None)
            client = APIClient()
            assert client._model == "gemini-3.8-flash"

    def test_call_api_model_unavailable_error(self):
        """404 or model not found error should return distinct model unavailable message."""
        client = APIClient()
        client._client = MagicMock()
        mock_exc = Exception("404 NOT_FOUND: models/gemini-2.5-flash is not available to new users")
        mock_exc.code = 404
        client._client.models.generate_content.side_effect = mock_exc

        res = client._call_api("Nice product")
        assert res["raw"] is None
        assert "Model unavailable" in res["error"]
        assert "gemini-3.8-flash" in res["error"]
        assert "Authentication failed" not in res["error"]
        assert "Quota" not in res["error"]

    def test_call_api_auth_error(self):
        client = APIClient()
        client._client = MagicMock()
        mock_exc = Exception("API_KEY_INVALID: 401 UNAUTHENTICATED")
        mock_exc.code = 401
        client._client.models.generate_content.side_effect = mock_exc

        res = client._call_api("Nice product")
        assert res["raw"] is None
        assert "Authentication failed" in res["error"]
        assert "Model unavailable" not in res["error"]

    def test_call_api_quota_rate_limit(self):
        client = APIClient()
        client._client = MagicMock()
        mock_exc = Exception("RESOURCE_EXHAUSTED: Quota exceeded")
        mock_exc.code = 429
        client._client.models.generate_content.side_effect = mock_exc

        res = client._call_api("Nice product")
        assert res["raw"] is None
        assert "quota or rate limit" in res["error"].lower()
        assert "Model unavailable" not in res["error"]

    def test_call_api_server_error(self):
        client = APIClient()
        client._client = MagicMock()
        mock_exc = Exception("503 Service Unavailable")
        mock_exc.code = 503
        client._client.models.generate_content.side_effect = mock_exc

        res = client._call_api("Nice product")
        assert res["raw"] is None
        assert "server error" in res["error"].lower()

    def test_call_api_timeout_error(self):
        client = APIClient()
        client._client = MagicMock()
        client._client.models.generate_content.side_effect = TimeoutError("Connection timed out")

        res = client._call_api("Nice product")
        assert res["raw"] is None
        assert "network error" in res["error"].lower()
