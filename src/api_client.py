"""
External AI API client for SmartReview.

Supported provider: Google Gemini (Developer API via official google-genai SDK).
The integration is intentionally optional and costs ₹0 on the Google AI Studio free tier.
If no API key is configured, the application falls back to the local ML model transparently.

Security:
- API key is loaded ONLY from environment variables (GEMINI_API_KEY) or Streamlit secrets.
- The key is NEVER printed, logged, or exposed in the UI.
- User review text is only sent when the user explicitly selects API mode.

Usage flow:
    client = APIClient()
    if client.is_configured():
        result = client.analyze(review_text)
"""

import json
import logging
import os
import re
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Environment variable names
ENV_KEY_NAME = "GEMINI_API_KEY"
ENV_MODEL_NAME = "GEMINI_MODEL"

# Standard fast, low-latency free-tier model on Google AI Studio
DEFAULT_MODEL = "gemini-3.8-flash"

# Default retry settings for transient API errors (503, 500, 429, timeouts)
DEFAULT_MAX_RETRIES = 2
DEFAULT_INITIAL_DELAY = 0.5  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_MAX_DELAY = 4.0

# Normalise external provider labels to our internal schema
_LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    # variants the model might return
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
    "mixed": "neutral",
}


class APIClient:
    """
    Wrapper around the Google Gemini API for sentiment analysis.

    The client remains disabled unless a GEMINI_API_KEY is configured.
    Includes bounded exponential backoff for transient server (503, 500) and rate-limit (429) errors.
    """

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        max_delay: float = DEFAULT_MAX_DELAY,
    ):
        self._api_key: Optional[str] = self._load_api_key()
        self._model: str = os.environ.get(ENV_MODEL_NAME, DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self._client = None
        self._max_retries = max(0, int(max_retries))
        self._initial_delay = max(0.0, float(initial_delay))
        self._backoff_factor = max(1.0, float(backoff_factor))
        self._max_delay = max(self._initial_delay, float(max_delay))

        if self._api_key:
            self._init_client()

    # ─── Private helpers ────────────────────────────────────────────────────

    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment variables or Streamlit secrets."""
        # 1. Environment variable
        key = os.environ.get(ENV_KEY_NAME, "").strip()
        if key:
            return key

        # 2. Streamlit secrets (when running inside Streamlit)
        try:
            import streamlit as st
            key = st.secrets.get(ENV_KEY_NAME, "").strip()
            if key:
                return key
        except Exception:
            pass

        return None

    def _init_client(self):
        """Initialise the Google GenAI client object."""
        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            logger.info("Google GenAI client initialised (model: %s).", self._model)
        except ImportError:
            logger.error("google-genai package not installed.")
            self._client = None
        except Exception as exc:
            logger.error("Failed to initialize Google GenAI client: %s", exc)
            self._client = None

    @staticmethod
    def _extract_status_code(exc: Exception) -> Optional[int]:
        """Extract HTTP status code from exception, response attribute, or error message."""
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            return code
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        resp = getattr(exc, "response", None)
        if resp is not None:
            resp_code = getattr(resp, "status_code", None)
            if isinstance(resp_code, int):
                return resp_code
        match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
        if match:
            return int(match.group(1))
        return None

    def _sdk_retries_configured(self) -> bool:
        """Check if underlying google-genai SDK already has active retry options."""
        try:
            api_client = getattr(self._client, "_api_client", None)
            http_opts = getattr(api_client, "_http_options", None)
            retry_opts = getattr(http_opts, "retry_options", None)
            if retry_opts and getattr(retry_opts, "attempts", 1) and retry_opts.attempts > 1:
                return True
        except Exception:
            pass
        return False

    def _is_transient_error(self, exc: Exception) -> bool:
        """
        Determine if an error is transient and safe to retry.
        Transient: 503, 500, 502, 504, 429, and network/timeouts.
        Permanent (401/403 auth, 404 model not found, 400 bad request) are never retried.
        """
        code = self._extract_status_code(exc)
        exc_name = type(exc).__name__
        msg = str(exc).lower()

        # Permanent failures: NEVER retry
        if code in (401, 403) or any(
            t in msg for t in ("unauthenticated", "permission_denied", "api_key_invalid", "invalid api key")
        ):
            return False
        if code == 404 or "not_found" in msg or "not available" in msg:
            return False
        if code == 400 or "invalid_argument" in msg:
            return False

        # Transient server errors: safe to retry (503, 500, 502, 504)
        if (code and code in (500, 502, 503, 504)) or "servererror" in exc_name.lower():
            return True
        if "service unavailable" in msg or "overloaded" in msg:
            return True

        # Transient quota / rate limit (429)
        if code == 429 or "resource_exhausted" in msg or "rate limit" in msg or "quota" in msg:
            return True

        # Transient network / timeout errors
        if any(term in exc_name for term in ("Timeout", "Connection", "Connect", "Network")):
            return True

        return False

    def _format_error(self, exc: Exception, attempts: int = 1) -> str:
        """Format clear, user-facing error message distinguishing failure types."""
        code = self._extract_status_code(exc)
        exc_name = type(exc).__name__
        msg = str(exc)

        # 1. Model Not Found / Unavailable (404 or specific message patterns)
        if code == 404 or "NOT_FOUND" in msg or "not available" in msg.lower() or "not found" in msg.lower():
            return (
                f"Model unavailable: Gemini model '{self._model}' was not found or is unavailable for this account. "
                "Verify that GEMINI_MODEL is set to a supported model (such as 'gemini-3.8-flash') or switch to the local ML model."
            )

        # 2. Authentication / Invalid API Key (401, 403, permission or key issues)
        if (
            code in (401, 403)
            or "UNAUTHENTICATED" in msg
            or "PERMISSION_DENIED" in msg
            or "API_KEY_INVALID" in msg
            or "invalid api key" in msg.lower()
        ):
            return (
                "Authentication failed: invalid or unauthorized GEMINI_API_KEY. "
                "Please verify your API key in .env or Streamlit secrets."
            )

        # 3. Quota / Rate Limits (429 or RESOURCE_EXHAUSTED)
        if code == 429 or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower() or "rate limit" in msg.lower():
            attempt_info = f" after {attempts} attempts" if attempts > 1 else ""
            return (
                f"Quota or rate limit reached: Gemini API free-tier request limit exceeded{attempt_info}. "
                "Please wait a moment before trying again or switch to the local ML model."
            )

        # 4. Server Errors (503, 500, 5xx, or ServerError class)
        if (code and code >= 500) or "ServerError" in exc_name or "service unavailable" in msg.lower():
            attempt_info = f" after {attempts} attempt(s)" if attempts > 1 else ""
            status_desc = f"{code or 503}"
            return (
                f"Google Gemini server error (status {status_desc}){attempt_info}. "
                "Google's service is temporarily unavailable or experiencing high traffic. "
                "Please retry shortly or switch to the local ML model."
            )

        # 5. Network / Timeout Errors
        if any(term in exc_name for term in ("Timeout", "Connection", "Connect", "Network")):
            attempt_info = f" after {attempts} attempt(s)" if attempts > 1 else ""
            return (
                f"Network error: connection to Google Gemini API timed out or unreachable{attempt_info}. "
                "Check your internet connection or switch to the local ML model."
            )

        # 6. Other Client Errors
        if code is not None:
            return f"Gemini client error (status {code}): {msg}"

        return f"Unexpected Gemini API error: {exc_name} ({msg})."

    def _call_api(self, text: str) -> Dict:
        """
        Send a review to the Google Gemini API with bounded exponential backoff for transient errors.

        Returns a raw dict with 'raw' response text and 'error'.
        """
        prompt = (
            "You are an expert sentiment analysis assistant. Analyze the given customer review "
            "and respond with ONLY a JSON object containing two keys: "
            '"label" (one of: positive, negative, neutral) and '
            '"explanation" (a single concise sentence explaining your classification).\n\n'
            f'Review: "{text}"'
        )

        effective_retries = 0 if self._sdk_retries_configured() else self._max_retries
        delay = self._initial_delay
        attempts = 0

        while True:
            attempts += 1
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config={"response_mime_type": "application/json", "temperature": 0.0},
                )
                raw = response.text.strip() if response and response.text else ""
                return {"raw": raw, "error": None}

            except Exception as exc:
                is_transient = self._is_transient_error(exc)
                if is_transient and attempts <= effective_retries:
                    logger.warning(
                        "Gemini API transient failure on attempt %d/%d: %s. Retrying in %.2fs...",
                        attempts,
                        effective_retries + 1,
                        exc,
                        delay,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    delay = min(delay * self._backoff_factor, self._max_delay)
                    continue

                return {"raw": None, "error": self._format_error(exc, attempts)}

    # ─── Public interface ────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """Return True if an API key is present and client is ready."""
        return bool(self._api_key and self._client)

    def analyze(self, text: str) -> Dict:
        """
        Analyse a review using the Google Gemini API.

        Returns
        -------
        dict with keys:
            label       : str | None    — normalised sentiment label
            explanation : str | None    — model's explanation
            source      : str           — always "api"
            model       : str           — model identifier used
            confidence  : None          — APIs don't return classification probabilities
            error       : str | None    — error message if call failed
        """
        if not self.is_configured():
            return {
                "label": None,
                "explanation": None,
                "source": "api",
                "model": self._model,
                "confidence": None,
                "error": "Gemini API is not configured. Set GEMINI_API_KEY.",
            }

        response = self._call_api(text)

        if response.get("error"):
            return {
                "label": None,
                "explanation": None,
                "source": "api",
                "model": self._model,
                "confidence": None,
                "error": response["error"],
            }

        raw_str = response.get("raw", "") or ""

        # Strip markdown fences if present (e.g. ```json ... ```)
        cleaned_json = re.sub(r"^```(?:json)?\s*", "", raw_str, flags=re.MULTILINE)
        cleaned_json = re.sub(r"\s*```$", "", cleaned_json, flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(cleaned_json)
            raw_label = str(parsed.get("label", "")).lower().strip()
            label = _LABEL_MAP.get(raw_label, None)
            explanation = str(parsed.get("explanation", "")).strip()

            if label is None:
                return {
                    "label": None,
                    "explanation": explanation or None,
                    "source": "api",
                    "model": self._model,
                    "confidence": None,
                    "error": f"Unrecognised label from Gemini API: '{raw_label}'",
                }

            return {
                "label": label,
                "explanation": explanation,
                "source": "api",
                "model": self._model,
                "confidence": None,
                "error": None,
            }
        except (json.JSONDecodeError, AttributeError) as exc:
            return {
                "label": None,
                "explanation": None,
                "source": "api",
                "model": self._model,
                "confidence": None,
                "error": f"Could not parse Gemini API response: {exc}",
            }
