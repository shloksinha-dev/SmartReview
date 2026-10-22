"""
Smoke tests for Streamlit application (app.py) using official AppTest framework.
"""

import pytest
from streamlit.testing.v1 import AppTest


from pathlib import Path

APP_FILE = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture
def app():
    """Create an AppTest instance for app.py with a safe timeout."""
    at = AppTest.from_file(APP_FILE, default_timeout=25)
    at.run()
    return at


def test_app_starts_without_unhandled_exceptions(app):
    """Verify app runs and produces no uncaught exceptions on initial launch."""
    assert not app.exception, f"App threw unexpected exception: {app.exception}"


def test_app_title_and_sidebar_present(app):
    """Verify that sidebar and main sections are populated."""
    assert len(app.sidebar) > 0


def test_app_navigation_to_model_info(app):
    """Navigate to Model Info and verify it renders without crashing."""
    app.radio[0].set_value("ℹ️ Model Info").run()
    assert not app.exception


def test_app_handles_gemini_api_error_with_fallback_ui():
    """Verify that an API error displays the error notice and fallback option cleanly without crashing."""
    at = AppTest.from_file(APP_FILE, default_timeout=25)
    at.session_state["single_result"] = (
        {"label": None, "confidence": None, "error": "Google Gemini server error (status 503)"},
        "api",
    )
    at.session_state["single_text"] = "This product broke after two days."
    at.run()
    assert not at.exception
    # Error message rendered
    assert any("503" in err.value for err in at.error)
    # Informational notice explaining local ML availability rendered
    assert any("Local ML Model" in info.value for info in at.info)
