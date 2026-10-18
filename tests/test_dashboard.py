"""
Tests for StockGuard Streamlit Dashboard (app.py)
Validates that the dashboard starts up, renders all tabs, and handles audit button clicks.
"""

import os
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))


def test_dashboard_startup_and_rendering():
    """Verify that app.py runs end-to-end with no exceptions and expected element counts."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert len(at.exception) == 0, f"App execution raised exceptions: {at.exception}"
    assert len(at.markdown) > 30, "Expected markdown elements rendered on page"
    assert len(at.button) >= 20, "Expected at least 20 interactive buttons on page"


def test_dashboard_shelf_audit_interaction():
    """Verify that clicking a SKU and selecting 'Shelf Empty' correctly triggers feedback."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Find SKU inspection button (e.g. SKU_133 if present)
    sku_btns = [b for b in at.button if "SKU_" in b.label]
    assert len(sku_btns) > 0, "Expected SKU selection buttons"

    # Click the first SKU button
    sku_btns[0].click().run()
    assert len(at.exception) == 0

    # Click 'Shelf Empty' audit button
    phantom_btns = [b for b in at.button if "Shelf Empty" in b.label]
    assert len(phantom_btns) > 0, "Expected 'Shelf Empty' audit action button"
    phantom_btns[0].click().run()

    assert len(at.exception) == 0
    # Either success (confirmed phantom) or warning (simulated manager override) must appear
    total_feedback = len(at.success) + len(at.warning)
    assert total_feedback > 0, "Expected visual feedback card after audit action"
