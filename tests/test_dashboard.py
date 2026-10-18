"""
Tests for StockGuard Streamlit Dashboard (app.py)
Validates the Store Manager View, Checklist Features, Detail Proof Charts, and Operational Audit Flow.
"""

import os
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))


def test_dashboard_startup_and_manager_view():
    """Verify that app.py launches into the manager view with expected metrics and controls."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert len(at.exception) == 0, f"App execution raised exceptions: {at.exception}"

    # Verify that product description tagline is present
    all_markdown_text = " ".join([m.value for m in at.markdown])
    assert "gives retail store managers" in all_markdown_text
    assert "Shelves to Check Today" in all_markdown_text
    assert "Total Margin at Risk" in all_markdown_text
    assert "Checklist Progress" in all_markdown_text

    # Verify absence of technical jargon in the manager view header
    assert "Bayesian Posterior" not in at.markdown[0].value

    # Verify download button exists
    assert len(at.download_button) >= 1, "Expected CSV download button for store manager"
    assert "Download CSV Checklist" in at.download_button[0].label


def test_dashboard_shelf_audit_and_feedback():
    """Verify selecting a SKU, inspecting plain-English charts, and recording an audit."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Pre-selected demo item is SKU_133 (Mosquito Repellent Item 133)
    # Check that plain-English chart title appears
    all_markdown_text = " ".join([m.value for m in at.markdown])
    assert "Recommended Action for Floor Staff" in all_markdown_text

    # Verify the four audit action buttons exist
    button_labels = [b.label for b in at.button]
    assert any("Shelf Empty" in l for l in button_labels), "Expected 'Shelf Empty' button"
    assert any("Found on Shelf" in l for l in button_labels), "Expected 'Found on Shelf' button"
    assert any("Found Wrong Bay" in l for l in button_labels), "Expected 'Found Wrong Bay' button"
    assert any("Item Discontinued" in l for l in button_labels), "Expected 'Item Discontinued' button"

    # Click 'Shelf Empty' for the selected SKU
    phantom_btns = [b for b in at.button if "Shelf Empty" in b.label]
    phantom_btns[0].click().run()

    assert len(at.exception) == 0
    # Operational outcome confirmation card must appear
    assert len(at.success) > 0, "Expected success confirmation card after confirming phantom"
    success_text = at.success[0].value
    assert "SHELF EMPTY CONFIRMED" in success_text
    assert "System Inventory Corrected" in success_text
    assert "Automated Replenishment Triggered" in success_text
    assert "Rupees Recovered" in success_text


def test_demo_shortcut_button():
    """Verify that clicking the demo video shortcut loads the pre-selected demo case."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    demo_btns = [b for b in at.button if "Load Demo Video Example" in b.label]
    assert len(demo_btns) > 0, "Expected demo shortcut button"
    demo_btns[0].click().run()

    assert len(at.exception) == 0
    assert at.session_state["curr_store"] == "STORE_02"
    assert at.session_state["curr_date"] == "2026-01-20"
    assert at.session_state["selected_sku_id"] == "SKU_133"
