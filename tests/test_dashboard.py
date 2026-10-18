"""
Tests for StockGuard Multi-Page Streamlit Application
Validates:
1. Main SaaS Store Operations Hub (app.py) from the shopkeeper perspective.
2. Executive Performance & ROI page (pages/1_📊_Performance_and_ROI.py).
3. Methodology & Grounding page (pages/2_🔬_Methodology_and_Research.py).
"""

import os
import pytest
from streamlit.testing.v1 import AppTest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PATH = os.path.join(BASE_DIR, "app.py")
PERF_PATH = os.path.join(BASE_DIR, "pages", "1_📊_Performance_and_ROI.py")
METHOD_PATH = os.path.join(BASE_DIR, "pages", "2_🔬_Methodology_and_Research.py")


def test_saas_store_portal_startup_and_kpis():
    """Verify that app.py launches into the shopkeeper view with plain-language metrics."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert len(at.exception) == 0, f"App execution raised exceptions: {at.exception}"

    all_markdown_text = " ".join([m.value for m in at.markdown])
    assert "gives retail store managers" in all_markdown_text
    assert "Shelves to Check Today" in all_markdown_text
    assert "Total Margin at Risk" in all_markdown_text
    assert "Checklist Progress" in all_markdown_text
    assert "Store Operations Hub" in all_markdown_text

    # Verify download button exists
    assert len(at.download_button) >= 1, "Expected CSV download button for store manager"
    assert "Download CSV Checklist" in at.download_button[0].label


def test_saas_shelf_audit_and_reorder_trigger():
    """Verify selecting a product, plain-English instructions, and tapping 'Shelf Empty'."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Pre-selected demo item is SKU_133 (Mosquito Repellent Item 133)
    all_markdown_text = " ".join([m.value for m in at.markdown])
    assert "Floor Walk Instructions for Store Staff" in all_markdown_text

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
    assert len(at.success) > 0, "Expected success confirmation card after confirming empty shelf"
    success_text = at.success[0].value
    assert "EMPTY SHELF CONFIRMED" in success_text
    assert "Computer Corrected" in success_text
    assert "Purchase Order Triggered" in success_text
    assert "Profit Saved" in success_text


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


def test_performance_and_roi_page():
    """Verify that the dedicated Performance & ROI page loads with benchmark table and simulator."""
    at = AppTest.from_file(PERF_PATH)
    at.run()
    assert len(at.exception) == 0, f"Performance page raised exceptions: {at.exception}"

    all_markdown_text = " ".join([m.value for m in at.markdown])
    assert "Executive Performance Benchmark" in all_markdown_text
    assert "StockGuard (Ours)" in all_markdown_text
    assert "Baseline A (Matched Volume)" in all_markdown_text
    assert "Chain-Wide Financial ROI Simulator" in all_markdown_text


def test_methodology_and_research_page():
    """Verify that the dedicated Methodology page loads with academic theory and robustness grid."""
    at = AppTest.from_file(METHOD_PATH)
    at.run()
    assert len(at.exception) == 0, f"Methodology page raised exceptions: {at.exception}"

    all_markdown_text = " ".join([m.value for m in at.markdown])
    assert "Mathematical Formulation" in all_markdown_text
    assert "DeHoratius & Raman" in all_markdown_text
    assert "Freezing Effect" in all_markdown_text
    assert "Stress-Test Robustness Grid" in all_markdown_text
