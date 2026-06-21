"""Home — professional dashboard overview (V2)."""

from src.i18n.translations import (
    feature_cards,
    page_header,
    pulse_cards,
    render_footer,
    section_header,
    stat_cards,
    t,
)

# --- Hero ------------------------------------------------------------------
page_header("home_welcome", "home_overview", icon="🎓")

# --- Top statistics cards (placeholder values) -----------------------------
stat_cards(
    [
        {"icon": "📄", "value": "0", "label_key": "stat_total_records"},
        {"icon": "🌐", "value": "2", "label_key": "stat_languages"},
        {"icon": "🧠", "value": "4", "label_key": "stat_models"},
        {"icon": "🎯", "value": "0", "label_key": "stat_predictions"},
    ]
)

# --- University Pulse ------------------------------------------------------
section_header("university_pulse", "university_pulse_sub")
pulse_cards(
    [
        {"icon": "🔥", "title_key": "pulse_most_discussed", "value_key": "no_data"},
        {"icon": "⚠️", "title_key": "pulse_most_complaint", "value_key": "no_data"},
        {"icon": "📢", "title_key": "pulse_latest_decision", "value_key": "no_data"},
    ]
)

# --- Explore features ------------------------------------------------------
section_header("quick_access", "quick_access_sub")
feature_cards()

render_footer()
