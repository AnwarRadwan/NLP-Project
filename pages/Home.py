"""Home — professional dashboard overview (real values from the dataset)."""

from src.i18n.translations import (
    feature_cards,
    page_header,
    pulse_cards,
    render_footer,
    section_header,
    stat_cards,
    t,
)
from src.services import load_dataset, model_count

# --- Hero ------------------------------------------------------------------
page_header("home_welcome", "home_overview", icon="🎓")

# --- Top statistics cards (real values) ------------------------------------
df = load_dataset()
if df is None:
    import streamlit as st
    st.error(f"⚠️ {t('err_dataset_missing')}")
    total_records, n_languages, n_categories = 0, 0, 0
else:
    total_records = len(df)
    n_languages = int(df["language"].nunique())
    n_categories = int(df["category"].nunique())

stat_cards(
    [
        {"icon": "📄", "value": f"{total_records:,}", "label_key": "stat_total_records"},
        {"icon": "🌐", "value": str(n_languages), "label_key": "stat_languages"},
        {"icon": "🗂️", "value": str(n_categories), "label_key": "stat_categories"},
        {"icon": "🧠", "value": str(model_count()), "label_key": "stat_models"},
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
