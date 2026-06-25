"""Analytics Dashboard — real dataset statistics and charts (functional)."""

import streamlit as st

from src.i18n.translations import page_header, render_footer, section_header, t
from src.services import load_dataset, pie_chart

page_header("analytics_title", "analytics_desc", icon="📊")

df = load_dataset()
if df is None:
    st.error(f"⚠️ {t('err_dataset_missing')}")
    render_footer()
    st.stop()

# --- Top metrics -----------------------------------------------------------
m1, m2, m3 = st.columns(3)
m1.metric(t("stat_total_records"), f"{len(df):,}")
m2.metric(t("stat_languages"), int(df["language"].nunique()))
m3.metric(t("stat_categories"), int(df["category"].nunique()))


def _distribution(column: str, title_key: str) -> None:
    """Render a bar chart + pie chart for the value counts of ``column``."""
    section_header(title_key)
    counts = df[column].value_counts()
    left, right = st.columns([3, 2])
    with left:
        st.bar_chart(counts)
    with right:
        st.pyplot(pie_chart(counts, t(title_key)))


_distribution("language", "ui_language_distribution")
_distribution("sentiment", "ui_sentiment_distribution")
_distribution("category", "ui_category_distribution")

render_footer()
