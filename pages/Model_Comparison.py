"""Model Comparison — placeholder page."""

import streamlit as st

from src.i18n.translations import page_setup, placeholder_banner, render_footer, t

lang = page_setup("model_comparison_title", icon="⚖️")

st.title(f"⚖️ {t('model_comparison_title')}")
st.markdown(f"#### {t('model_comparison_desc')}")
st.write("")

placeholder_banner()

# Models that will be compared once training is implemented.
models = ["Logistic Regression", "Neural Network", "LSTM", "AraBERT"]
cols = st.columns(len(models))
for col, name in zip(cols, models):
    col.markdown(
        f'<div class="bzu-card"><h3>{name}</h3>'
        f'<span class="bzu-badge">{t("coming_soon")}</span></div>',
        unsafe_allow_html=True,
    )

render_footer()
