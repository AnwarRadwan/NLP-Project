"""Model Comparison — placeholder page."""

from src.i18n.translations import page_header, placeholder_banner, render_footer, t
import streamlit as st

page_header("model_comparison_title", "model_comparison_desc", icon="⚖️")
placeholder_banner()

# Models that will be compared once training is implemented.
models = ["Logistic Regression", "Neural Network", "LSTM", "AraBERT"]
cards = "".join(
    f'<div class="feature-card"><div class="feature-icon">🧠</div>'
    f'<h3>{name}</h3><span class="bzu-badge">{t("coming_soon")}</span></div>'
    for name in models
)
st.markdown(f'<div class="bzu-grid">{cards}</div>', unsafe_allow_html=True)

render_footer()
