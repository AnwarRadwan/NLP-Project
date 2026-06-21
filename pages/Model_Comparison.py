"""Model Comparison — static evaluated results (group-aware split).

Displays previously-computed metrics for the five trained models. No model code
is imported and nothing is retrained; the numbers come from the group-aware
evaluations run in the model test scripts.
"""

import pandas as pd
import streamlit as st

from src.i18n.translations import page_header, render_footer, section_header, t

page_header("model_comparison_title", "model_comparison_desc", icon="⚖️")
st.caption(t("mc_caption"))

# Evaluated results (group-aware split): (sent_acc, sent_f1, cat_acc, cat_f1).
RESULTS: dict[str, tuple[float, float, float, float]] = {
    "Logistic Regression": (0.6838, 0.6369, 0.7374, 0.6808),
    "Neural Network": (0.6045, 0.6011, 1.0000, 1.0000),
    "LSTM": (0.4624, 0.3751, 0.4470, 0.4067),
    "BiLSTM": (0.5561, 0.5546, 0.9444, 0.9326),
    "AraBERT": (0.7333, 0.7433, 0.8641, 0.8095),
}

# --- Results table ---------------------------------------------------------
section_header("mc_table_heading")
table = pd.DataFrame(
    [
        {
            t("mc_col_model"): name,
            t("mc_col_sent_acc"): vals[0],
            t("mc_col_sent_f1"): vals[1],
            t("mc_col_cat_acc"): vals[2],
            t("mc_col_cat_f1"): vals[3],
        }
        for name, vals in RESULTS.items()
    ]
).set_index(t("mc_col_model"))
st.dataframe(table.style.format("{:.4f}"), use_container_width=True)

# --- Charts ----------------------------------------------------------------
section_header("mc_charts_heading")
sent_f1 = pd.Series({name: vals[1] for name, vals in RESULTS.items()})
cat_f1 = pd.Series({name: vals[3] for name, vals in RESULTS.items()})

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**{t('mc_chart_sent_f1')}**")
    st.bar_chart(sent_f1, color="#5b8cff")
with col2:
    st.markdown(f"**{t('mc_chart_cat_f1')}**")
    st.bar_chart(cat_f1, color="#8a6cff")

# --- Highlights ------------------------------------------------------------
section_header("mc_highlights_heading")
h1, h2 = st.columns(2)
with h1:
    st.markdown(
        f'<div class="feature-card"><div class="feature-icon">🏆</div>'
        f'<h3>{t("mc_best_sentiment")}</h3>'
        f'<p><b>AraBERT</b> — Macro-F1 0.7433</p></div>',
        unsafe_allow_html=True,
    )
with h2:
    st.markdown(
        f'<div class="feature-card"><div class="feature-icon">🥇</div>'
        f'<h3>{t("mc_best_category")}</h3>'
        f'<p><b>BiLSTM</b> — Macro-F1 0.9326</p></div>',
        unsafe_allow_html=True,
    )

st.warning(t("mc_leakage_note"))

# --- Conclusions -----------------------------------------------------------
section_header("mc_conclusions_heading")
st.markdown(
    "\n".join(f"- {t(f'mc_concl_{i}')}" for i in range(1, 6))
)

render_footer()
