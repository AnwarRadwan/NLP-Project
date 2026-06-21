"""About the NLP Pipeline — informational page."""

import streamlit as st

from src.i18n.translations import page_setup, render_footer, t

lang = page_setup("about_title", icon="ℹ️")

st.title(f"ℹ️ {t('about_title')}")
st.markdown(f"#### {t('about_desc')}")
st.write("")

# --- Pipeline stages -------------------------------------------------------
st.subheader(t("about_pipeline_heading"))
pipeline = [
    "Dataset generation",
    "Preprocessing",
    "Tokenization",
    "N-grams",
    "TF-IDF",
    "Embeddings",
    "Logistic Regression",
    "Neural Network",
    "LSTM",
    "AraBERT",
    "Information Retrieval",
    "Evaluation & Confusion Matrices",
]
st.markdown(
    " ➜ ".join(f"**{stage}**" for stage in pipeline)
)

st.write("")

# --- Technology stack ------------------------------------------------------
st.subheader(t("about_stack_heading"))
stack = [
    "Python", "Streamlit", "scikit-learn", "PyTorch",
    "HuggingFace Transformers", "SQLite", "TF-IDF + Cosine Similarity",
]
st.markdown(
    " ".join(
        f'<span class="bzu-badge">{tech}</span>' for tech in stack
    ),
    unsafe_allow_html=True,
)

render_footer()
