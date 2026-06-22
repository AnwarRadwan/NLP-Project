"""About the NLP Pipeline — informational page."""

import streamlit as st

from src.i18n.translations import page_header, render_footer, section_header, t
from src.services import db_status

page_header("about_title", "about_desc", icon="ℹ️")

# --- Pipeline stages -------------------------------------------------------
section_header("about_pipeline_heading")
pipeline = [
    "Dataset generation", "Preprocessing", "Tokenization", "N-grams",
    "TF-IDF", "Embeddings", "Logistic Regression", "Neural Network",
    "LSTM", "AraBERT", "Information Retrieval", "Evaluation & Confusion Matrices",
]
st.markdown(" ➜ ".join(f"**{stage}**" for stage in pipeline))

# --- Technology stack ------------------------------------------------------
section_header("about_stack_heading")
stack = [
    "Python", "Streamlit", "scikit-learn", "PyTorch",
    "HuggingFace Transformers", "SQLite", "TF-IDF + Cosine Similarity",
]
st.markdown(
    " ".join(f'<span class="bzu-badge">{tech}</span>' for tech in stack),
    unsafe_allow_html=True,
)

# --- Data storage (SQLite) -------------------------------------------------
section_header("about_storage_heading")
st.write(t("about_storage_note"))
status = db_status()
s1, s2 = st.columns(2)
s1.metric(t("about_storage_records"), f"{status['dataset_records']:,}")
s2.metric(t("about_storage_logs"), f"{status['prediction_logs']:,}")

render_footer()
