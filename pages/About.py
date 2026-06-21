"""About the NLP Pipeline — informational page."""

import streamlit as st

from src.i18n.translations import page_header, render_footer, section_header, t

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

render_footer()
