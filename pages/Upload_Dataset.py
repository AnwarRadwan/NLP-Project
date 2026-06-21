"""Upload & Analyze Dataset — placeholder page."""

import streamlit as st

from src.i18n.translations import page_setup, placeholder_banner, render_footer, t

lang = page_setup("upload_dataset_title", icon="📤")

st.title(f"📤 {t('upload_dataset_title')}")
st.markdown(f"#### {t('upload_dataset_desc')}")
st.write("")

placeholder_banner()

# Upload widget is shown but not wired to any processing yet.
st.file_uploader(t("upload_prompt"), type=["csv"], disabled=True)

render_footer()
