"""Upload & Analyze Dataset — placeholder page."""

import streamlit as st

from src.i18n.translations import page_header, placeholder_banner, render_footer, t

page_header("upload_dataset_title", "upload_dataset_desc", icon="📤")
placeholder_banner()

# Upload widget shown but not wired to any processing yet.
st.file_uploader(t("upload_prompt"), type=["csv"], disabled=True)

render_footer()
