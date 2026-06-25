"""Upload & Analyze Dataset — CSV upload + preview (no retraining)."""

import pandas as pd
import streamlit as st

from src.i18n.translations import page_header, render_footer, t

page_header("upload_dataset_title", "upload_dataset_desc", icon="📤")

uploaded = st.file_uploader(t("upload_prompt"), type=["csv"])

if uploaded is None:
    st.info(t("ui_upload_help"))
else:
    # Try utf-8-sig first (handles Arabic + BOM), then fall back.
    df = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding=encoding)
            break
        except Exception:
            continue

    if df is None:
        st.error(f"⚠️ {t('ui_upload_error')}")
    else:
        c1, c2 = st.columns(2)
        c1.metric(t("ui_rows"), f"{df.shape[0]:,}")
        c2.metric(t("ui_columns"), df.shape[1])

        st.markdown(f"**{t('ui_shape')}:** {df.shape[0]} × {df.shape[1]}")
        st.markdown(f"**{t('ui_columns')}:** {', '.join(map(str, df.columns))}")

        st.markdown(f"#### {t('ui_preview')}")
        st.dataframe(df.head(10), use_container_width=True)

render_footer()
