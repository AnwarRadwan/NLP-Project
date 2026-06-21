"""Course Feedback Analysis — sentiment prediction (functional)."""

import streamlit as st

from src.i18n.translations import page_header, render_footer, t
from src.services import analyze_sentiment, get_sentiment_bundle, localize_sentiment

page_header("course_feedback_title", "course_feedback_desc", icon="📝")

# Guard: model must be trained.
if get_sentiment_bundle() is None:
    st.error(f"⚠️ {t('err_model_missing')}")
    render_footer()
    st.stop()

text = st.text_area(t("ui_input_text"), height=140,
                    placeholder="…", key="cf_text")

if st.button(t("ui_analyze"), type="primary"):
    if not text.strip():
        st.warning(t("ui_enter_text_warn"))
    else:
        result = analyze_sentiment(text)
        if result is None:
            st.error(f"⚠️ {t('err_model_missing')}")
        else:
            label, confidence = result
            localized = localize_sentiment(label)
            emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(label, "")

            c1, c2 = st.columns(2)
            c1.metric(t("ui_predicted_sentiment"), f"{emoji} {localized}")
            c2.metric(t("ui_confidence"), f"{confidence * 100:.1f}%")
            st.progress(min(max(confidence, 0.0), 1.0))

render_footer()
