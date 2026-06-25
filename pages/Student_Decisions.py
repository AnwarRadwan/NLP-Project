"""Student Decisions — category prediction (functional)."""

import streamlit as st

from src.i18n.translations import page_header, render_footer, t
from src.services import (
    analyze_category,
    get_category_bundle,
    localize_category,
    log_prediction,
)

page_header("student_decisions_title", "student_decisions_desc", icon="📢")

# Guard: model must be trained.
if get_category_bundle() is None:
    st.error(f"⚠️ {t('err_model_missing')}")
    render_footer()
    st.stop()

text = st.text_area(t("ui_input_text"), height=140,
                    placeholder="…", key="sd_text")

if st.button(t("ui_analyze"), type="primary"):
    if not text.strip():
        st.warning(t("ui_enter_text_warn"))
    else:
        result = analyze_category(text)
        if result is None:
            st.error(f"⚠️ {t('err_model_missing')}")
        else:
            label, confidence = result
            localized = localize_category(label)

            c1, c2 = st.columns(2)
            c1.metric(t("ui_predicted_category"), localized)
            c2.metric(t("ui_confidence"), f"{confidence * 100:.1f}%")
            st.progress(min(max(confidence, 0.0), 1.0))

            # Persist the prediction (fail-safe).
            log_prediction("Student Decisions", text,
                           predicted_category=label, category_confidence=confidence)

render_footer()
