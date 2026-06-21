"""AI Student Assistant — retrieval + classification (no LLM, no external APIs).

Pure rule-based assistant built entirely from existing project components:
classifies the message (sentiment + category via the trained Logistic Regression
services), retrieves similar records (TF-IDF search engine), and composes a
templated response. No model training and no network calls.
"""

import streamlit as st

from src.i18n.translations import page_header, render_footer, section_header, t
from src.services import (
    analyze_category,
    analyze_sentiment,
    get_category_bundle,
    get_search_engine,
    get_sentiment_bundle,
    localize_category,
    localize_sentiment,
)

page_header("ai_assistant_title", "ai_assistant_desc", icon="🤖")

# Guard: the classifiers must be trained.
if get_sentiment_bundle() is None or get_category_bundle() is None:
    st.error(f"⚠️ {t('err_model_missing')}")
    render_footer()
    st.stop()

# Category -> rule-based response template.
RESPONSE_KEYS = {
    "Course Feedback": "aia_resp_course",
    "Student Decisions": "aia_resp_decisions",
    "University Discussions": "aia_resp_discussions",
}

text = st.text_area(t("aia_input_label"), placeholder=t("ai_input_placeholder"),
                    height=130, key="aia_text")

if st.button(t("ui_analyze"), type="primary"):
    if not text.strip():
        st.warning(t("ui_enter_text_warn"))
    else:
        sentiment = analyze_sentiment(text)
        category = analyze_category(text)
        if sentiment is None or category is None:
            st.error(f"⚠️ {t('err_model_missing')}")
        else:
            s_label, s_conf = sentiment
            c_label, c_conf = category
            s_loc = localize_sentiment(s_label)
            c_loc = localize_category(c_label)

            # --- Predictions + confidence --------------------------------
            col1, col2 = st.columns(2)
            col1.metric(t("ui_predicted_sentiment"), s_loc)
            col1.caption(f"{t('ui_confidence')}: {s_conf * 100:.1f}%")
            col2.metric(t("ui_predicted_category"), c_loc)
            col2.caption(f"{t('ui_confidence')}: {c_conf * 100:.1f}%")

            # --- Rule-based assistant response ---------------------------
            section_header("aia_response_heading")
            resp_key = RESPONSE_KEYS.get(c_label)
            response = t(resp_key).format(sentiment=s_loc) if resp_key else ""
            st.markdown(f"🤖 {response}")

            # --- Top 3 similar records (retrieval) -----------------------
            section_header("aia_similar_heading")
            engine = get_search_engine()
            if engine is None:
                st.warning(f"⚠️ {t('err_engine')}")
            else:
                results = engine.search(text, top_k=3)
                if not results:
                    st.info(t("ui_no_results"))
                for r in results:
                    score = r["similarity_score"]
                    rc = localize_category(r["category"])
                    rs = localize_sentiment(r["sentiment"])
                    rt = str(r["text"]).replace("<", "&lt;").replace(">", "&gt;")
                    st.markdown(
                        f'<div class="feature-card">'
                        f'<div><span class="bzu-badge">{t("ui_similarity")}: {score:.3f}</span>'
                        f'&nbsp;&nbsp;<b>{rc}</b> · {rs}</div>'
                        f'<p style="margin-top:8px;">{rt}</p></div>',
                        unsafe_allow_html=True,
                    )

render_footer()
