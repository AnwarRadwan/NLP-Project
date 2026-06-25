"""Search Engine — TF-IDF + cosine similarity retrieval (functional)."""

import streamlit as st

from src.i18n.translations import page_header, render_footer, t
from src.services import get_search_engine, localize_category, localize_sentiment

page_header("search_engine_title", "search_engine_desc", icon="🔍")

engine = get_search_engine()
if engine is None:
    st.error(f"⚠️ {t('err_engine')}")
    render_footer()
    st.stop()

# --- Query controls --------------------------------------------------------
col_q, col_k = st.columns([4, 1])
with col_q:
    query = st.text_input(t("search_placeholder"), key="search_query",
                          label_visibility="collapsed")
with col_k:
    top_k = st.number_input(t("ui_top_k"), min_value=1, max_value=20, value=5, step=1)

run = st.button(t("search_button"), type="primary")

# --- Run search ------------------------------------------------------------
if run:
    if not query.strip():
        st.warning(t("ui_enter_query"))
    else:
        results = engine.search(query, top_k=int(top_k))
        if not results:
            st.info(t("ui_no_results"))
        else:
            st.caption(f"{t('ui_search_results')}: {len(results)}")
            for r in results:
                score = r["similarity_score"]
                cat = localize_category(r["category"])
                sent = localize_sentiment(r["sentiment"])
                text = str(r["text"]).replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(
                    f'<div class="feature-card">'
                    f'<div><span class="bzu-badge">{t("ui_similarity")}: {score:.3f}</span>'
                    f'&nbsp;&nbsp;<b>{cat}</b> · {sent}</div>'
                    f'<p style="margin-top:8px;">{text}</p>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

render_footer()
