"""Professor & Course Insights — aggregated insights from Course Feedback.

Read-only analytics over the existing processed dataset (no models, no
training). Lets the user search by professor or course and see the sentiment
breakdown, top keywords, and sample comments.
"""

from collections import Counter

import streamlit as st

from src.i18n.translations import page_header, render_footer, section_header, t
from src.services import load_dataset, localize_sentiment, pie_chart

page_header("professor_insights_title", "professor_insights_desc", icon="👨‍🏫")

df = load_dataset()
if df is None:
    st.error(f"⚠️ {t('err_dataset_missing')}")
    render_footer()
    st.stop()

# 2) Course Feedback records only.
cf = df[df["category"] == "Course Feedback"].copy()

# 1) Search box (matches professor_name OR course_name, case-insensitive).
query = st.text_input(t("pi_search_label"), placeholder=t("pi_search_placeholder"),
                      key="pi_query")

if query.strip():
    q = query.strip()
    mask = (
        cf["professor_name"].astype(str).str.contains(q, case=False, na=False)
        | cf["course_name"].astype(str).str.contains(q, case=False, na=False)
    )
    results = cf[mask]
else:
    st.caption(t("pi_hint"))
    results = cf

total = len(results)

# 6) Friendly message if nothing matches.
if total == 0:
    st.info(f"🔎 {t('pi_no_results')}")
    render_footer()
    st.stop()

# 3) Sentiment distribution + percentages.
order = ["positive", "negative", "neutral"]
counts = results["sentiment"].value_counts().reindex(order, fill_value=0)

m1, m2, m3, m4 = st.columns(4)
m1.metric(t("pi_matching_comments"), f"{total:,}")
m2.metric(t("pi_pct_positive"), f"{counts['positive'] / total * 100:.1f}%")
m3.metric(t("pi_pct_negative"), f"{counts['negative'] / total * 100:.1f}%")
m4.metric(t("pi_pct_neutral"), f"{counts['neutral'] / total * 100:.1f}%")

# 4) Charts — localized labels for display.
section_header("ui_sentiment_distribution")
display_counts = counts.rename(index={k: localize_sentiment(k) for k in order})
left, right = st.columns([3, 2])
with left:
    st.bar_chart(display_counts)
with right:
    pie_data = display_counts[display_counts > 0]   # drop empty slices
    st.pyplot(pie_chart(pie_data, t("ui_sentiment_distribution")))

# 3) Most common keywords from tokens.
section_header("pi_keywords")
keywords: Counter = Counter()
for tokens in results["tokens"].fillna(""):
    keywords.update(str(tokens).split())
top_keywords = [w for w, _ in keywords.most_common(15)]
if top_keywords:
    st.markdown(
        " ".join(f'<span class="bzu-badge">{w}</span>' for w in top_keywords),
        unsafe_allow_html=True,
    )
else:
    st.caption("—")

# 3) Sample comments per sentiment.
section_header("pi_samples")
for sentiment in order:
    samples = results[results["sentiment"] == sentiment]["text"].head(3).tolist()
    with st.expander(f"{localize_sentiment(sentiment)} ({counts[sentiment]})"):
        if samples:
            for text in samples:
                st.markdown(f"- {text}")
        else:
            st.caption("—")

render_footer()
