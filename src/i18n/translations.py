"""Internationalization (i18n) and shared UI shell — V2.

Provides everything the website chrome needs:
- Translation dictionaries for English (``en``) and Arabic (``ar``).
- ``t(key)`` lookup helper with safe fallback.
- A modern dark-dashboard theme (CSS) with RTL (Arabic) / LTR (English) support.
- A sidebar brand block + language switcher (persisted in ``st.session_state``).
- ``st.navigation``-based, translated, icon-rich sidebar navigation.
- Reusable UI builders: page headers, stat cards, "University Pulse" cards,
  and feature cards.

No machine-learning logic lives here; this is purely the website shell.
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #
# Display label -> language code
LANGUAGES: dict[str, str] = {
    "English": "en",
    "العربية": "ar",
}

DEFAULT_LANG = "en"
RTL_LANGS = {"ar"}


# --------------------------------------------------------------------------- #
# Translations
# --------------------------------------------------------------------------- #
TRANSLATIONS: dict[str, dict[str, str]] = {
    # --- Brand / shell -----------------------------------------------------
    "app_name": {
        "en": "BZU Student Intelligence",
        "ar": "منصة ذكاء طلبة بيرزيت",
    },
    "app_tagline": {
        "en": "Bilingual NLP Platform · ENCS5342",
        "ar": "منصة معالجة لغات طبيعية ثنائية اللغة · ENCS5342",
    },
    "language": {"en": "Language", "ar": "اللغة"},
    "coming_soon": {"en": "Coming soon", "ar": "قريباً"},
    "no_data": {"en": "No data yet", "ar": "لا توجد بيانات بعد"},
    "placeholder_notice": {
        "en": "This page is a placeholder. The NLP features will be added in a later step.",
        "ar": "هذه الصفحة مبدئية. ستتم إضافة ميزات معالجة اللغة في مرحلة لاحقة.",
    },
    "footer": {
        "en": "Birzeit University · Natural Language Processing (ENCS5342)",
        "ar": "جامعة بيرزيت · معالجة اللغات الطبيعية (ENCS5342)",
    },

    # --- Navigation labels -------------------------------------------------
    "nav_home": {"en": "Dashboard", "ar": "اللوحة الرئيسية"},
    "nav_course_feedback": {"en": "Course Feedback", "ar": "تقييم المساقات"},
    "nav_student_decisions": {"en": "Student Decisions", "ar": "قرارات الطلبة"},
    "nav_professor_insights": {"en": "Professor Insights", "ar": "رؤى المدرّسين"},
    "nav_upload_dataset": {"en": "Upload Dataset", "ar": "رفع البيانات"},
    "nav_search_engine": {"en": "Search Engine", "ar": "محرك البحث"},
    "nav_analytics": {"en": "Analytics", "ar": "لوحة التحليلات"},
    "nav_ai_assistant": {"en": "AI Assistant", "ar": "المساعد الذكي"},
    "nav_model_comparison": {"en": "Model Comparison", "ar": "مقارنة النماذج"},
    "nav_about": {"en": "About", "ar": "حول المشروع"},

    # --- Section groups (sidebar) -----------------------------------------
    "group_overview": {"en": "Overview", "ar": "نظرة عامة"},
    "group_features": {"en": "NLP Features", "ar": "ميزات المعالجة"},
    "group_tools": {"en": "Tools & Info", "ar": "الأدوات والمعلومات"},

    # --- Home dashboard ----------------------------------------------------
    "home_title": {"en": "Dashboard", "ar": "اللوحة الرئيسية"},
    "home_welcome": {
        "en": "Welcome to the BZU Student Intelligence Platform",
        "ar": "مرحباً بك في منصة ذكاء طلبة بيرزيت",
    },
    "home_overview": {
        "en": "A bilingual (English / Arabic) NLP platform for understanding BZU student & community content.",
        "ar": "منصة معالجة لغات طبيعية ثنائية اللغة (عربي / إنجليزي) لفهم محتوى طلبة ومجتمع بيرزيت.",
    },

    # Stat cards
    "stat_total_records": {"en": "Total Records", "ar": "إجمالي السجلات"},
    "stat_languages": {"en": "Languages", "ar": "اللغات"},
    "stat_models": {"en": "Models", "ar": "النماذج"},
    "stat_predictions": {"en": "Predictions", "ar": "التنبؤات"},

    # University Pulse
    "university_pulse": {"en": "University Pulse", "ar": "نبض الجامعة"},
    "university_pulse_sub": {
        "en": "Live highlights from across campus (placeholders for now).",
        "ar": "أبرز ما يجري في الحرم الجامعي (قيم مبدئية حالياً).",
    },
    "pulse_most_discussed": {"en": "Most Discussed Topic", "ar": "الموضوع الأكثر نقاشاً"},
    "pulse_most_complaint": {"en": "Most Common Complaint", "ar": "الشكوى الأكثر شيوعاً"},
    "pulse_latest_decision": {"en": "Latest Student Decision", "ar": "آخر قرار طلابي"},

    # Quick access / features
    "quick_access": {"en": "Explore Features", "ar": "استكشف الميزات"},
    "quick_access_sub": {
        "en": "Jump into any module from the sidebar.",
        "ar": "انتقل إلى أي وحدة من الشريط الجانبي.",
    },

    # --- Course Feedback ---------------------------------------------------
    "course_feedback_title": {"en": "Course Feedback Analysis", "ar": "تحليل تقييم المساقات"},
    "course_feedback_desc": {
        "en": "Analyze sentiment and themes in student course feedback (Arabic & English).",
        "ar": "تحليل المشاعر والمواضيع في تقييمات الطلبة للمساقات (بالعربية والإنجليزية).",
    },

    # --- Student Decisions -------------------------------------------------
    "student_decisions_title": {
        "en": "Student Movements & University Decisions",
        "ar": "حراك الطلبة وقرارات الجامعة",
    },
    "student_decisions_desc": {
        "en": "Track and analyze student movements and official university decisions.",
        "ar": "متابعة وتحليل حراك الطلبة والقرارات الرسمية للجامعة.",
    },

    # --- Professor Insights ------------------------------------------------
    "professor_insights_title": {"en": "Professor & Course Insights", "ar": "رؤى حول المدرّسين والمساقات"},
    "professor_insights_desc": {
        "en": "Explore aggregated insights about professors and courses.",
        "ar": "استكشاف رؤى مجمّعة حول المدرّسين والمساقات.",
    },

    # --- Upload Dataset ----------------------------------------------------
    "upload_dataset_title": {"en": "Upload & Analyze Dataset", "ar": "رفع وتحليل البيانات"},
    "upload_dataset_desc": {
        "en": "Upload your own bilingual dataset to run the NLP pipeline on it.",
        "ar": "ارفع مجموعة بياناتك ثنائية اللغة لتشغيل مسار معالجة اللغة عليها.",
    },
    "upload_prompt": {"en": "Upload a CSV file", "ar": "ارفع ملف CSV"},

    # --- Search Engine -----------------------------------------------------
    "search_engine_title": {"en": "Search Engine", "ar": "محرك البحث"},
    "search_engine_desc": {
        "en": "Search the corpus using TF-IDF and cosine similarity.",
        "ar": "ابحث في النصوص باستخدام TF-IDF وتشابه جيب التمام.",
    },
    "search_placeholder": {"en": "Enter your query...", "ar": "اكتب استعلامك..."},
    "search_button": {"en": "Search", "ar": "بحث"},

    # --- Analytics ---------------------------------------------------------
    "analytics_title": {"en": "Analytics Dashboard", "ar": "لوحة التحليلات"},
    "analytics_desc": {
        "en": "Visualize corpus statistics, trends, and model performance.",
        "ar": "عرض إحصاءات النصوص والاتجاهات وأداء النماذج بصرياً.",
    },

    # --- AI Assistant ------------------------------------------------------
    "ai_assistant_title": {"en": "AI Student Assistant", "ar": "المساعد الذكي للطلبة"},
    "ai_assistant_desc": {
        "en": "Ask questions about courses, decisions, and campus life.",
        "ar": "اطرح أسئلة حول المساقات والقرارات والحياة الجامعية.",
    },
    "ai_input_placeholder": {"en": "Ask me anything...", "ar": "اسألني أي شيء..."},

    # --- Model Comparison --------------------------------------------------
    "model_comparison_title": {"en": "Model Comparison", "ar": "مقارنة النماذج"},
    "model_comparison_desc": {
        "en": "Compare Logistic Regression, Neural Network, LSTM, BiLSTM, and AraBERT.",
        "ar": "قارن بين الانحدار اللوجستي والشبكة العصبية وLSTM وBiLSTM وAraBERT.",
    },

    # --- About -------------------------------------------------------------
    "about_title": {"en": "About the NLP Pipeline", "ar": "حول مسار معالجة اللغة"},
    "about_desc": {
        "en": "An overview of the end-to-end NLP pipeline powering this platform.",
        "ar": "نظرة عامة على مسار معالجة اللغة الكامل الذي يشغّل هذه المنصة.",
    },
    "about_pipeline_heading": {"en": "Pipeline Stages", "ar": "مراحل المسار"},
    "about_stack_heading": {"en": "Technology Stack", "ar": "التقنيات المستخدمة"},

    # --- Functional UI (integrated pages) ----------------------------------
    "stat_categories": {"en": "Categories", "ar": "الفئات"},

    "ui_top_k": {"en": "Results to show (Top K)", "ar": "عدد النتائج (Top K)"},
    "ui_search_results": {"en": "Results", "ar": "النتائج"},
    "ui_no_results": {"en": "No matching results.", "ar": "لا توجد نتائج مطابقة."},
    "ui_enter_query": {"en": "Please enter a search query.", "ar": "الرجاء إدخال نص للبحث."},
    "ui_similarity": {"en": "Similarity", "ar": "نسبة التشابه"},

    "ui_input_text": {"en": "Enter your text", "ar": "أدخل النص"},
    "ui_analyze": {"en": "Analyze", "ar": "تحليل"},
    "ui_enter_text_warn": {"en": "Please enter some text.", "ar": "الرجاء إدخال نص."},
    "ui_predicted_sentiment": {"en": "Predicted sentiment", "ar": "المشاعر المتوقعة"},
    "ui_predicted_category": {"en": "Predicted category", "ar": "الفئة المتوقعة"},
    "ui_confidence": {"en": "Confidence", "ar": "نسبة الثقة"},

    "ui_language_distribution": {"en": "Language Distribution", "ar": "توزيع اللغات"},
    "ui_sentiment_distribution": {"en": "Sentiment Distribution", "ar": "توزيع المشاعر"},
    "ui_category_distribution": {"en": "Category Distribution", "ar": "توزيع الفئات"},
    "ui_count": {"en": "Count", "ar": "العدد"},

    "ui_preview": {"en": "Preview (first rows)", "ar": "معاينة (أول الصفوف)"},
    "ui_shape": {"en": "Shape", "ar": "الأبعاد"},
    "ui_columns": {"en": "Columns", "ar": "الأعمدة"},
    "ui_rows": {"en": "Rows", "ar": "الصفوف"},
    "ui_upload_help": {"en": "Upload a CSV file to preview it.", "ar": "ارفع ملف CSV لمعاينته."},
    "ui_upload_error": {"en": "Could not read the file as CSV.", "ar": "تعذّر قراءة الملف كـ CSV."},

    "err_model_missing": {
        "en": "Model files are missing. Train them first:  python src/models/test_logistic_regression.py",
        "ar": "ملفات النموذج غير موجودة. درّب النماذج أولاً:  python src/models/test_logistic_regression.py",
    },
    "err_dataset_missing": {
        "en": "Dataset not found. Generate it (generate_dataset.py) then preprocess (run_preprocessing.py).",
        "ar": "البيانات غير موجودة. أنشئها (generate_dataset.py) ثم عالجها (run_preprocessing.py).",
    },
    "err_engine": {
        "en": "Search engine is unavailable. Make sure the dataset and TF-IDF artifacts exist.",
        "ar": "محرك البحث غير متاح. تأكد من وجود البيانات وملفات TF-IDF.",
    },

    # Localized class labels
    "sent_positive": {"en": "Positive", "ar": "إيجابي"},
    "sent_negative": {"en": "Negative", "ar": "سلبي"},
    "sent_neutral": {"en": "Neutral", "ar": "محايد"},
    "cat_course_feedback": {"en": "Course Feedback", "ar": "تقييم المساقات"},
    "cat_student_decisions": {"en": "Student Decisions", "ar": "قرارات الطلبة"},
    "cat_university_discussions": {"en": "University Discussions", "ar": "نقاشات جامعية"},

    # --- Model Comparison page ---------------------------------------------
    "mc_table_heading": {"en": "Results Table", "ar": "جدول النتائج"},
    "mc_charts_heading": {"en": "Charts", "ar": "الرسوم البيانية"},
    "mc_highlights_heading": {"en": "Highlights", "ar": "أبرز النتائج"},
    "mc_conclusions_heading": {"en": "Conclusions", "ar": "الاستنتاجات"},
    "mc_caption": {
        "en": "All scores use the same group-aware split (held-out template skeletons).",
        "ar": "جميع النتائج تستخدم نفس التقسيم المراعي للمجموعات (قوالب محجوبة).",
    },

    "mc_col_model": {"en": "Model", "ar": "النموذج"},
    "mc_col_sent_acc": {"en": "Sentiment Accuracy", "ar": "دقة المشاعر"},
    "mc_col_sent_f1": {"en": "Sentiment Macro-F1", "ar": "F1 الكلي للمشاعر"},
    "mc_col_cat_acc": {"en": "Category Accuracy", "ar": "دقة الفئة"},
    "mc_col_cat_f1": {"en": "Category Macro-F1", "ar": "F1 الكلي للفئة"},

    "mc_chart_sent_f1": {"en": "Sentiment Macro-F1 by model", "ar": "F1 الكلي للمشاعر حسب النموذج"},
    "mc_chart_cat_f1": {"en": "Category Macro-F1 by model", "ar": "F1 الكلي للفئة حسب النموذج"},

    "mc_best_sentiment": {"en": "Best Sentiment Model", "ar": "أفضل نموذج للمشاعر"},
    "mc_best_category": {"en": "Best Category Model", "ar": "أفضل نموذج للفئة"},
    "mc_leakage_note": {
        "en": ("Note: the Neural Network's category score (1.0000) is affected by "
               "embedding leakage — its Word2Vec vectors were trained on the full "
               "corpus, including the test rows. BiLSTM is the best leakage-free "
               "category model."),
        "ar": ("ملاحظة: نتيجة الفئة للشبكة العصبية (1.0000) متأثرة بتسرّب التمثيلات — "
               "دُرِّبت متجهات Word2Vec على كامل البيانات بما فيها بيانات الاختبار. "
               "BiLSTM هو أفضل نموذج للفئة بدون تسرّب."),
    },

    "mc_concl_1": {
        "en": "TF-IDF + Logistic Regression is a very strong baseline.",
        "ar": "‏TF-IDF مع الانحدار اللوجستي أساس قوي جداً.",
    },
    "mc_concl_2": {
        "en": "LSTM underperformed on this dataset.",
        "ar": "كان أداء LSTM ضعيفاً على هذه البيانات.",
    },
    "mc_concl_3": {
        "en": "BiLSTM significantly improved over LSTM.",
        "ar": "حسّن BiLSTM الأداء بشكل كبير مقارنةً بـ LSTM.",
    },
    "mc_concl_4": {
        "en": "AraBERT achieved the best sentiment performance.",
        "ar": "حقّق AraBERT أفضل أداء في تحليل المشاعر.",
    },
    "mc_concl_5": {
        "en": "Transformer models are strongest for Arabic sentiment understanding.",
        "ar": "نماذج المحوّلات (Transformers) هي الأقوى لفهم المشاعر بالعربية.",
    },

    # --- Professor & Course Insights page ----------------------------------
    "pi_search_label": {
        "en": "Search by professor or course",
        "ar": "ابحث باسم المدرّس أو المساق",
    },
    "pi_search_placeholder": {
        "en": "e.g. Ahmad Saleh or Algorithms",
        "ar": "مثال: أحمد صالح أو خوارزميات",
    },
    "pi_hint": {
        "en": "Showing all course feedback. Type above to filter by professor or course.",
        "ar": "عرض جميع تقييمات المساقات. اكتب بالأعلى للتصفية حسب المدرّس أو المساق.",
    },
    "pi_no_results": {
        "en": "No course feedback matches your search.",
        "ar": "لا توجد تقييمات مساقات مطابقة لبحثك.",
    },
    "pi_matching_comments": {"en": "Matching comments", "ar": "التعليقات المطابقة"},
    "pi_pct_positive": {"en": "Positive %", "ar": "نسبة الإيجابي"},
    "pi_pct_negative": {"en": "Negative %", "ar": "نسبة السلبي"},
    "pi_pct_neutral": {"en": "Neutral %", "ar": "نسبة المحايد"},
    "pi_keywords": {"en": "Top Keywords", "ar": "أبرز الكلمات"},
    "pi_samples": {"en": "Sample Comments", "ar": "نماذج من التعليقات"},

    # --- AI Student Assistant page -----------------------------------------
    "aia_input_label": {
        "en": "Ask a question or describe a complaint",
        "ar": "اطرح سؤالاً أو صف شكوى",
    },
    "aia_response_heading": {"en": "Assistant Response", "ar": "رد المساعد"},
    "aia_similar_heading": {"en": "Top Similar Records", "ar": "أقرب السجلات"},
    "aia_resp_course": {
        "en": ("This message appears related to **course feedback**. Detected "
               "sentiment: **{sentiment}**. Consider discussing it with your "
               "instructor, the teaching assistant, or your course representative."),
        "ar": ("يبدو أن رسالتك متعلقة بـ **تقييم المساقات**. المشاعر المكتشفة: "
               "**{sentiment}**. ننصحك بمناقشتها مع المدرّس أو مساعد التدريس أو "
               "ممثل المساق."),
    },
    "aia_resp_decisions": {
        "en": ("This message appears related to **student decisions or student "
               "movements**. Detected sentiment: **{sentiment}**. Check the official "
               "student council or university announcements for accurate updates."),
        "ar": ("يبدو أن رسالتك متعلقة بـ **قرارات الطلبة أو الحراك الطلابي**. "
               "المشاعر المكتشفة: **{sentiment}**. تابع إعلانات مجلس الطلبة أو "
               "الإعلانات الرسمية للجامعة."),
    },
    "aia_resp_discussions": {
        "en": ("This message appears related to **university services** "
               "(registration, fees, facilities, or campus discussions). Detected "
               "sentiment: **{sentiment}**. Check the relevant university office or "
               "official communication channels."),
        "ar": ("يبدو أن رسالتك متعلقة بـ **خدمات الجامعة** (التسجيل، الرسوم، المرافق، "
               "أو نقاشات الحرم). المشاعر المكتشفة: **{sentiment}**. راجع المكتب "
               "المختص أو قنوات التواصل الرسمية."),
    },
}


# --------------------------------------------------------------------------- #
# Page & feature catalog
# --------------------------------------------------------------------------- #
# Drives both the sidebar navigation and the landing feature grid.
PAGES: list[dict] = [
    {"module": "pages/Home.py",              "key": "home",               "icon": "🏠", "group": "group_overview",  "default": True},
    {"module": "pages/Course_Feedback.py",   "key": "course_feedback",    "icon": "📝", "group": "group_features"},
    {"module": "pages/Student_Decisions.py", "key": "student_decisions",  "icon": "📢", "group": "group_features"},
    {"module": "pages/Professor_Insights.py","key": "professor_insights", "icon": "👨‍🏫", "group": "group_features"},
    {"module": "pages/Search_Engine.py",     "key": "search_engine",      "icon": "🔍", "group": "group_features"},
    {"module": "pages/Analytics.py",         "key": "analytics",          "icon": "📊", "group": "group_features"},
    {"module": "pages/AI_Assistant.py",      "key": "ai_assistant",       "icon": "🤖", "group": "group_features"},
    {"module": "pages/Upload_Dataset.py",    "key": "upload_dataset",     "icon": "📤", "group": "group_tools"},
    {"module": "pages/Model_Comparison.py",  "key": "model_comparison",   "icon": "⚖️", "group": "group_tools"},
    {"module": "pages/About.py",             "key": "about",              "icon": "ℹ️", "group": "group_tools"},
]

# Feature cards shown on the dashboard (everything except Home itself).
FEATURES: list[dict] = [p for p in PAGES if p["key"] != "home"]


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #
def init_language() -> str:
    if "lang" not in st.session_state:
        st.session_state["lang"] = DEFAULT_LANG
    return st.session_state["lang"]


def get_lang() -> str:
    return st.session_state.get("lang", DEFAULT_LANG)


def t(key: str, lang: str | None = None) -> str:
    """Translate ``key``; fall back to English, then the raw key."""
    lang = lang or get_lang()
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang) or entry.get("en") or key


def is_rtl(lang: str | None = None) -> bool:
    return (lang or get_lang()) in RTL_LANGS


# --------------------------------------------------------------------------- #
# Theme (modern dark dashboard) + RTL/LTR
# --------------------------------------------------------------------------- #
def _inject_css(lang: str) -> None:
    direction = "rtl" if is_rtl(lang) else "ltr"
    text_align = "right" if is_rtl(lang) else "left"

    st.markdown(
        f"""
        <style>
            /* ---- Base dark theme ---- */
            .stApp {{
                background: radial-gradient(1200px 700px at 85% -10%, #16233a 0%, #0b1020 45%, #060912 100%);
                color: #e6ebf5;
            }}
            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #0e1730 0%, #0a1124 100%);
                border-right: 1px solid rgba(255,255,255,0.06);
            }}
            [data-testid="stHeader"] {{ background: transparent; }}
            .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}

            /* ---- Direction / RTL-LTR ---- */
            .stApp, section[data-testid="stSidebar"] {{
                direction: {direction};
                text-align: {text_align};
            }}

            /* ---- Headings ---- */
            h1, h2, h3 {{ color: #f4f7ff; font-weight: 700; }}
            h1 {{ letter-spacing: 0.2px; }}
            .bzu-muted {{ color: #9fb0cc; }}

            /* ---- Page hero ---- */
            .bzu-hero {{
                background: linear-gradient(135deg, rgba(63,94,168,0.30), rgba(20,28,52,0.30));
                border: 1px solid rgba(120,160,255,0.16);
                border-radius: 20px;
                padding: 26px 30px;
                margin-bottom: 22px;
            }}
            .bzu-hero h1 {{ margin: 0 0 6px 0; font-size: 1.9rem; }}
            .bzu-hero p {{ margin: 0; color: #b9c6e4; font-size: 1.02rem; }}

            /* ---- Generic grid (responsive, wraps) ---- */
            .bzu-grid {{
                display: flex;
                flex-wrap: wrap;
                gap: 16px;
                margin: 6px 0 10px 0;
            }}

            /* ---- Stat cards ---- */
            .stat-card {{
                flex: 1 1 200px;
                background: linear-gradient(160deg, rgba(40,58,96,0.55), rgba(16,24,44,0.6));
                border: 1px solid rgba(120,160,255,0.16);
                border-radius: 18px;
                padding: 18px 20px;
                box-shadow: 0 10px 26px rgba(0,0,0,0.28);
                transition: transform .15s ease, border-color .15s ease;
            }}
            .stat-card:hover {{ transform: translateY(-3px); border-color: rgba(120,160,255,0.4); }}
            .stat-card .stat-icon {{
                font-size: 1.4rem;
                width: 44px; height: 44px;
                display: inline-flex; align-items: center; justify-content: center;
                border-radius: 12px;
                background: rgba(120,160,255,0.14);
                border: 1px solid rgba(120,160,255,0.22);
                margin-bottom: 12px;
            }}
            .stat-card .stat-value {{ font-size: 2.0rem; font-weight: 800; color: #ffffff; line-height: 1; }}
            .stat-card .stat-label {{ margin-top: 6px; color: #9fb0cc; font-size: 0.92rem; }}

            /* ---- Pulse cards ---- */
            .pulse-card {{
                flex: 1 1 240px;
                background: linear-gradient(160deg, rgba(48,40,72,0.5), rgba(18,20,40,0.6));
                border: 1px solid rgba(170,140,255,0.18);
                border-radius: 18px;
                padding: 18px 20px;
            }}
            .pulse-card .pulse-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
            .pulse-card .pulse-icon {{ font-size: 1.3rem; }}
            .pulse-card .pulse-title {{ color: #cdbcff; font-weight: 700; font-size: 0.95rem; }}
            .pulse-card .pulse-value {{ color: #eef0ff; font-size: 1.05rem; font-weight: 600; }}

            /* ---- Feature cards ---- */
            .feature-card {{
                flex: 1 1 280px;
                background: linear-gradient(160deg, rgba(36,52,84,0.5), rgba(16,24,44,0.55));
                border: 1px solid rgba(120,160,255,0.14);
                border-radius: 18px;
                padding: 20px 22px;
                transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
            }}
            .feature-card:hover {{
                transform: translateY(-4px);
                border-color: rgba(120,160,255,0.45);
                box-shadow: 0 14px 30px rgba(0,0,0,0.3);
            }}
            .feature-card .feature-icon {{
                font-size: 1.6rem;
                width: 50px; height: 50px;
                display: inline-flex; align-items: center; justify-content: center;
                border-radius: 14px;
                background: rgba(120,160,255,0.12);
                border: 1px solid rgba(120,160,255,0.2);
                margin-bottom: 14px;
            }}
            .feature-card h3 {{ margin: 0 0 6px 0; font-size: 1.12rem; }}
            .feature-card p {{ margin: 0; color: #aab8d4; font-size: 0.92rem; }}

            /* ---- Section headers ---- */
            .bzu-section {{ margin: 26px 0 8px 0; }}
            .bzu-section h2 {{ margin: 0; font-size: 1.35rem; }}
            .bzu-section .bzu-muted {{ font-size: 0.92rem; }}

            /* ---- Badge ---- */
            .bzu-badge {{
                display: inline-block;
                padding: 3px 12px;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 600;
                background: rgba(255,176,32,0.15);
                color: #ffcf66;
                border: 1px solid rgba(255,176,32,0.35);
            }}

            /* ---- Sidebar brand ---- */
            .bzu-brand {{
                padding: 4px 4px 14px 4px;
                border-bottom: 1px solid rgba(255,255,255,0.07);
                margin-bottom: 14px;
            }}
            .bzu-brand .bzu-title {{ font-size: 1.12rem; font-weight: 800; color: #f4f7ff; }}
            .bzu-brand .bzu-sub {{ font-size: 0.76rem; color: #9fb0cc; }}

            /* ---- Footer ---- */
            .bzu-footer {{
                margin-top: 30px;
                padding-top: 12px;
                border-top: 1px solid rgba(255,255,255,0.07);
                color: #8294b3;
                font-size: 0.8rem;
            }}

            /* ---- Buttons ---- */
            .stButton > button {{
                border-radius: 10px;
                border: 1px solid rgba(120,160,255,0.25);
                background: rgba(120,160,255,0.12);
                color: #e6ebf5;
            }}
            .stButton > button:hover {{
                border-color: rgba(120,160,255,0.5);
                background: rgba(120,160,255,0.2);
            }}

            /* ---- Small screens ---- */
            @media (max-width: 640px) {{
                .bzu-hero {{ padding: 20px; }}
                .bzu-hero h1 {{ font-size: 1.5rem; }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar (brand + language switcher)
# --------------------------------------------------------------------------- #
def _render_brand_and_switcher() -> None:
    with st.sidebar:
        st.markdown(
            f'<div class="bzu-brand">'
            f'<div class="bzu-title">🎓 {t("app_name")}</div>'
            f'<div class="bzu-sub">{t("app_tagline")}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        labels = list(LANGUAGES.keys())
        current_code = get_lang()
        current_label = next(
            (lbl for lbl, code in LANGUAGES.items() if code == current_code),
            labels[0],
        )
        chosen = st.radio(
            t("language"),
            labels,
            index=labels.index(current_label),
            horizontal=True,
            key="lang_selector",
        )
        new_code = LANGUAGES[chosen]
        if new_code != current_code:
            st.session_state["lang"] = new_code
            st.rerun()


# --------------------------------------------------------------------------- #
# App bootstrap + navigation (called by app.py controller)
# --------------------------------------------------------------------------- #
def app_bootstrap() -> None:
    """Configure the app once per run: page config, theme, sidebar shell."""
    init_language()
    st.set_page_config(
        page_title=t("app_name"),
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css(get_lang())
    _render_brand_and_switcher()


def build_navigation():
    """Build translated, grouped, icon-rich navigation via ``st.navigation``."""
    grouped: dict[str, list] = {}
    for page in PAGES:
        st_page = st.Page(
            page["module"],
            title=t(f"nav_{page['key']}"),
            icon=page["icon"],
            default=page.get("default", False),
        )
        grouped.setdefault(t(page["group"]), []).append(st_page)
    return st.navigation(grouped, position="sidebar")


# --------------------------------------------------------------------------- #
# Reusable page UI builders
# --------------------------------------------------------------------------- #
def page_header(title_key: str, desc_key: str | None = None, icon: str = "") -> None:
    """Render the standard page hero header."""
    desc = f"<p>{t(desc_key)}</p>" if desc_key else ""
    prefix = f"{icon} " if icon else ""
    html = (
        f'<div class="bzu-hero"><h1>{prefix}{t(title_key)}</h1>{desc}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def section_header(title_key: str, sub_key: str | None = None) -> None:
    sub = f'<div class="bzu-muted">{t(sub_key)}</div>' if sub_key else ""
    st.markdown(
        f'<div class="bzu-section"><h2>{t(title_key)}</h2>{sub}</div>',
        unsafe_allow_html=True,
    )


def stat_cards(stats: list[dict]) -> None:
    """stats: list of {icon, value, label_key}."""
    cards = "".join(
        f'<div class="stat-card">'
        f'<div class="stat-icon">{s["icon"]}</div>'
        f'<div class="stat-value">{s["value"]}</div>'
        f'<div class="stat-label">{t(s["label_key"])}</div>'
        f"</div>"
        for s in stats
    )
    st.markdown(f'<div class="bzu-grid">{cards}</div>', unsafe_allow_html=True)


def pulse_cards(items: list[dict]) -> None:
    """items: list of {icon, title_key, value_key}."""
    cards = "".join(
        f'<div class="pulse-card">'
        f'<div class="pulse-head">'
        f'<span class="pulse-icon">{i["icon"]}</span>'
        f'<span class="pulse-title">{t(i["title_key"])}</span>'
        f"</div>"
        f'<div class="pulse-value">{t(i["value_key"])}</div>'
        f"</div>"
        for i in items
    )
    st.markdown(f'<div class="bzu-grid">{cards}</div>', unsafe_allow_html=True)


def feature_cards() -> None:
    """Render the dashboard feature grid from the PAGES catalog."""
    cards = "".join(
        f'<div class="feature-card">'
        f'<div class="feature-icon">{f["icon"]}</div>'
        f'<h3>{t("nav_" + f["key"])}</h3>'
        f'<p>{t(f["key"] + "_desc")}</p>'
        f"</div>"
        for f in FEATURES
    )
    st.markdown(f'<div class="bzu-grid">{cards}</div>', unsafe_allow_html=True)


def placeholder_banner() -> None:
    st.info(f"🚧 {t('placeholder_notice')}", icon="🚧")


def render_footer() -> None:
    st.markdown(
        f'<div class="bzu-footer">{t("footer")}</div>',
        unsafe_allow_html=True,
    )
