"""Internationalization (i18n) and shared UI helpers.

Provides:
- Translation dictionaries for English (``en``) and Arabic (``ar``).
- ``t(key)`` lookup helper.
- A sidebar language switcher (persisted in ``st.session_state``).
- A modern dark-dashboard theme (CSS) with RTL support for Arabic.
- ``page_setup()`` — a one-call helper every page uses to configure itself.

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
# Each key maps to a {language_code: text} dictionary.
TRANSLATIONS: dict[str, dict[str, str]] = {
    # --- Brand / shell -----------------------------------------------------
    "app_name": {
        "en": "BZU Student Intelligence",
        "ar": "منصة ذكاء طلبة بيرزيت",
    },
    "app_tagline": {
        "en": "Bilingual NLP platform · ENCS5342",
        "ar": "منصة معالجة لغات طبيعية ثنائية اللغة · ENCS5342",
    },
    "language": {"en": "Language", "ar": "اللغة"},
    "navigation": {"en": "Navigation", "ar": "التنقل"},
    "coming_soon": {"en": "Coming soon", "ar": "قريباً"},
    "placeholder_notice": {
        "en": "This page is a placeholder. The NLP features will be added in a later step.",
        "ar": "هذه الصفحة مبدئية. ستتم إضافة ميزات معالجة اللغة في مرحلة لاحقة.",
    },
    "footer": {
        "en": "Birzeit University · Natural Language Processing (ENCS5342)",
        "ar": "جامعة بيرزيت · معالجة اللغات الطبيعية (ENCS5342)",
    },

    # --- Navigation labels -------------------------------------------------
    "nav_home": {"en": "Home", "ar": "الرئيسية"},
    "nav_course_feedback": {"en": "Course Feedback", "ar": "تقييم المساقات"},
    "nav_student_decisions": {"en": "Student Decisions", "ar": "قرارات الطلبة"},
    "nav_professor_insights": {"en": "Professor Insights", "ar": "رؤى المدرّسين"},
    "nav_upload_dataset": {"en": "Upload Dataset", "ar": "رفع البيانات"},
    "nav_search_engine": {"en": "Search Engine", "ar": "محرك البحث"},
    "nav_analytics": {"en": "Analytics", "ar": "لوحة التحليلات"},
    "nav_ai_assistant": {"en": "AI Assistant", "ar": "المساعد الذكي"},
    "nav_model_comparison": {"en": "Model Comparison", "ar": "مقارنة النماذج"},
    "nav_about": {"en": "About", "ar": "حول المشروع"},

    # --- Landing (app.py) --------------------------------------------------
    "landing_title": {
        "en": "BZU Student Intelligence Platform",
        "ar": "منصة ذكاء طلبة بيرزيت",
    },
    "landing_subtitle": {
        "en": "A bilingual (English / Arabic) NLP platform for understanding BZU student & community content.",
        "ar": "منصة معالجة لغات طبيعية ثنائية اللغة (عربي / إنجليزي) لفهم محتوى طلبة ومجتمع بيرزيت.",
    },
    "landing_get_started": {
        "en": "Use the sidebar to switch language and navigate between features.",
        "ar": "استخدم الشريط الجانبي لتبديل اللغة والتنقل بين الميزات.",
    },
    "features_heading": {"en": "Platform Features", "ar": "ميزات المنصة"},

    # --- Home dashboard ----------------------------------------------------
    "home_title": {"en": "Dashboard Home", "ar": "الصفحة الرئيسية للوحة"},
    "home_welcome": {
        "en": "Welcome to the BZU Student Intelligence Platform.",
        "ar": "مرحباً بك في منصة ذكاء طلبة بيرزيت.",
    },
    "home_overview": {
        "en": "This dashboard will surface insights from student feedback, university decisions, and course data once the NLP pipeline is connected.",
        "ar": "ستعرض هذه اللوحة رؤى مستخلصة من تقييمات الطلبة وقرارات الجامعة وبيانات المساقات بعد ربط مسار معالجة اللغة.",
    },
    "metric_documents": {"en": "Documents", "ar": "المستندات"},
    "metric_models": {"en": "Models", "ar": "النماذج"},
    "metric_languages": {"en": "Languages", "ar": "اللغات"},
    "metric_features": {"en": "Features", "ar": "الميزات"},

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
        "en": "Compare Logistic Regression, Neural Network, LSTM, and AraBERT.",
        "ar": "قارن بين الانحدار اللوجستي والشبكة العصبية وLSTM وAraBERT.",
    },

    # --- About -------------------------------------------------------------
    "about_title": {"en": "About the NLP Pipeline", "ar": "حول مسار معالجة اللغة"},
    "about_desc": {
        "en": "An overview of the end-to-end NLP pipeline powering this platform.",
        "ar": "نظرة عامة على مسار معالجة اللغة الكامل الذي يشغّل هذه المنصة.",
    },
    "about_pipeline_heading": {"en": "Pipeline Stages", "ar": "مراحل المسار"},
    "about_stack_heading": {"en": "Technology Stack", "ar": "التقنيات المستخدمة"},
}


# --------------------------------------------------------------------------- #
# Feature catalog (used on the landing page)
# --------------------------------------------------------------------------- #
FEATURES: list[dict[str, str]] = [
    {"icon": "📝", "key": "course_feedback"},
    {"icon": "📢", "key": "student_decisions"},
    {"icon": "👨‍🏫", "key": "professor_insights"},
    {"icon": "📤", "key": "upload_dataset"},
    {"icon": "🔍", "key": "search_engine"},
    {"icon": "📊", "key": "analytics"},
    {"icon": "🤖", "key": "ai_assistant"},
    {"icon": "⚖️", "key": "model_comparison"},
    {"icon": "ℹ️", "key": "about"},
]


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #
def init_language() -> str:
    """Ensure a language is set in session state and return its code."""
    if "lang" not in st.session_state:
        st.session_state["lang"] = DEFAULT_LANG
    return st.session_state["lang"]


def get_lang() -> str:
    """Return the active language code."""
    return st.session_state.get("lang", DEFAULT_LANG)


def t(key: str, lang: str | None = None) -> str:
    """Translate ``key`` into the active (or given) language.

    Falls back to English, then to the raw key, so missing keys never crash.
    """
    lang = lang or get_lang()
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang) or entry.get("en") or key


def is_rtl(lang: str | None = None) -> bool:
    return (lang or get_lang()) in RTL_LANGS


# --------------------------------------------------------------------------- #
# Theme (modern dark dashboard) + RTL
# --------------------------------------------------------------------------- #
def _inject_css(lang: str) -> None:
    direction = "rtl" if is_rtl(lang) else "ltr"
    text_align = "right" if is_rtl(lang) else "left"

    st.markdown(
        f"""
        <style>
            /* ---- Base dark theme ---- */
            .stApp {{
                background: radial-gradient(1200px 600px at 80% -10%, #16233a 0%, #0b1020 45%, #070a14 100%);
                color: #e6ebf5;
            }}
            section[data-testid="stSidebar"] {{
                background-color: #0d1426;
                border-right: 1px solid rgba(255,255,255,0.06);
            }}
            [data-testid="stHeader"] {{ background: transparent; }}

            /* ---- Direction / RTL ---- */
            .stApp, section[data-testid="stSidebar"] {{
                direction: {direction};
                text-align: {text_align};
            }}

            /* ---- Headings ---- */
            h1, h2, h3 {{ color: #f4f7ff; font-weight: 700; }}
            h1 {{ letter-spacing: 0.2px; }}

            /* ---- Dashboard card ---- */
            .bzu-card {{
                background: linear-gradient(160deg, rgba(36,52,84,0.55), rgba(18,26,46,0.55));
                border: 1px solid rgba(120,160,255,0.14);
                border-radius: 16px;
                padding: 20px 22px;
                margin-bottom: 16px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            }}
            .bzu-card h3 {{ margin-top: 0; }}
            .bzu-card .bzu-icon {{ font-size: 1.8rem; }}

            /* ---- Metric tiles ---- */
            [data-testid="stMetric"] {{
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(120,160,255,0.12);
                border-radius: 14px;
                padding: 14px 18px;
            }}

            /* ---- Badges ---- */
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

            /* ---- Brand block in sidebar ---- */
            .bzu-brand {{
                padding: 6px 4px 14px 4px;
                border-bottom: 1px solid rgba(255,255,255,0.07);
                margin-bottom: 12px;
            }}
            .bzu-brand .bzu-title {{ font-size: 1.15rem; font-weight: 800; color: #f4f7ff; }}
            .bzu-brand .bzu-sub {{ font-size: 0.78rem; color: #9fb0cc; }}

            /* ---- Footer ---- */
            .bzu-footer {{
                margin-top: 28px;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="bzu-brand">
                <div class="bzu-title">🎓 {t('app_name')}</div>
                <div class="bzu-sub">{t('app_tagline')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Language switcher
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


def render_footer() -> None:
    st.markdown(
        f'<div class="bzu-footer">{t("footer")}</div>',
        unsafe_allow_html=True,
    )


def placeholder_banner() -> None:
    """Standard 'coming soon' notice shown on feature placeholder pages."""
    st.info(f"🚧 {t('placeholder_notice')}", icon="🚧")


# --------------------------------------------------------------------------- #
# One-call page setup
# --------------------------------------------------------------------------- #
def page_setup(title_key: str, icon: str = "🎓") -> str:
    """Configure a Streamlit page (theme, language switcher) and return lang.

    Must be the first Streamlit call in each page script.
    """
    st.set_page_config(
        page_title=f"{t(title_key)} · {t('app_name')}",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    lang = init_language()
    _inject_css(lang)
    _render_sidebar()
    return lang
