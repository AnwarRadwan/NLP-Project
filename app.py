"""BZU Student Intelligence Platform — Streamlit entry point (controller).

Bilingual (English / Arabic) NLP application for ENCS5342.

This file is the multipage controller: it configures the app once (theme,
sidebar brand, language switcher) and then dispatches to the selected page
via ``st.navigation``. Page content lives under ``pages/``.

Run with:
    streamlit run app.py
"""

from src.i18n.translations import app_bootstrap, build_navigation

# One-time setup: page config + dark theme + sidebar brand + language switcher.
app_bootstrap()

# Translated, grouped, icon-rich navigation -> run the selected page.
navigation = build_navigation()
navigation.run()
