import streamlit as st
from i18n import LANGUAGES
from api_client import query
from components import render_trace, render_legend, render_issues, render_confidence_banner
from styles import inject

st.set_page_config(page_title="Jatayu", layout="wide")
inject()

st.title("Jatayu")

# --- Sidebar: language selector ---
lang_display = st.sidebar.selectbox("Language", list(LANGUAGES.keys()))
lang_code = LANGUAGES[lang_display]

# --- Upload + query ---
uploaded_files = st.file_uploader(
    "Upload satellite image(s)", type=["tif", "tiff"], accept_multiple_files=True
)
question = st.text_input("Ask a question about the image")

if st.button("Analyse") and uploaded_files and question:
    data = query(uploaded_files, question, lang=lang_code)

    render_issues(data.get("issues", []))
    render_confidence_banner(data["confidence"])

    # --- Answer, translated if needed ---
    if lang_code != "en":
        st.markdown(f"### {data['answer_translated']}")
        with st.expander("Original (English)"):
            st.write(data["answer_english"])
    else:
        st.markdown(f"### {data['answer']}")

    # --- Overlay image + legend ---
    if data.get("overlay_png"):
        st.image(data["overlay_png"], caption="Analysis result")
    render_legend(data["legend"])

    # --- Trace panel ---
    render_trace(data["trace"])
