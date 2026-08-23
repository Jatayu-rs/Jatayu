import streamlit as st
from i18n import LANGUAGES
from api_client import query
from state import init_session, get_current_chat, add_message
from auth import render_login_widget
from components import (
    render_trace, render_legend, render_issues, render_confidence_banner,
    render_chat_sidebar, render_profile_icon,
)
from styles import inject

st.set_page_config(page_title="Jatayu", layout="wide")
inject()
init_session()

# --- Sidebar, top to bottom ---
render_chat_sidebar()
st.sidebar.markdown("---")
lang_display = st.sidebar.selectbox("Language", list(LANGUAGES.keys()))
lang_code = LANGUAGES[lang_display]
render_login_widget()
render_profile_icon()   # last call = renders at the bottom

# --- Main area ---
st.title("Jatayu")

uploaded_files = st.file_uploader("Upload satellite image(s)", type=["tif", "tiff"], accept_multiple_files=True)
question = st.text_input("Ask a question about the image")

if st.button("Analyse") and uploaded_files and question:
    data = query(uploaded_files, question, lang=lang_code)
    add_message(question, data)
    st.rerun()

# --- Replay current chat's history ---
chat = get_current_chat()
for msg in chat["messages"]:
    data = msg["data"]
    render_issues(data.get("issues", []))
    render_confidence_banner(data["confidence"])
    if lang_code != "en":
        st.markdown(f"### {data['answer_translated']}")
        with st.expander("Original (English)"):
            st.write(data["answer_english"])
    else:
        st.markdown(f"### {data['answer']}")
    if data.get("overlay_png"):
        st.image(data["overlay_png"], caption="Analysis result")
    render_legend(data["legend"])
    render_trace(data["trace"])
    st.divider()