import uuid
import streamlit as st


def init_session():
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        new_chat()
    if "user" not in st.session_state:
        st.session_state.user = None  # None = guest, not signed in


def new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"title": "New chat", "messages": []}
    st.session_state.current_chat_id = chat_id
    return chat_id


def get_current_chat():
    return st.session_state.chats[st.session_state.current_chat_id]


def switch_chat(chat_id):
    st.session_state.current_chat_id = chat_id


def add_message(question, data):
    chat = get_current_chat()
    chat["messages"].append({"question": question, "data": data})
    if chat["title"] == "New chat":
        chat["title"] = question[:40] + ("…" if len(question) > 40 else "")