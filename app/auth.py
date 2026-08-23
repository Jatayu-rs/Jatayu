import streamlit as st


def render_login_widget():
    with st.sidebar.expander("Account", expanded=False):
        if st.session_state.user is None:
            st.caption("You're browsing as a guest — signing in is optional.")
            username = st.text_input("Username", key="login_username")
            if st.button("Sign in", key="login_button"):
                if username.strip():
                    st.session_state.user = {"name": username.strip()}
                    st.rerun()
        else:
            st.write(f"Signed in as **{st.session_state.user['name']}**")
            if st.button("Sign out", key="logout_button"):
                st.session_state.user = None
                st.rerun()