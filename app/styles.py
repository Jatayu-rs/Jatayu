import streamlit as st

def inject():
    st.markdown(
        """
        <style>
        .stage-label {
            color: white;
            padding: 2px 10px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        .legend-swatch {
            display: inline-block;
            width: 16px;
            height: 16px;
            border-radius: 3px;
            margin-right: 6px;
            vertical-align: middle;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )