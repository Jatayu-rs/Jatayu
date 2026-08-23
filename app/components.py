import json
import streamlit as st

STAGE_COLORS = {
    "check": "#3B82F6",
    "task": "#8B5CF6",
    "select": "#F59E0B",
    "run": "#10B981",
    "assemble": "#EF4444",
}

CONFIDENCE_THRESHOLD = 0.5


def render_trace(trace_steps: list[dict]):
    st.subheader("How this answer was produced")
    for i, step in enumerate(trace_steps, start=1):
        color = STAGE_COLORS.get(step["stage"], "#6B7280")
        with st.container(border=True):
            st.markdown(
                f"<span class='stage-label' style='background:{color}'>"
                f"{step['stage'].upper()}</span>&nbsp;&nbsp;**{i}. {step['detail']}**",
                unsafe_allow_html=True,
            )
            st.caption(f"{step['duration_ms']} ms")

    st.download_button(
        "Copy trace as JSON",
        data=json.dumps(trace_steps, indent=2),
        file_name="trace.json",
        mime="application/json",
    )


def render_legend(legend: dict[str, str]):
    if not legend:
        return
    st.subheader("Legend")
    cols = st.columns(len(legend))
    for col, (label, color) in zip(cols, legend.items()):
        with col:
            st.markdown(
                f"<span class='legend-swatch' style='background:{color}'></span>{label}",
                unsafe_allow_html=True,
            )
def render_issues(issues: list[dict]):
    for issue in issues:
        if issue["severity"] == "error":
            st.error(issue["message"])
        else:
            st.warning(issue["message"])


def render_confidence_banner(confidence: float):
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ The system is not confident in this answer. Treat it as a lead, not a fact.")
