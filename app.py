import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="UniFit",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 UniFit")
st.caption("Your AI-powered university research assistant")


# ---------- SIDEBAR ----------

sidebar = st.sidebar

sidebar.markdown("### University")

universities = sorted(
    folder.name
    for folder in Path("data").iterdir()
    if folder.is_dir()
)

selected_university = sidebar.selectbox(
    "Select a university",
    universities,
    label_visibility="collapsed",
    filter_mode=None
)


university_path = Path("data") / selected_university

programs = sorted(
    file.stem
    for file in university_path.glob("*.md")
)

sidebar.divider()

sidebar.markdown("### Program")

selected_program = sidebar.radio(
    "Select a program",
    programs,
    label_visibility="collapsed"
)

sidebar.write("")
sidebar.divider()

sidebar.markdown("### Search Techniques")

selected_search_technique = sidebar.radio(
    "Select a search technique",
    ["Vector Search", "BM25 Search", "Hybrid Search", "Multi-query Search"],
    label_visibility="collapsed"
)



# ---------- MAIN ----------

st.write(f"**University:** {selected_university}")
st.write(f"**Program:** {selected_program}")
st.write(f"**Search Technique:** {selected_search_technique}")