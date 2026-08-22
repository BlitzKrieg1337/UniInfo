import streamlit as st
from pathlib import Path

from query import Query


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="UniInfo",
    page_icon="🎓",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================

sidebar = st.sidebar

# University
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

# Program
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

# Retrieval strategy
sidebar.write("")
sidebar.divider()
sidebar.markdown("### Search Techniques")

selected_search_technique = sidebar.radio(
    "Select a search technique",
    [
        "Vector Search",
        "BM25 Search",
        "Hybrid Search",
        "Multi-query Search"
    ],
    label_visibility="collapsed"
)
strategy_map = {
    "Vector Search": "similarity",
    "BM25 Search": "bm25",
    "Hybrid Search": "hybrid_search",
    "Multi-query Search": "multi_query"
}

strategy = strategy_map[selected_search_technique]


# =========================
# CURRENT SELECTION
# =========================

st.markdown(
    """
    <h1 style="
        text-align: center;
        transform: translateY(-50px);
        margin-bottom: 0;
    ">
        🎓 UniInfo
    </h1>
    """,
    unsafe_allow_html=True
)

st.write(f"**University:** {selected_university} \t**Program:** {selected_program}")

# =========================
# QUERY ENGINE
# =========================

@st.cache_resource(show_spinner=False)
def load_query(college, program):
    return Query(college, program)

with st.spinner(f"Just a second..."):
    query_engine = load_query(
        selected_university,
        selected_program
    )

# =========================
# CHAT INPUT
# =========================

user_query = st.chat_input("Enter your question here...")


if user_query:

    with st.chat_message("user"):
        st.write(user_query)

    answer, documents = query_engine.answer_query(
        user_query,
        strategy=strategy
    )

    with st.chat_message("assistant"):
        st.write(answer)

        if documents:
            st.write("**Source:**")
            st.write(
                documents[0].metadata.get("source", "Unknown")
            )
