import streamlit as st

st.title("My Streamlit App")

college = st.selectbox(
    "Select a university",
    [
        "Trinity College Dublin - TCD",
        "University College Dublin - UCD",
        "Dublin City University - DCU"
    ]
)

st.write("Selected:", college)