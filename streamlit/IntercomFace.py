import streamlit as st

st.set_page_config(
    page_title="IntercomFace",
)

with open("README.md") as fh:
    long_description = fh.read()

st.write(long_description)
