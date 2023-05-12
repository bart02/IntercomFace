import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Database",
    page_icon="💿",
)
st.title("💿 Database")

p = Path("faces")

for name in p.iterdir():
    if not name.is_dir():
        continue
    name_col, *cols = st.columns(5)
    with name_col:
        st.write(name.name)
    for i, file in enumerate(name.glob("*.jpg")):
        with cols[i % len(cols)]:
            st.image(str(file), caption=file.name, use_column_width=True)
