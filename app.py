
import streamlit as st

st.set_page_config(
    page_title="VYRON | DiffLens AI",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 VYRON — DiffLens AI")
st.subheader("Visual Intelligence Workspace")

st.write(
    "Welcome to VYRON. Your journey into visual intelligence starts here."
)

st.info(
    "Upload two images to prepare for visual comparison."
)

col1, col2 = st.columns(2)

with col1:
    st.file_uploader(
        "Upload the original image",
        type=["png", "jpg", "jpeg"],
        key="original"
    )

with col2:
    st.file_uploader(
        "Upload the comparison image",
        type=["png", "jpg", "jpeg"],
        key="comparison"
    )

st.caption("VYRON is under development. More analysis tools are coming soon.")

import streamlit as st

st.set_page_config(
    page_title="VYRON | DiffLens AI",
    page_icon="🔎",
    layout="wide"
)

st.title("VYRON — DiffLens AI")
st.subheader("Visual Intelligence Workspace")

st.write(
    "Welcome to VYRON. Your journey into visual intelligence starts here."
)

st.info("Upload two images to compare them.")

col1, col2 = st.columns(2)

with col1:
    original = st.file_uploader(
        "Upload the original image",
        type=["png", "jpg", "jpeg"],
        key="original"
    )

with col2:
    comparison = st.file_uploader(
        "Upload the comparison image",
        type=["png", "jpg", "jpeg"],
        key="comparison"
    )

if original and comparison:
    st.success("Both images uploaded successfully!")

st.caption("VYRON is under development. More analysis tools are coming soon.")
