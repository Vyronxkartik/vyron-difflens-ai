import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="VYRON | DiffLens AI", page_icon="🔍", layout="wide")

st.title("VYRON")
st.subheader("DiffLens AI — Image Difference Detector")
st.write("Upload two images to discover and highlight what changed.")

col1, col2 = st.columns(2)

with col1:
    image_a = st.file_uploader("Upload Original Image", type=["png", "jpg", "jpeg"], key="image_a")

with col2:
    image_b = st.file_uploader("Upload Modified Image", type=["png", "jpg", "jpeg"], key="image_b")

if image_a and image_b:
    original = Image.open(image_a).convert("RGB")
    modified = Image.open(image_b).convert("RGB")

    if original.size != modified.size:
        st.warning("The images have different dimensions. The second image will be resized to match the first.")
        modified = modified.resize(original.size)

    arr_a = np.array(original)
    arr_b = np.array(modified)

    difference = cv2.absdiff(arr_a, arr_b)
    gray = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    highlighted = arr_a.copy()
    highlighted[mask > 0] = [255, 40, 40]

    changed_pixels = int(np.count_nonzero(mask))
    total_pixels = mask.size
    percentage = changed_pixels / total_pixels * 100

    st.divider()
    st.subheader("Comparison Results")

    result1, result2, result3 = st.columns(3)
    result1.metric("Changed pixels", f"{changed_pixels:,}")
    result2.metric("Image area changed", f"{percentage:.2f}%")
    result3.metric("Status", "Differences found" if changed_pixels else "No major differences")

    left, middle, right = st.columns(3)

    with left:
        st.image(original, caption="Original Image", use_container_width=True)
    with middle:
        st.image(modified, caption="Modified Image", use_container_width=True)
    with right:
        st.image(highlighted, caption="Differences highlighted in red", use_container_width=True)

    result_image = Image.fromarray(highlighted)
    output = BytesIO()
    result_image.save(output, format="PNG")

    st.download_button(
        "Download Highlighted Image",
        data=output.getvalue(),
        file_name="vyron_differences.png",
        mime="image/png"
    )
else:
    st.info("Upload both images to start comparing.")