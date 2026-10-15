import streamlit as st
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="VYRON | DiffLens AI",
    page_icon="🔎",
    layout="wide",
)


def load_image(file):
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def block_ssim(a, b, block=8):
    """Simple structural similarity (0 to 1) using numpy only."""
    a = np.array(a.convert("L"), dtype=np.float64)
    b = np.array(b.convert("L"), dtype=np.float64)
    h = (a.shape[0] // block) * block
    w = (a.shape[1] // block) * block
    if h == 0 or w == 0:
        return 1.0
    a = a[:h, :w].reshape(h // block, block, w // block, block)
    b = b[:h, :w].reshape(h // block, block, w // block, block)

    mu_a = a.mean(axis=(1, 3))
    mu_b = b.mean(axis=(1, 3))
    var_a = a.var(axis=(1, 3))
    var_b = b.var(axis=(1, 3))
    cov = ((a - mu_a[:, None, :, None]) * (b - mu_b[:, None, :, None])).mean(axis=(1, 3))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    )
    return float(np.clip(ssim_map.mean(), 0, 1))


st.title("🔎 VYRON - DiffLens AI")
st.subheader("Visual Intelligence Workspace")
st.write("Welcome to VYRON. Your journey into visual intelligence starts here.")

col1, col2 = st.columns(2)

with col1:
    original = st.file_uploader(
        "Upload the original image",
        type=["png", "jpg", "jpeg"],
        key="original",
    )

with col2:
    comparison = st.file_uploader(
        "Upload the comparison image",
        type=["png", "jpg", "jpeg"],
        key="comparison",
    )

if original is None or comparison is None:
    st.info("Upload two images to start the visual comparison.")
else:
    try:
        img1 = load_image(original)
        img2 = load_image(comparison)

        if img2.size != img1.size:
            st.warning("Images have different sizes. The comparison image was resized to match the original.")
            img2 = img2.resize(img1.size)

        st.divider()
        p1, p2 = st.columns(2)
        with p1:
            st.image(img1, caption="Original", use_container_width=True)
        with p2:
            st.image(img2, caption="Comparison", use_container_width=True)

        st.divider()
        st.subheader("Analysis")

        threshold = st.slider(
            "Sensitivity (lower = detects smaller changes)",
            min_value=5,
            max_value=100,
            value=30,
        )

        arr1 = np.array(img1, dtype=np.float64)
        arr2 = np.array(img2, dtype=np.float64)

        diff = np.abs(arr1 - arr2).mean(axis=2)
        mask = diff > threshold

        similarity = block_ssim(img1, img2) * 100
        changed = mask.mean() * 100
        avg_diff = diff.mean()

        m1, m2, m3 = st.columns(3)
        m1.metric("Similarity", f"{similarity:.2f}%")
        m2.metric("Changed area", f"{changed:.2f}%")
        m3.metric("Average difference", f"{avg_diff:.2f}")

        if similarity > 99.9 and changed == 0:
            st.success("The images look identical.")
        elif changed < 1:
            st.info("The images are very similar with only tiny differences.")
        else:
            st.warning("Noticeable differences found.")

        highlight = np.array(img1).copy()
        red = np.array([255, 0, 0], dtype=np.float64)
        highlight[mask] = (0.4 * highlight[mask] + 0.6 * red).astype(np.uint8)

        diff_img = np.clip(diff * 3, 0, 255).astype(np.uint8)

        d1, d2 = st.columns(2)
        with d1:
            st.image(highlight, caption="Changes highlighted in red", use_container_width=True)
        with d2:
            st.image(diff_img, caption="Difference map (brighter = bigger change)", use_container_width=True)

    except Exception as e:
        st.error(f"Could not analyze these images: {e}")

st.caption("VYRON is under development. More analysis tools are coming soon.")