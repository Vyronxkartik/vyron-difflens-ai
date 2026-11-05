"""VYRON - DiffLens AI (Visual Intelligence Workspace)

Files:
  app.py          this file: layout and flow
  engine.py       alignment + difference detection
  chatbot.py      assistant (Claude if a key is set, offline otherwise)
  ui.py           animated components
  assets/styles.css
"""
import csv
import io
import json

import streamlit as st

import chatbot
import engine
import ui

st.set_page_config(
    page_title="VYRON - DiffLens AI | Compare two images",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.load_css()
ui.hero()

# ------------------------------------------------------------------ settings
with st.sidebar:
    st.markdown("### Settings")
    sensitivity = st.slider("Sensitivity", 1, 100, 60, help="Higher finds subtler changes, but may flag noise.")
    min_region = st.slider("Minimum region size (%)", 0.01, 2.0, 0.05, 0.01, help="Ignore changed areas smaller than this share of the image.")
    smoothing = st.slider("Smoothing", 0, 5, 2, help="Blurs grain and compression noise before comparing.")
    auto_align = st.toggle("Auto-align image B to A", value=True, help="Fixes small shifts, zoom and tilt between the two photos.")
    st.caption("Settings apply instantly. Change them and the result refreshes.")


@st.cache_data(show_spinner=False, max_entries=6)
def run_compare(a_bytes, b_bytes, sens, min_pct, blur, align):
    return engine.compare(a_bytes, b_bytes, sens, min_pct, blur, align)


# ---------------------------------------------------------------- session
st.session_state.setdefault("chat", [])
st.session_state.setdefault("analysis", None)

tab_compare, tab_chat = st.tabs(["Compare", "Assistant"])

# ------------------------------------------------------------------ compare
with tab_compare:
    col_a, col_b = st.columns(2)
    with col_a:
        up_a = st.file_uploader("Image A (original)", type=["png", "jpg", "jpeg", "webp", "bmp"], key="up_a")
    with col_b:
        up_b = st.file_uploader("Image B (changed)", type=["png", "jpg", "jpeg", "webp", "bmp"], key="up_b")

    result = None
    if not (up_a and up_b):
        st.session_state["analysis"] = None
        st.info("Upload both images to start. Photos or screenshots both work.")
    else:
        try:
            bytes_a, bytes_b = up_a.getvalue(), up_b.getvalue()
            slot = ui.scanning_bar()
            result = run_compare(bytes_a, bytes_b, sensitivity, min_region, smoothing, auto_align)
            slot.empty()
            st.session_state["analysis"] = result.summary()
        except Exception as exc:
            st.error(f"Could not compare these images ({type(exc).__name__}). Try a different file format, such as PNG or JPG.")

    if result is not None:
        top_left, top_right = st.columns([1, 1.4])
        with top_left:
            ui.score_ring(result.score, result.verdict)
        with top_right:
            changed_tone = "good" if result.changed_pct < 1 else "warn" if result.changed_pct > 10 else ""
            ui.stat_cards(
                [
                    (str(len(result.regions)), "Changed regions", changed_tone),
                    (f"{result.changed_pct:.2f}%", "Area changed", changed_tone),
                    (f"{result.ssim:.3f}", "Structure match (SSIM)", ""),
                    ("Yes" if result.aligned else "No", "Auto-aligned", "good" if result.aligned else ""),
                ]
            )

        if auto_align and not result.aligned and result.size:
            st.caption("Auto-align found too little shared detail, so image B was only resized to match A.")

        t_swipe, t_marked, t_heat, t_mask = st.tabs(["Swipe A / B", "Marked changes", "Heatmap", "Change mask"])
        with t_swipe:
            ui.swipe_compare(engine.to_jpeg_b64(result.img_a), engine.to_jpeg_b64(result.img_b), "A", "B")
            st.caption("Drag the handle, or use the arrow keys, to reveal the difference.")
        with t_marked:
            st.image(result.annotated, use_container_width=True, caption="Numbered boxes match the table below.")
        with t_heat:
            st.image(result.heatmap, use_container_width=True, caption="Warmer colors mean stronger change.")
        with t_mask:
            st.image(result.mask, use_container_width=True, caption="White areas are detected changes.")

        st.markdown("#### Changed regions")
        if result.regions:
            rows = [
                {
                    "Region": r.id,
                    "Position (x, y)": f"{r.x}, {r.y}",
                    "Size (px)": f"{r.w} x {r.h}",
                    "Area %": round(r.area_pct, 2),
                    "Intensity": round(r.intensity, 1),
                    "Severity": r.severity,
                }
                for r in result.regions
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.success("No changes found at the current settings. Raise Sensitivity to look for subtler differences.")

        st.markdown("#### Export")
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Marked image", engine.to_png_bytes(result.annotated), "vyron_marked.png", "image/png", use_container_width=True)
        d2.download_button("Heatmap", engine.to_png_bytes(result.heatmap), "vyron_heatmap.png", "image/png", use_container_width=True)
        d3.download_button("Report (JSON)", json.dumps(result.summary(), indent=2), "vyron_report.json", "application/json", use_container_width=True)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["region", "x", "y", "width", "height", "area_pct", "intensity", "severity"])
        for r in result.regions:
            writer.writerow([r.id, r.x, r.y, r.w, r.h, round(r.area_pct, 3), round(r.intensity, 1), r.severity])
        d4.download_button("Regions (CSV)", buf.getvalue(), "vyron_regions.csv", "text/csv", use_container_width=True)

# ---------------------------------------------------------------- assistant
with tab_chat:
    live = chatbot.has_api_key(st.secrets)
    st.caption(
        "Live AI assistant is on." if live
        else "Built-in assistant is on. Add ANTHROPIC_API_KEY in your Streamlit secrets for the live AI version."
    )
    st.caption("The assistant sees only the numbers from your comparison, never your photos.")

    pending = None
    chip_cols = st.columns(2)
    for i, prompt in enumerate(chatbot.QUICK_PROMPTS):
        if chip_cols[i % 2].button(prompt, key=f"chip{i}", use_container_width=True):
            pending = prompt

    typed = st.chat_input("Ask about your comparison...")
    question = typed or pending

    if question:
        reply = chatbot.ask(question, st.session_state["chat"], st.session_state["analysis"], st.secrets)
        st.session_state["chat"] += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": reply},
        ]

    if not st.session_state["chat"]:
        with st.chat_message("assistant"):
            st.write("Hi, I'm the VYRON assistant. Upload two images, then ask me what changed or how to tune the result.")
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if st.session_state["chat"] and st.button("Clear chat"):
        st.session_state["chat"] = []
        st.rerun()
