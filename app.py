"""
VYRON - DiffLens AI  |  Visual Intelligence Workspace
Professional image comparison: robust alignment, SSIM + perceptual colour (Lab)
change detection, heatmaps, edges, histograms, flicker GIF, full reports.
"""
import io
import json
import time
import zipfile
import hashlib
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ExifTags
from skimage.metrics import structural_similarity

st.set_page_config(page_title="VYRON | DiffLens AI", page_icon="🔍", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1200px}
.vy-hero{padding:1rem 1.3rem;border-radius:16px;
  background:linear-gradient(135deg,#4f46e5,#06b6d4);color:#fff;margin-bottom:1rem}
.vy-hero h1{margin:0;font-size:1.7rem;color:#fff}
.vy-hero p{margin:.25rem 0 0;opacity:.92}
.vy-verdict{padding:.8rem 1rem;border-radius:12px;font-weight:600;margin:.5rem 0 1rem 0;color:#fff}
</style>
<div class="vy-hero"><h1>🔍 VYRON · DiffLens AI</h1>
<p>Visual Intelligence Workspace: precise, explainable image comparison</p></div>
""",
    unsafe_allow_html=True,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]

PRESETS = {
    "Balanced (recommended)": dict(sens=70, blur=2, min_area=0.05, align=True, light=False, otsu=False, border=1),
    "Strict: pixel-perfect (screenshots, UI, docs)": dict(sens=92, blur=1, min_area=0.005, align=False, light=False, otsu=False, border=0),
    "Tolerant: photos (lighting / camera shift)": dict(sens=50, blur=5, min_area=0.2, align=True, light=True, otsu=False, border=2),
}


# ----------------------------------------------------------------------------
# LOADING & PREPROCESSING
# ----------------------------------------------------------------------------
def file_info(f):
    raw = f.getvalue()
    img = Image.open(io.BytesIO(raw))
    info = {"Name": f.name, "Format": img.format, "Mode": img.mode,
            "Original size": f"{img.width} x {img.height}", "File size": f"{len(raw) / 1024:.1f} KB"}
    try:
        for tid, val in img.getexif().items():
            tag = ExifTags.TAGS.get(tid, tid)
            if tag in ("Make", "Model", "DateTime", "Software"):
                info[str(tag)] = str(val)[:60]
    except Exception:
        pass
    return info
def load_image(data: bytes, max_side: int) -> np.ndarray:
    """Load, fix EXIF rotation, flatten transparency on white, RGB uint8."""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    im = im.convert("RGB")
    w, h = im.size
    s = max_side / max(w, h)
    if s < 1:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return np.array(im)


def prep_b(arr, p):
    im = Image.fromarray(arr)
    if p["rot_b"]:
        im = im.rotate(-p["rot_b"], expand=True)
    if p["flip_h"]:
        im = ImageOps.mirror(im)
    if p["flip_v"]:
        im = ImageOps.flip(im)
    if p["bright_b"] != 1.0:
        im = ImageEnhance.Brightness(im).enhance(p["bright_b"])
    if p["contrast_b"] != 1.0:
        im = ImageEnhance.Contrast(im).enhance(p["contrast_b"])
    return np.array(im)


def to_gray_rgb(arr):
    g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return np.dstack([g, g, g])


def resize_to(arr, w, h):
    interp = cv2.INTER_AREA if arr.shape[1] > w else cv2.INTER_CUBIC
    return cv2.resize(arr, (w, h), interpolation=interp)


def align_orb(a, b):
    """Align B onto A: ORB + ratio test + RANSAC homography, with safety checks."""
    h, w = a.shape[:2]
    full = np.ones((h, w), np.uint8)
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=4000)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return b, full, {"status": "skipped", "reason": "not enough visual features"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(db, da, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 15:
        return b, full, {"status": "skipped", "reason": f"only {len(good)} good matches"}
    src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inl is None:
        return b, full, {"status": "skipped", "reason": "no stable transform found"}
    inliers = int(inl.sum())
    ratio = inliers / max(1, len(good))
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    shift = float(np.max(np.linalg.norm(cv2.perspectiveTransform(corners, H) - corners, axis=2)))
    info = {"status": "aligned", "matches": len(good), "inliers": inliers,
            "inlier_ratio": round(ratio, 3), "max_shift_px": round(shift, 1)}
    if inliers < 15 or ratio < 0.3:
        info.update(status="rejected", reason="low alignment confidence")
        return b, full, info
    if shift > 0.5 * float(np.hypot(w, h)):
        info.update(status="rejected", reason="implausible transform")
        return b, full, info
    if shift < 0.75:
        info.update(status="already_aligned")
        return b, full, info
    b_al = cv2.warpPerspective(b, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpPerspective(full, H, (w, h), flags=cv2.INTER_NEAREST)
    ek = max(9, int(min(h, w) * 0.02)) | 1
    valid = cv2.erode(valid, np.ones((ek, ek), np.uint8))
    return b_al, valid, info
def match_lighting(a, b):
    la = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
    lb = cv2.cvtColor(b, cv2.COLOR_RGB2LAB).astype(np.float32)
    sa, sb = la[..., 0], lb[..., 0]
    s = float(np.clip(sa.std() / (sb.std() + 1e-6), 0.5, 2.0))
    lb[..., 0] = np.clip((sb - sb.mean()) * s + sa.mean(), 0, 255)
    return cv2.cvtColor(lb.astype(np.uint8), cv2.COLOR_LAB2RGB)


# ----------------------------------------------------------------------------
# METRICS & DRAWING
# ----------------------------------------------------------------------------
def _hash_bits(img, kind):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if kind == "a":
        s = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        return (s > s.mean()).flatten()
    if kind == "d":
        s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        return (s[:, 1:] > s[:, :-1]).flatten()
    s = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(s)[:8, :8]
    return (d > np.median(d)).flatten()


def hash_similarity(a, b, kind):
    return float(1.0 - np.count_nonzero(_hash_bits(a, kind) != _hash_bits(b, kind)) / 64.0)


def hist_similarity(a, b):
    vals = []
    for ch in range(3):
        h1 = cv2.calcHist([a], [ch], None, [64], [0, 256])
        h2 = cv2.calcHist([b], [ch], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        vals.append(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    return float(np.clip(np.mean(vals), 0, 1))


def edge_analysis(a, b, low=80):
    ga = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    ea, eb = cv2.Canny(ga, low, low * 2), cv2.Canny(gb, low, low * 2)
    k = np.ones((3, 3), np.uint8)
    da, db = cv2.dilate(ea, k), cv2.dilate(eb, k)
    na, nb = int(np.count_nonzero(ea)), int(np.count_nonzero(eb))
    sim = 1.0 if na + nb == 0 else float((np.count_nonzero(ea & db) + np.count_nonzero(eb & da)) / (na + nb))
    vis = np.full((*ea.shape, 3), 18, np.uint8)
    vis[ea > 0] = (255, 90, 90)
    vis[eb > 0] = (80, 200, 255)
    vis[(ea > 0) & (eb > 0)] = (255, 255, 255)
    return sim, vis
def verdict(score):
    if score >= 0.98:
        return "Identical / near-identical", "#16a34a"
    if score >= 0.92:
        return "Very similar (minor differences)", "#65a30d"
    if score >= 0.80:
        return "Similar, with visible changes", "#d97706"
    if score >= 0.60:
        return "Noticeably different", "#ea580c"
    return "Substantially different", "#dc2626"


def where(cx, cy, w, h):
    v = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "middle"
    hz = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
    return "center" if (v == "middle" and hz == "center") else f"{v}-{hz}"


def annotate(img, regions, color, thick):
    out = img.copy()
    t = max(1, int(thick))
    fs = max(0.5, 0.5 + 0.1 * t)
    for r in regions:
        x, y, bw, bh = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, t)
        txt = str(r["id"])
        (tw, th), _ = cv2.getTextSize(txt, FONT, fs, 2)
        top = max(0, y - th - 8)
        cv2.rectangle(out, (x, top), (x + tw + 8, top + th + 8), color, -1)
        cv2.putText(out, txt, (x + 4, top + th + 2), FONT, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def heat_views(diff, base, gain):
    d8 = (np.clip(diff * gain / 3.0, 0, 1) * 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    alpha = np.clip(diff * gain, 0, 0.85)[..., None]
    overlay = (base * (1 - alpha) + heat * alpha).astype(np.uint8)
    return heat, overlay


# ----------------------------------------------------------------------------
# MAIN ANALYSIS (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=6)
def analyze(bytes_a: bytes, bytes_b: bytes, p: dict) -> dict:
    t0 = time.time()
    a = load_image(bytes_a, p["max_side"])
    b = prep_b(load_image(bytes_b, p["max_side"]), p)
    if p["gray"]:
        a, b = to_gray_rgb(a), to_gray_rgb(b)
    notes = []

    if a.shape[:2] != b.shape[:2]:
        ra, rb = a.shape[1] / a.shape[0], b.shape[1] / b.shape[0]
        if abs(ra - rb) / ra > 0.02:
            notes.append("Aspect ratios differ: one image was stretched. Results may overstate differences.")
        if p["size_mode"] == "Resize A to match B":
            a = resize_to(a, b.shape[1], b.shape[0])
        elif p["size_mode"] == "Resize both to smallest size":
            tw, th = min(a.shape[1], b.shape[1]), min(a.shape[0], b.shape[0])
            a, b = resize_to(a, tw, th), resize_to(b, tw, th)
        else:
            b = resize_to(b, a.shape[1], a.shape[0])
        notes.append(f"Sizes differed: {p['size_mode'].lower()}.")
    h, w = a.shape[:2]
    if min(h, w) < 16:
        raise ValueError("Images are too small to analyse (minimum 16 px).")

    valid = np.ones((h, w), np.uint8)
    align_info = {"status": "off"}
    if p["align"]:
        b, valid, align_info = align_orb(a, b)
        s_ = align_info["status"]
        if s_ == "aligned":
            notes.append(f"Auto-aligned B to A ({align_info['inliers']} inlier matches, max shift {align_info['max_shift_px']} px).")
        elif s_ == "already_aligned":
            notes.append("Images already aligned: no warping applied (pixels stay untouched).")
        else:
            notes.append(f"Alignment {s_}: {align_info.get('reason', '')}. Compared as-is.")

    b_cmp = b
    if p["light"]:
        b_cmp = match_lighting(a, b)
        notes.append("Lighting compensation applied to B for the comparison.")

    # --- SSIM (structure) ---
    ws = min(7, h, w)
    ws = ws if ws % 2 == 1 else ws - 1
    ssim_score, smap = structural_similarity(
        a, b_cmp, channel_axis=2, data_range=255, full=True,
        win_size=ws, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    smap = smap.mean(axis=2).astype(np.float32)

    # --- Perceptual colour difference (CIE Lab) ---
    fa = cv2.cvtColor(a.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    fb = cv2.cvtColor(b_cmp.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    dE = np.sqrt(((fa - fb) ** 2).sum(axis=2))

    # --- Combined change map ---
    diff = np.clip(0.6 * np.clip(1 - smap, 0, 1) + 0.4 * np.clip(dE / 40.0, 0, 1), 0, 1).astype(np.float32)
    diff *= valid.astype(np.float32)
    bpx = int(min(h, w) * p["border"] / 100)
    if bpx > 0:
        diff[:bpx, :] = 0
        diff[-bpx:, :] = 0
        diff[:, :bpx] = 0
        diff[:, -bpx:] = 0
    if p["blur"] > 0:
        k = 2 * p["blur"] + 1
        diff = cv2.GaussianBlur(diff, (k, k), 0)

    thr = 0.02 + 0.30 * (1 - p["sens"] / 100.0)
    if p["otsu"]:
        otsu, _ = cv2.threshold((diff * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(otsu / 255.0, 0.04)
    mask = (diff > thr).astype(np.uint8) * 255
    if p["clean"] > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((p["clean"], p["clean"]), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # --- Regions ---
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_px = p["min_area"] / 100.0 * h * w
    raw, final_mask = [], np.zeros((h, w), np.uint8)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_px:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.drawContours(final_mask, [c], -1, 255, -1)
        sub_d = diff[y:y + bh, x:x + bw]
        sub_m = mask[y:y + bh, x:x + bw] > 0
        vals = sub_d[sub_m] if sub_m.any() else sub_d.reshape(-1)
        raw.append((area, x, y, bw, bh, float(vals.mean()), float(vals.max())))
    raw.sort(key=lambda r: -r[0])
    regions = []
    for i, (area, x, y, bw, bh, md, pk) in enumerate(raw[:200], start=1):
        regions.append({
            "id": i, "location": where(x + bw / 2, y + bh / 2, w, h),
            "severity": "High" if md > 0.5 else "Medium" if md > 0.25 else "Low",
            "area_pct": round(100 * area / (h * w), 3), "mean_diff": round(md, 3), "peak_diff": round(pk, 3),
            "x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "area_px": int(area)})
    changed_pct = float(np.count_nonzero(final_mask) / (h * w) * 100)

    # --- Other metrics ---
    d32 = a.astype(np.float32) - b_cmp.astype(np.float32)
    mse, mae = float(np.mean(d32 ** 2)), float(np.mean(np.abs(d32)))
    psnr = 100.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    vmask = valid > 0
    de_mean = float(dE[vmask].mean()) if vmask.any() else float(dE.mean())
    de_p95 = float(np.percentile(dE[vmask], 95)) if vmask.any() else float(np.percentile(dE, 95))
    phash, dhash, ahash = (hash_similarity(a, b_cmp, k) for k in "pda")
    hist = hist_similarity(a, b_cmp)
    edge_sim, _ = edge_analysis(a, b_cmp)

    score = (0.35 * float(np.clip(ssim_score, 0, 1)) + 0.20 * float(np.clip(1 - changed_pct / 30.0, 0, 1))
             + 0.10 * phash + 0.10 * hist + 0.15 * float(np.clip(1 - de_mean / 25.0, 0, 1)) + 0.10 * edge_sim)

    color = tuple(p["box_color"])
    tint = np.where((final_mask > 0)[..., None],
                    (0.55 * b + 0.45 * np.array(color, np.float32)).astype(np.uint8), b)
    metrics = {
        "similarity_score": round(score * 100, 2), "ssim": round(float(ssim_score), 4),
        "psnr_db": round(psnr, 2), "mse": round(mse, 3), "mae": round(mae, 3),
        "delta_e_mean": round(de_mean, 3), "delta_e_p95": round(de_p95, 3),
        "changed_area_pct": round(changed_pct, 3), "change_regions": len(regions),
        "phash_similarity": round(phash, 4), "dhash_similarity": round(dhash, 4),
        "ahash_similarity": round(ahash, 4), "histogram_similarity": round(hist, 4),
        "edge_similarity": round(edge_sim, 4)}
    return {"a": a, "b": b, "b_cmp": b_cmp, "diff": diff, "ssim_map": smap, "mask": final_mask, "tint": tint,
            "annot_a": annotate(a, regions, color, p["box_thick"]),
            "annot_b": annotate(b, regions, color, p["box_thick"]),
            "regions": regions, "metrics": metrics, "notes": notes, "align": align_info,
            "threshold": round(float(thr), 4), "size": (w, h), "elapsed": round(time.time() - t0, 2)}


# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------
def to_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=4)
def make_gif(a, b, ms):
    s = min(640 / max(a.shape[:2]), 1.0)
    size = (max(1, int(a.shape[1] * s)), max(1, int(a.shape[0] * s)))
    fr = [Image.fromarray(cv2.resize(x, size, interpolation=cv2.INTER_AREA)) for x in (a, b)]
    buf = io.BytesIO()
    fr[0].save(buf, format="GIF", save_all=True, append_images=fr[1:], duration=ms, loop=0)
    return buf.getvalue()


def metric_grid(items, cols=2):
    for i in range(0, len(items), cols):
        for col, (label, val, hlp) in zip(st.columns(cols), items[i:i + cols]):
            col.metric(label, val, help=hlp)


def build_zip(files: dict):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ Settings")
    preset_name = st.selectbox("Preset", list(PRESETS))
    P = PRESETS[preset_name]
    K = lambda n: f"{n}::{preset_name}"  # noqa: E731  (widgets reset when preset changes)

    st.subheader("Preparation")
    size_mode = st.selectbox("If sizes differ", ["Resize B to match A", "Resize A to match B", "Resize both to smallest size"])
    max_side = st.select_slider("Analysis resolution (long side, px)", [640, 960, 1280, 1600, 1920, 2400], value=1280,
                                help="Higher = more detail but slower.")
    align = st.checkbox("Auto-align B to A (shift / rotation / zoom)", P["align"], key=K("align"))
    light = st.checkbox("Compensate lighting / exposure", P["light"], key=K("light"))
    gray = st.checkbox("Compare in grayscale", False)

    st.subheader("Adjust image B")
    rot_b = st.selectbox("Rotate B", [0, 90, 180, 270])
    flip_h = st.checkbox("Flip B horizontally")
    flip_v = st.checkbox("Flip B vertically")
    bright_b = st.slider("Brightness B", 0.5, 1.5, 1.0, 0.05)
    contrast_b = st.slider("Contrast B", 0.5, 1.5, 1.0, 0.05)

    st.subheader("Change detection")
    sens = st.slider("Sensitivity", 1, 100, P["sens"], key=K("sens"), help="Higher catches smaller differences.")
    otsu = st.checkbox("Auto threshold (Otsu)", P["otsu"], key=K("otsu"))
    blur = st.slider("Noise suppression", 0, 10, P["blur"], key=K("blur"))
    clean = st.slider("Merge nearby changes", 0, 25, 5)
    min_area = st.slider("Ignore regions smaller than (% of image)", 0.0, 5.0, float(P["min_area"]), 0.005,
                         key=K("minarea"), format="%.3f")
    border = st.slider("Ignore border (% of image)", 0, 10, P["border"], key=K("border"),
                       help="Useful for watermarks, scrollbars, status bars.")

    st.subheader("Display")
    box_hex = st.color_picker("Box colour", "#ff3b30")
    box_color = tuple(int(box_hex[i:i + 2], 16) for i in (1, 3, 5))
    box_thick = st.slider("Box thickness", 1, 8, 2)
    heat_gain = st.slider("Heatmap gain", 1.0, 8.0, 3.0, 0.5)

    st.divider()
    with st.expander("🕘 Session history"):
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()
        else:
            st.caption("No comparisons yet.")

params = dict(size_mode=size_mode, max_side=max_side, align=align, light=light, gray=gray,
              rot_b=rot_b, flip_h=flip_h, flip_v=flip_v, bright_b=bright_b, contrast_b=contrast_b,
              sens=sens, otsu=otsu, blur=blur, clean=clean, min_area=float(min_area), border=border,
              box_color=box_color, box_thick=box_thick)
up_a = st.file_uploader("Image A (original)", type=["png", "jpg", "jpeg", "webp", "bmp"], key="vyron_up_a")
up_b = st.file_uploader("Image B (changed)", type=["png", "jpg", "jpeg", "webp", "bmp"], key="vyron_up_b")
if not (up_a and up_b):
    st.info("Upload two images to start. **Balanced** works for most cases; use **Strict** for screenshots "
            "and **Tolerant** for photos. Open the sidebar (top-left arrow) for all settings.")
    st.stop()

bytes_a, bytes_b = up_a.getvalue(), up_b.getvalue()
try:
    with st.spinner("Analysing images..."):
        res = analyze(bytes_a, bytes_b, params)
    info_a, info_b = file_info(up_a), file_info(up_b)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not analyse these images: {e}")
    st.stop()

M = res["metrics"]
A, B = res["a"], res["b"]
w_px, h_px = res["size"]
heat, heat_overlay = heat_views(res["diff"], A, heat_gain)

hkey = hashlib.md5(bytes_a + bytes_b + repr(sorted(params.items())).encode()).hexdigest()
if st.session_state.get("last_hkey") != hkey:
    st.session_state.last_hkey = hkey
    st.session_state.history.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "A": up_a.name[:18],
                                        "B": up_b.name[:18], "score": M["similarity_score"],
                                        "regions": M["change_regions"]})
    st.session_state.history = st.session_state.history[:12]

cols_reg = ["id", "location", "severity", "area_pct", "mean_diff", "peak_diff", "x", "y", "w", "h", "area_px"]
regions_df = pd.DataFrame(res["regions"], columns=cols_reg)
metrics_df = pd.DataFrame({"metric": list(M.keys()), "value": list(M.values())})
report = {
    "app": "VYRON - DiffLens AI", "generated": datetime.now().isoformat(timespec="seconds"),
    "image_a": info_a, "image_b": info_b, "analysis_size_px": {"width": w_px, "height": h_px},
    "verdict": verdict(M["similarity_score"] / 100)[0], "metrics": M, "threshold_used": res["threshold"],
    "alignment": res["align"], "notes": res["notes"],
    "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
    "regions": res["regions"]}

tabs = st.tabs(["📊 Overview", "🔀 Compare", "🔥 Heatmap", "🎯 Changes", "🧠 SSIM map",
                "✏️ Edges", "🎨 Colors", "🎞️ Flicker", "💾 Export"])

# --- Overview
with tabs[0]:
    label, vcol = verdict(M["similarity_score"] / 100)
    st.markdown(f'<div class="vy-verdict" style="background:{vcol}">{label}: '
                f'{M["similarity_score"]:.1f}% similar</div>', unsafe_allow_html=True)
    st.progress(min(1.0, M["similarity_score"] / 100))
    metric_grid([
        ("Similarity score", f'{M["similarity_score"]:.1f}%', "Weighted blend of all metrics."),
        ("Structural (SSIM)", f'{M["ssim"]:.3f}', "1.0 = structurally identical."),
        ("Changed area", f'{M["changed_area_pct"]:.2f}%', "Share of the image flagged as changed."),
        ("Change regions", M["change_regions"], "Distinct changed areas after filtering."),
        ("Colour diff (ΔE mean)", f'{M["delta_e_mean"]:.2f}', "Under ~2 is hard to see; over ~10 is obvious."),
        ("PSNR", "∞" if M["psnr_db"] >= 100 else f'{M["psnr_db"]:.1f} dB', "Higher = closer."),
    ])
    for n in res["notes"]:
        st.caption("ℹ️ " + n)
    st.caption(f"Analysed at {w_px}×{h_px}px in {res['elapsed']}s · change threshold {res['threshold']}")
    with st.expander("All metrics"):
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    with st.expander("Image info"):
        i1, i2 = st.columns(2)
        i1.write("**Image A**")
        i1.json(info_a)
        i2.write("**Image B**")
        i2.json(info_b)
# ----------------------------------------------------------------------------
# UPLOAD + ANALYSIS
# ----------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    up_a = st.file_uploader("📁 Image A (original)", type=TYPES, key="a")
with c2:
    up_b = st.file_uploader("📁 Image B (modified)", type=TYPES, key="b")

# --- Compare
with tabs[1]:
    mode = st.radio("View", ["Side by side", "Split slider", "Opacity blend", "Red / Cyan overlay", "Difference only"],
                    horizontal=True)
    if mode == "Side by side":
        s1, s2 = st.columns(2)
        s1.image(A, caption="Image A", use_container_width=True)
        s2.image(B, caption="Image B (prepared / aligned)", use_container_width=True)
        shown = None
    elif mode == "Split slider":
        orient = st.radio("Direction", ["Vertical", "Horizontal"], horizontal=True)
        pos = st.slider("Move the divider", 0, 100, 50)
        shown = A.copy()
        if orient == "Vertical":
            x = int(w_px * pos / 100)
            shown[:, x:] = B[:, x:]
            cv2.line(shown, (x, 0), (x, h_px), (255, 255, 255), max(2, w_px // 400))
        else:
            y = int(h_px * pos / 100)
            shown[y:, :] = B[y:, :]
            cv2.line(shown, (0, y), (w_px, y), (255, 255, 255), max(2, h_px // 400))
        st.image(shown, use_container_width=True)
    elif mode == "Opacity blend":
        al = st.slider("Image B opacity", 0.0, 1.0, 0.5, 0.05)
        shown = cv2.addWeighted(A, 1 - al, B, al, 0)
        st.image(shown, use_container_width=True)
    elif mode == "Red / Cyan overlay":
        ga, gb = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY), cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
        shown = np.dstack([ga, gb, gb])
        st.caption("Red = only in A · Cyan = only in B · Gray = same.")
        st.image(shown, use_container_width=True)
    else:
        gain = st.slider("Amplify", 1, 10, 3)
        shown = np.clip(cv2.absdiff(A, res["b_cmp"]).astype(np.int32) * gain, 0, 255).astype(np.uint8)
        st.image(shown, use_container_width=True)
    if shown is not None:
        st.download_button("⬇️ Download this view", to_png(shown), "vyron_compare.png", "image/png")

# --- Heatmap
with tabs[2]:
    on = st.checkbox("Overlay on Image A", True)
    hs = heat_overlay if on else heat
    st.image(hs, caption="Hotter = bigger perceptual difference", use_container_width=True)
    st.download_button("⬇️ Download heatmap", to_png(hs), "vyron_heatmap.png", "image/png")

# --- Changes
with tabs[3]:
    if not res["regions"]:
        st.success("No significant changes detected with the current settings.")
        st.caption("Raise Sensitivity or lower the minimum region size to catch subtler differences.")
    else:
        view = st.radio("View", ["Boxes on B", "Boxes on A", "Tinted pixels", "Binary mask"], horizontal=True)
        shown = {"Boxes on B": res["annot_b"], "Boxes on A": res["annot_a"], "Tinted pixels": res["tint"],
                 "Binary mask": np.dstack([res["mask"]] * 3)}[view]
        st.image(shown, use_container_width=True)
        st.download_button("⬇️ Download this view", to_png(shown), "vyron_changes.png", "image/png")
        sev = st.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        st.dataframe(regions_df[regions_df["severity"].isin(sev)][["id", "location", "severity", "area_pct",
                                                                    "mean_diff", "peak_diff"]],
                     hide_index=True, use_container_width=True)
        rid = st.number_input("Zoom into region #", 1, len(res["regions"]), 1)
        r = res["regions"][int(rid) - 1]
        pad = 20
        y0, y1 = max(0, r["y"] - pad), min(h_px, r["y"] + r["h"] + pad)
        x0, x1 = max(0, r["x"] - pad), min(w_px, r["x"] + r["w"] + pad)
        z1, z2 = st.columns(2)
        z1.image(A[y0:y1, x0:x1], caption=f"A · region {r['id']}", use_container_width=True)
        z2.image(B[y0:y1, x0:x1], caption=f"B · region {r['id']}", use_container_width=True)

# --- SSIM map
with tabs[4]:
    sv = np.clip((1 - res["ssim_map"]) * 255 * heat_gain / 2, 0, 255).astype(np.uint8)
    ssim_img = cv2.cvtColor(cv2.applyColorMap(sv, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    st.image(ssim_img, caption="Structural dissimilarity: bright = changed most", use_container_width=True)
    st.caption("SSIM compares structure, contrast and luminance, so it matches human perception better than raw pixel difference.")
    st.download_button("⬇️ Download SSIM map", to_png(ssim_img), "vyron_ssim.png", "image/png")

# --- Edges
with tabs[5]:
    low = st.slider("Edge sensitivity", 20, 200, 80)
    esim, evis = edge_analysis(A, res["b_cmp"], low)
    st.image(evis, use_container_width=True)
    st.caption(f"🔴 only in A · 🔵 only in B · ⚪ both. Edge similarity: {esim * 100:.1f}%")
    st.download_button("⬇️ Download edge map", to_png(evis), "vyron_edges.png", "image/png")

# --- Colors
with tabs[6]:
    names = ["Red", "Green", "Blue"]
    chan = st.radio("Channel", ["Brightness"] + names, horizontal=True)
    if chan == "Brightness":
        va, vb = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY), cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
    else:
        i = names.index(chan)
        va, vb = A[..., i], B[..., i]
    st.line_chart(pd.DataFrame({"Image A": np.bincount(va.ravel(), minlength=256)[:256],
                                "Image B": np.bincount(vb.ravel(), minlength=256)[:256]}))
    st.caption(f"Histogram similarity {M['histogram_similarity'] * 100:.1f}% · "
               f"perceptual hash similarity {M['phash_similarity'] * 100:.1f}%")
    avg = pd.DataFrame({"Image A": A.reshape(-1, 3).mean(axis=0), "Image B": B.reshape(-1, 3).mean(axis=0)},
                       index=names).round(1)
    avg["Difference"] = (avg["Image B"] - avg["Image A"]).round(1)
    st.dataframe(avg, use_container_width=True)
    s1, s2 = st.columns(2)
    s1.image(np.full((60, 200, 3), tuple(int(v) for v in A.reshape(-1, 3).mean(axis=0)), np.uint8), caption="Average colour A")
    s2.image(np.full((60, 200, 3), tuple(int(v) for v in B.reshape(-1, 3).mean(axis=0)), np.uint8), caption="Average colour B")

# --- Flicker
with tabs[7]:
    ms = st.slider("Flip speed (ms)", 200, 1500, 600, 50)
    gif = make_gif(A, B, ms)
    st.image(gif, caption="Animated A ⇄ B: differences jump out", use_container_width=True)
    st.download_button("⬇️ Download GIF", gif, "vyron_flicker.gif", "image/gif")

# --- Export
with tabs[8]:
    zip_bytes = build_zip({
        "report.json": json.dumps(report, indent=2), "metrics.csv": metrics_df.to_csv(index=False),
        "regions.csv": regions_df.to_csv(index=False), "image_a.png": to_png(A), "image_b_prepared.png": to_png(B),
        "annotated_b.png": to_png(res["annot_b"]), "heatmap_overlay.png": to_png(heat_overlay),
        "change_mask.png": to_png(res["mask"]), "ssim_map.png": to_png(ssim_img),
        "edges.png": to_png(edge_analysis(A, res["b_cmp"])[1])})
    st.download_button("📦 Download everything (ZIP)", zip_bytes, "vyron_report.zip", "application/zip",
                       use_container_width=True)
    e1, e2 = st.columns(2)
    e1.download_button("Annotated image (PNG)", to_png(res["annot_b"]), "vyron_annotated.png", "image/png", use_container_width=True)
    e2.download_button("Report (JSON)", json.dumps(report, indent=2), "vyron_report.json", "application/json", use_container_width=True)
    e3, e4 = st.columns(2)
    e3.download_button("Metrics (CSV)", metrics_df.to_csv(index=False), "vyron_metrics.csv", "text/csv", use_container_width=True)
    e4.download_button("Regions (CSV)", regions_df.to_csv(index=False), "vyron_regions.csv", "text/csv", use_container_width=True)

st.divider()
st.caption("VYRON · DiffLens AI. Visual Intelligence Workspace. Scores are analytical estimates; tune the settings for your use case.")
