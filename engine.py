"""VYRON engine: image alignment + multi-signal difference detection."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity

MAX_SIDE = 1600


@dataclass
class Region:
    id: int
    x: int
    y: int
    w: int
    h: int
    area_pct: float
    intensity: float  # 0-100, how strong the change is
    severity: str


@dataclass
class Result:
    score: float
    verdict: str
    ssim: float
    changed_pct: float
    hist_similarity: float
    hash_similarity: float
    color_shift: float
    aligned: bool
    size: tuple
    regions: list = field(default_factory=list)
    img_a: np.ndarray | None = None
    img_b: np.ndarray | None = None
    annotated: np.ndarray | None = None
    heatmap: np.ndarray | None = None
    mask: np.ndarray | None = None

    def summary(self) -> dict:
        return {
            "similarity_score": round(self.score, 1),
            "verdict": self.verdict,
            "ssim": round(self.ssim, 4),
            "changed_area_pct": round(self.changed_pct, 2),
            "histogram_similarity": round(self.hist_similarity, 3),
            "perceptual_hash_similarity": round(self.hash_similarity, 3),
            "average_color_shift": round(self.color_shift, 2),
            "auto_aligned": self.aligned,
            "analysed_size": f"{self.size[0]}x{self.size[1]}",
            "region_count": len(self.regions),
            "top_regions": [
                {
                    "id": r.id,
                    "position": f"x={r.x}, y={r.y}",
                    "size": f"{r.w}x{r.h}",
                    "area_pct": round(r.area_pct, 2),
                    "intensity": round(r.intensity, 1),
                    "severity": r.severity,
                }
                for r in self.regions[:8]
            ],
        }


def load_image(data: bytes) -> np.ndarray:
    """Decode bytes to an RGB uint8 array, honouring EXIF rotation."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def _align(a: np.ndarray, b: np.ndarray):
    """Warp b onto a using ORB features + RANSAC homography.

    Returns (b_aligned, success). Falls back to a plain resize.
    """
    h, w = a.shape[:2]
    plain = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    try:
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(plain, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(4000)
        ka, da = orb.detectAndCompute(ga, None)
        kb, db = orb.detectAndCompute(gb, None)
        if da is None or db is None or len(ka) < 30 or len(kb) < 30:
            return plain, False
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(db, da), key=lambda m: m.distance)[:600]
        if len(matches) < 30:
            return plain, False
        src = np.float32([kb[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([ka[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or inliers is None or int(inliers.sum()) < 25:
            return plain, False
        # Reject wild warps (guards against bad matches on unrelated images).
        if abs(np.linalg.det(H[:2, :2])) < 0.5 or abs(np.linalg.det(H[:2, :2])) > 2.0:
            return plain, False
        warped = cv2.warpPerspective(
            plain, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return warped, True
    except cv2.error:
        return plain, False


def _dhash(img: np.ndarray, size: int = 16) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def _verdict(score: float) -> str:
    if score >= 98:
        return "Practically identical"
    if score >= 92:
        return "Very similar"
    if score >= 80:
        return "Minor differences"
    if score >= 60:
        return "Noticeable differences"
    if score >= 40:
        return "Major differences"
    return "Mostly different"


def _severity(intensity: float) -> str:
    if intensity >= 65:
        return "High"
    if intensity >= 35:
        return "Medium"
    return "Low"


def compare(
    data_a: bytes,
    data_b: bytes,
    sensitivity: int = 60,
    min_region_pct: float = 0.05,
    blur: int = 3,
    auto_align: bool = True,
) -> Result:
    a = load_image(data_a)
    b_raw = load_image(data_b)
    h, w = a.shape[:2]

    if auto_align:
        b, aligned = _align(a, b_raw)
    else:
        b, aligned = cv2.resize(b_raw, (w, h), interpolation=cv2.INTER_AREA), False

    k = max(1, int(blur) * 2 + 1)
    a_s = cv2.GaussianBlur(a, (k, k), 0) if blur else a
    b_s = cv2.GaussianBlur(b, (k, k), 0) if blur else b

    # Signal 1: structural difference (SSIM on grayscale).
    ga = cv2.cvtColor(a_s, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b_s, cv2.COLOR_RGB2GRAY)
    ssim_val, ssim_map = structural_similarity(ga, gb, full=True, data_range=255)
    struct_diff = np.clip(1.0 - ssim_map, 0, 1)

    # Signal 2: perceptual colour difference in Lab space.
    la = cv2.cvtColor(a_s, cv2.COLOR_RGB2LAB).astype(np.float32)
    lb = cv2.cvtColor(b_s, cv2.COLOR_RGB2LAB).astype(np.float32)
    delta_e = np.sqrt(((la - lb) ** 2).sum(axis=2))
    color_diff = np.clip(delta_e / 100.0, 0, 1)

    combined = np.clip(0.5 * struct_diff + 0.5 * color_diff * 1.6, 0, 1)
    combined_u8 = (combined * 255).astype(np.uint8)

    thr = int(np.interp(sensitivity, [1, 100], [95, 14]))
    _, mask = cv2.threshold(combined_u8, thr, 255, cv2.THRESH_BINARY)
    ksz = max(3, int(round(min(h, w) / 220)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = float(h * w)
    regions: list[Region] = []
    clean_mask = np.zeros_like(mask)
    for c in contours:
        area_pct = cv2.contourArea(c) / total * 100
        if area_pct < min_region_pct:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.drawContours(clean_mask, [c], -1, 255, -1)
        roi = combined[y : y + bh, x : x + bw]
        roi_mask = clean_mask[y : y + bh, x : x + bw] > 0
        intensity = float(roi[roi_mask].mean() * 100) if roi_mask.any() else 0.0
        regions.append(Region(0, x, y, bw, bh, area_pct, intensity, _severity(intensity)))
    regions.sort(key=lambda r: r.area_pct * (0.5 + r.intensity / 100), reverse=True)
    for i, r in enumerate(regions, 1):
        r.id = i

    changed_pct = float((clean_mask > 0).sum()) / total * 100

    # Signal 3 & 4: global colour histogram + perceptual hash.
    hist_scores = []
    for ch in range(3):
        ha = cv2.calcHist([a], [ch], None, [64], [0, 256])
        hb = cv2.calcHist([b], [ch], None, [64], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        hist_scores.append(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    hist_sim = float(np.clip(np.mean(hist_scores), 0, 1))
    hash_sim = float(1.0 - np.mean(_dhash(a) != _dhash(b)))

    pixel_score = 1.0 - min(1.0, changed_pct / 100 * 1.5)
    score = 100 * (
        0.40 * max(0.0, ssim_val)
        + 0.30 * pixel_score
        + 0.15 * hist_sim
        + 0.15 * hash_sim
    )
    if not regions:
        score = max(score, 97.0)
    score = float(np.clip(score, 0, 100))

    # Visual outputs.
    annotated = a.copy()
    tint = np.zeros_like(a)
    tint[:] = (255, 80, 90)
    m3 = (clean_mask > 0)[..., None]
    annotated = np.where(m3, (annotated * 0.55 + tint * 0.45).astype(np.uint8), annotated)
    thick = max(2, int(min(h, w) / 300))
    for r in regions:
        cv2.rectangle(annotated, (r.x, r.y), (r.x + r.w, r.y + r.h), (255, 255, 255), thick + 2)
        cv2.rectangle(annotated, (r.x, r.y), (r.x + r.w, r.y + r.h), (255, 90, 100), thick)
        label = str(r.id)
        fs = max(0.6, min(h, w) / 900)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        ty = max(th + 8, r.y)
        cv2.rectangle(annotated, (r.x, ty - th - 8), (r.x + tw + 10, ty), (255, 90, 100), -1)
        cv2.putText(annotated, label, (r.x + 5, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)

    heat = cv2.applyColorMap(combined_u8, cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heatmap = cv2.addWeighted(a, 0.35, heat, 0.65, 0)

    return Result(
        score=score,
        verdict=_verdict(score),
        ssim=float(ssim_val),
        changed_pct=changed_pct,
        hist_similarity=hist_sim,
        hash_similarity=hash_sim,
        color_shift=float(delta_e.mean()),
        aligned=aligned,
        size=(w, h),
        regions=regions,
        img_a=a,
        img_b=b,
        annotated=annotated,
        heatmap=heatmap,
        mask=clean_mask,
    )


def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def to_jpeg_b64(arr: np.ndarray, quality: int = 85, max_side: int = 1100) -> str:
    import base64

    img = Image.fromarray(arr)
    w, h = img.size
    s = max_side / max(w, h)
    if s < 1:
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()
