"""VYRON assistant.

Uses Claude when an API key is available (Streamlit secret ANTHROPIC_API_KEY),
otherwise falls back to a built-in offline assistant that answers from the
analysis results. Either way it only ever sees the numbers, not your photos.
"""
from __future__ import annotations

import json

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = (
    "You are the assistant inside VYRON, an image comparison app (DiffLens AI). "
    "You are given a JSON analysis of two uploaded images. Explain results in plain, "
    "friendly language, point to numbered regions, suggest settings to try "
    "(sensitivity, minimum region size, smoothing, auto-align), and say honestly when "
    "the data cannot answer a question. You cannot see the images themselves, only the "
    "analysis. Keep answers short: 2 to 5 sentences unless asked for more. "
    "Do not invent numbers that are not in the analysis."
)

QUICK_PROMPTS = [
    "Summarize the result",
    "What changed the most?",
    "Why is the score not 100?",
    "How do I get more accurate results?",
]


def has_api_key(secrets) -> bool:
    try:
        return bool(secrets.get("ANTHROPIC_API_KEY"))
    except Exception:
        return False


def ask(question: str, history: list[dict], analysis: dict | None, secrets) -> str:
    """Return the assistant's reply to `question`."""
    if has_api_key(secrets):
        try:
            return _ask_claude(question, history, analysis, secrets)
        except Exception as exc:  # network, quota, bad key...
            return _offline(question, analysis) + f"\n\n_(Live AI is unavailable right now: {type(exc).__name__}. Showing the built-in answer.)_"
    return _offline(question, analysis)


def _ask_claude(question, history, analysis, secrets) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=secrets["ANTHROPIC_API_KEY"])
    model = secrets.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    context = json.dumps(analysis) if analysis else "No analysis yet: the user has not compared two images."
    messages = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
    messages.append({"role": "user", "content": question})
    resp = client.messages.create(
        model=model,
        max_tokens=600,
        system=f"{SYSTEM_PROMPT}\n\nAnalysis:\n{context}",
        messages=messages,
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


# ---------------------------------------------------------------- offline mode
def _offline(question: str, a: dict | None) -> str:
    q = question.lower()

    if any(w in q for w in ("hello", "hi ", "hey")) and len(q) < 20:
        return "Hi! Upload two images and I'll explain what changed between them."

    if not a:
        return (
            "I don't have a comparison yet. Upload image A and image B on the Compare tab "
            "and I'll walk you through the result."
        )

    regions = a["top_regions"]
    n = a["region_count"]

    if any(w in q for w in ("accura", "better", "improve", "precise", "wrong", "false", "noise", "too many", "missed")):
        tips = []
        if not a["auto_aligned"]:
            tips.append("turn on **Auto-align** if the photos were taken from slightly different positions")
        if n > 8:
            tips.append("raise **Minimum region size** and **Smoothing** to drop tiny noise")
        if n == 0 and a["similarity_score"] < 99:
            tips.append("raise **Sensitivity** so subtle changes are picked up")
        tips.append("use two images with the same framing and lighting, and avoid screenshots of screenshots")
        return "To get cleaner results: " + "; ".join(tips) + "."

    if any(w in q for w in ("not 100", "why", "score")):
        return (
            f"The score is {a['similarity_score']}/100, a blend of structure (SSIM {a['ssim']}), "
            f"changed area ({a['changed_area_pct']}%), colour histogram match ({a['histogram_similarity']}) "
            f"and perceptual hash match ({a['perceptual_hash_similarity']}). "
            "Anything below 100 means at least one of those signals saw a difference, "
            "even a tiny one like compression or lighting."
        )

    if any(w in q for w in ("most", "biggest", "largest", "main", "top", "where")):
        if not regions:
            return "I found no regions above the current thresholds, so nothing stands out."
        r = regions[0]
        return (
            f"Region {r['id']} is the biggest change: {r['size']} px at {r['position']}, "
            f"covering {r['area_pct']}% of the image with {r['severity'].lower()} intensity ({r['intensity']}/100)."
        )

    if any(w in q for w in ("region", "list", "all", "how many", "count")):
        if not regions:
            return "No regions were detected."
        lines = [f"Region {r['id']}: {r['area_pct']}% of the image, {r['severity'].lower()} intensity" for r in regions[:5]]
        more = f" (showing 5 of {n})" if n > 5 else ""
        return f"I found {n} changed region{'s' if n != 1 else ''}{more}:\n\n" + "\n".join(f"- {l}" for l in lines)

    if any(w in q for w in ("align", "shift", "rotate", "crop")):
        return (
            "Auto-align was used and matched the images." if a["auto_aligned"]
            else "Auto-align did not find enough shared detail, so image B was only resized to match A. "
            "If the framing differs, try images that share more visible detail."
        )

    if any(w in q for w in ("sensitiv", "setting", "threshold")):
        return (
            "Higher **Sensitivity** catches subtler changes but may flag noise. "
            "**Minimum region size** ignores tiny blobs, and **Smoothing** blurs away grain before comparing."
        )

    # Default: summary
    top = ""
    if regions:
        r = regions[0]
        top = f" The largest change is region {r['id']} ({r['area_pct']}% of the image, {r['severity'].lower()} intensity)."
    return (
        f"**{a['verdict']}** - similarity {a['similarity_score']}/100. "
        f"{n} changed region{'s' if n != 1 else ''} covering {a['changed_area_pct']}% of the image.{top}"
    )
