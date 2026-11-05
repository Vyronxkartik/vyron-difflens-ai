"""ui.py - Glassmorphism UI for VYRON - DiffLens AI (Streamlit)."""

import streamlit as st

# ====== CHANGE THIS TO YOUR NAME ======
OWNER_NAME = "Your Name"
APP_NAME = "VYRON"
APP_TAGLINE = "DiffLens AI · Visual Intelligence Workspace"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

:root {
  --glass-bg: rgba(255, 255, 255, 0.08);
  --glass-border: rgba(255, 255, 255, 0.22);
  --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
  --accent1: #7f5af0;
  --accent2: #2cb67d;
  --accent3: #00d1ff;
  --text: #f4f4ff;
  --muted: rgba(244, 244, 255, 0.65);
}

html, body, [class*="css"], .stApp {
  font-family: 'Poppins', sans-serif !important;
  color: var(--text);
}

/* ---------- Animated gradient background ---------- */
.stApp {
  background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0b3d5c);
  background-size: 400% 400%;
  animation: bgShift 18s ease infinite;
  background-attachment: fixed;
}
@keyframes bgShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* Floating blurred orbs */
.stApp::before, .stApp::after {
  content: "";
  position: fixed;
  border-radius: 50%;
  filter: blur(80px);
  opacity: 0.45;
  z-index: 0;
  pointer-events: none;
  animation: floatOrb 14s ease-in-out infinite;
}
.stApp::before {
  width: 320px; height: 320px;
  background: var(--accent1);
  top: -80px; left: -80px;
}
.stApp::after {
  width: 380px; height: 380px;
  background: var(--accent3);
  bottom: -120px; right: -100px;
  animation-delay: -7s;
}
@keyframes floatOrb {
  0%, 100% { transform: translate(0, 0) scale(1); }
  50% { transform: translate(40px, 60px) scale(1.15); }
}

header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1100px; position: relative; z-index: 1; }

/* ---------- Glass card ---------- */
.glass {
  background: var(--glass-bg);
  backdrop-filter: blur(16px) saturate(160%);
  -webkit-backdrop-filter: blur(16px) saturate(160%);
  border: 1px solid var(--glass-border);
  border-radius: 22px;
  box-shadow: var(--glass-shadow);
  padding: 1.2rem 1.4rem;
  margin: 0.9rem 0;
  transition: transform 0.35s ease, box-shadow 0.35s ease, border-color 0.35s ease;
}
.glass:hover {
  transform: translateY(-4px);
  border-color: rgba(255, 255, 255, 0.4);
  box-shadow: 0 14px 40px rgba(127, 90, 240, 0.35);
}

/* ---------- Scroll-reveal animation ---------- */
.reveal { animation: fadeUp 0.9s ease both; }
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(40px) scale(0.97); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
@supports (animation-timeline: view()) {
  .reveal {
    animation: fadeUp linear both;
    animation-timeline: view();
    animation-range: entry 0% entry 55%;
  }
}

/* ---------- Hero ---------- */
.hero { text-align: center; padding: 2.2rem 1rem 1.6rem; }
.hero h1 {
  font-size: clamp(2.2rem, 7vw, 4rem);
  font-weight: 800;
  letter-spacing: 3px;
  margin: 0;
  background: linear-gradient(90deg, var(--accent1), var(--accent3), var(--accent2), var(--accent1));
  background-size: 300% 100%;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: textFlow 6s linear infinite;
}
@keyframes textFlow { to { background-position: 300% 0; } }
.hero p { color: var(--muted); font-weight: 300; margin-top: 0.4rem; font-size: 1.05rem; }

/* ---------- Name badge (always on display) ---------- */
.name-badge {
  position: fixed;
  top: 14px; right: 16px;
  z-index: 9999;
  padding: 6px 16px;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  background: rgba(255, 255, 255, 0.12);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--glass-border);
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
  animation: pulseGlow 3s ease-in-out infinite;
}
@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 8px rgba(127, 90, 240, 0.4); }
  50% { box-shadow: 0 0 22px rgba(0, 209, 255, 0.7); }
}

/* ---------- Scrolling marquee ---------- */
.marquee {
  overflow: hidden;
  white-space: nowrap;
  border-radius: 999px;
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  backdrop-filter: blur(12px);
  padding: 8px 0;
  margin: 0.6rem 0 1rem;
}
.marquee span {
  display: inline-block;
  padding-left: 100%;
  animation: scrollText 22s linear infinite;
  color: var(--muted);
  font-size: 0.9rem;
}
@keyframes scrollText { to { transform: translateX(-100%); } }

/* ---------- Streamlit widgets ---------- */
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(135deg, var(--accent1), var(--accent3));
  color: #fff;
  border: 1px solid var(--glass-border);
  border-radius: 14px;
  padding: 0.6rem 1.4rem;
  font-weight: 600;
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-3px) scale(1.03);
  box-shadow: 0 10px 28px rgba(0, 209, 255, 0.4);
  color: #fff;
}

[data-testid="stFileUploader"] section {
  background: var(--glass-bg);
  border: 2px dashed var(--glass-border);
  border-radius: 18px;
  backdrop-filter: blur(10px);
}
[data-testid="stFileUploader"] section:hover { border-color: var(--accent3); }

[data-testid="stSidebar"] {
  background: rgba(15, 12, 41, 0.55) !important;
  backdrop-filter: blur(18px);
  border-right: 1px solid var(--glass-border);
}

[data-testid="stMetric"] {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: 18px;
  padding: 0.8rem 1rem;
  backdrop-filter: blur(12px);
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  padding: 6px 16px;
  backdrop-filter: blur(10px);
}
.stTabs [aria-selected="true"] { background: rgba(127, 90, 240, 0.35); }

[data-testid="stChatMessage"] {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: 18px;
  backdrop-filter: blur(12px);
  animation: fadeUp 0.6s ease both;
}

img { border-radius: 16px; }

/* ---------- Footer ---------- */
.footer { text-align: center; color: var(--muted); font-size: 0.85rem; padding: 1.6rem 0 0.8rem; }
.footer b { color: var(--text); }

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: linear-gradient(var(--accent1), var(--accent3)); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }
</style>
"""


def inject_css():
    """Inject all glassmorphism CSS inline (no external file needed)."""
    st.markdown(CSS, unsafe_allow_html=True)


# Backward-compatible aliases in case app.py uses older names
load_css = inject_css
load_styles = inject_css
apply_styles = inject_css


def name_badge(name: str = OWNER_NAME):
    st.markdown(f'<div class="name-badge">✨ {name}</div>', unsafe_allow_html=True)


def render_header(title: str = APP_NAME, tagline: str = APP_TAGLINE, name: str = OWNER_NAME):
    inject_css()
    name_badge(name)
    st.markdown(
        f"""
        <div class="hero reveal">
          <h1>{title}</h1>
          <p>{tagline}</p>
        </div>
        <div class="marquee">
          <span>🔍 {title} — compare images with precision &nbsp;•&nbsp; Built by {name}
          &nbsp;•&nbsp; Upload two images and see every difference &nbsp;•&nbsp; Powered by {tagline}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def glass_card(title: str = "", body: str = "", icon: str = ""):
    head = f"<h3 style='margin:0 0 .4rem'>{icon} {title}</h3>" if title else ""
    st.markdown(
        f'<div class="glass reveal">{head}<div style="color:var(--muted)">{body}</div></div>',
        unsafe_allow_html=True,
    )


def section_title(text: str, icon: str = ""):
    st.markdown(
        f'<h2 class="reveal" style="margin-top:1.6rem">{icon} {text}</h2>',
        unsafe_allow_html=True,
    )


def glass_open():
    st.markdown('<div class="glass reveal">', unsafe_allow_html=True)


def glass_close():
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer(name: str = OWNER_NAME):
    st.markdown(
        f'<div class="footer reveal">Designed &amp; built by <b>{name}</b> · {APP_NAME} © 2026</div>',
        unsafe_allow_html=True,
    )
def hero(*args, **kwargs):
    render_header()


def footer(*args, **kwargs):
    render_footer()


def card(title="", body="", icon=""):
    glass_card(title, body, icon)


def __getattr__(name):
    # any other missing ui.xxx() call does nothing instead of crashing
    if name.startswith("__"):
        raise AttributeError(name)

    def _fallback(*args, **kwargs):
        pass

    return _fallback
