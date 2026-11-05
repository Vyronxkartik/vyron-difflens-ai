"""VYRON UI helpers: styling, hero, animated score ring and swipe comparer."""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ASSETS = Path(__file__).parent


def load_css() -> None:
    css = (ASSETS / "Styles").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        """
        <div class="vy-hero">
          <div class="vy-lens"><i></i><i></i><i></i><b></b></div>
          <div>
            <h1>VYRON <span style="color:var(--muted);font-weight:500">DiffLens AI</span></h1>
            <p>Drop in two images. VYRON lines them up, finds every change and explains it.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scanning_bar():
    return st.markdown('<div class="vy-scan"></div>', unsafe_allow_html=True)


def stat_cards(items: list[tuple[str, str, str]]) -> None:
    """items: (value, label, tone) where tone is '', 'good' or 'warn'."""
    cards = "".join(
        f'<div class="vy-card vy-stat {tone}"><div class="v">{v}</div><div class="l">{l}</div></div>'
        for v, l, tone in items
    )
    st.markdown(f'<div class="vy-stats">{cards}</div>', unsafe_allow_html=True)


def score_ring(score: float, verdict: str) -> None:
    """Animated ring that counts up to the similarity score."""
    color = "#46e3d1" if score >= 92 else "#ffc46b" if score >= 60 else "#ff6a7a"
    html = f"""
    <style>
      body {{ margin:0; background:transparent; color:#e9ecff; font-family:'Manrope',system-ui,sans-serif; }}
      .wrap {{ display:flex; align-items:center; gap:20px; padding:4px 2px; }}
      svg {{ width:130px; height:130px; transform:rotate(-90deg); flex:0 0 130px; }}
      .track {{ stroke:rgba(160,170,255,.18); }}
      .bar {{ stroke:{color}; stroke-linecap:round; filter:drop-shadow(0 0 6px {color}88);
              stroke-dasharray:326.7; stroke-dashoffset:326.7;
              transition:stroke-dashoffset 1.4s cubic-bezier(.2,.8,.2,1); }}
      .num {{ font-size:2.6rem; font-weight:700; line-height:1; font-family:'Sora',system-ui,sans-serif; }}
      .lbl {{ color:#a3a9d6; font-size:.9rem; margin-top:6px; }}
      .verdict {{ font-size:1.2rem; font-weight:600; margin-top:2px; color:{color}; }}
    </style>
    <div class="wrap">
      <svg viewBox="0 0 120 120" aria-hidden="true">
        <circle class="track" cx="60" cy="60" r="52" fill="none" stroke-width="10"/>
        <circle id="bar" class="bar" cx="60" cy="60" r="52" fill="none" stroke-width="10"/>
      </svg>
      <div>
        <div class="num"><span id="n">0</span><span style="font-size:1.1rem;color:#a3a9d6"> / 100</span></div>
        <div class="verdict">{verdict}</div>
        <div class="lbl">Similarity score</div>
      </div>
    </div>
    <script>
      const target = {score:.1f}, bar = document.getElementById('bar'), n = document.getElementById('n');
      const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
      requestAnimationFrame(() => {{ bar.style.strokeDashoffset = 326.7 * (1 - target / 100); }});
      if (reduce) {{ n.textContent = target.toFixed(1); }}
      else {{
        const t0 = performance.now(), dur = 1400;
        (function tick(t) {{
          const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
          n.textContent = (target * e).toFixed(1);
          if (p < 1) requestAnimationFrame(tick);
        }})(t0);
      }}
    </script>
    """
    components.html(html, height=150)


def swipe_compare(a_b64: str, b_b64: str, label_a: str = "A", label_b: str = "B", height: int = 460) -> None:
    """Drag-to-reveal comparer. Sweeps once on load, then follows your finger/mouse."""
    html = f"""
    <style>
      body {{ margin:0; background:transparent; font-family:'Manrope',system-ui,sans-serif; }}
      #box {{ position:relative; height:{height - 10}px; border-radius:16px; overflow:hidden; touch-action:pan-y;
              background:#0b0e29; border:1px solid rgba(160,170,255,.2); user-select:none; -webkit-user-select:none; cursor:ew-resize; }}
      .layer {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; pointer-events:none; }}
      #top {{ clip-path:inset(0 50% 0 0); }}
      #bar {{ position:absolute; top:0; bottom:0; left:50%; width:3px; margin-left:-1.5px; background:#fff;
              box-shadow:0 0 14px rgba(70,227,209,.9); pointer-events:none; }}
      #knob {{ position:absolute; top:50%; left:50%; width:44px; height:44px; margin:-22px 0 0 -22px; border-radius:50%;
               background:#8b7bff; border:3px solid #fff; display:flex; align-items:center; justify-content:center;
               color:#fff; font-size:15px; box-shadow:0 4px 18px rgba(0,0,0,.45); pointer-events:none; }}
      .tag {{ position:absolute; top:10px; padding:4px 11px; border-radius:999px; font-size:12px; font-weight:700;
              color:#fff; background:rgba(16,20,54,.78); border:1px solid rgba(160,170,255,.35); pointer-events:none; }}
    </style>
    <div id="box" role="slider" aria-label="Compare images" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50" tabindex="0">
      <img class="layer" src="data:image/jpeg;base64,{b_b64}" alt="{label_b}">
      <img class="layer" id="top" src="data:image/jpeg;base64,{a_b64}" alt="{label_a}">
      <span class="tag" style="left:10px">{label_a}</span><span class="tag" style="right:10px">{label_b}</span>
      <div id="bar"></div><div id="knob">&#8596;</div>
    </div>
    <script>
      const box = document.getElementById('box'), topImg = document.getElementById('top'),
            bar = document.getElementById('bar'), knob = document.getElementById('knob');
      let pct = 50, touched = false;
      function set(p) {{
        pct = Math.max(0, Math.min(100, p));
        topImg.style.clipPath = 'inset(0 ' + (100 - pct) + '% 0 0)';
        bar.style.left = knob.style.left = pct + '%';
        box.setAttribute('aria-valuenow', Math.round(pct));
      }}
      function fromEvent(e) {{ const r = box.getBoundingClientRect(); set((e.clientX - r.left) / r.width * 100); }}
      box.addEventListener('pointerdown', e => {{ touched = true; box.setPointerCapture(e.pointerId); fromEvent(e); }});
      box.addEventListener('pointermove', e => {{ if (e.buttons || e.pointerType === 'touch') fromEvent(e); }});
      box.addEventListener('keydown', e => {{
        if (e.key === 'ArrowLeft') {{ touched = true; set(pct - 4); }}
        if (e.key === 'ArrowRight') {{ touched = true; set(pct + 4); }}
      }});
      // One-time intro sweep: right, left, back to centre.
      if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {{
        const t0 = performance.now(), dur = 2600;
        (function sweep(t) {{
          if (touched) return;
          const p = Math.min(1, (t - t0) / dur);
          set(50 + 48 * Math.sin(p * Math.PI * 2) * (1 - p * 0.15));
          if (p < 1) requestAnimationFrame(sweep); else set(50);
        }})(t0);
      }}
      set(50);
    </script>
    """
    components.html(html, height=height)
