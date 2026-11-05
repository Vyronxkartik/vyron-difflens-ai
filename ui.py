# ui.py
# Vyron DiffLense AI - Modern UI with professional scrolling animations

import gradio as gr
import time

# ====================== CUSTOM CSS ======================
custom_css = """
/* ===== Global Smooth Scroll ===== */
html {
    scroll-behavior: smooth;
}

/* ===== Base Theme ===== */
.gradio-container {
    background: #0a0a0f !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* ===== Sticky Header ===== */
.header-bar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(10, 10, 15, 0.85);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding: 16px 24px;
    margin-bottom: 32px;
}

/* ===== Reveal Animation Classes ===== */
.reveal {
    opacity: 0;
    transform: translateY(48px) scale(0.96);
    transition: 
        opacity 0.85s cubic-bezier(0.22, 1, 0.36, 1),
        transform 0.85s cubic-bezier(0.22, 1, 0.36, 1);
    will-change: opacity, transform;
}

.reveal.visible {
    opacity: 1;
    transform: translateY(0) scale(1);
}

/* Stagger delays */
.reveal.delay-1 { transition-delay: 0.08s; }
.reveal.delay-2 { transition-delay: 0.16s; }
.reveal.delay-3 { transition-delay: 0.24s; }
.reveal.delay-4 { transition-delay: 0.32s; }

/* ===== Cards ===== */
.card {
    background: #12121a;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: border-color 0.3s ease, transform 0.3s ease;
}

.card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}

/* ===== Chat Area ===== */
.chatbot {
    min-height: 480px !important;
    border-radius: 16px !important;
    background: #0f0f16 !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
}

/* ===== Input ===== */
.input-row {
    background: #12121a;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 8px;
}

/* ===== Buttons ===== */
.primary-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35) !important;
}

/* ===== Section Titles ===== */
.section-title {
    font-size: 1.4rem;
    font-weight: 650;
    letter-spacing: -0.02em;
    margin-bottom: 16px;
    color: #f1f1f5;
}

/* ===== Subtle Parallax helper ===== */
.parallax-slow {
    transition: transform 0.1s linear;
}
"""

# ====================== JS for Scroll Animations ======================
scroll_js = """
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {
        threshold: 0.12,
        rootMargin: "0px 0px -60px 0px"
    });

    document.querySelectorAll('.reveal').forEach(el => {
        observer.observe(el);
    });
}

// Run after Gradio loads
document.addEventListener('DOMContentLoaded', initScrollAnimations);
setTimeout(initScrollAnimations, 800);   // safety for Gradio dynamic load
setTimeout(initScrollAnimations, 1800);
"""

# ====================== UI BUILD ======================
def create_ui(chat_fn, clear_fn=None):
    with gr.Blocks(
        css=custom_css,
        js=scroll_js,
        title="Vyron DiffLense AI",
        theme=gr.themes.Base(
            primary_hue="indigo",
            neutral_hue="slate",
            font=["Inter", "system-ui", "sans-serif"]
        )
    ) as demo:

        # Sticky Header
        with gr.Row(elem_classes="header-bar"):
            gr.Markdown(
                """
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="width:36px; height:36px; background:linear-gradient(135deg,#6366f1,#8b5cf6); 
                                border-radius:10px; display:flex; align-items:center; justify-content:center;
                                font-weight:700; color:white; font-size:18px;">V</div>
                    <div>
                        <div style="font-size:1.25rem; font-weight:700; letter-spacing:-0.03em; color:#f8f8ff;">
                            Vyron DiffLense AI
                        </div>
                        <div style="font-size:0.8rem; color:#94a3b8; margin-top:1px;">
                            Professional visual intelligence
                        </div>
                    </div>
                </div>
                """,
                elem_classes="reveal"
            )

        # Hero / Intro Section
        with gr.Column(elem_classes="reveal delay-1"):
            gr.Markdown(
                """
                <div class="section-title">What can I help you create or analyze?</div>
                <p style="color:#94a3b8; margin-top:-8px; margin-bottom:24px;">
                    Upload images, describe ideas, or ask questions. DiffLense processes with high precision.
                </p>
                """
            )

        # Main Chat Card
        with gr.Column(elem_classes=["card", "reveal", "delay-2"]):
            chatbot = gr.Chatbot(
                label="",
                height=520,
                show_label=False,
                elem_classes="chatbot",
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=Vyron"),
                bubble_full_width=False
            )

            with gr.Row(elem_classes="input-row"):
                msg = gr.Textbox(
                    placeholder="Describe an image, ask for analysis, or give a creative direction...",
                    show_label=False,
                    container=False,
                    scale=6
                )
                submit_btn = gr.Button("Send", elem_classes="primary-btn", scale=1)

        # Feature Cards (scroll reveal)
        with gr.Row():
            with gr.Column(elem_classes=["card", "reveal", "delay-1"]):
                gr.Markdown("### 🔍 Precise Analysis")
                gr.Markdown("Deep visual understanding with structured insights.")

            with gr.Column(elem_classes=["card", "reveal", "delay-2"]):
                gr.Markdown("### 🎨 Creative Generation")
                gr.Markdown("High-quality diffusion-style outputs guided by your intent.")

            with gr.Column(elem_classes=["card", "reveal", "delay-3"]):
                gr.Markdown("### ⚡ Fast Iteration")
                gr.Markdown("Refine results quickly with natural language.")

        # Footer
        gr.Markdown(
            """
            <div style="text-align:center; color:#64748b; font-size:0.85rem; margin-top:48px; padding-bottom:40px;" 
                 class="reveal delay-4">
                Vyron DiffLense AI • Built for clarity and speed
            </div>
            """,
        )

        # Event handlers
        def user_message(user_msg, history):
            return "", history + [[user_msg, None]]

        def bot_response(history):
            user_msg = history[-1][0]
            # Call your real engine here
            response = chat_fn(user_msg) if chat_fn else "Processing with DiffLense..."
            history[-1][1] = response
            return history

        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, chatbot, chatbot
        )
        submit_btn.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, chatbot, chatbot
        )

        if clear_fn:
            clear_btn = gr.Button("Clear conversation", variant="secondary")
            clear_btn.click(clear_fn, None, chatbot)

    return demo


# ====================== Example usage ======================
if __name__ == "__main__":
    def dummy_chat(message):
        time.sleep(0.6)
        return f"**DiffLense response:**\n\nI analyzed your request: *{message}*\n\nHere's a refined professional interpretation."

    demo = create_ui(chat_fn=dummy_chat)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)