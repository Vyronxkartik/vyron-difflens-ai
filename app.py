
import os
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")

client = OpenAI(api_key=API_KEY) if API_KEY else None

SYSTEM_PROMPT = """
You are a professional, friendly, intelligent AI assistant.
Give accurate, clear, and useful answers.
Explain complex topics in simple language when appropriate.
If you are uncertain, be honest instead of inventing facts.
Use readable formatting and concise answers by default.
Never claim to have performed an action you did not perform.
"""

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nova AI Chat</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #101218;
    color: #f5f5f5;
}
header {
    padding: 18px;
    text-align: center;
    border-bottom: 1px solid #30333d;
    font-size: 22px;
    font-weight: bold;
}
header span { color: #8b9cff; }
#chat {
    max-width: 850px;
    margin: auto;
    padding: 20px;
    height: calc(100dvh - 160px);
    overflow-y: auto;
}
.message {
    padding: 14px 16px;
    margin: 12px 0;
    border-radius: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
.user {
    background: #303b67;
    margin-left: 12%;
}
.bot {
    background: #20232c;
    margin-right: 8%;
}
#composer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #101218;
    padding: 14px;
    border-top: 1px solid #30333d;
}
form {
    display: flex;
    gap: 10px;
    max-width: 850px;
    margin: auto;
}
textarea {
    flex: 1;
    resize: none;
    min-width: 0;
    height: 52px;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #414653;
    background: #20232c;
    color: white;
    font: inherit;
}
button {
    border: 0;
    border-radius: 12px;
    padding: 0 20px;
    cursor: pointer;
    font-weight: bold;
}
#send { background: #8b9cff; color: #101218; }
#reset { background: #30333d; color: white; padding: 10px 16px; }
button:disabled { opacity: 0.5; cursor: wait; }
footer { text-align: center; color: #999; font-size: 12px; padding: 8px; }
</style>
</head>
<body>
<header>✦ Nova <span>AI</span></header>

<div style="text-align:center;padding:12px">
    <button id="reset" type="button">New conversation</button>
</div>

<main id="chat" aria-live="polite">
    <div class="message bot">Hello! I'm Nova AI. How can I help you today?</div>
</main>

<div id="composer">
    <form id="form">
        <textarea id="input" maxlength="8000"
            placeholder="Message Nova AI..." required></textarea>
        <button id="send" type="submit">Send</button>
    </form>
    <footer>AI can make mistakes. Verify important information.</footer>
</div>

<script>
const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");
const reset = document.getElementById("reset");

let history = [];
let busy = false;

function addMessage(role, text) {
    const element = document.createElement("div");
    element.className = "message " + role;
    element.textContent = text;
    chat.appendChild(element);
    chat.scrollTop = chat.scrollHeight;
    return element;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = input.value.trim();
    if (!message || busy) return;

    busy = true;
    send.disabled = true;
    input.disabled = true;
    addMessage("user", message);
    input.value = "";

    const pending = addMessage("bot", "Thinking...");
    const previousHistory = history.slice();

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                history: previousHistory
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "The request failed.");
        }

        pending.textContent = data.reply;

        history.push(
            { role: "user", content: message },
            { role: "assistant", content: data.reply }
        );

        // Keep recent conversation context bounded.
        history = history.slice(-20);

    } catch (error) {
        pending.textContent = "Sorry, something went wrong: " + error.message;
    } finally {
        busy = false;
        send.disabled = false;
        input.disabled = false;
        input.focus();
        chat.scrollTop = chat.scrollHeight;
    }
});

reset.addEventListener("click", () => {
    history = [];
    chat.replaceChildren();
    addMessage("bot", "New conversation started. How can I help?");
    input.focus();
});
</script>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(HTML)


@app.post("/api/chat")
def chat_api():
    if client is None:
        return jsonify({
            "error": "API key is missing. Configure OPENAI_API_KEY in your .env file."
        }), 503

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON request."}), 400

    message = data.get("message")
    history = data.get("history", [])

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Please enter a message."}), 400

    if len(message) > 8000:
        return jsonify({"error": "Message is too long."}), 400

    if not isinstance(history, list):
        return jsonify({"error": "Invalid conversation history."}), 400

    # Accept only simple user/assistant text messages.
    safe_history = []
    for item in history[-20:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if (
            role in ("user", "assistant")
            and isinstance(content, str)
            and len(content) <= 8000
        ):
            safe_history.append({
                "role": role,
                "content": content
            })

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=safe_history + [
                {"role": "user", "content": message.strip()}
            ],
            max_output_tokens=1200
        )

        reply = response.output_text

        if not reply:
            reply = "I couldn't generate a response. Please try again."

        return jsonify({"reply": reply})

    except Exception:
        app.logger.exception("AI response generation failed")
        return jsonify({
            "error": "The AI service could not complete your request. Check your API configuration and try again."
        }), 502


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=False
    )
