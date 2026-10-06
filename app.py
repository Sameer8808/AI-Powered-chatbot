from flask import Flask, render_template, request, jsonify
from chatbot import SupportChatbot
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)
bot = SupportChatbot()
DB_PATH = os.getenv("CHATBOT_DB", "chatbot.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                intent TEXT,
                confidence REAL,
                created_at TEXT NOT NULL
            )
        """)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = (data.get("session_id") or "anonymous").strip()[:100]
    if not message:
        return jsonify({"error": "Message is required"}), 400

    result = bot.respond(message, session_id)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO chat_logs
               (session_id,user_message,bot_response,intent,confidence,created_at)
               VALUES (?,?,?,?,?,?)""",
            (session_id, message, result["response"], result["intent"],
             result["confidence"], datetime.utcnow().isoformat())
        )
    return jsonify(result)

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "ai-powered-chatbot"})

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
