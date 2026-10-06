# AI-Powered Customer Support Chatbot

A production-style internship project using **Python, Flask, NLTK, scikit-learn TF-IDF, and SQLite**.

## Features
- NLP preprocessing with stop-word removal and lemmatization
- TF-IDF intent retrieval for FAQ responses
- Short-term conversation context
- Confidence-based fallback responses
- SQLite interaction logs
- Responsive browser chat UI
- Health-check endpoint

## Run locally

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
python -c "import nltk; [nltk.download(x) for x in ['stopwords','wordnet','omw-1.4']]"
python app.py
```

Open http://127.0.0.1:5000

## API
`POST /api/chat`
```json
{"message":"Where is my order?","session_id":"demo-1"}
```

`GET /api/health`

## Database
`chatbot.db` is created automatically. Each request stores the session ID, user message, response, detected intent, confidence, and timestamp.

## Deployment
Works with Gunicorn:
```bash
gunicorn app:app
```
Set `PORT` and `CHATBOT_DB` environment variables as needed.

## Project outcome
A working NLP customer-support chatbot with contextual interaction handling and persistent user interaction logs.
