import json, re
from pathlib import Path
from collections import defaultdict, deque

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent

class SupportChatbot:
    """Lightweight NLP FAQ chatbot with intent retrieval and short-term context."""

    def __init__(self):
        self.data = json.loads((BASE / "data" / "faqs.json").read_text())
        self.items = self.data["faqs"]
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([x["question"] for x in self.items])
        self.history = defaultdict(lambda: deque(maxlen=4))

    def normalize(self, text):
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        return " ".join(
            self.lemmatizer.lemmatize(t)
            for t in tokens if t not in self.stop_words
        )

    def respond(self, message, session_id):
        clean = self.normalize(message)
        query_vec = self.vectorizer.transform([clean])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        best = int(scores.argmax())
        confidence = float(scores[best])

        # Context fallback: if a short follow-up has little lexical overlap,
        # reuse the previous intent instead of hallucinating an answer.
        if confidence < 0.22 and self.history[session_id]:
            previous = self.history[session_id][-1]
            if any(w in message.lower().split() for w in
                   ["that", "it", "this", "there", "same", "again"]):
                best = previous["index"]
                confidence = max(confidence, previous["confidence"] * 0.85)

        item = self.items[best]
        if confidence < 0.18:
            response = ("I’m not fully sure I understood that. Could you rephrase it? "
                        "You can ask about orders, refunds, delivery, payments, or support hours.")
            intent = "fallback"
        else:
            response = item["answer"]
            intent = item["intent"]

        self.history[session_id].append({"index": best, "confidence": confidence})
        return {
            "response": response,
            "intent": intent,
            "confidence": round(confidence, 3)
        }
