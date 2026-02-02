from memory import calendar_events, event_count, embeddings, next_id
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from numpy.linalg import norm

# Embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ollama LLM
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:270m"

def call_ollama(prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data.get("response", "No reply from Ollama")
    except Exception as e:
        print("Ollama error:", e)
        return "Sorry, I couldn't process your request."


def store_embedding(content, doc_type="conversation"):
    """Store embedding in the in-memory dictionary"""
    global next_id
    vector = embed_model.encode(content)
    embeddings[next_id] = {"content": content, "vector": vector, "type": doc_type}
    next_id += 1
    return next_id - 1

def retrieve_top_k(query_vector, k=3):
    """Retrieve top-k most similar embeddings"""

    scored = [(doc, cosine_similarity(query_vector, data["vector"]))
              for doc, data in embeddings.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [embeddings[doc]["content"] for doc, score in scored[:k]]

def cosine_similarity(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))


def create_event(title, start_time, end_time, participants = None, notes=""):
    global event_count
    calendar_events[event_count] = {
        "event_id": event_count, 
        "title": title, 
        "start_time": start_time, 
        "end_time": end_time, 
        "participants": participants or [], 
        "notes": notes
    }
    event = calendar_events[event_count]
    event_count += 1
    return event

    
def update_event(event_id, **updates):
    if event_id in calendar_events:
        calendar_events[event_id].update(updates)
        return calendar_events[event_id]
    return None

def delete_event(event_id):
    event = calendar_events.pop(event_id)
    return event

def query_event(start_date=None, end_date=None, participants=None, keyword=None):
    '''
    Parameters:
        start_date (str) - "YYYY-MM-DD" format or None
        end_date (str)   - "YYYY-MM-DD" format or None
        participants (list of str) - participants to match, or None
        keyword (str)    - keyword to search in title or notes, or None
    '''
    results = []

    for event in calendar_events.values():
        match = True

        # Check date range
        event_date = datetime.fromisoformat(event["start_time"]).date()
        if start_date:
            start_dt = datetime.fromisoformat(start_date).date()
            if event_date < start_dt:
                match = False
        if end_date:
            end_dt = datetime.fromisoformat(end_date).date()
            if event_date > end_dt:
                match = False

        # Check participants
        if participants:
            if not any(p.lower() in [ep.lower() for ep in event.get("participants", [])] for p in participants):
                match = False

        # Check keyword in title or notes
        if keyword:
            keyword_lower = keyword.lower()
            title_lower = event.get("title", "").lower()
            notes_lower = event.get("notes", "").lower()
            if keyword_lower not in title_lower and keyword_lower not in notes_lower:
                match = False

        if match:
            results.append(event)

    return results