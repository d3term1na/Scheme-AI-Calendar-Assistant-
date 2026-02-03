from memory import calendar_events, event_count, embeddings, next_id
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
import re
from numpy.linalg import norm

# Today's date
today = datetime.now().strftime("%Y-%m-%d")
now = datetime.now()

# Embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ollama LLM
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3:4b"

def call_ollama(user_message, context=""):
    context_section = f"Context:\n{context}\n\n" if context else ""
    payload = {"model": MODEL_NAME,
               "messages": [
                    {
                        "role": "system",
                        "content": f"You are an AI calendar assistant. Only answer the user query. Today's date is {today}. Interpret phrases like 'this week', 'next week', and 'tomorrow' relative to today."
                    },
                    {
                        "role": "user",
                        "content": f"{context_section}User query: {user_message}"
                    }
                ],
                "stream": False}
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data["message"]["content"]
    except Exception as e:
        print("Ollama error:", e)
        return "Sorry, I couldn't process your request."


def classify_intent(user_message):
    """Use LLM to classify user intent semantically."""
    classification_prompt = f"""Classify the user's intent into exactly ONE of these categories:

- CREATE: User wants to schedule, add, or create a new event/meeting/appointment
- DELETE: User wants to remove, cancel, or delete an existing event
- QUERY: User wants to see, list, check, or ask about their events/calendar
- UPDATE: User wants to change, modify, reschedule, or update an existing event
- GENERAL: Greetings, help requests, or questions not about specific calendar operations

Important:
- "Don't delete" or "I don't want to remove" = NOT delete intent
- "Can you help me create..." = CREATE intent
- "What's on my calendar?" or "Am I free tomorrow?" = QUERY intent
- "Move my meeting to..." or "Change the time of..." = UPDATE intent

Message: {user_message}

Respond with ONLY the category name (CREATE, DELETE, QUERY, UPDATE, or GENERAL):"""
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You classify user intents. Respond with only one word: CREATE, DELETE, QUERY, UPDATE, or GENERAL."},
            {"role": "user", "content": classification_prompt}
        ],
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=30)
        res.raise_for_status()
        response = res.json()["message"]["content"].strip().upper()

        # Extract just the intent keyword if there's extra text
        for intent in ["CREATE", "DELETE", "QUERY", "UPDATE", "GENERAL"]:
            if intent in response:
                return intent

        return "GENERAL"
    except Exception as e:
        print("Intent classification error:", e)
        return "GENERAL"


def extract_event_details(user_message):
    """Use LLM to extract event details from natural language."""
    extraction_prompt = f"""Extract event details from this message. Today is {today} (current time: {now.strftime("%H:%M")}).

Return ONLY valid JSON with these fields:
- title: string (the event name/description)
- start_time: string in "YYYY-MM-DD HH:MM:SS" format
- end_time: string in "YYYY-MM-DD HH:MM:SS" format (default to 45 min after start if not specified)
- participants: array of strings (names mentioned, empty array if none)

Interpret relative dates like "tomorrow", "next Monday", "this Friday" relative to today.
If no time specified, default to 09:00:00.

Message: {user_message}

JSON:"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You extract event details and return only valid JSON. No explanations."},
            {"role": "user", "content": extraction_prompt}
        ],
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()
        response_text = res.json()["message"]["content"]

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            response_text = json_match.group(1)

        details = json.loads(response_text.strip())

        # Validate required fields
        if not all(k in details for k in ["title", "start_time", "end_time"]):
            raise ValueError("Missing required fields")

        return details
    except Exception as e:
        print("Extraction error:", e)
        # Fallback: use message as title with default times
        default_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if default_start < now:
            default_start += timedelta(days=1)
        return {
            "title": user_message,
            "start_time": default_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": (default_start + timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S"),
            "participants": []
        }

def extract_query_filters(user_message):
    """Use LLM to extract query/filter criteria from natural language."""
    extraction_prompt = f"""Extract search filters from this message. Today is {today}.

Return ONLY valid JSON with these fields (use null if NOT EXPLICITLY specified):
- start_date: string in "YYYY-MM-DD" format or null (start of date range)
- end_date: string in "YYYY-MM-DD" format or null (end of date range)
- participants: array of strings (names mentioned) or null
- keyword: string (event title/topic to search for) or null

IMPORTANT: Only extract filters that are EXPLICITLY mentioned. Do NOT infer or assume dates.
- If no date is mentioned, use null for both start_date and end_date
- "What's on my calendar?" with no date = all nulls (show ALL events)
- "Show my events" with no date = all nulls (show ALL events)

Examples:
- "delete my meeting with Bob tomorrow" -> {{"start_date": "{today}", "end_date": "{today}", "participants": ["Bob"], "keyword": "meeting"}}
- "show events this week" -> {{"start_date": "...", "end_date": "...", "participants": null, "keyword": null}}
- "what's on my calendar next Friday" -> {{"start_date": "next Friday date", "end_date": "next Friday date", "participants": null, "keyword": null}}
- "what's on my calendar?" -> {{"start_date": null, "end_date": null, "participants": null, "keyword": null}}
- "show all my events" -> {{"start_date": null, "end_date": null, "participants": null, "keyword": null}}
- "list events with Alice" -> {{"start_date": null, "end_date": null, "participants": ["Alice"], "keyword": null}}

Message: {user_message}

JSON:"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You extract search filters and return only valid JSON. No explanations."},
            {"role": "user", "content": extraction_prompt}
        ],
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()
        response_text = res.json()["message"]["content"]

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            response_text = json_match.group(1)

        filters = json.loads(response_text.strip())
        return {
            "start_date": filters.get("start_date"),
            "end_date": filters.get("end_date"),
            "participants": filters.get("participants"),
            "keyword": filters.get("keyword")
        }
    except Exception as e:
        print("Filter extraction error:", e)
        return {"start_date": None, "end_date": None, "participants": None, "keyword": None}


def extract_event_identifier(user_message):
    """Use LLM to identify WHICH event the user wants to update (not the new values)."""
    extraction_prompt = f"""Identify which EXISTING event the user is referring to. Today is {today}.

The user wants to UPDATE an event. Extract identifiers for the CURRENT event (NOT the new values):
- keyword: the event type/name (e.g., "meeting", "standup", "lunch")
- participants: people currently in the event (NOT people being added)
- current_date: the CURRENT date of the event, ONLY if explicitly stated (e.g., "my 2pm meeting" or "tomorrow's standup")

IMPORTANT: Distinguish between CURRENT event info vs NEW values:
- "reschedule my meeting to 3pm tomorrow" -> keyword="meeting", current_date=null (tomorrow is the NEW time)
- "move tomorrow's standup to Friday" -> keyword="standup", current_date=tomorrow (tomorrow is CURRENT, Friday is NEW)
- "change my 2pm meeting to 4pm" -> keyword="meeting", current_date=null (no current DATE specified, just time)
- "rename the team meeting to Sprint Review" -> keyword="team meeting", current_date=null

Return ONLY valid JSON:
- keyword: string or null (event type/title to search for)
- participants: array of strings or null (current participants)
- current_date: string in "YYYY-MM-DD" format or null (current event date, ONLY if explicitly stated)

Message: {user_message}

JSON:"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You identify which event the user is referring to. Return only valid JSON. No explanations."},
            {"role": "user", "content": extraction_prompt}
        ],
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()
        response_text = res.json()["message"]["content"]

        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            response_text = json_match.group(1)

        identifier = json.loads(response_text.strip())
        return {
            "keyword": identifier.get("keyword"),
            "participants": identifier.get("participants"),
            "current_date": identifier.get("current_date")
        }
    except Exception as e:
        print("Event identifier extraction error:", e)
        return {"keyword": None, "participants": None, "current_date": None}


def extract_update_details(user_message):
    """Use LLM to extract what changes the user wants to make to an event."""
    extraction_prompt = f"""Extract the UPDATE details from this message. Today is {today} (current time: {now.strftime("%H:%M")}).

The user wants to modify an existing event. Extract ONLY the fields they want to CHANGE (use null for fields not being changed):

Return ONLY valid JSON with these fields:
- new_title: string or null (new event name if changing)
- new_start_time: string in "YYYY-MM-DD HH:MM:SS" format or null (new start time if rescheduling)
- new_end_time: string in "YYYY-MM-DD HH:MM:SS" format or null (new end time if changing duration)
- new_participants: array of strings or null (replacement participants list if changing)
- add_participants: array of strings or null (participants to ADD to existing list)
- remove_participants: array of strings or null (participants to REMOVE from existing list)

Interpret relative dates like "tomorrow", "next Monday" relative to today.

Examples:
- "reschedule my meeting to 3pm tomorrow" -> {{"new_start_time": "tomorrow 15:00:00", "new_end_time": "tomorrow 15:45:00", ...rest null}}
- "change the team standup to 10am" -> {{"new_start_time": "... 10:00:00", ...rest null}}
- "rename my meeting to Project Review" -> {{"new_title": "Project Review", ...rest null}}
- "add Bob to the meeting" -> {{"add_participants": ["Bob"], ...rest null}}
- "move my 2pm meeting to Friday at 4pm" -> {{"new_start_time": "Friday 16:00:00", "new_end_time": "Friday 16:45:00", ...rest null}}

Message: {user_message}

JSON:"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You extract event update details and return only valid JSON. No explanations."},
            {"role": "user", "content": extraction_prompt}
        ],
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()
        response_text = res.json()["message"]["content"]

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            response_text = json_match.group(1)

        details = json.loads(response_text.strip())
        return {
            "new_title": details.get("new_title"),
            "new_start_time": details.get("new_start_time"),
            "new_end_time": details.get("new_end_time"),
            "new_participants": details.get("new_participants"),
            "add_participants": details.get("add_participants"),
            "remove_participants": details.get("remove_participants")
        }
    except Exception as e:
        print("Update extraction error:", e)
        return {
            "new_title": None, "new_start_time": None, "new_end_time": None,
            "new_participants": None, "add_participants": None, "remove_participants": None
        }


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
        match_score = 0  # Higher score = better match

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

        # Check keyword in title or notes (matches if ANY word in keyword matches)
        if keyword:
            keyword_lower = keyword.lower().strip()
            title_lower = event.get("title", "").lower().strip()
            notes_lower = event.get("notes", "").lower().strip()
            print(f"Matching keyword='{keyword_lower}' against title='{title_lower}'")  # debug

            # Score by match quality (higher = better)
            if keyword_lower == title_lower:
                match_score = 100  # Exact title match
            elif keyword_lower in title_lower:
                match_score = 80  # Exact phrase in title
            elif keyword_lower in notes_lower:
                match_score = 60  # Exact phrase in notes
            else:
                # Check if any word in the keyword matches
                keyword_words = [w for w in keyword_lower.split() if len(w) > 2]
                matching_words = sum(1 for w in keyword_words if w in title_lower or w in notes_lower)
                if matching_words > 0:
                    match_score = 20 + (matching_words * 10)  # Partial match
                else:
                    match = False

        if match:
            results.append((event, match_score))

    # Sort by match score (best matches first), then by event_id for consistency
    results.sort(key=lambda x: (-x[1], x[0].get("event_id", 0)))
    return [event for event, score in results]