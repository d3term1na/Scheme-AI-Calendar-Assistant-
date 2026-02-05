from memory import calendar_events, event_count, embeddings, next_id
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
import re
from numpy.linalg import norm
from zoneinfo import ZoneInfo

# Today's date
today = datetime.now().strftime("%Y-%m-%d")
now = datetime.now()

# Local timezone (Singapore)
LOCAL_TZ = ZoneInfo("Asia/Singapore")

# Ollama LLM config (must be defined before functions that use them)
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3:4b"

# Embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_timezone_from_message(user_message):
    """Use LLM to detect if a timezone is mentioned in the user message."""
    extraction_prompt = f"""Analyze this message and determine if a timezone is mentioned.

Common timezone abbreviations and their IANA names:
- PST/PDT/Pacific -> America/Los_Angeles
- EST/EDT/Eastern -> America/New_York
- CST/CDT/Central -> America/Chicago
- MST/MDT/Mountain -> America/Denver
- GMT/UTC -> UTC
- BST/London -> Europe/London
- CET/Paris/Berlin -> Europe/Paris
- IST/India -> Asia/Kolkata
- JST/Tokyo -> Asia/Tokyo
- SGT/Singapore -> Asia/Singapore
- AEST/Sydney -> Australia/Sydney
- HKT/Hong Kong -> Asia/Hong_Kong
- KST/Seoul -> Asia/Seoul
- CST (China) -> Asia/Shanghai

If a timezone IS mentioned, return the IANA timezone name (e.g., "America/Los_Angeles").
If NO timezone is mentioned, return "null".

Message: {user_message}

Return ONLY the IANA timezone name or "null" (no quotes, no explanation):"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You detect timezones in messages. Return only the IANA timezone name or null."},
            {"role": "user", "content": extraction_prompt}
        ],
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=30)
        res.raise_for_status()
        raw_response = res.json()["message"]["content"].strip()
        print(f"Timezone detection raw response: '{raw_response}'")  # debug

        # Clean up response (keep case for ZoneInfo validation)
        response = raw_response.replace('"', '').replace("'", "").strip()
        response_lower = response.lower()
        print(f"Timezone detection cleaned response: '{response}'")  # debug

        if response_lower == "null" or response_lower == "none" or not response:
            print("No timezone detected")  # debug
            return None

        # Validate it's a real timezone (try original case first)
        try:
            ZoneInfo(response)
            print(f"Valid IANA timezone found: {response}")  # debug
            return response
        except:
            pass

        # Try common IANA formats with proper casing
        iana_formats = {
            "america/los_angeles": "America/Los_Angeles",
            "america/new_york": "America/New_York",
            "america/chicago": "America/Chicago",
            "america/denver": "America/Denver",
            "europe/london": "Europe/London",
            "europe/paris": "Europe/Paris",
            "asia/tokyo": "Asia/Tokyo",
            "asia/singapore": "Asia/Singapore",
            "asia/hong_kong": "Asia/Hong_Kong",
            "asia/kolkata": "Asia/Kolkata",
            "asia/shanghai": "Asia/Shanghai",
            "asia/seoul": "Asia/Seoul",
            "australia/sydney": "Australia/Sydney",
        }
        if response_lower in iana_formats:
            tz = iana_formats[response_lower]
            print(f"Matched IANA timezone: {tz}")  # debug
            return tz

        # Try to match common abbreviations
        tz_mapping = {
            "pst": "America/Los_Angeles",
            "pdt": "America/Los_Angeles",
            "pacific": "America/Los_Angeles",
            "est": "America/New_York",
            "edt": "America/New_York",
            "eastern": "America/New_York",
            "cst": "America/Chicago",
            "cdt": "America/Chicago",
            "central": "America/Chicago",
            "mst": "America/Denver",
            "mdt": "America/Denver",
            "mountain": "America/Denver",
            "gmt": "UTC",
            "utc": "UTC",
            "bst": "Europe/London",
            "london": "Europe/London",
            "jst": "Asia/Tokyo",
            "tokyo": "Asia/Tokyo",
            "sgt": "Asia/Singapore",
            "singapore": "Asia/Singapore",
            "hkt": "Asia/Hong_Kong",
            "ist": "Asia/Kolkata",
            "aest": "Australia/Sydney",
        }
        for key, tz in tz_mapping.items():
            if key in response_lower:
                print(f"Matched timezone abbreviation '{key}' -> {tz}")  # debug
                return tz

        print(f"Could not match timezone: {response}")  # debug
        return None
    except Exception as e:
        print(f"Timezone extraction error: {e}")
        return None


def convert_to_local_tz(datetime_str, source_tz_name):
    """Convert a datetime string from source timezone to local timezone (SGT)."""
    if not datetime_str or not source_tz_name:
        return datetime_str

    try:
        print(f"Converting: {datetime_str} from {source_tz_name} to SGT")  # debug
        # Parse the datetime
        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

        # Attach the source timezone
        source_tz = ZoneInfo(source_tz_name)
        dt_with_tz = dt.replace(tzinfo=source_tz)

        # Convert to local timezone
        dt_local = dt_with_tz.astimezone(LOCAL_TZ)

        result = dt_local.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Converted result: {result}")  # debug
        # Return as string without timezone info (for storage)
        return result
    except Exception as e:
        print(f"Timezone conversion error: {e}")
        return datetime_str


def convert_time_to_local_tz(time_str, source_tz_name):
    """Convert a time string (HH:MM:SS) from source timezone to local timezone (SGT)."""
    if not time_str or not source_tz_name:
        return time_str

    try:
        # Create a datetime using today's date + the time
        today_date = datetime.now().strftime("%Y-%m-%d")
        full_datetime = f"{today_date} {time_str}"

        # Convert using the full datetime function
        converted = convert_to_local_tz(full_datetime, source_tz_name)

        # Extract just the time portion
        return converted.split(" ")[1] if converted else time_str
    except Exception as e:
        print(f"Time timezone conversion error: {e}")
        return time_str


def normalize_datetime(datetime_str, original_message=None):
    """
    Normalize a datetime string to YYYY-MM-DD HH:MM:SS format.
    Handles relative dates like 'tomorrow', 'next week', etc.
    If original_message is provided, check it for day names to override extracted date.
    """
    if not datetime_str:
        return None

    datetime_str = datetime_str.strip()
    current = datetime.now()

    # Try to extract time portion if present
    time_match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', datetime_str)
    time_str = time_match.group(1) if time_match else "09:00:00"
    if len(time_str) == 5:  # HH:MM format
        time_str += ":00"

    # Check the original message for day names (more reliable than LLM extraction)
    check_str = (original_message or datetime_str).lower()
    target_date = None

    # Day name map
    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }

    # Handle relative date keywords
    if "tomorrow" in check_str:
        target_date = current.date() + timedelta(days=1)
    elif "today" in check_str and "not today" not in check_str:
        target_date = current.date()
    elif "next week" in check_str:
        target_date = current.date() + timedelta(weeks=1)
    elif "next month" in check_str:
        target_date = current.date() + timedelta(days=30)
    else:
        # Check for day names
        for day_name, day_num in day_map.items():
            if day_name in check_str:
                days_ahead = day_num - current.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                target_date = current.date() + timedelta(days=days_ahead)
                break

    if target_date:
        return f"{target_date.strftime('%Y-%m-%d')} {time_str}"

    # If already in correct format and no day name found, return as-is
    try:
        datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        return datetime_str
    except ValueError:
        pass

    # Couldn't normalize - return None to indicate failure
    print(f"Warning: Could not normalize datetime: {datetime_str}")
    return None


# Answers context questions
def call_ollama(user_message, context=""):

    # Context
    context_section = f"Context from your calendar and meeting notes:\n{context}\n\n" if context else ""

    # Prompt
    context_prompt = f"""You are an AI calendar assistant. Today's date is {today}.

When answering questions:
- Use the provided context from calendar events and meeting notes to answer questions
- If the context contains relevant meeting notes, reference which meeting the information is from
- If asked about decisions, discussions, or action items, look for them in the meeting notes
- Be concise but informative
- If the context doesn't contain relevant information, say so honestly"""

    # LLM
    payload = {"model": MODEL_NAME,
               "messages": [
                    {
                        "role": "system",
                        "content": context_prompt
                    },
                    {
                        "role": "user",
                        "content": f"{context_section}User question: {user_message}"
                    }
                ],
                "stream": False}
    try:
        # Asking LLM
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()

        # LLM's response
        data = res.json()
        return data["message"]["content"]
    except Exception as e:
        print("Ollama error:", e)
        return "Sorry, I couldn't process your request."


def classify_intent(user_message):
    """Use LLM to classify user intent semantically."""
    classification_prompt = f"""Classify the user's intent into exactly ONE of these categories:

- CREATE_RECURRING: User wants to create RECURRING/REPEATED events (every week, every day, weekly, daily, "for the next X weeks")
- CREATE: User wants to schedule a SINGLE one-time event/meeting/appointment
- DELETE: User wants to remove, cancel, or delete an existing event
- QUERY: User wants to see their SCHEDULE - list events, check availability, see what's on calendar (NOT asking about meeting content)
- UPDATE: User wants to change, modify, reschedule, or update an existing event (time, title, participants)
- ADD_NOTES: User wants to add notes, comments, or a summary to an existing event (STATEMENTS like "We discussed X", "The meeting covered Y")
- BULK_RESCHEDULE: User wants to move/push/reschedule ALL events from one date to another ("push everything today to tomorrow", "move all my meetings from Friday to Monday")
- BULK_CANCEL: User wants to cancel/delete ALL events on a specific date ("cancel everything today", "clear my calendar tomorrow")
- GENERAL: Questions about meeting CONTENT/DISCUSSIONS (like "What did we discuss?", "What was decided?", "How much was X increased?")

Important:
- "every Friday", "weekly", "every week", "daily", "for the next 4 weeks" = CREATE_RECURRING (not CREATE)
- "schedule a meeting tomorrow" = CREATE (single event, no recurrence)
- "Don't delete" or "I don't want to remove" = NOT delete intent
- "What's on my calendar?" or "Am I free tomorrow?" = QUERY intent
- "Move my meeting to..." or "Change the time of..." = UPDATE intent (single event)
- "Push everything today to tomorrow" or "Move all meetings from X to Y" = BULK_RESCHEDULE
- "Cancel everything today" or "Clear my calendar for tomorrow" = BULK_CANCEL
- "Add notes to my meeting..." or "We discussed X" (STATEMENT) = ADD_NOTES intent
- "What did we discuss?" (QUESTION about past meetings) = GENERAL intent

Message: {user_message}

Respond with ONLY the category name (CREATE_RECURRING, CREATE, DELETE, QUERY, UPDATE, ADD_NOTES, BULK_RESCHEDULE, BULK_CANCEL, or GENERAL):"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You classify user intents. Respond with only one word: CREATE_RECURRING, CREATE, DELETE, QUERY, UPDATE, ADD_NOTES, BULK_RESCHEDULE, BULK_CANCEL, or GENERAL."},
            {"role": "user", "content": classification_prompt}
        ],
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=30)
        res.raise_for_status()
        response = res.json()["message"]["content"].strip().upper()

        # Extract just the intent keyword if there's extra text
        # Check longer intents first to avoid partial matches
        for intent in ["CREATE_RECURRING", "BULK_RESCHEDULE", "BULK_CANCEL", "CREATE", "DELETE", "QUERY", "UPDATE", "ADD_NOTES", "GENERAL"]:
            if intent in response:
                return intent

        return "GENERAL"
    except Exception as e:
        print("Intent classification error:", e)
        return "GENERAL"

# Extracts event details from natural language into JSON format using LLM
def extract_event_details(user_message):

    # Prompt to feed
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

    # LLM
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You extract event details and return only valid JSON. No explanations."},
            {"role": "user", "content": extraction_prompt}
        ],
        "stream": False
    }

    try:
        # Asking LLM
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()

        # LLM's response
        response_text = res.json()["message"]["content"]

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            response_text = json_match.group(1)

        details = json.loads(response_text.strip())

        # Validate required fields
        if not all(k in details for k in ["title", "start_time", "end_time"]):
            raise ValueError("Missing required fields")

        # Normalize datetime values
        start_time = normalize_datetime(details["start_time"])
        end_time = normalize_datetime(details["end_time"])

        # If normalization failed, use defaults
        if not start_time:
            default_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
            if default_start < now:
                default_start += timedelta(days=1)
            start_time = default_start.strftime("%Y-%m-%d %H:%M:%S")

        if not end_time:
            try:
                start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                end_time = (start_dt + timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S")
            except:
                end_time = start_time

        # Check for timezone in user message and convert to local (SGT)
        print(f"About to check timezone in: {user_message}")  # debug
        source_tz = extract_timezone_from_message(user_message)
        print(f"Timezone extraction result: {source_tz}")  # debug
        if source_tz:
            print(f"Detected timezone: {source_tz}, converting to SGT")
            print(f"Before conversion - start: {start_time}, end: {end_time}")  # debug
            start_time = convert_to_local_tz(start_time, source_tz)
            end_time = convert_to_local_tz(end_time, source_tz)
            print(f"After conversion - start: {start_time}, end: {end_time}")  # debug

        details["start_time"] = start_time
        details["end_time"] = end_time

        return details
    except Exception as e:
        print("Extraction error:", e)
        # Fallback: use message as title with default time 9:00am
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
    current_year = datetime.now().year
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    extraction_prompt = f"""Extract search filters from this message. Today is {today}, tomorrow is {tomorrow}.

Return ONLY valid JSON with these fields (use null if NOT EXPLICITLY specified):
- start_date: string in "YYYY-MM-DD" format or null (start of date range)
- end_date: string in "YYYY-MM-DD" format or null (end of date range)
- participants: array of strings (names mentioned) or null
- keyword: string (event title/topic to search for) or null

IMPORTANT:
- "today" = {today}
- "tomorrow" = {tomorrow}
- "Feb 2" or "February 2" = "{current_year}-02-02"
- If no date is mentioned, use null for both start_date and end_date

Examples:
- "Who is the Project Review tomorrow with?" -> {{"start_date": "{tomorrow}", "end_date": "{tomorrow}", "participants": null, "keyword": "Project Review"}}
- "Who was the Morning Planning with on Feb 2?" -> {{"start_date": "{current_year}-02-02", "end_date": "{current_year}-02-02", "participants": null, "keyword": "Morning Planning"}}
- "What's on my calendar today?" -> {{"start_date": "{today}", "end_date": "{today}", "participants": null, "keyword": null}}
- "what's on my calendar?" -> {{"start_date": null, "end_date": null, "participants": null, "keyword": null}}
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

        # Post-process dates to handle relative date strings the LLM might return
        start_date = filters.get("start_date")
        end_date = filters.get("end_date")

        if start_date:
            start_lower = start_date.lower().strip()
            if start_lower == "today" or "today" in start_lower:
                start_date = today
            elif start_lower == "tomorrow" or "tomorrow" in start_lower:
                start_date = tomorrow

        if end_date:
            end_lower = end_date.lower().strip()
            if end_lower == "today" or "today" in end_lower:
                end_date = today
            elif end_lower == "tomorrow" or "tomorrow" in end_lower:
                end_date = tomorrow

        return {
            "start_date": start_date,
            "end_date": end_date,
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

        # Normalize datetime values to ensure proper format
        # Pass the original user_message so day names like "Saturday" can be detected
        new_start = details.get("new_start_time")
        new_end = details.get("new_end_time")

        if new_start:
            new_start = normalize_datetime(new_start, user_message)
        if new_end:
            new_end = normalize_datetime(new_end, user_message)

        # If we have start but no end, calculate end as 45 min later
        if new_start and not new_end:
            try:
                start_dt = datetime.strptime(new_start, "%Y-%m-%d %H:%M:%S")
                end_dt = start_dt + timedelta(minutes=45)
                new_end = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass

        # Check for timezone in user message and convert to local (SGT)
        source_tz = extract_timezone_from_message(user_message)
        if source_tz and new_start:
            print(f"Detected timezone: {source_tz}, converting to SGT")
            new_start = convert_to_local_tz(new_start, source_tz)
            if new_end:
                new_end = convert_to_local_tz(new_end, source_tz)

        return {
            "new_title": details.get("new_title"),
            "new_start_time": new_start,
            "new_end_time": new_end,
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


def extract_notes_details(user_message):
    """Use LLM to extract which event to add notes to and what the notes are."""
    extraction_prompt = f"""Extract note details from this message. Today is {today}.

The user wants to ADD NOTES to an existing event. Extract:
1. Which event they're referring to (keyword, participants, date)
2. The actual notes/content to add

Return ONLY valid JSON with these fields:
- keyword: string or null (event type/title to search for, e.g., "meeting", "standup")
- participants: array of strings or null (people in the event)
- event_date: string in "YYYY-MM-DD" format or null (when the event was/is)
- notes: string (the actual notes content to add - extract the meaningful content)

Examples:
- "Add notes to my meeting with Bob yesterday: we discussed the Q1 budget" -> {{"keyword": "meeting", "participants": ["Bob"], "event_date": "yesterday's date", "notes": "Discussed the Q1 budget"}}
- "The standup this morning covered sprint progress" -> {{"keyword": "standup", "participants": null, "event_date": "{today}", "notes": "Covered sprint progress"}}
- "Notes for yesterday's team meeting: action items - finish design doc, review PRs" -> {{"keyword": "team meeting", "participants": null, "event_date": "yesterday's date", "notes": "Action items: finish design doc, review PRs"}}

Message: {user_message}

JSON:"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You extract event notes details and return only valid JSON. No explanations."},
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

        details = json.loads(response_text.strip())

        # Normalize event_date to handle relative dates like "today", "yesterday"
        event_date = details.get("event_date")
        if event_date:
            event_date_lower = event_date.lower().strip()
            if event_date_lower == "today" or "today" in event_date_lower:
                event_date = datetime.now().strftime("%Y-%m-%d")
            elif event_date_lower == "yesterday" or "yesterday" in event_date_lower:
                event_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            elif event_date_lower == "this morning" or "this morning" in event_date_lower:
                event_date = datetime.now().strftime("%Y-%m-%d")
            # Keep as-is if already in YYYY-MM-DD format or other format

        return {
            "keyword": details.get("keyword"),
            "participants": details.get("participants"),
            "event_date": event_date,
            "notes": details.get("notes", "")
        }
    except Exception as e:
        print("Notes extraction error:", e)
        return {"keyword": None, "participants": None, "event_date": None, "notes": ""}


def extract_recurring_details(user_message):
    """Use LLM to extract recurring event details from natural language."""
    extraction_prompt = f"""Extract recurring event details from this message. Today is {today} ({datetime.now().strftime("%A")}).

The user wants to create RECURRING events. Extract:
1. Event details (title, time, participants)
2. Recurrence pattern (which day, frequency)
3. Limit (how many occurrences or end date)

Return ONLY valid JSON with these fields:
- title: string (the event name/description)
- time: string in "HH:MM:SS" format (the time of day for the event)
- duration_minutes: integer (duration in minutes, default 45)
- participants: array of strings (names mentioned, empty array if none)
- frequency: string - "weekly" or "daily" (default "weekly")
- day_of_week: string or null (for weekly: "monday", "tuesday", etc. - extract from message or use today's day)
- occurrence_limit: integer or null (number of events to create, e.g., "3 meetings" = 3)
- end_date: string in "YYYY-MM-DD" format or null (e.g., "till March" or "this month")

IMPORTANT:
- If no limit specified, use occurrence_limit: 4 (default)
- "every Friday at 5pm" -> day_of_week: "friday", time: "17:00:00"
- "daily standup at 9am" -> frequency: "daily", time: "09:00:00"
- "for the next 3 weeks" -> occurrence_limit: 3
- For end_date, ALWAYS use YYYY-MM-DD format or use these EXACT keywords: "end_of_month", "end_of_year"
- "till March" or "until March" -> end_date: "2026-03-01" (first day of that month)
- "this month" or "end of month" -> end_date: "end_of_month" (special keyword)

Examples:
- "Set a progress meeting for every friday 5pm" -> {{"title": "Progress Meeting", "time": "17:00:00", "duration_minutes": 45, "participants": [], "frequency": "weekly", "day_of_week": "friday", "occurrence_limit": 4, "end_date": null}}
- "Weekly standup with the team every Monday 9am for 3 weeks" -> {{"title": "Weekly Standup", "time": "09:00:00", "duration_minutes": 45, "participants": [], "frequency": "weekly", "day_of_week": "monday", "occurrence_limit": 3, "end_date": null}}
- "Daily check-in at 10am till end of month" -> {{"title": "Daily Check-in", "time": "10:00:00", "duration_minutes": 45, "participants": [], "frequency": "daily", "day_of_week": null, "occurrence_limit": null, "end_date": "end_of_month"}}

Message: {user_message}

JSON:"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You extract recurring event details and return only valid JSON. No explanations."},
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

        details = json.loads(response_text.strip())

        # Get time value
        time_value = details.get("time") or "09:00:00"

        # Check for timezone in user message and convert to local (SGT)
        source_tz = extract_timezone_from_message(user_message)
        if source_tz:
            print(f"Detected timezone: {source_tz}, converting time to SGT")
            time_value = convert_time_to_local_tz(time_value, source_tz)

        # Use 'or' to handle None values from LLM returning null
        return {
            "title": details.get("title") or "Recurring Event",
            "time": time_value,
            "duration_minutes": details.get("duration_minutes") or 45,
            "participants": details.get("participants") or [],
            "frequency": details.get("frequency") or "weekly",
            "day_of_week": details.get("day_of_week"),
            "occurrence_limit": details.get("occurrence_limit") or 4,  # Default to 4 if null/None
            "end_date": details.get("end_date")
        }
    except Exception as e:
        print("Recurring extraction error:", e)
        return {
            "title": "Recurring Event",
            "time": "09:00:00",
            "duration_minutes": 45,
            "participants": [],
            "frequency": "weekly",
            "day_of_week": None,
            "occurrence_limit": 4,
            "end_date": None
        }


def calculate_recurring_dates(details):
    """Calculate the dates for recurring events based on the extracted details."""
    dates = []

    frequency = details.get("frequency") or "weekly"
    day_of_week = details.get("day_of_week")
    occurrence_limit = details.get("occurrence_limit") or 4  # Default to 4 events
    end_date_str = details.get("end_date")

    # Parse end_date if provided
    end_date = None
    if end_date_str:
        end_date_lower = end_date_str.lower().strip()
        # Handle special keywords
        if "end_of_month" in end_date_lower or "end of month" in end_date_lower or "this month" in end_date_lower:
            # Calculate last day of current month
            current = datetime.now()
            if current.month == 12:
                end_date = current.replace(year=current.year + 1, month=1, day=1).date() - timedelta(days=1)
            else:
                end_date = current.replace(month=current.month + 1, day=1).date() - timedelta(days=1)
        elif "end_of_year" in end_date_lower or "end of year" in end_date_lower:
            # Calculate last day of current year
            end_date = datetime(datetime.now().year, 12, 31).date()
        else:
            # Try to parse as YYYY-MM-DD
            try:
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            except:
                # Try other common formats
                for fmt in ["%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"]:
                    try:
                        end_date = datetime.strptime(end_date_str, fmt).date()
                        break
                    except:
                        pass

    # Map day names to weekday numbers (0=Monday, 6=Sunday)
    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }

    current_date = datetime.now().date()

    if frequency == "weekly" and day_of_week:
        # Find the target weekday
        target_day = day_map.get(day_of_week.lower(), current_date.weekday())

        # Find the next occurrence of that day
        days_ahead = target_day - current_date.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        next_date = current_date + timedelta(days=days_ahead)

        # Generate dates
        count = 0
        max_iterations = occurrence_limit  # Already defaults to 4

        while count < max_iterations:
            if end_date and next_date > end_date:
                break
            dates.append(next_date)
            count += 1
            next_date += timedelta(weeks=1)

    elif frequency == "daily":
        # Start from tomorrow for daily events
        next_date = current_date + timedelta(days=1)

        count = 0
        max_iterations = occurrence_limit  # Already defaults to 4

        while count < max_iterations:
            if end_date and next_date > end_date:
                break
            dates.append(next_date)
            count += 1
            next_date += timedelta(days=1)
    else:
        # Default: weekly starting from today's day of week, next week
        next_date = current_date + timedelta(weeks=1)
        count = 0
        max_iterations = occurrence_limit  # Already defaults to 4

        while count < max_iterations:
            if end_date and next_date > end_date:
                break
            dates.append(next_date)
            count += 1
            next_date += timedelta(weeks=1)

    return dates


def extract_bulk_operation_details(user_message):
    """Extract source date and destination date for bulk reschedule/cancel operations."""
    extraction_prompt = f"""Extract the dates from this bulk calendar operation. Today is {today} ({datetime.now().strftime("%A")}).

The user wants to move/reschedule/cancel ALL events from one date. Extract:
- source_date: The date to move events FROM (or cancel events on)
- destination_date: The date to move events TO (null if canceling)

Return ONLY valid JSON with these fields:
- source_date: string in "YYYY-MM-DD" format (the date events are being moved FROM)
- destination_date: string in "YYYY-MM-DD" format or null (the date events are being moved TO, null if canceling)

IMPORTANT:
- "today" = {today}
- "tomorrow" = the day after today
- "Push everything today to tomorrow" -> source_date: today, destination_date: tomorrow
- "Cancel everything on Friday" -> source_date: next Friday, destination_date: null
- "Move all meetings from Feb 10 to Feb 12" -> source_date: "2026-02-10", destination_date: "2026-02-12"

Message: {user_message}

JSON:"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You extract dates for bulk calendar operations. Return only valid JSON."},
            {"role": "user", "content": extraction_prompt}
        ],
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=30)
        res.raise_for_status()
        response_text = res.json()["message"]["content"]

        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            response_text = json_match.group(1)

        details = json.loads(response_text.strip())

        # Handle relative dates
        source_date = details.get("source_date")
        destination_date = details.get("destination_date")

        if source_date:
            source_lower = source_date.lower().strip()
            if source_lower == "today":
                source_date = today
            elif source_lower == "tomorrow":
                source_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        if destination_date:
            dest_lower = destination_date.lower().strip()
            if dest_lower == "today":
                destination_date = today
            elif dest_lower == "tomorrow":
                destination_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        return {
            "source_date": source_date,
            "destination_date": destination_date
        }
    except Exception as e:
        print("Bulk operation extraction error:", e)
        return {
            "source_date": today,
            "destination_date": None
        }


# -----------------------------
# RAG
# -----------------------------
# Vectorisation
def store_embedding(content, doc_type="conversation"):
    """Store embedding in the in-memory dictionary"""
    global next_id
    vector = embed_model.encode(content)
    embeddings[next_id] = {"content": content, "vector": vector, "type": doc_type}
    next_id += 1
    return next_id - 1


def embed_existing_event_notes():
    """
    Embed notes from all existing calendar events into RAG.
    Called on startup to make pre-populated event notes searchable.
    """
    embedded_count = 0
    for event in calendar_events.values():
        notes = event.get("notes", "")
        if notes and notes.strip():
            # Format date naturally (e.g., "January 29")
            try:
                event_dt = datetime.strptime(event['start_time'], "%Y-%m-%d %H:%M:%S")
                date_str = event_dt.strftime("%B %d").replace(" 0", " ").lstrip("0")
            except:
                date_str = event['start_time'].split(' ')[0]
            # Embed the event notes with context
            content = f"Meeting '{event['title']}' on {date_str}: {notes}"
            store_embedding(content, doc_type="meeting_notes")
            embedded_count += 1
            print(f"[RAG] Embedded notes for: {event['title']} on {date_str}")

    print(f"[RAG] Embedded {embedded_count} event notes into RAG")
    return embedded_count

# Compare and rank vectors
def retrieve_top_k(query_vector, k=3):
    """Retrieve top-k most similar embeddings"""

    scored = [(doc, cosine_similarity(query_vector, data["vector"]))
              for doc, data in embeddings.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [embeddings[doc]["content"] for doc, score in scored[:k]]

# Similarity score
def cosine_similarity(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

# -----------------------------
# Calendar event CRUD functions
# -----------------------------
def check_time_conflict(start_time, end_time, exclude_event_id=None):
    """
    Check if a new event would conflict with existing events.
    Returns a list of conflicting events, or empty list if no conflicts.

    A conflict occurs when:
    - New event starts during an existing event
    - New event ends during an existing event
    - New event completely contains an existing event
    """
    conflicts = []

    try:
        new_start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        new_end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return []  # Can't check conflicts with invalid times

    for event in calendar_events.values():
        # Skip the event we're updating (if any)
        if exclude_event_id is not None and event.get("event_id") == exclude_event_id:
            continue

        try:
            existing_start = datetime.strptime(event["start_time"], "%Y-%m-%d %H:%M:%S")
            existing_end = datetime.strptime(event["end_time"], "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            continue

        # Check for overlap
        # Two events overlap if one starts before the other ends AND ends after the other starts
        if new_start < existing_end and new_end > existing_start:
            conflicts.append(event)

    return conflicts


def format_conflict_message(conflicts):
    """Format a user-friendly message about conflicting events."""
    if len(conflicts) == 1:
        event = conflicts[0]
        time_str = event["start_time"]
        try:
            parsed = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            time_str = parsed.strftime("%B %d at %I:%M %p").replace(" 0", " ").lstrip("0")
        except:
            pass
        return f"This conflicts with '{event['title']}' scheduled for {time_str}."
    else:
        event_names = [f"'{e['title']}'" for e in conflicts[:3]]
        if len(conflicts) > 3:
            return f"This conflicts with {', '.join(event_names)} and {len(conflicts) - 3} other event(s)."
        return f"This conflicts with {' and '.join(event_names)}."


def create_event(title, start_time, end_time, participants=None, notes="", recurrence_group=None):
    global event_count
    calendar_events[event_count] = {
        "event_id": event_count,
        "title": title,
        "start_time": start_time,
        "end_time": end_time,
        "participants": participants or [],
        "notes": notes,
        "recurrence_group": recurrence_group  # Groups recurring events together
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

        # Check date range - handle invalid date formats gracefully
        try:
            event_date = datetime.fromisoformat(event["start_time"]).date()
        except (ValueError, TypeError):
            # Try parsing with strptime as fallback
            try:
                event_date = datetime.strptime(event["start_time"], "%Y-%m-%d %H:%M:%S").date()
            except (ValueError, TypeError):
                # Skip date filtering for events with invalid dates
                print(f"Warning: Event '{event.get('title')}' has invalid date format: {event.get('start_time')}")
                event_date = None
        if start_date and event_date:
            try:
                start_dt = datetime.fromisoformat(start_date).date()
                if event_date < start_dt:
                    match = False
            except (ValueError, TypeError):
                pass  # Skip date filter if invalid
        if end_date and event_date:
            try:
                end_dt = datetime.fromisoformat(end_date).date()
                if event_date > end_dt:
                    match = False
            except (ValueError, TypeError):
                pass  # Skip date filter if invalid

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


# -----------------------------
# Agenda Suggestions for Recurring Meetings
# -----------------------------
def get_upcoming_recurring_meetings():
    """
    Get recurring meetings that are coming up (in the future)
    that have a past occurrence with notes.
    Returns list of {event, last_occurrence_notes, suggested_agenda}
    """
    now = datetime.now()
    print(f"[Agenda] Current time: {now}")  # debug

    # Group events by recurrence_group
    recurrence_groups = {}
    for event in calendar_events.values():
        group_id = event.get("recurrence_group")
        if group_id:
            if group_id not in recurrence_groups:
                recurrence_groups[group_id] = []
            recurrence_groups[group_id].append(event)

    suggestions = []

    for group_id, events in recurrence_groups.items():
        # Separate past and upcoming events using full datetime comparison
        past_events = []
        upcoming_events = []

        for event in events:
            try:
                event_datetime = datetime.strptime(event["start_time"], "%Y-%m-%d %H:%M:%S")
                # Compare full datetime, not just date
                # An event that started before now is "past"
                if event_datetime < now:
                    past_events.append(event)
                    print(f"[Agenda] Past event: {event['title']} at {event['start_time']} (has notes: {bool(event.get('notes'))})")  # debug
                else:
                    upcoming_events.append(event)
                    print(f"[Agenda] Upcoming event: {event['title']} at {event['start_time']}")  # debug
            except (ValueError, TypeError):
                continue

        # Sort past events by date descending (most recent first)
        past_events.sort(key=lambda e: e["start_time"], reverse=True)
        # Sort upcoming events by date ascending (soonest first)
        upcoming_events.sort(key=lambda e: e["start_time"])

        # Find the immediate last past occurrence with notes
        last_with_notes = None
        for event in past_events:
            if event.get("notes") and event["notes"].strip():
                last_with_notes = event
                print(f"[Agenda] Found last event with notes: {event['title']} at {event['start_time']}")  # debug
                print(f"[Agenda] Notes content: {event['notes'][:100]}...")  # debug
                break

        # If we have an upcoming event and a past event with notes, generate suggestion
        if upcoming_events and last_with_notes:
            upcoming_event = upcoming_events[0]
            notes = last_with_notes["notes"]

            # Use LLM to generate agenda suggestions from notes
            suggested_agenda = generate_agenda_from_notes(
                last_with_notes["title"],
                notes
            )

            suggestions.append({
                "upcoming_event": upcoming_event,
                "last_occurrence": last_with_notes,
                "suggested_agenda": suggested_agenda
            })

    return suggestions


def generate_agenda_from_notes(meeting_title, notes):
    """Use LLM to extract agenda items/follow-ups from previous meeting notes."""
    extraction_prompt = f"""Based on these meeting notes from a previous "{meeting_title}" meeting, suggest 2-3 concise agenda items or follow-ups for the next meeting.

Previous meeting notes:
{notes}

Return ONLY a brief bullet list of suggested agenda items (no explanations, just the items). Focus on:
- Action items that were mentioned
- Topics that need follow-up
- Unresolved issues

Keep each item under 15 words. Format as a simple bullet list starting with "-"."""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You extract actionable agenda items from meeting notes. Be concise."},
            {"role": "user", "content": extraction_prompt}
        ],
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=30)
        res.raise_for_status()
        response_text = res.json()["message"]["content"].strip()
        return response_text
    except Exception as e:
        print(f"Error generating agenda suggestions: {e}")
        # Fallback: extract simple action items from notes
        return extract_simple_agenda(notes)


def extract_simple_agenda(notes):
    """Fallback: Extract action items without LLM."""
    items = []

    # Look for common action item patterns
    patterns = [
        "follow up",
        "action item",
        "need to",
        "should",
        "will"
    ]

    sentences = notes.replace(". ", ".\n").split("\n")
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_lower = sentence.lower()
        if any(pattern in sentence_lower for pattern in patterns):
            if len(sentence) > 10:
                items.append(f"- {sentence[:80]}")
                if len(items) >= 3:
                    break

    return "\n".join(items) if items else "- Review previous meeting notes"


# -----------------------------
# Scheduling Insights (inferred from calendar history)
# -----------------------------
def analyze_scheduling_patterns():
    """
    Analyze calendar events to infer user scheduling patterns.
    Detects implicit recurring patterns even for non-recurring events.
    """
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Group events by title + day of week to detect recurring patterns
    title_day_patterns = {}  # {(title, day_of_week): [list of events]}

    for event in calendar_events.values():
        try:
            start_dt = datetime.strptime(event["start_time"], "%Y-%m-%d %H:%M:%S")
            day_of_week = day_names[start_dt.weekday()]
            hour = start_dt.hour
            title = event["title"]

            # Track title + day patterns
            key = (title, day_of_week)
            if key not in title_day_patterns:
                title_day_patterns[key] = []
            title_day_patterns[key].append({
                "event": event,
                "datetime": start_dt,
                "hour": hour
            })

        except (ValueError, TypeError):
            continue

    # Identify recurring patterns (2+ occurrences on the same day of week)
    recurring_patterns = []
    for (title, day), occurrences in title_day_patterns.items():
        if len(occurrences) >= 2:
            # Calculate typical time
            hours = [o["hour"] for o in occurrences]
            typical_hour = max(set(hours), key=hours.count)  # Most common hour

            recurring_patterns.append({
                "title": title,
                "day": day,
                "typical_hour": typical_hour,
                "occurrences": occurrences,
                "count": len(occurrences)
            })

    return {
        "recurring_patterns": recurring_patterns,
        "title_day_patterns": title_day_patterns
    }


def get_current_week_range():
    """Get the start and end dates of the current week (Monday to Sunday)."""
    now = datetime.now()
    start_of_week = now - timedelta(days=now.weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
    return start_of_week, end_of_week


def get_scheduling_insight():
    """
    Generate a contextual scheduling insight based on inferred patterns.
    Detects missing recurring meetings and suggests based on history.
    """
    patterns = analyze_scheduling_patterns()
    now = datetime.now()
    current_day = now.strftime("%A")
    current_hour = now.hour
    start_of_week, end_of_week = get_current_week_range()

    insights = []

    # Check each recurring pattern
    for pattern in patterns["recurring_patterns"]:
        title = pattern["title"]
        usual_day = pattern["day"]
        typical_hour = pattern["typical_hour"]
        occurrences = pattern["occurrences"]

        # Check if this meeting is scheduled for the current week
        has_current_week_occurrence = False
        for occ in occurrences:
            if start_of_week <= occ["datetime"] <= end_of_week:
                has_current_week_occurrence = True
                break

        # Format time nicely
        if typical_hour < 12:
            time_str = f"{typical_hour}am" if typical_hour > 0 else "12am"
        elif typical_hour == 12:
            time_str = "12pm"
        else:
            time_str = f"{typical_hour - 12}pm"

        # If it's the usual day and no meeting scheduled this week
        if usual_day == current_day and not has_current_week_occurrence:
            insights.append({
                "priority": 1,
                "text": f"You usually have {title} on {usual_day}s at {time_str}"
            })
        # If it's before the usual day this week and not scheduled
        elif not has_current_week_occurrence:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            usual_day_idx = day_order.index(usual_day)
            current_day_idx = day_order.index(current_day)

            if usual_day_idx > current_day_idx:
                # Upcoming day this week, meeting not scheduled
                insights.append({
                    "priority": 2,
                    "text": f"You usually have {title} on {usual_day}s"
                })

    # Check for upcoming meetings today
    today_str = now.strftime("%Y-%m-%d")
    upcoming_today = []
    for event in calendar_events.values():
        if event["start_time"].startswith(today_str):
            try:
                event_time = datetime.strptime(event["start_time"], "%Y-%m-%d %H:%M:%S")
                if event_time > now:
                    upcoming_today.append(event)
            except (ValueError, TypeError):
                continue

    if upcoming_today:
        upcoming_today.sort(key=lambda e: e["start_time"])
        next_event = upcoming_today[0]
        event_time = datetime.strptime(next_event["start_time"], "%Y-%m-%d %H:%M:%S")
        time_until = event_time - now
        mins = int(time_until.total_seconds() / 60)
        if mins <= 60:
            insights.append({
                "priority": 0,  # Highest priority
                "text": f"You have '{next_event['title']}' in {mins} minutes"
            })

    # Sort by priority and return the top insight
    if insights:
        insights.sort(key=lambda x: x["priority"])
        return insights[0]["text"]
    else:
        # Default based on time
        if current_hour < 9:
            return "Good morning! What would you like to schedule today?"
        elif current_day == "Friday" and current_hour >= 14:
            return "Friday afternoon - good time for focused work"
        else:
            return "What would you like to add to your calendar?"