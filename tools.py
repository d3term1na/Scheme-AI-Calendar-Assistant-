import db
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer, util
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
    # Pre-check: only call LLM if a known timezone keyword exists in the message
    tz_keywords = [
        "pst", "pdt", "est", "edt", "cst", "cdt", "mst", "mdt",
        "gmt", "utc", "bst", "cet", "ist", "jst", "sgt", "aest", "hkt", "kst",
        "pacific time", "eastern time", "central time", "mountain time",
        "singapore time", "tokyo time", "london time",
    ]
    msg_lower = user_message.lower()
    if not any(kw in msg_lower for kw in tz_keywords):
        print("No timezone keyword found in message, skipping LLM call")
        return None

    extraction_prompt = f"""Does this message contain an EXPLICIT timezone abbreviation or name?

Rules:
- "am" and "pm" are NOT timezones. "7pm", "9am", "3:00pm" have NO timezone.
- "Sun", "Mon", "Tue" etc. are days of the week, NOT timezones.
- Only these count as timezones: PST, PDT, EST, EDT, CST, CDT, MST, MDT, GMT, UTC, BST, CET, IST, JST, SGT, AEST, HKT, KST, or full names like "Pacific time", "Eastern time", "Singapore time".

Examples:
- "Reschedule all standups to Sun 7pm" -> null
- "meeting at 6pm" -> null
- "Friday 6pm" -> null
- "meeting at 3pm PST" -> America/Los_Angeles
- "call at 9am Eastern" -> America/New_York
- "schedule for 2pm SGT" -> Asia/Singapore

Message: {user_message}

Return ONLY "null" or the IANA timezone name. Nothing else:"""

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
        offset = 0
        if datetime.strptime(converted, "%Y-%m-%d %H:%M:%S").date() < datetime.strptime(full_datetime, "%Y-%m-%d %H:%M:%S").date():
            offset = -1
        elif datetime.strptime(converted, "%Y-%m-%d %H:%M:%S").date() > datetime.strptime(full_datetime, "%Y-%m-%d %H:%M:%S").date():
            offset = 1
        # Extract just the time portion
        final_str = ""
        if converted:
            final_str = converted.split(" ")[1]
        else:
            final_str = time_str
        return final_str, offset
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
- DELETE: User wants to remove, cancel, or delete a SINGLE existing event
- DELETE_RECURRING: User wants to remove/cancel ALL events in a recurring series ("remove all my standups", "delete all Team Meetings", "cancel all my 1:1s")
- QUERY: User wants to see their SCHEDULE - list events, check availability, see what's on calendar (NOT asking about meeting content)
- UPDATE: User wants to change, modify, reschedule, or update a SINGLE existing event (time, title, participants)
- UPDATE_RECURRING: User wants to change ALL events in a recurring series ("rename all my Project Reviews to Budget Reviews", "move all my Morning Plannings to Tuesday", "change all standups to 10am")
- ADD_NOTES: User wants to add notes, comments, or a summary to an existing event (STATEMENTS like "We discussed X", "The meeting covered Y")
- BULK_RESCHEDULE: User wants to move/push/reschedule ALL events from one DATE to another DATE ("push everything today to tomorrow", "move all my meetings from Friday to Monday")
- BULK_CANCEL: User wants to cancel/delete ALL events on a specific DATE ("cancel everything today", "clear my calendar tomorrow")
- GENERAL: Questions about meeting CONTENT/DISCUSSIONS (like "What did we discuss?", "What was decided?", "How much was X increased?")

CRITICAL DISTINCTION - Recurring series vs Date-based:
- UPDATE_RECURRING/DELETE_RECURRING: Targets a recurring SERIES by name ("all my standups", "all Project Reviews")
- BULK_RESCHEDULE/BULK_CANCEL: Targets a specific DATE ("everything today", "all meetings on Friday")

Important:
- "every Friday", "weekly", "every week", "daily", "for the next 4 weeks" = CREATE_RECURRING (not CREATE)
- "schedule a meeting tomorrow" = CREATE (single event, no recurrence)
- "Delete the standup" or "Cancel tomorrow's meeting" = DELETE (single event)
- "Remove ALL my standups" or "Delete all Team Meetings" = DELETE_RECURRING (recurring series)
- "Move my meeting to 3pm" or "Reschedule Product Meeting to Feb 10" = UPDATE (single event)
- "Reschedule my dinner with Charlie to tomorrow 8am" = UPDATE (single event, no "all")
- "Reschedule the Team Standup on Feb 11 to tomorrow" = UPDATE (targets ONE specific occurrence by date)
- "Change ALL my standups to 10am" or "Rename all Project Reviews" = UPDATE_RECURRING (recurring series, must say "all")
- "Push everything today to tomorrow" = BULK_RESCHEDULE (date-based)
- "Cancel everything today" = BULK_CANCEL (date-based)
- "Add notes to my meeting..." or "We discussed X" (STATEMENT) = ADD_NOTES intent
- "What did we discuss?" (QUESTION about past meetings) = GENERAL intent

KEY RULE: UPDATE_RECURRING and DELETE_RECURRING require the word "all" (e.g., "all my standups", "all Project Reviews"). Without "all", it is UPDATE or DELETE (single event). Also, if the user mentions a SPECIFIC DATE (like "on Feb 11", "tomorrow"), it is UPDATE or DELETE, NOT UPDATE_RECURRING or DELETE_RECURRING.

Message: {user_message}

Respond with ONLY the category name:"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You classify user intents. Respond with only one word: CREATE_RECURRING, CREATE, DELETE, DELETE_RECURRING, QUERY, UPDATE, UPDATE_RECURRING, ADD_NOTES, BULK_RESCHEDULE, BULK_CANCEL, or GENERAL."},
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
        classified = "GENERAL"
        for intent in ["CREATE_RECURRING", "UPDATE_RECURRING", "DELETE_RECURRING", "BULK_RESCHEDULE", "BULK_CANCEL", "CREATE", "DELETE", "QUERY", "UPDATE", "ADD_NOTES", "GENERAL"]:
            if intent in response:
                classified = intent
                break

        # Guard: UPDATE_RECURRING/DELETE_RECURRING require "all" in the message
        msg_lower = user_message.lower()
        if classified == "UPDATE_RECURRING" and "all" not in msg_lower:
            print(f"Intent guard: {classified} -> UPDATE (no 'all' in message)")
            classified = "UPDATE"
        elif classified == "DELETE_RECURRING" and "all" not in msg_lower:
            print(f"Intent guard: {classified} -> DELETE (no 'all' in message)")
            classified = "DELETE"

        return classified
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
    # Calculate next week's Monday and Sunday
    now = datetime.now()
    days_until_next_monday = (7 - now.weekday()) % 7
    if days_until_next_monday == 0:
        days_until_next_monday = 7
    next_monday = (now + timedelta(days=days_until_next_monday)).strftime("%Y-%m-%d")
    next_sunday = (now + timedelta(days=days_until_next_monday + 6)).strftime("%Y-%m-%d")

    extraction_prompt = f"""Extract search filters from this message. Today is {today}, tomorrow is {tomorrow}.

Return ONLY valid JSON with these fields (use null if NOT EXPLICITLY specified):
- start_date: string in "YYYY-MM-DD" format or null (start of date range)
- end_date: string in "YYYY-MM-DD" format or null (end of date range)
- participants: array of strings (names mentioned) or null
- keyword: string (specific event title/topic to search for) or null

IMPORTANT:
- "today" = {today}
- "tomorrow" = {tomorrow}
- "next week" = start_date: "{next_monday}", end_date: "{next_sunday}"
- "Feb 2" or "February 2" = "{current_year}-02-02"
- If no date is mentioned, use null for both start_date and end_date
- "keyword" is ONLY for specific event names like "Project Review", "Standup", "Lunch with Bob"
- Generic words like "events", "meetings", "calendar", "schedule" are NOT keywords — use null

Examples:
- "Who is the Project Review tomorrow with?" -> {{"start_date": "{tomorrow}", "end_date": "{tomorrow}", "participants": null, "keyword": "Project Review"}}
- "Who was the Morning Planning with on Feb 2?" -> {{"start_date": "{current_year}-02-02", "end_date": "{current_year}-02-02", "participants": null, "keyword": "Morning Planning"}}
- "What's on my calendar today?" -> {{"start_date": "{today}", "end_date": "{today}", "participants": null, "keyword": null}}
- "what's on my calendar?" -> {{"start_date": null, "end_date": null, "participants": null, "keyword": null}}
- "What are my events next week?" -> {{"start_date": "{next_monday}", "end_date": "{next_sunday}", "participants": null, "keyword": null}}
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
- "change my product meeting to 9 feb 5pm" -> keyword="product meeting", current_date=null (9 feb 5pm is NEW)
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

The user wants to modify an existing event. Extract ONLY the NEW values they want to CHANGE TO (use null for fields not being changed).

CRITICAL: When the message has TWO dates (e.g., "on 11 Feb to 13 Feb 8pm"), the date BEFORE "to" identifies WHICH event, the date AFTER "to" is the NEW date/time. Only return the NEW date/time.

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
- "Reschedule the Team Standup on 11 Feb to 13 Feb 8pm" -> {{"new_start_time": "2026-02-13 20:00:00", "new_end_time": "2026-02-13 20:45:00", ...rest null}} (13 Feb is the NEW date, 11 Feb is ignored)

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
        print(f"UPDATE details LLM response: {details}")  # debug

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
- "Add Notes to Product meeting on 6 Feb: look into Bob's issue with authentication" -> {{"keyword": "Product meeting", "participants": null, "event_date": "2026-02-06", "notes": "Look into Bob's issue with authentication"}}

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
- "Weekly standup with Bob and Charlie every Monday 9am for 3 weeks" -> {{"title": "Weekly Standup", "time": "09:00:00", "duration_minutes": 45, "participants": ["Bob", "Charlie"], "frequency": "weekly", "day_of_week": "monday", "occurrence_limit": 3, "end_date": null}}
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
        day_of_event = details.get("day_of_week")
        # Check for timezone in user message and convert to local (SGT)
        source_tz = extract_timezone_from_message(user_message)
        if source_tz:
            days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            print(f"Detected timezone: {source_tz}, converting time to SGT")
            time_value, offset = convert_time_to_local_tz(time_value, source_tz)
            if offset == -1 and details.get("day_of_week") == "monday":
                day_of_event = "sunday"
            elif offset == 1 and details.get("day_of_week") == "sunday":
                day_of_event = "monday"
            else:
                day_of_event = days_of_week[days_of_week.index(details.get("day_of_week")) + offset]
                
        # Use 'or' to handle None values from LLM returning null
        return {
            "title": details.get("title") or "Recurring Event",
            "time": time_value,
            "duration_minutes": details.get("duration_minutes") or 45,
            "participants": details.get("participants") or [],
            "frequency": details.get("frequency") or "weekly",
            "day_of_week": day_of_event,
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
        if days_ahead <= -1:  # Target day already happened this week
            days_ahead += 7
        elif days_ahead == 0 and datetime.now().time() > datetime.strptime(details.get("time"), '%H:%M:%S').time():
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


def extract_recurring_operation_details(user_message):
    """
    Extract details for recurring series operations (UPDATE_RECURRING, DELETE_RECURRING).

    These target a recurring SERIES by name, not a specific date.
    Examples:
    - "Change the title of all my Project Reviews to Budget Reviews"
    - "Reschedule all my Morning Plannings to every Tuesday 3pm"
    - "Remove all my Team Standups"
    """
    extraction_prompt = f"""Extract details for an operation on a RECURRING event series. Today is {today} ({datetime.now().strftime("%A")}).

The user wants to modify or delete ALL occurrences in a recurring series. Extract:

1. series_keyword: The name/title of the recurring series to target (e.g., "standups", "Project Review", "Morning Planning")
2. For updates, what changes to make:
   - new_title: New name for the series (null if not changing title)
   - new_day: New day of week (null if not changing day) - use lowercase: "monday", "tuesday", etc.
   - new_time: New time in HH:MM:SS format (null if not changing time)

Return ONLY valid JSON with these fields:
- series_keyword: string (the recurring series name to search for)
- new_title: string or null (new name if renaming)
- new_day: string or null (new day of week if rescheduling, lowercase)
- new_time: string in "HH:MM:SS" format or null (new time if rescheduling)
- new_participants: array of strings (names mentioned, empty array if none)
IMPORTANT: Use the EXACT event name from the user's message for series_keyword. Do NOT rename or map to other event names.

series_keyword examples:
- "all my standups" -> series_keyword: "standup"
- "all Project Reviews" -> series_keyword: "Project Review"
- "all Budget Reviews" -> series_keyword: "Budget Review"
- "all my Morning Plannings" -> series_keyword: "Morning Planning"
- "all 1:1s" or "all one-on-ones" -> series_keyword: "1:1"

TIME CONVERSION (use 24-hour format):
- AM times: "9am" -> "09:00:00", "10am" -> "10:00:00", "11am" -> "11:00:00"
- PM times: Add 12 to the hour! "1pm" -> "13:00:00", "2pm" -> "14:00:00", "3pm" -> "15:00:00", "4pm" -> "16:00:00", "5pm" -> "17:00:00", "6pm" -> "18:00:00", "7pm" -> "19:00:00"
- Special: "12pm" (noon) -> "12:00:00", "12am" (midnight) -> "00:00:00"

Examples:
- "Change the title of all my Project Reviews to Budget Reviews" -> {{"series_keyword": "Project Review", "new_title": "Budget Reviews", "new_day": null, "new_time": null, "new_participants": []}}
- "Reschedule all my Morning Plannings to every Tuesday 3pm" -> {{"series_keyword": "Morning Planning", "new_title": null, "new_day": "tuesday", "new_time": "15:00:00", "new_participants": []}}
- "Move all standups to 10am" -> {{"series_keyword": "standup", "new_title": null, "new_day": null, "new_time": "10:00:00", "new_participants": []}}
- "Reschedule all 1:1s to Friday 6pm" -> {{"series_keyword": "1:1", "new_title": null, "new_day": "friday", "new_time": "18:00:00", "new_participants": []}}
- "Remove all my Team Standups" -> {{"series_keyword": "Team Standup", "new_title": null, "new_day": null, "new_time": null, "new_participants": []}}
- "Delete all 1:1s with Manager" -> {{"series_keyword": "1:1", "new_title": null, "new_day": null, "new_time": null, "new_participants": []}}
- "Change the participants of all my Team Standups to Bob and Charlie" -> {{"series_keyword": "Team Standup", "new_title": null, "new_day": null, "new_time": null, "new_participants": ["Bob", "Charlie"]}}

Message: {user_message}

JSON:"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You extract recurring series operation details. Return only valid JSON. CRITICAL: For PM times, ADD 12 to the hour (1pm=13:00, 2pm=14:00, 3pm=15:00, 4pm=16:00, 5pm=17:00, 6pm=18:00, 7pm=19:00, 8pm=20:00)."},
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
        print(f"LLM extracted details: {details}")  # debug

        day_of_event = details.get("new_day")
        # Handle timezone conversion for new_time if present
        new_time = details.get("new_time")
        if new_time:
            source_tz = extract_timezone_from_message(user_message)
            if source_tz:
                days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                print(f"Detected timezone: {source_tz}, converting time to SGT")
                new_time, offset = convert_time_to_local_tz(new_time, source_tz)
                if offset == -1 and details.get("new_day") == "monday":
                    day_of_event = "sunday"
                elif offset == 1 and details.get("new_day") == "sunday":
                    day_of_event = "monday"
                else:
                    day_of_event = days_of_week[days_of_week.index(details.get("new_day")) + offset]

        return {
            "series_keyword": details.get("series_keyword"),
            "new_title": details.get("new_title"),
            "new_day": day_of_event,
            "new_time": new_time,
            "new_participants": details.get("new_participants")
        }
    except Exception as e:
        print("Recurring operation extraction error:", e)
        return {
            "series_keyword": None,
            "new_title": None,
            "new_day": None,
            "new_time": None,
            "new_participants": []
        }


def find_recurring_series_events(username, series_keyword):
    """
    Find all events that belong to a recurring series matching the keyword.
    Returns list of events that share the same recurrence_group.
    """
    if not series_keyword:
        return []

    keyword_lower = series_keyword.lower().strip()
    all_events = db.get_user_events(username)

    # First, find events that match the keyword
    matching_groups = set()

    for event in all_events:
        title_lower = event.get("title", "").lower().strip()
        recurrence_group = event.get("recurrence_group")

        # Check if title matches the keyword
        if keyword_lower in title_lower or title_lower in keyword_lower:
            if recurrence_group:
                matching_groups.add(recurrence_group)

    # Get only current/future events in those recurrence groups
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = []
    for event in all_events:
        if event.get("recurrence_group") in matching_groups:
            if event.get("end_time", "") >= now_str:
                results.append(event)

    # Sort by start_time
    results.sort(key=lambda e: e.get("start_time", ""))

    return results


def update_recurring_series(username, series_keyword, new_title=None, new_day=None, new_time=None, new_participants=[]):
    """
    Update all events in a recurring series.

    Parameters:
    - username: The user who owns the events
    - series_keyword: Name of the series to find
    - new_title: New title for all events (optional)
    - new_day: New day of week, e.g., "tuesday" (optional)
    - new_time: New time in HH:MM:SS format (optional)

    Returns dict with count of updated events and list of updated events.
    """
    events = find_recurring_series_events(username, series_keyword)
    print("Series keyword: ", series_keyword)
    print(f"UPDATE_RECURRING_SERIES - Found {len(events)} events for '{series_keyword}'")  # debug

    if not events:
        return {"count": 0, "events": [], "error": f"No recurring series found matching '{series_keyword}'"}

    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }

    updated_events = []

    for event in events:
        updates = {}

        # Update title if specified
        if new_title:
            updates["title"] = new_title

        # Update time/day if specified
        if new_day or new_time:
            try:
                current_start = datetime.strptime(event["start_time"], "%Y-%m-%d %H:%M:%S")
                current_end = datetime.strptime(event["end_time"], "%Y-%m-%d %H:%M:%S")
                duration = current_end - current_start

                # If changing day of week
                if new_day:
                    target_day = day_map.get(new_day.lower())
                    if target_day is not None:
                        current_day = current_start.weekday()
                        days_diff = target_day - current_day
                        current_start = current_start + timedelta(days=days_diff)

                # If changing time
                if new_time:
                    time_parts = new_time.split(":")
                    hour = int(time_parts[0])
                    minute = int(time_parts[1]) if len(time_parts) > 1 else 0
                    second = int(time_parts[2]) if len(time_parts) > 2 else 0
                    current_start = current_start.replace(hour=hour, minute=minute, second=second)

                # Calculate new end time preserving duration
                new_end = current_start + duration

                updates["start_time"] = current_start.strftime("%Y-%m-%d %H:%M:%S")
                updates["end_time"] = new_end.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                print(f"Error updating time for event {event.get('event_id')}: {e}")
                continue
        if new_participants:
            updates["participants"] = new_participants
        print(f"UPDATE_RECURRING_SERIES - Event {event.get('event_id')} '{event.get('title')}': updates={updates}")  # debug
        if updates:
            event_id = event.get("event_id")
            updated_event = db.update_event(event_id, **updates)
            print(f"UPDATE_RECURRING_SERIES - db.update_event returned: {updated_event is not None}")  # debug
            if updated_event:
                updated_events.append(updated_event)

    return {
        "count": len(updated_events),
        "events": updated_events,
        "series_keyword": series_keyword
    }


def delete_recurring_series(username, series_keyword):
    """
    Delete all events in a recurring series.

    Parameters:
    - username: The user who owns the events
    - series_keyword: Name of the series to find and delete

    Returns dict with count of deleted events.
    """
    events = find_recurring_series_events(username, series_keyword)

    if not events:
        return {"count": 0, "deleted": [], "error": f"No recurring series found matching '{series_keyword}'"}

    deleted_events = []

    for event in events:
        event_id = event.get("event_id")
        try:
            deleted = db.delete_event(event_id)
            if deleted:
                deleted_events.append(deleted)
        except Exception as e:
            print(f"Error deleting event {event_id}: {e}")

    return {
        "count": len(deleted_events),
        "deleted": deleted_events,
        "series_keyword": series_keyword
    }


# -----------------------------
# RAG
# -----------------------------
# Vectorisation
def store_event_embedding(event_id, content):
    """Store embedding for an event in the database."""
    vector = embed_model.encode(content)
    db.update_event_embedding(event_id, vector)
    return vector

def embed_existing_event_notes(username):
    """
    Embed notes from all existing calendar events into RAG.
    Called on startup to make pre-populated event notes searchable.
    """
    embedded_count = 0
    all_events = db.get_user_events(username)
    for event in all_events:
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
            store_event_embedding(event["event_id"], content)
            embedded_count += 1
            print(f"[RAG] Embedded notes for: {event['title']} on {date_str}")

    print(f"[RAG] Embedded {embedded_count} event notes into RAG")
    return embedded_count

# Compare and rank vectors, retrieve top 3 contexts
def retrieve_top_k(username, query_vector, k=3):
    """Retrieve top-k most similar embeddings from events and conversations."""
    results = []

    # Get events with embeddings
    events_with_embeddings = db.get_events_with_embeddings(username)
    for event in events_with_embeddings:
        if event["embedding"] is not None:
            similarity = util.cos_sim(query_vector, event["embedding"]).item()
            content = f"Meeting '{event['title']}' on {event['start_time'].split(' ')[0]}: {event['notes']}"
            results.append((content, similarity))

    # Get conversations with embeddings
    conversations_with_embeddings = db.get_conversations_with_embeddings(username)
    for conv in conversations_with_embeddings:
        if conv["embedding"] is not None:
            similarity = util.cos_sim(query_vector, conv["embedding"]).item()
            results.append((conv["content"], similarity))

    # Sort by similarity and return top k
    results.sort(key=lambda x: x[1], reverse=True)
    return [content for content, _ in results[:k]]

# -----------------------------
# Calendar event CRUD functions
# -----------------------------
def check_time_conflict(username, start_time, end_time, exclude_event_id=None):
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

    all_events = db.get_user_events(username)
    for event in all_events:
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

# -----------------------------
# Agenda Suggestions for Recurring Meetings
# -----------------------------
def get_upcoming_recurring_meetings(username):
    """
    Get recurring meetings that are coming up (in the future)
    that have a past occurrence with notes.
    Returns list of {event, last_occurrence_notes, suggested_agenda}
    """
    now = datetime.now()
    print(f"[Agenda] Current time: {now}")  # debug

    all_events = db.get_user_events(username)

    # Group events by recurrence_group
    recurrence_groups = {}
    for event in all_events:
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
def analyze_scheduling_patterns(username):
    """
    Analyze calendar events to infer user scheduling patterns.
    Detects implicit recurring patterns even for non-recurring events.
    """
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Group events by title + day of week to detect recurring patterns
    title_day_patterns = {}  # {(title, day_of_week): [list of events]}

    all_events = db.get_user_events(username)
    for event in all_events:
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


def get_scheduling_insight(username):
    """
    Generate a contextual scheduling insight based on inferred patterns.
    Detects missing recurring meetings and suggests based on history.
    """
    patterns = analyze_scheduling_patterns(username)
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

        # If it's the usual day, before the usual time, and no meeting scheduled this week
        if usual_day == current_day and not has_current_week_occurrence and current_hour < typical_hour:
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
    all_events = db.get_user_events(username)
    for event in all_events:
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
        elif current_day not in ("Saturday", "Sunday") and current_hour >= 14:
            return f"{current_day} afternoon - good time for focused work"
        else:
            return "What would you like to add to your calendar?"
        
def get_current_week_range():
    """Get the start and end dates of the current week (Monday to Sunday)."""
    now = datetime.now()
    start_of_week = now - timedelta(days=now.weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
    return start_of_week, end_of_week