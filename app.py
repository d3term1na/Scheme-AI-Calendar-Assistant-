from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import db
from tools import store_event_embedding, embed_model, retrieve_top_k, call_ollama, extract_event_details, extract_query_filters, extract_event_identifier, extract_update_details, extract_notes_details, extract_recurring_details, calculate_recurring_dates, classify_intent, get_upcoming_recurring_meetings, get_scheduling_insight, check_time_conflict, format_conflict_message, extract_bulk_operation_details, embed_existing_event_notes, extract_recurring_operation_details, update_recurring_series, delete_recurring_series
from datetime import datetime as dt, timedelta
import bcrypt
import uuid

# In-memory session store {session_id: username}
sessions = {}


@asynccontextmanager
async def lifespan(app):
    """Embed event notes for pre-populated test users on startup."""
    embed_existing_event_notes("Alice")
    yield

app = FastAPI(lifespan=lifespan)

def get_current_user(request: Request) -> str:
    """Get the current logged-in user from session cookie. Returns None if not logged in."""
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        return sessions[session_id]
    return None


def require_auth(request: Request) -> str:
    """Get the current user or raise 401 if not logged in."""
    username = get_current_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not logged in")
    return username


# -----------------------------
# Authentication Endpoints
# -----------------------------
@app.post("/register")
async def register(request: Request):
    """Register a new user."""
    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")

    if len(password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")

    # Hash password and create user
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    success = db.create_user(username, password_hash)

    if not success:
        raise HTTPException(status_code=409, detail="Username already exists")

    # Auto-login after registration
    session_id = str(uuid.uuid4())
    sessions[session_id] = username

    response = JSONResponse({"message": "Registration successful", "username": username})
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return response


@app.post("/login")
async def login(request: Request):
    """Login with username and password."""
    body = await request.json()
    username = body.get("username", "").strip()
    password = body.get("password", "")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    user = db.get_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Verify password
    if not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Create session
    session_id = str(uuid.uuid4())
    sessions[session_id] = username

    response = JSONResponse({"message": "Login successful", "username": username})
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return response


@app.post("/logout")
async def logout(request: Request):
    """Logout the current user."""
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        del sessions[session_id]

    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie("session_id")
    return response


@app.get("/me")
async def get_current_user_info(request: Request):
    """Get the current logged-in user info."""
    username = get_current_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not logged in")
    return {"username": username}

# -----------------------------
# Agenda Suggestions Endpoint
# -----------------------------
@app.get("/agenda-suggestions")
async def get_agenda_suggestions(request: Request):
    """Get agenda suggestions for upcoming recurring meetings based on past notes."""
    username = require_auth(request)
    suggestions = get_upcoming_recurring_meetings(username)

    # Format for frontend consumption
    formatted = []
    for item in suggestions:
        formatted.append({
            "event_id": item["upcoming_event"]["event_id"],
            "event_title": item["upcoming_event"]["title"],
            "event_time": item["upcoming_event"]["start_time"],
            "last_meeting_date": item["last_occurrence"]["start_time"],
            "suggested_agenda": item["suggested_agenda"],
            "recurrence_group": item["upcoming_event"].get("recurrence_group")
        })

    return {"suggestions": formatted}


@app.get("/events")
async def get_all_events(request: Request):
    """Get all calendar events for initial page load."""
    username = require_auth(request)
    return {"events": db.get_user_events(username)}


@app.get("/scheduling-insight")
async def get_insight(request: Request):
    """Get a contextual scheduling insight based on user's calendar patterns."""
    username = require_auth(request)
    insight = get_scheduling_insight(username)
    return {"insight": insight}


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/chat")
async def chat_endpoint(request: Request):
    username = require_auth(request)
    body = await request.json()
    conversation_id = body.get("conversation_id", "default")
    user_message = body.get("message", "")

    reply, metadata = agent_process(username, user_message, conversation_id)
    print("Sending reply:", reply)  # debug
    return {
        "reply": reply,
        "requires_clarification": False,
        "metadata": metadata
    }

# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
# -----------------------------
# Agent logic
# -----------------------------
def agent_process(username, user_message, conversation_id="default"):
    # Get conversation history from database
    history = db.get_conversation_history(username, conversation_id)
    history.append({"user": user_message})

    reply = ""
    metadata = {}

    # Classify intent using LLM
    intent = classify_intent(user_message)
    metadata["intent"] = intent

    if intent == "CREATE":
        details = extract_event_details(user_message)

        # Check for time conflicts before creating
        conflicts = check_time_conflict(username, details["start_time"], details["end_time"])
        if conflicts:
            conflict_msg = format_conflict_message(conflicts)
            # Format the requested time nicely
            time_str = details['start_time']
            try:
                parsed = dt.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                time_str = parsed.strftime("%B %d at %I:%M %p").replace(" 0", " ").lstrip("0")
            except:
                pass
            reply = f"I can't schedule '{details['title']}' for {time_str}. {conflict_msg} Would you like to pick a different time?"
            metadata["conflict"] = True
            metadata["conflicting_events"] = conflicts
        else:
            event = db.create_event(
                username=username,
                title=details["title"],
                start_time=details["start_time"],
                end_time=details["end_time"],
                participants=details.get("participants", [])
            )
            # Build natural language response
            participants_str = ""
            if event.get('participants'):
                participants_str = f" with {', '.join(event['participants'])}"
            # Format time nicely
            time_str = event['start_time']
            try:
                parsed = dt.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                time_str = parsed.strftime("%B %d at %I:%M %p").replace(" 0", " ").lstrip("0")
            except:
                pass
            reply = f"Got it! Scheduled '{event['title']}'{participants_str} for {time_str}."
            metadata["events_created"] = [event]
            # Store richer event info in RAG
            participants_info = f" with {', '.join(event['participants'])}" if event.get('participants') else ""
            content = f"Event: {event['title']}{participants_info} scheduled for {event['start_time']}"
            store_event_embedding(event["event_id"], content)
    elif intent == "CREATE_RECURRING":
        details = extract_recurring_details(user_message)
        print(f"Recurring details: {details}")  # debug

        # Calculate the dates for recurring events
        dates = calculate_recurring_dates(details)

        if not dates:
            reply = "I couldn't determine the recurring schedule. Please specify the day and frequency."
        else:
            created_events = []
            skipped_dates = []  # Dates skipped due to conflicts
            time_str = details.get("time", "09:00:00")
            duration = details.get("duration_minutes", 45)

            # Generate a unique recurrence_group ID for this series
            recurrence_group = str(uuid.uuid4())[:8]
            event_participants = details.get("participants").copy()
            event_participants.append(username)
            for event_date in dates:
                # Build start and end times
                start_datetime = f"{event_date.strftime('%Y-%m-%d')} {time_str}"
                try:
                    start_dt = dt.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
                except:
                    start_dt = dt.strptime(f"{event_date.strftime('%Y-%m-%d')} 09:00:00", "%Y-%m-%d %H:%M:%S")
                end_dt = start_dt + timedelta(minutes=duration)

                start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                end_time_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

                # Check for conflicts
                conflicts = check_time_conflict(username, start_time_str, end_time_str)
                if conflicts:
                    skipped_dates.append({
                        "date": event_date,
                        "conflicts": conflicts
                    })
                    continue  # Skip this occurrence
                
                event = db.create_event(
                    username=username,
                    title=details["title"],
                    start_time=start_time_str,
                    end_time=end_time_str,
                    participants=event_participants,
                    recurrence_group=recurrence_group
                )
                created_events.append(event)
                print(event)
                # Store in RAG
                participants_info = f" with {', '.join(event['participants'])}" if event.get('participants') else ""
                content = f"Event: {event['title']}{participants_info} scheduled for {event['start_time']}"
                store_event_embedding(event["event_id"], content)

            # Build natural language response
            num_events = len(created_events)
            frequency = details.get("frequency", "weekly")
            day_of_week = details.get("day_of_week", "")
            participants_str = ""
            if event.get('participants'):
                participants_str = f" with {', '.join(details.get("participants"))}"
            # Format time nicely
            try:
                time_parsed = dt.strptime(time_str, "%H:%M:%S")
                time_formatted = time_parsed.strftime("%I:%M %p").lstrip("0")
            except:
                time_formatted = time_str

            if num_events == 0:
                # All dates had conflicts
                conflict_dates = [s["date"].strftime("%B %d") for s in skipped_dates]
                reply = f"I couldn't schedule any '{details['title']}' events. All requested times conflict with existing events on {', '.join(conflict_dates)}."
                metadata["conflict"] = True
            elif skipped_dates:
                # Some dates had conflicts
                first_date = created_events[0]["start_time"].split(" ")[0]
                first_date_str = dt.strptime(first_date, "%Y-%m-%d").strftime("%B %d")
                skipped_date_strs = [s["date"].strftime("%B %d") for s in skipped_dates]

                if frequency == "weekly" and day_of_week:
                    reply = f"I've scheduled '{details['title']}'{participants_str} for every {day_of_week.capitalize()} at {time_formatted}, starting {first_date_str} ({num_events} events). Skipped {', '.join(skipped_date_strs)} due to conflicts."
                else:
                    reply = f"Created {num_events} '{details['title']}' events{participants_str} starting {first_date_str}. Skipped {', '.join(skipped_date_strs)} due to conflicts."

                metadata["events_created"] = created_events
                metadata["skipped_due_to_conflict"] = skipped_dates
            else:
                # No conflicts
                first_date = dates[0]
                first_date_str = first_date.strftime("%B %d")
                
                if frequency == "weekly" and day_of_week:
                    reply = f"Done! I've scheduled '{details['title']}'{participants_str} for every {day_of_week.capitalize()} at {time_formatted}, starting {first_date_str} ({num_events} events total)."
                elif frequency == "daily":
                    reply = f"Done! I've scheduled '{details['title']}'{participants_str} daily at {time_formatted}, starting {first_date_str} ({num_events} events total)."
                else:
                    reply = f"Done! I've created {num_events} recurring '{details['title']}' events{participants_str} starting {first_date_str} at {time_formatted}."

                metadata["events_created"] = created_events
    elif intent == "DELETE":
        print(f"DELETE - Getting events for user: {username}")  # debug
        filters = extract_query_filters(user_message)
        print(f"DELETE - Extracted filters: {filters}")  # debug
        events = db.query_events(
            username=username,
            start_date=filters["start_date"],
            end_date=filters["end_date"],
            participants=filters["participants"],
            keyword=filters["keyword"]
        )

        # Fallback: if no events found with keyword, try without keyword
        if not events and filters["keyword"]:
            print("DELETE - Trying fallback without keyword filter...")  # debug
            events = db.query_events(
                username=username,
                start_date=filters["start_date"],
                end_date=filters["end_date"],
                participants=filters["participants"],
                keyword=None
            )

        # Fallback: if still no events, try without date filter
        if not events and (filters["start_date"] or filters["end_date"]):
            print("DELETE - Trying fallback without date filter...")  # debug
            events = db.query_events(
                username=username,
                participants=filters["participants"],
                keyword=filters["keyword"]
            )

        if events:
            deleted = db.delete_event(events[0]["event_id"])
            reply = f"Deleted event: {deleted['title']} ({deleted['start_time']})"
            metadata["events_deleted"] = [deleted]
        else:
            reply = "No matching events found to delete."
    elif intent == "QUERY":
        filters = extract_query_filters(user_message)
        print(f"QUERY - Extracted filters: {filters}")  # debug

        # Default to current datetime onwards when no date filters are specified
        query_start = filters["start_date"]
        if not query_start and not filters["end_date"]:
            query_start = dt.now().strftime("%Y-%m-%d %H:%M:%S")

        events = db.query_events(
            username=username,
            start_date=query_start,
            end_date=filters["end_date"],
            participants=filters["participants"],
            keyword=filters["keyword"]
        )
        print(f"QUERY - Found {len(events)} events")  # debug

        # Detect if user is asking a specific question about event details
        msg_lower = user_message.lower()
        is_asking_participants = any(phrase in msg_lower for phrase in ["who was", "who is", "who are", "who's in", "with whom", "participants", "attendees", "who attended", "who will be"])
        is_asking_notes = any(phrase in msg_lower for phrase in ["what notes", "what was discussed", "notes from", "summary of"])

        if events:
            if is_asking_participants and len(events) == 1:
                # User is asking about participants for a specific event
                event = events[0]
                participants = event.get("participants", [])
                if participants:
                    reply = f"'{event['title']}' on {event['start_time'].split(' ')[0]} had these participants: {', '.join(participants)}."
                else:
                    reply = f"'{event['title']}' on {event['start_time'].split(' ')[0]} had no participants listed - it appears to be a solo event."
            elif is_asking_notes and len(events) == 1:
                # User is asking about notes for a specific event
                event = events[0]
                notes = event.get("notes", "")
                if notes:
                    reply = f"Notes from '{event['title']}': {notes}"
                else:
                    reply = f"No notes recorded for '{event['title']}'."
            else:
                # Default: list events
                event_lines = []
                for e in events:
                    line = f"- {e['title']} at {e['start_time']}"
                    if e.get("participants"):
                        line += f" (with {', '.join(e['participants'])})"
                    event_lines.append(line)
                reply = "Your events:\n" + "\n".join(event_lines)
            metadata["events_queried"] = events
        else:
            reply = "No events found matching your criteria."
    elif intent == "UPDATE":
        print(f"UPDATE - Getting events for user: {username}")  # debug
        identifier = extract_event_identifier(user_message)
        print(f"UPDATE - Extracted identifier: {identifier}")  # debug
        events = db.query_events(
            username=username,
            start_date=identifier["current_date"],
            end_date=identifier["current_date"],
            participants=identifier["participants"],
            keyword=identifier["keyword"]
        )

        # Fallback: if no events found with keyword, try without keyword
        if not events and identifier["keyword"]:
            print("UPDATE - Trying fallback without keyword filter...")  # debug
            print(identifier["keyword"])
            events = db.query_events(
                username=username,
                start_date=identifier["current_date"],
                end_date=identifier["current_date"],
                participants=identifier["participants"],
                keyword=None
            )
            print(events)
            print(f"UPDATE - Events found without keyword: {len(events)}")  # debug

        # Fallback: if still no events, try without date filter
        if not events and identifier["current_date"]:
            print("UPDATE - Trying fallback without date filter...")  # debug
            events = db.query_events(
                username=username,
                participants=identifier["participants"],
                keyword=identifier["keyword"]
            )
            print(f"UPDATE - Events found without date: {len(events)}")  # debug

        # Fallback: get all events if still none found
        if not events:
            print("UPDATE - Trying fallback to get all events...")  # debug
            events = db.query_events(username=username)
            print(f"UPDATE - Total events in calendar: {len(events)}")  # debug

        if events:
            event = events[0]
            update_details = extract_update_details(user_message)

            # Build updates dict with only non-null values
            updates = {}
            if update_details["new_title"] and update_details["new_title"] != event["title"]:
                updates["title"] = update_details["new_title"]
            if update_details["new_start_time"]:
                updates["start_time"] = update_details["new_start_time"]
            if update_details["new_end_time"]:
                updates["end_time"] = update_details["new_end_time"]
            if update_details["new_participants"] is not None:
                updates["participants"] = update_details["new_participants"] + [username]
            elif update_details["add_participants"]:
                current = event.get("participants", [])
                updates["participants"] = current + update_details["add_participants"]
            elif update_details["remove_participants"]:
                current = event.get("participants", [])
                updates["participants"] = [p for p in current if p not in update_details["remove_participants"]]

            # Check for time conflicts if rescheduling
            if "start_time" in updates or "end_time" in updates:
                new_start = updates.get("start_time", event["start_time"])
                new_end = updates.get("end_time", event["end_time"])

                # Exclude the current event from conflict check (don't conflict with itself)
                conflicts = check_time_conflict(username, new_start, new_end, exclude_event_id=event["event_id"])

                if conflicts:
                    conflict_msg = format_conflict_message(conflicts)
                    # Format the requested time nicely
                    time_str = new_start
                    try:
                        parsed = dt.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                        time_str = parsed.strftime("%B %d at %I:%M %p").replace(" 0", " ").lstrip("0")
                    except:
                        pass
                    reply = f"I can't reschedule '{event['title']}' to {time_str}. {conflict_msg} Would you like to pick a different time?"
                    metadata["conflict"] = True
                    metadata["conflicting_events"] = conflicts
                else:
                    # No conflicts, proceed with update
                    updated = db.update_event(event["event_id"], **updates)
                    # Build natural language response
                    change_parts = []
                    if "start_time" in updates:
                        time_str = updates['start_time']
                        try:
                            parsed = dt.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                            time_str = parsed.strftime("%B %d at %I:%M %p").replace(" 0", " ").lstrip("0")
                        except:
                            pass
                        change_parts.append(f"rescheduled it to {time_str}")
                    if "title" in updates:
                        change_parts.append(f"renamed it to '{updates['title']}'")
                    if "participants" in updates:
                        change_parts.append("updated the participants")

                    if change_parts:
                        reply = f"Done! I've {' and '.join(change_parts)} for '{event['title']}'."
                    else:
                        reply = f"Updated '{updated['title']}' successfully."
                    metadata["events_updated"] = [updated]
            elif updates:
                # No time change, just update other fields (no conflict check needed)
                updated = db.update_event(event["event_id"], **updates)
                change_parts = []
                if "title" in updates:
                    change_parts.append(f"renamed it to '{updates['title']}'")
                if "participants" in updates:
                    change_parts.append("updated the participants")

                if change_parts:
                    reply = f"Done! I've {' and '.join(change_parts)} for '{event['title']}'."
                else:
                    reply = f"Updated '{updated['title']}' successfully."
                metadata["events_updated"] = [updated]
            else:
                reply = "I couldn't understand what you want to change. Please specify the new time, title, or participants."
        else:
            reply = "No matching events found to update."
    elif intent == "ADD_NOTES":
        notes_details = extract_notes_details(user_message)
        print(f"ADD_NOTES - Notes details: {notes_details}")  # debug
        print(f"ADD_NOTES - Searching for event_date: {notes_details['event_date']}, keyword: {notes_details['keyword']}")  # debug

        # Find the event to add notes to
        events = db.query_events(
            username=username,
            start_date=notes_details["event_date"],
            end_date=notes_details["event_date"],
            participants=notes_details["participants"],
            keyword=notes_details["keyword"]
        )
        print(f"ADD_NOTES - Found {len(events)} matching events")  # debug
        if events:
            print(f"ADD_NOTES - First match: {events[0]['title']} at {events[0]['start_time']}")  # debug

        # Fallback: try without date filter
        if not events and notes_details["keyword"]:
            events = db.query_events(
                username=username,
                participants=notes_details["participants"],
                keyword=notes_details["keyword"]
            )

        # Fallback: get all events if still none found
        if not events:
            events = db.query_events(username=username)

        if events and notes_details["notes"]:
            event = events[0]
            # Append to existing notes or create new
            existing_notes = event.get("notes", "")
            if existing_notes:
                new_notes = existing_notes + "\n" + notes_details["notes"]
            else:
                new_notes = notes_details["notes"]

            updated = db.update_event(event["event_id"], notes=new_notes)
            reply = f"Added notes to '{event['title']}': \"{notes_details['notes']}\""
            metadata["events_updated"] = [updated]
            # Store notes in RAG for future questions
            notes_content = f"Meeting '{event['title']}' on {event['start_time']}: {notes_details['notes']}"
            store_event_embedding(event["event_id"], notes_content)
        elif not notes_details["notes"]:
            reply = "I couldn't understand what notes you want to add. Please try again."
        else:
            reply = "No matching events found to add notes to."
    elif intent == "BULK_RESCHEDULE":
        bulk_details = extract_bulk_operation_details(user_message)
        source_date = bulk_details["source_date"]
        destination_date = bulk_details["destination_date"]
        print(f"BULK_RESCHEDULE - source: {source_date}, destination: {destination_date}")  # debug

        if not source_date or not destination_date:
            reply = "I couldn't understand which dates you want to move events between. Please specify the source and destination dates."
        else:
            # Find all events on the source date
            events_to_move = db.query_events(username=username, start_date=source_date, end_date=source_date)
            print(f"BULK_RESCHEDULE - Found {len(events_to_move)} events on {source_date}")  # debug

            if not events_to_move:
                # Format date nicely
                try:
                    source_parsed = dt.strptime(source_date, "%Y-%m-%d")
                    source_str = source_parsed.strftime("%B %d")
                except:
                    source_str = source_date
                reply = f"You don't have any events scheduled on {source_str}."
            else:
                moved_events = []
                conflict_events = []

                for event in events_to_move:
                    # Calculate new start and end times
                    try:
                        old_start = dt.strptime(event["start_time"], "%Y-%m-%d %H:%M:%S")
                        old_end = dt.strptime(event["end_time"], "%Y-%m-%d %H:%M:%S")
                        duration = old_end - old_start

                        # Keep the same time, just change the date
                        new_start = dt.strptime(f"{destination_date} {old_start.strftime('%H:%M:%S')}", "%Y-%m-%d %H:%M:%S")
                        new_end = new_start + duration

                        new_start_str = new_start.strftime("%Y-%m-%d %H:%M:%S")
                        new_end_str = new_end.strftime("%Y-%m-%d %H:%M:%S")

                        # Check for conflicts on the destination date
                        conflicts = check_time_conflict(username, new_start_str, new_end_str, exclude_event_id=event["event_id"])

                        if conflicts:
                            conflict_events.append({
                                "event": event,
                                "conflicts": conflicts
                            })
                        else:
                            # Move the event
                            updated = db.update_event(event["event_id"], start_time=new_start_str, end_time=new_end_str)
                            moved_events.append(updated)
                    except Exception as e:
                        print(f"BULK_RESCHEDULE - Error moving event {event['title']}: {e}")
                        continue

                # Build response
                try:
                    source_parsed = dt.strptime(source_date, "%Y-%m-%d")
                    dest_parsed = dt.strptime(destination_date, "%Y-%m-%d")
                    source_str = source_parsed.strftime("%B %d")
                    dest_str = dest_parsed.strftime("%B %d")
                except:
                    source_str = source_date
                    dest_str = destination_date

                if moved_events and not conflict_events:
                    event_names = [f"'{e['title']}'" for e in moved_events]
                    if len(event_names) == 1:
                        reply = f"Done! I've moved {event_names[0]} from {source_str} to {dest_str}."
                    else:
                        reply = f"Done! I've moved {len(moved_events)} events from {source_str} to {dest_str}: {', '.join(event_names)}."
                    metadata["events_updated"] = moved_events
                elif moved_events and conflict_events:
                    moved_names = [f"'{e['title']}'" for e in moved_events]
                    conflict_names = [f"'{c['event']['title']}'" for c in conflict_events]
                    reply = f"I've moved {len(moved_events)} events from {source_str} to {dest_str}: {', '.join(moved_names)}. However, {', '.join(conflict_names)} could not be moved due to conflicts."
                    metadata["events_updated"] = moved_events
                    metadata["conflicts"] = conflict_events
                else:
                    conflict_names = [f"'{c['event']['title']}'" for c in conflict_events]
                    reply = f"I couldn't move any events from {source_str} to {dest_str}. All events conflict with existing events on {dest_str}: {', '.join(conflict_names)}."
                    metadata["conflicts"] = conflict_events
    elif intent == "BULK_CANCEL":
        bulk_details = extract_bulk_operation_details(user_message)
        source_date = bulk_details["source_date"]
        print(f"BULK_CANCEL - source: {source_date}")  # debug

        if not source_date:
            reply = "I couldn't understand which date you want to cancel events on. Please specify the date."
        else:
            # Find all events on the source date
            events_to_cancel = db.query_events(username=username, start_date=source_date, end_date=source_date)
            print(f"BULK_CANCEL - Found {len(events_to_cancel)} events on {source_date}")  # debug

            if not events_to_cancel:
                try:
                    source_parsed = dt.strptime(source_date, "%Y-%m-%d")
                    source_str = source_parsed.strftime("%B %d")
                except:
                    source_str = source_date
                reply = f"You don't have any events scheduled on {source_str}."
            else:
                deleted_events = []
                for event in events_to_cancel:
                    deleted = db.delete_event(event["event_id"])
                    if deleted:
                        deleted_events.append(deleted)

                # Build response
                try:
                    source_parsed = dt.strptime(source_date, "%Y-%m-%d")
                    source_str = source_parsed.strftime("%B %d")
                except:
                    source_str = source_date

                event_names = [f"'{e['title']}'" for e in deleted_events]
                if len(event_names) == 1:
                    reply = f"Done! I've cancelled {event_names[0]} on {source_str}."
                else:
                    reply = f"Done! I've cancelled {len(deleted_events)} events on {source_str}: {', '.join(event_names)}."
                metadata["events_deleted"] = deleted_events
    elif intent == "UPDATE_RECURRING":
        # Update all events in a recurring series
        recurring_details = extract_recurring_operation_details(user_message)
        series_keyword = recurring_details.get("series_keyword")
        print(f"UPDATE_RECURRING - series_keyword: {series_keyword}, details: {recurring_details}")  # debug

        if not series_keyword:
            reply = "I couldn't understand which recurring series you want to update. Please specify the meeting name."
        else:
            update_participants = recurring_details.get("new_participants").copy()
            update_participants.append(username)
            result = update_recurring_series(
                username=username,
                series_keyword=series_keyword,
                new_title=recurring_details.get("new_title"),
                new_day=recurring_details.get("new_day"),
                new_time=recurring_details.get("new_time"),
                new_participants=update_participants
            )

            if result.get("error"):
                reply = result["error"]
            elif result["count"] == 0:
                # Not a recurring series — fall back to single-event update
                print(f"UPDATE_RECURRING fallback - treating as single event update")  # debug
                identifier = extract_event_identifier(user_message)
                events = db.query_events(
                    username=username,
                    start_date=identifier["current_date"],
                    end_date=identifier["current_date"],
                    participants=identifier["participants"],
                    keyword=identifier["keyword"]
                )
                if not events and identifier["keyword"]:
                    events = db.query_events(username=username, keyword=identifier["keyword"])
                if not events:
                    events = db.query_events(username=username)

                if events:
                    event = events[0]
                    update_details = extract_update_details(user_message)
                    updates = {}
                    if update_details["new_title"] and update_details["new_title"] != event["title"]:
                        updates["title"] = update_details["new_title"]
                    if update_details["new_start_time"]:
                        updates["start_time"] = update_details["new_start_time"]
                    if update_details["new_end_time"]:
                        updates["end_time"] = update_details["new_end_time"]

                    if "start_time" in updates or "end_time" in updates:
                        new_start = updates.get("start_time", event["start_time"])
                        new_end = updates.get("end_time", event["end_time"])
                        conflicts = check_time_conflict(username, new_start, new_end, exclude_event_id=event["event_id"])
                        if conflicts:
                            conflict_msg = format_conflict_message(conflicts)
                            time_str = new_start
                            try:
                                parsed = dt.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                                time_str = parsed.strftime("%B %d at %I:%M %p").replace(" 0", " ").lstrip("0")
                            except:
                                pass
                            reply = f"I can't reschedule '{event['title']}' to {time_str}. {conflict_msg} Would you like to pick a different time?"
                            metadata["conflict"] = True
                            metadata["conflicting_events"] = conflicts
                        else:
                            updated = db.update_event(event["event_id"], **updates)
                            time_str = updates.get('start_time', '')
                            try:
                                parsed = dt.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                                time_str = parsed.strftime("%B %d at %I:%M %p").replace(" 0", " ").lstrip("0")
                            except:
                                pass
                            reply = f"Done! I've rescheduled '{event['title']}' to {time_str}."
                            metadata["events_updated"] = [updated]
                    elif updates:
                        updated = db.update_event(event["event_id"], **updates)
                        reply = f"Updated '{updated['title']}' successfully."
                        metadata["events_updated"] = [updated]
                    else:
                        reply = "I couldn't understand what you want to change."
                else:
                    reply = f"No events found matching '{series_keyword}'."
            else:
                # Build a descriptive response
                change_parts = []
                if recurring_details.get("new_title"):
                    change_parts.append(f"renamed to '{recurring_details['new_title']}'")
                if recurring_details.get("new_day"):
                    change_parts.append(f"moved to {recurring_details['new_day'].capitalize()}s")
                if recurring_details.get("new_time"):
                    try:
                        time_parsed = dt.strptime(recurring_details['new_time'], "%H:%M:%S")
                        time_formatted = time_parsed.strftime("%I:%M %p").lstrip("0")
                        change_parts.append(f"rescheduled to {time_formatted}")
                    except:
                        change_parts.append(f"rescheduled to {recurring_details['new_time']}")
                if recurring_details.get("new_participants"):
                    change_parts.append(f"updated participants to {', '.join(recurring_details.get("new_participants"))}")
                changes_str = " and ".join(change_parts) if change_parts else "updated"
                reply = f"Done! I've {changes_str} all {result['count']} events in the '{series_keyword}' series."
                metadata["events_updated"] = result["events"]
    elif intent == "DELETE_RECURRING":
        # Delete all events in a recurring series
        recurring_details = extract_recurring_operation_details(user_message)
        series_keyword = recurring_details.get("series_keyword")
        print(f"DELETE_RECURRING - series_keyword: {series_keyword}")  # debug

        if not series_keyword:
            reply = "I couldn't understand which recurring series you want to delete. Please specify the meeting name."
        else:
            result = delete_recurring_series(username, series_keyword)

            if result.get("error"):
                reply = result["error"]
            elif result["count"] == 0:
                reply = f"No recurring events found matching '{series_keyword}'."
            else:
                reply = f"Done! I've deleted all {result['count']} events in the '{series_keyword}' series."
                metadata["events_deleted"] = result["deleted"]
    else:  # GENERAL or fallback
        # First, try to find relevant calendar events with notes
        # Extract filters to find specific events the user is asking about
        filters = extract_query_filters(user_message)
        print(f"GENERAL - Extracted filters: {filters}")  # debug

        # Search for events matching the query
        relevant_events = db.query_events(
            username=username,
            start_date=filters["start_date"],
            end_date=filters["end_date"],
            participants=filters["participants"],
            keyword=filters["keyword"]
        )

        # Build context from matching events with notes
        event_context = []
        for event in relevant_events:
            notes = event.get("notes", "")
            if notes and notes.strip():
                # Format date naturally (e.g., "January 29")
                try:
                    event_dt = dt.strptime(event['start_time'], "%Y-%m-%d %H:%M:%S")
                    date_str = event_dt.strftime("%B %d").replace(" 0", " ").lstrip("0")
                except:
                    date_str = event['start_time'].split(' ')[0]
                event_context.append(f"Meeting '{event['title']}' on {date_str}: {notes}")
                print(f"GENERAL - Found event with notes: {event['title']}")  # debug

        # Also get RAG context
        query_vec = embed_model.encode(user_message)
        top_docs = retrieve_top_k(username, query_vec, k=3)

        # Combine event notes context with RAG context
        all_context = event_context + top_docs
        context_text = "\n".join(all_context) if all_context else "No relevant context."
        print("Context text:", context_text)

        reply = call_ollama(user_message, context_text)
        metadata["retrieved_docs"] = top_docs
        metadata["relevant_events"] = relevant_events

    # Save conversation to database
    db.save_conversation_message(username, conversation_id, user_message, reply)
    print("This is the conversation id", conversation_id)

    return reply, metadata
