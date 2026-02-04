from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from memory import conversation_memory, calendar_events
from tools import create_event, query_event, update_event, delete_event, store_embedding, embed_model, retrieve_top_k, call_ollama, extract_event_details, extract_query_filters, extract_event_identifier, extract_update_details, extract_notes_details, extract_recurring_details, calculate_recurring_dates, classify_intent, get_upcoming_recurring_meetings
from datetime import datetime as dt, timedelta

app = FastAPI()

# -----------------------------
# Agenda Suggestions Endpoint
# -----------------------------
@app.get("/agenda-suggestions")
async def get_agenda_suggestions():
    """Get agenda suggestions for upcoming recurring meetings based on past notes."""
    suggestions = get_upcoming_recurring_meetings()

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
async def get_all_events():
    """Get all calendar events for initial page load."""
    return {"events": list(calendar_events.values())}


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    conversation_id = body.get("conversation_id", "default")
    user_message = body.get("message", "")

    reply, metadata = agent_process(user_message, conversation_id)
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
def agent_process(user_message, conversation_id="default"):
    history = conversation_memory.get(conversation_id, [])
    history.append({"user": user_message})

    reply = ""
    metadata = {}

    # Classify intent using LLM
    intent = classify_intent(user_message)
    metadata["intent"] = intent

    if intent == "CREATE":
        details = extract_event_details(user_message)
        event = create_event(
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
        store_embedding(content, doc_type="event")
    elif intent == "CREATE_RECURRING":
        details = extract_recurring_details(user_message)
        print(f"Recurring details: {details}")  # debug

        # Calculate the dates for recurring events
        dates = calculate_recurring_dates(details)

        if not dates:
            reply = "I couldn't determine the recurring schedule. Please specify the day and frequency."
        else:
            created_events = []
            time_str = details.get("time", "09:00:00")
            duration = details.get("duration_minutes", 45)

            # Generate a unique recurrence_group ID for this series
            import uuid
            recurrence_group = str(uuid.uuid4())[:8]

            for event_date in dates:
                # Build start and end times
                start_datetime = f"{event_date.strftime('%Y-%m-%d')} {time_str}"
                try:
                    start_dt = dt.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
                except:
                    start_dt = dt.strptime(f"{event_date.strftime('%Y-%m-%d')} 09:00:00", "%Y-%m-%d %H:%M:%S")
                end_dt = start_dt + timedelta(minutes=duration)

                event = create_event(
                    title=details["title"],
                    start_time=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    end_time=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    participants=details.get("participants", []),
                    recurrence_group=recurrence_group
                )
                created_events.append(event)

                # Store in RAG
                participants_info = f" with {', '.join(event['participants'])}" if event.get('participants') else ""
                content = f"Event: {event['title']}{participants_info} scheduled for {event['start_time']}"
                store_embedding(content, doc_type="event")

            # Build natural language response
            num_events = len(created_events)
            frequency = details.get("frequency", "weekly")
            day_of_week = details.get("day_of_week", "")

            # Format first date nicely
            first_date = dates[0]
            first_date_str = first_date.strftime("%B %d")

            # Format time nicely
            try:
                time_parsed = dt.strptime(time_str, "%H:%M:%S")
                time_formatted = time_parsed.strftime("%I:%M %p").lstrip("0")
            except:
                time_formatted = time_str

            if frequency == "weekly" and day_of_week:
                reply = f"Done! I've scheduled '{details['title']}' for every {day_of_week.capitalize()} at {time_formatted}, starting {first_date_str} ({num_events} events total)."
            elif frequency == "daily":
                reply = f"Done! I've scheduled '{details['title']}' daily at {time_formatted}, starting {first_date_str} ({num_events} events total)."
            else:
                reply = f"Done! I've created {num_events} recurring '{details['title']}' events starting {first_date_str} at {time_formatted}."

            metadata["events_created"] = created_events
    elif intent == "DELETE":
        print(f"DELETE - All calendar events: {list(calendar_events.values())}")  # debug
        filters = extract_query_filters(user_message)
        print(f"DELETE - Extracted filters: {filters}")  # debug
        events = query_event(
            start_date=filters["start_date"],
            end_date=filters["end_date"],
            participants=filters["participants"],
            keyword=filters["keyword"]
        )

        # Fallback: if no events found with keyword, try without keyword
        if not events and filters["keyword"]:
            print("DELETE - Trying fallback without keyword filter...")  # debug
            events = query_event(
                start_date=filters["start_date"],
                end_date=filters["end_date"],
                participants=filters["participants"],
                keyword=None
            )

        # Fallback: if still no events, try without date filter
        if not events and (filters["start_date"] or filters["end_date"]):
            print("DELETE - Trying fallback without date filter...")  # debug
            events = query_event(
                participants=filters["participants"],
                keyword=filters["keyword"]
            )

        # Fallback: get all events if still none found
        if not events:
            print("DELETE - Trying fallback to get all events...")  # debug
            events = query_event()
            print(f"DELETE - Total events in calendar: {len(events)}")  # debug

        if events:
            deleted = delete_event(events[0]["event_id"])
            reply = f"Deleted event: {deleted['title']} ({deleted['start_time']})"
            metadata["events_deleted"] = [deleted]
        else:
            reply = "No matching events found to delete."
    elif intent == "QUERY":
        filters = extract_query_filters(user_message)
        events = query_event(
            start_date=filters["start_date"],
            end_date=filters["end_date"],
            participants=filters["participants"],
            keyword=filters["keyword"]
        )
        if events:
            event_lines = [f"- {e['title']} at {e['start_time']}" for e in events]
            reply = "Your events:\n" + "\n".join(event_lines)
        else:
            reply = "No events found matching your criteria."
    elif intent == "UPDATE":
        print(f"UPDATE - All calendar events: {list(calendar_events.values())}")  # debug
        identifier = extract_event_identifier(user_message)
        print(f"UPDATE - Extracted identifier: {identifier}")  # debug
        events = query_event(
            start_date=identifier["current_date"],
            end_date=identifier["current_date"],
            participants=identifier["participants"],
            keyword=identifier["keyword"]
        )

        # Fallback: if no events found with keyword, try without keyword
        if not events and identifier["keyword"]:
            print("UPDATE - Trying fallback without keyword filter...")  # debug
            events = query_event(
                start_date=identifier["current_date"],
                end_date=identifier["current_date"],
                participants=identifier["participants"],
                keyword=None
            )
            print(f"UPDATE - Events found without keyword: {len(events)}")  # debug

        # Fallback: if still no events, try without date filter
        if not events and identifier["current_date"]:
            print("UPDATE - Trying fallback without date filter...")  # debug
            events = query_event(
                participants=identifier["participants"],
                keyword=identifier["keyword"]
            )
            print(f"UPDATE - Events found without date: {len(events)}")  # debug

        # Fallback: get all events if still none found
        if not events:
            print("UPDATE - Trying fallback to get all events...")  # debug
            events = query_event()
            print(f"UPDATE - Total events in calendar: {len(events)}")  # debug

        if events:
            event = events[0]
            update_details = extract_update_details(user_message)

            # Build updates dict with only non-null values
            updates = {}
            if update_details["new_title"]:
                updates["title"] = update_details["new_title"]
            if update_details["new_start_time"]:
                updates["start_time"] = update_details["new_start_time"]
            if update_details["new_end_time"]:
                updates["end_time"] = update_details["new_end_time"]
            if update_details["new_participants"] is not None:
                updates["participants"] = update_details["new_participants"]
            elif update_details["add_participants"]:
                current = event.get("participants", [])
                updates["participants"] = current + update_details["add_participants"]
            elif update_details["remove_participants"]:
                current = event.get("participants", [])
                updates["participants"] = [p for p in current if p not in update_details["remove_participants"]]

            if updates:
                updated = update_event(event["event_id"], **updates)
                # Build natural language response
                change_parts = []
                if "start_time" in updates:
                    # Format time nicely (remove seconds if present)
                    time_str = updates['start_time']
                    try:
                        parsed = dt.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                        time_str = parsed.strftime("%B %d at %I:%M %p").replace(" 0", " ").lstrip("0")
                    except:
                        pass  # Keep original if parsing fails
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
            else:
                reply = "I couldn't understand what you want to change. Please specify the new time, title, or participants."
        else:
            reply = "No matching events found to update."
    elif intent == "ADD_NOTES":
        notes_details = extract_notes_details(user_message)
        print(f"ADD_NOTES - Notes details: {notes_details}")  # debug
        print(f"ADD_NOTES - Searching for event_date: {notes_details['event_date']}, keyword: {notes_details['keyword']}")  # debug

        # Find the event to add notes to
        events = query_event(
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
            events = query_event(
                participants=notes_details["participants"],
                keyword=notes_details["keyword"]
            )

        # Fallback: get all events if still none found
        if not events:
            events = query_event()

        if events and notes_details["notes"]:
            event = events[0]
            # Append to existing notes or create new
            existing_notes = event.get("notes", "")
            if existing_notes:
                new_notes = existing_notes + "\n" + notes_details["notes"]
            else:
                new_notes = notes_details["notes"]

            updated = update_event(event["event_id"], notes=new_notes)
            reply = f"Added notes to '{event['title']}': \"{notes_details['notes']}\""
            metadata["events_updated"] = [updated]
            # Store notes in RAG for future questions
            notes_content = f"Meeting '{event['title']}' on {event['start_time']}: {notes_details['notes']}"
            store_embedding(notes_content, doc_type="meeting_notes")
        elif not notes_details["notes"]:
            reply = "I couldn't understand what notes you want to add. Please try again."
        else:
            reply = "No matching events found to add notes to."
    else:  # GENERAL or fallback
        query_vec = embed_model.encode(user_message)
        top_docs = retrieve_top_k(query_vec, k=3)
        context_text = "\n".join(top_docs) or "No relevant context."
        print("Context text:", context_text)
        # prompt = f"""
        # You are an AI calendar assistant.
        # Only answer the user query.

        # Context:
        # {context_text}

        # User query: {user_message}

        # """
        # print(prompt)
        reply = call_ollama(user_message,context_text)
        metadata["retrieved_docs"] = top_docs
        store_embedding(f"User: {user_message}", doc_type="conversation")
    history.append({"agent": reply})
    conversation_memory[conversation_id] = history

    return reply, metadata
