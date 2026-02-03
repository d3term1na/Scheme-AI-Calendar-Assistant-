from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from memory import conversation_memory
from tools import create_event, query_event, update_event, delete_event, store_embedding, embed_model, retrieve_top_k, call_ollama, extract_event_details, extract_query_filters, extract_event_identifier, extract_update_details, classify_intent

app = FastAPI()


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
            from datetime import datetime as dt
            parsed = dt.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            time_str = parsed.strftime("%B %d at %I:%M %p").replace(" 0", " ").lstrip("0")
        except:
            pass
        reply = f"Got it! Scheduled '{event['title']}'{participants_str} for {time_str}."
        metadata["events_created"] = [event]
        content = f"{event['title']} from {event['start_time']} to {event['end_time']}"
        store_embedding(content, doc_type="event")
    elif intent == "DELETE":
        filters = extract_query_filters(user_message)
        events = query_event(
            start_date=filters["start_date"],
            end_date=filters["end_date"],
            participants=filters["participants"],
            keyword=filters["keyword"]
        )
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
        from memory import calendar_events
        print(f"All calendar events: {list(calendar_events.values())}")  # debug
        identifier = extract_event_identifier(user_message)
        print(f"Update identifier: {identifier}")  # debug
        events = query_event(
            start_date=identifier["current_date"],
            end_date=identifier["current_date"],
            participants=identifier["participants"],
            keyword=identifier["keyword"]
        )
        print(f"Events found with filters: {len(events)}")  # debug

        # Fallback: if no events found with keyword, try without keyword
        if not events and identifier["keyword"]:
            print("Trying fallback without keyword filter...")  # debug
            events = query_event(
                start_date=identifier["current_date"],
                end_date=identifier["current_date"],
                participants=identifier["participants"],
                keyword=None
            )
            print(f"Events found without keyword: {len(events)}")  # debug

        # Fallback: if still no events and no date filter, get all events
        if not events and not identifier["current_date"]:
            print("Trying fallback to get all events...")  # debug
            events = query_event()
            print(f"Total events in calendar: {len(events)}")  # debug
            
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
                        from datetime import datetime as dt
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

    history.append({"agent": reply})
    conversation_memory[conversation_id] = history
    return reply, metadata
    # # Initialize conversation memory if needed
    # history = conversation_memory.get(conversation_id, [])
    # history.append({"user": user_message})

    # # 1️⃣ Store embedding for this user message
    # store_embedding(f"User: {user_message}", doc_type="conversation")

    # # 2️⃣ Retrieve top relevant context
    # query_vec = embed_model.encode(user_message)
    # top_docs = retrieve_top_k(query_vec, k=3)
    # context_text = "\n".join(top_docs) or "No relevant context."

    # # 3️⃣ Build prompt
    # prompt = f"""
    # You are a helpful AI calendar assistant.
    # Use the following context to answer the user's question:

    # Context:
    # {context_text}

    # User message: {user_message}

    # Provide a concise, helpful, friendly response.
    # """

    # # 4️⃣ Call Ollama
    # reply = call_ollama(prompt)
    # history.append({"agent": reply})

    # conversation_memory[conversation_id] = history
    # return reply, {"retrieved_docs": top_docs}

# -----------------------------
# Run server
# -----------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5500)