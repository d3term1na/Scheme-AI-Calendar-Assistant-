from fastapi import FastAPI, Request
import json
from memory import conversation_memory
from tools import create_event, query_event, update_event, delete_event

app = FastAPI()

@app.post("/chat")
# async def chat_endpoint(request: Request):
#     body = await request.json()  # <-- Parse JSON
#     user_message = body["message"]
#     # Pass message to agent logic
#     agent_reply = agent_process(user_message)
#     return {"reply": agent_reply, "requires_clarification": False, "metadata": {}}

async def chat_endpoint(request: Request):
    body = await request.json()
    conversation_id = body.get("conversation_id", "default")
    user_message = body.get("message", "")

    reply, metadata = agent_process(user_message, conversation_id)

    return {
        "reply": reply,
        "requires_clarification": False,
        "metadata": metadata
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

def agent_process(user_message, conversation_id):
    history = conversation_memory.get(conversation_id, [])
    history.append({"user": user_message})

    # Simple intent detection
    message_lower = user_message.lower()
    reply = ""
    metadata = {}

    if "schedule" in message_lower or "create" in message_lower:
        # naive parsing for demo
        event = create_event(title=user_message, start="2026-02-02T12:00", end="2026-02-02T12:45")
        reply = f"Got it! Scheduled: {event['title']} at {event['start']}"
        metadata["events_created"] = [event]
    elif "delete" in message_lower:
        events = query_event()
        if events:
            deleted = delete_event(events[0]["id"])
            reply = f"Deleted event: {deleted['title']}"
            metadata["events_deleted"] = [deleted]
        else:
            reply = "No events to delete."
    elif "show" in message_lower or "list" in message_lower:
        events = query_event()
        if events:
            reply = "Your events:\n" + "\n".join([e["title"] for e in events])
        else:
            reply = "No events found."
    else:
        reply = "I can schedule, delete, or list events. Try something like 'Schedule lunch with Bob'."

    history.append({"agent": reply})
    conversation_memory[conversation_id] = history

    return reply, metadata