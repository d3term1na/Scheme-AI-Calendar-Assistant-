from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from memory import conversation_memory
from tools import create_event, query_event, update_event, delete_event, store_embedding, embed_model, retrieve_top_k, call_ollama

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

    message_lower = user_message.lower()
    reply = ""
    metadata = {}

    if "schedule" in message_lower or "create" in message_lower:
        event = create_event(user_message, "2026-02-02T12:00", "2026-02-02T12:45")
        reply = f"Got it! Scheduled: {event['title']} at {event['start_time']}"
        metadata["events_created"] = [event]
    elif "delete" in message_lower:
        events = query_event()
        if events:
            deleted = delete_event(events[0]["event_id"])
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
    elif "update" in message_lower or "change" in message_lower:
        events = query_event()
        if events:
            updated = update_event(events[0]["event_id"],title=user_message)
            reply = f"Updated event: {updated['title']}"
            metadata["events_updated"] = [updated]
    else:
        store_embedding(f"User: {user_message}", doc_type="conversation")
        query_vec = embed_model.encode(user_message)
        top_docs = retrieve_top_k(query_vec, k=3)
        context_text = "\n".join(top_docs) or "No relevant context."
        prompt = f"""
        You are an AI calendar assistant.

        Your job:
        - Help users schedule, update, delete, and query calendar events.
        - Ask clarifying questions if information is missing.
        - Be concise and practical.
        - Do NOT hallucinate events.
        - If no action is required, respond conversationally.

        Always respond in plain English.

        User message:
        {context_text}

        User message: {user_message}

        Assistant:
        """
        reply = call_ollama(prompt)
        metadata["retrieved_docs"] = top_docs
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