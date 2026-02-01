from fastapi import FastAPI, Request
import json
from memory import conversation_memory

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

def agent_process(user_message, conversation_id="abc123"):
    history = conversation_memory.get(conversation_id, [])
    history.append({"user": user_message})
    # run simple intent detection -> call tools
    reply = "Got it!"
    history.append({"agent": reply})
    conversation_memory[conversation_id] = history
    return reply