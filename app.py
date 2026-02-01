from fastapi import FastAPI, Request
import json

app = FastAPI()

# @app.post("/chat")
# async def chat_endpoint(request: Request):
#     body = await request.json()  # <-- Parse JSON
#     user_message = body["message"]
#     # Pass message to agent logic
#     agent_reply = agent_process(user_message)
#     return {"reply": agent_reply, "requires_clarification": False, "metadata": {}}