# Scheme - AI Calendar Assistant

## Overview

A proof-of-concept AI-powered calendar management system that enables natural language scheduling through a RAG (Retrieval-Augmented Generation) pipeline. Users interact via a chat interface to create, query, update, and delete calendar events.

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Backend | FastAPI + Uvicorn | REST API server |
| LLM | Ollama (gemma3:4b) | Natural language understanding |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) | Vector encoding for RAG |
| Frontend | Vanilla HTML/CSS/JS | Chat interface |
| Storage | In-memory dictionaries | Volatile data storage |

## Project Structure

```
├── app.py          # FastAPI app, API endpoints, agent orchestration
├── tools.py        # LLM integration, embeddings, CRUD operations
├── memory.py       # Global state: events, embeddings, conversations
├── frontend/
│   └── index.html  # Chat UI (served at /)
├── requirements.txt
└── README.md
```

## Key Files & Entry Points

| File | Key Functions | Description |
|------|---------------|-------------|
| [app.py](app.py) | `chat_endpoint()` (L12), `agent_process()` (L31) | Request handling and intent routing |
| [tools.py](tools.py) | `call_ollama()` (L18), `create_event()` (L61), `query_event()` (L86) | Business logic and LLM calls |
| [memory.py](memory.py) | Global dicts (L1-5) | State storage |

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama (required - runs locally)
ollama serve

# Start the server
python -m uvicorn app:app --host 0.0.0.0 --port 5500
```

**Prerequisites:**
- Python 3.10+
- Ollama installed with `gemma3:4b` model pulled
- Dependencies from requirements.txt

## API

Single endpoint:
- `POST /chat` - Send user message, receive agent response
  - Body: `{"conversation_id": "string", "message": "string"}`
  - Response: `{"reply": "string", "metadata": {...}, "requires_clarification": bool}`

## Data Flow

1. User sends message via frontend → `/chat` endpoint
2. `agent_process()` routes by intent keywords ([app.py:35-65](app.py#L35-L65))
3. For calendar ops: direct CRUD via tools.py
4. For queries: RAG pipeline (embed → retrieve → LLM → respond)
5. Conversation stored in `memory.conversation_memory`

## Intent Routing Keywords

| Keywords | Action | Handler |
|----------|--------|---------|
| "schedule", "create" | Create event | [app.py:39-44](app.py#L39-L44) |
| "delete" | Delete event | [app.py:45-52](app.py#L45-L52) |
| "show", "list" | List events | [app.py:53-58](app.py#L53-L58) |
| "update", "change" | Update event | [app.py:59-64](app.py#L59-L64) |
| (default) | RAG query | [app.py:66-83](app.py#L66-L83) |

## Known Limitations or Bugs to be fixed
**IMPORTANT**: When you work on a new feature or improvement or bug, create a git branch first. Then work on changes in that branch for the remaining of the session.

- **Volatile storage**: All data lost on restart (no database)
- **Hardcoded context**: LLM context in [tools.py:27](tools.py#L27) is static
- **Basic NLU**: Intent detection via substring matching, not semantic
- **Date parsing**: Events use hardcoded date (2026-02-02)

## Additional Documentation

When working on specific areas, consult:

| Topic | File |
|-------|------|
| Architecture & design patterns | [.claude/docs/architectural_patterns.md](.claude/docs/architectural_patterns.md) |
