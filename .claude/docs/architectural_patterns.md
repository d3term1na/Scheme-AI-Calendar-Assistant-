# Architectural Patterns

This document describes the recurring patterns and design decisions observed across the Scheme codebase.

## 1. Layered Architecture

The application follows a clear layered structure:

```
Frontend (index.html)
    ↓
API Layer (app.py - FastAPI endpoints)
    ↓
Agent/Orchestration Layer (app.py - agent_process)
    ↓
Service Layer (tools.py - business logic)
    ↓
Data Layer (memory.py - global state)
```

**Files implementing this:**
- [app.py:12-24](../../../app.py#L12-L24) - API layer
- [app.py:31-87](../../../app.py#L31-L87) - Orchestration layer
- [tools.py:18-125](../../../tools.py#L18-L125) - Service layer
- [memory.py:1-5](../../../memory.py#L1-L5) - Data layer

## 2. Global State Pattern

All application state is stored in module-level global dictionaries in `memory.py`:

```python
conversation_memory = {}  # {conversation_id: [messages]}
calendar_events = {}      # {event_id: event_dict}
embeddings = {}           # {doc_id: {content, vector, type}}
```

**Rationale:** Simple POC approach avoiding database complexity.

**Trade-offs:**
- Pros: Simple, no setup required, fast access
- Cons: No persistence, no concurrency safety, memory-bound

**Files using this pattern:**
- [memory.py](../../../memory.py) - Defines globals
- [tools.py:61-73](../../../tools.py#L61-L73) - Mutates `calendar_events`
- [tools.py:41-47](../../../tools.py#L41-L47) - Mutates `embeddings`
- [app.py:32](../../../app.py#L32) - Reads/writes `conversation_memory`

## 3. Auto-Incrementing ID Pattern

Both events and embeddings use global counters for ID generation:

```python
# In memory.py
event_count = 0
next_id = 1
```

**Implementation locations:**
- [tools.py:62](../../../tools.py#L62) - `global event_count`
- [tools.py:72](../../../tools.py#L72) - Increment and assign ID
- [tools.py:42](../../../tools.py#L42) - `global next_id` for embeddings

## 4. Intent-Based Routing Pattern

User messages are classified by keyword matching to determine handling path:

```python
message_lower = user_message.lower()
if "schedule" in message_lower or "create" in message_lower:
    # handle create
elif "delete" in message_lower:
    # handle delete
# ... etc
```

**Location:** [app.py:35-65](../../../app.py#L35-L65)

**Trade-offs:**
- Pros: Simple, predictable, fast
- Cons: No semantic understanding, order-dependent, brittle

## 5. RAG (Retrieval-Augmented Generation) Pipeline

For non-CRUD queries, the system uses a RAG pattern:

1. **Embed query** - Convert user message to vector
2. **Retrieve** - Find top-k similar documents by cosine similarity
3. **Augment** - Build context from retrieved documents
4. **Generate** - Call LLM with context + query

**Implementation:**
- [app.py:66](../../../app.py#L66) - Query embedding
- [app.py:67](../../../app.py#L67) - Retrieval call
- [app.py:68](../../../app.py#L68) - Context building
- [app.py:81](../../../app.py#L81) - LLM generation
- [tools.py:49-55](../../../tools.py#L49-L55) - `retrieve_top_k()` implementation
- [tools.py:57-58](../../../tools.py#L57-L58) - Cosine similarity calculation

## 6. Conversation Memory Pattern

Each conversation maintains its own message history keyed by `conversation_id`:

```python
if conversation_id not in conversation_memory:
    conversation_memory[conversation_id] = []
# ... process ...
conversation_memory[conversation_id].append({"user": message, "agent": reply})
```

**Locations:**
- [app.py:32-33](../../../app.py#L32-L33) - Initialize conversation
- [app.py:85](../../../app.py#L85) - Append to history

## 7. External Service Integration Pattern

LLM calls are wrapped in a dedicated function with error handling:

```python
def call_ollama(user_message, context=""):
    payload = {...}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json()["message"]["content"]
    except Exception as e:
        return f"Error calling Ollama: {e}"
```

**Location:** [tools.py:18-38](../../../tools.py#L18-L38)

**Pattern elements:**
- Centralized configuration (URL, model) at module level
- Request construction with system/user roles
- Try-except with fallback message

## 8. CRUD Service Functions Pattern

Calendar operations follow a consistent function signature pattern:

```python
def create_event(title, start, end, participants, notes) -> dict
def update_event(event_id, updates) -> None
def delete_event(event_id) -> None
def query_event(start_date, end_date, participants, keyword) -> list
```

**Locations:**
- [tools.py:61-73](../../../tools.py#L61-L73) - Create
- [tools.py:76-80](../../../tools.py#L76-L80) - Update
- [tools.py:82-84](../../../tools.py#L82-L84) - Delete
- [tools.py:86-125](../../../tools.py#L86-L125) - Query

**Conventions:**
- Global state mutation via `global` keyword
- Return created object on create, None on update/delete
- Query returns filtered list with multiple optional filters

## 9. Frontend-Backend Communication Pattern

JSON-based REST communication with consistent request/response structure:

**Request (frontend → backend):**
```javascript
fetch("/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({conversation_id, message})
})
```

**Response (backend → frontend):**
```python
return {"reply": str, "metadata": dict, "requires_clarification": bool}
```

**Locations:**
- [frontend/index.html:82-91](../../../frontend/index.html#L82-L91) - Client fetch
- [app.py:20-24](../../../app.py#L20-L24) - Response construction
