conversation_memory = {}  # {conversation_id: [ {"user": msg}, {"agent": reply}, ... ] }
calendar_events = {}       # {event_id: {title, start, end, participants, notes}}
event_count = 0
embeddings = {}
next_id = 1  # Simple counter for doc_id