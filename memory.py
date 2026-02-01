conversation_memory = {}  # {conversation_id: [ {"user": msg}, {"agent": reply}, ... ] }
calendar_events = {}       # {event_id: {title, start, end, participants, notes}}
global event_count
event_count = 0