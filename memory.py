conversation_memory = {}  # {conversation_id: [ {"user": msg}, {"agent": reply}, ... ] }
calendar_events = {}       # {event_id: {title, start, end, participants, notes, recurrence_group}}
event_count = 0
embeddings = {} # {"content": content, "vector": vector, "type": doc_type}
next_id = 1  # Simple counter for doc_id


# =============================================================================
# Sample data for testing agenda suggestions feature
# These are recurring meetings with notes from past occurrences
# =============================================================================
def populate_sample_events():
    """Populate calendar with sample recurring events for testing."""
    global calendar_events, event_count

    # Weekly Team Standup - recurring series (recurrence_group: "standup1")
    # Past occurrence with notes (Jan 28, 2026)
    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Team Standup",
        "start_time": "2026-01-28 08:00:00",
        "end_time": "2026-01-28 08:30:00",
        "participants": ["Alice", "Bob", "Charlie"],
        "notes": "Discussed blockers on the API integration. Bob needs help with authentication. Action items: Review PR #42, Update documentation for new endpoints.",
        "recurrence_group": "standup1"
    }
    event_count += 1

    # Current week occurrence (Feb 4, 2026 - today)
    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Team Standup",
        "start_time": "2026-02-04 08:00:00",
        "end_time": "2026-02-04 08:30:00",
        "participants": ["Alice", "Bob", "Charlie"],
        "notes": "",
        "recurrence_group": "standup1"
    }
    event_count += 1

    # Future occurrence (Feb 11, 2026)
    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Team Standup",
        "start_time": "2026-02-11 08:00:00",
        "end_time": "2026-02-11 08:30:00",
        "participants": ["Alice", "Bob", "Charlie"],
        "notes": "",
        "recurrence_group": "standup1"
    }
    event_count += 1

    # Weekly Project Review - recurring series (recurrence_group: "review1")
    # Past occurrence with notes (Jan 30, 2026)
    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Project Review",
        "start_time": "2026-01-30 14:00:00",
        "end_time": "2026-01-30 15:00:00",
        "participants": ["Alice", "David"],
        "notes": "Sprint velocity was 85%. Need to address tech debt in the payment module. Follow up: Schedule meeting with finance team about Q2 budget.",
        "recurrence_group": "review1"
    }
    event_count += 1

    # Upcoming occurrence (Feb 6, 2026)
    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Project Review",
        "start_time": "2026-02-06 14:00:00",
        "end_time": "2026-02-06 15:00:00",
        "participants": ["Alice", "David"],
        "notes": "",
        "recurrence_group": "review1"
    }
    event_count += 1

    # Weekly 1:1 Meeting - recurring series (recurrence_group: "one2one1")
    # Past occurrence with notes (Jan 29, 2026)
    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "1:1 with Manager",
        "start_time": "2026-01-29 11:00:00",
        "end_time": "2026-01-29 11:30:00",
        "participants": ["Alice", "Manager"],
        "notes": "Discussed career growth path. Manager suggested taking the tech lead course. Need to prepare presentation for Q1 review. Follow up on promotion timeline.",
        "recurrence_group": "one2one1"
    }
    event_count += 1

    # Upcoming occurrence (Feb 5, 2026)
    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "1:1 with Manager",
        "start_time": "2026-02-05 11:00:00",
        "end_time": "2026-02-05 11:30:00",
        "participants": ["Alice", "Manager"],
        "notes": "",
        "recurrence_group": "one2one1"
    }
    event_count += 1

    # =============================================================================
    # Deep Work blocks - Friday afternoons pattern
    # Note: Jan 9, 16, 23 are Fridays in 2026. No Deep Work scheduled for Feb 6
    # to demonstrate the "You usually have Deep Work on Fridays" insight
    # =============================================================================
    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Deep Work",
        "start_time": "2026-01-09 14:00:00",
        "end_time": "2026-01-09 18:00:00",
        "participants": [],
        "notes": "Focused coding session - completed API refactoring",
        "recurrence_group": "deepwork1"
    }
    event_count += 1

    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Deep Work",
        "start_time": "2026-01-16 14:00:00",
        "end_time": "2026-01-16 18:00:00",
        "participants": [],
        "notes": "Documentation and code review",
        "recurrence_group": "deepwork1"
    }
    event_count += 1

    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Deep Work",
        "start_time": "2026-01-23 14:00:00",
        "end_time": "2026-01-23 18:00:00",
        "participants": [],
        "notes": "Sprint planning prep",
        "recurrence_group": "deepwork1"
    }
    event_count += 1
    # No Deep Work scheduled for this week's Friday (Feb 6) to trigger insight

    # =============================================================================
    # Additional meetings to show patterns
    # =============================================================================
    # Past lunch meetings with Bob (shows frequent participant)
    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Lunch with Bob",
        "start_time": "2026-01-21 12:00:00",
        "end_time": "2026-01-21 13:00:00",
        "participants": ["Bob"],
        "notes": "Discussed new project ideas",
        "recurrence_group": None
    }
    event_count += 1

    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Lunch with Bob",
        "start_time": "2026-01-28 15:00:00",
        "end_time": "2026-01-28 15:30:00",
        "participants": ["Bob"],
        "notes": "Caught up on sprint progress",
        "recurrence_group": None
    }
    event_count += 1

    # Morning Planning - Monday mornings pattern
    # Jan 19, 26 and Feb 2 are Mondays in 2026
    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Morning Planning",
        "start_time": "2026-01-19 08:30:00",
        "end_time": "2026-01-19 09:00:00",
        "participants": [],
        "notes": "Weekly planning session",
        "recurrence_group": "morningplan1"
    }
    event_count += 1

    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Morning Planning",
        "start_time": "2026-01-26 08:30:00",
        "end_time": "2026-01-26 09:00:00",
        "participants": [],
        "notes": "Reviewed sprint goals",
        "recurrence_group": "morningplan1"
    }
    event_count += 1

    calendar_events[event_count] = {
        "event_id": event_count,
        "title": "Morning Planning",
        "start_time": "2026-02-02 08:30:00",
        "end_time": "2026-02-02 09:00:00",
        "participants": [],
        "notes": "",
        "recurrence_group": "morningplan1"
    }
    event_count += 1

# Auto-populate on import
populate_sample_events()