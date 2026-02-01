from memory import calendar_events, event_count
from datetime import datetime

def create_event(title, start_time, end_time, participants = None, notes=""):
    global event_count
    event_count += 1
    calendar_events[event_count] = {
        "event_id": event_count, 
        "title": title, 
        "start_time": start_time, 
        "end_time": end_time, 
        "participants": participants or [], 
        "notes": notes
    }
    return calendar_events[event_count]

    
def update_event(event_id, **updates):
    if event_id in calendar_events:
        calendar_events[event_id].update(updates)
        return calendar_events[event_id]
    return None

def delete_event(event_id):
    calendar_events.pop(event_count)
    return

def query_event(start_date=None, end_date=None, participants=None, keyword=None):
    '''
    Parameters:
        start_date (str) - "YYYY-MM-DD" format or None
        end_date (str)   - "YYYY-MM-DD" format or None
        participants (list of str) - participants to match, or None
        keyword (str)    - keyword to search in title or notes, or None
    '''
    results = []

    for event in calendar_events.values():
        match = True

        # Check date range
        event_date = datetime.fromisoformat(event["start_time"]).date()
        if start_date:
            start_dt = datetime.fromisoformat(start_date).date()
            if event_date < start_dt:
                match = False
        if end_date:
            end_dt = datetime.fromisoformat(end_date).date()
            if event_date > end_dt:
                match = False

        # Check participants
        if participants:
            if not any(p.lower() in [ep.lower() for ep in event.get("participants", [])] for p in participants):
                match = False

        # Check keyword in title or notes
        if keyword:
            keyword_lower = keyword.lower()
            title_lower = event.get("title", "").lower()
            notes_lower = event.get("notes", "").lower()
            if keyword_lower not in title_lower and keyword_lower not in notes_lower:
                match = False

        if match:
            results.append(event)

    return results