from datetime import datetime, timedelta
from typing import List

from sqlalchemy.orm import Session

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool

from . import models


# --- DB helper functions ---


def db_create_event(
    db: Session,
    user_id: int,
    *,
    title: str,
    start: datetime,
    end: datetime,
    location: str | None = None,
    description: str | None = None,
    assignment_type: str | None = None,
    class_name: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> str:
    event = models.Event(
        title=title,
        start=start,
        end=end,
        location=location,
        description=description,
        assignment_type=assignment_type,
        class_name=class_name,
        status=status,
        priority=priority,
        owner_id=user_id,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return f"Created event id={event.id} titled '{event.title}' starting {event.start}"


def db_update_event(
    db: Session,
    user_id: int,
    *,
    event_id: int,
    title: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    location: str | None = None,
    description: str | None = None,
    assignment_type: str | None = None,
    class_name: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> str:
    event = (
        db.query(models.Event)
        .filter(models.Event.id == event_id, models.Event.owner_id == user_id)
        .first()
    )
    if not event:
        return "Event not found for this user."

    if title is not None:
        event.title = title
    if start is not None:
        event.start = start
    if end is not None:
        event.end = end
    if location is not None:
        event.location = location
    if description is not None:
        event.description = description
    if assignment_type is not None:
        event.assignment_type = assignment_type
    if class_name is not None:
        event.class_name = class_name
    if status is not None:
        event.status = status
    if priority is not None:
        event.priority = priority

    db.commit()
    db.refresh(event)
    return f"Updated event id={event.id} titled '{event.title}'"


def db_delete_event(db: Session, user_id: int, *, event_id: int) -> str:
    event = (
        db.query(models.Event)
        .filter(models.Event.id == event_id, models.Event.owner_id == user_id)
        .first()
    )
    if not event:
        return "Event not found for this user."
    db.delete(event)
    db.commit()
    return f"Deleted event id={event_id}"


def db_list_events(
    db: Session,
    user_id: int,
    *,
    date: datetime | None = None,
    title_query: str | None = None,
) -> List[models.Event]:
    q = db.query(models.Event).filter(models.Event.owner_id == user_id)
    if date is not None:
        # Filter by same calendar date
        q = q.filter(models.Event.start.between(date.replace(hour=0, minute=0, second=0, microsecond=0),
                                               date.replace(hour=23, minute=59, second=59, microsecond=999999)))
    if title_query:
        q = q.filter(models.Event.title.ilike(f"%{title_query}%"))
    return q.order_by(models.Event.start.asc()).all()


# --- Tool factory ---


def make_calendar_tools(db: Session, user_id: int):
    """Create per-user tools so the LLM can never access other users' events."""

    def create_event_tool(
        title: str,
        due_iso: str,
        location: str | None = None,
        description: str | None = None,
        assignment_type: str | None = None,
        class_name: str | None = None,
        status: str | None = None,
        priority: str | None = None,
    ) -> str:
        """Create a new calendar task/event for the current user.

        Treat "due_iso" as the single due date & time (no separate start/end).
        Use a standardized title format: "<Subject> - <Assignment Type>: <Short Description>".
        Examples:
        - "English - Homework: Read Chapter 3"
        - "Math - Exam: Midterm 1"
        """

        due = datetime.fromisoformat(due_iso)
        start = due
        end = due
        return db_create_event(
            db,
            user_id,
            title=title,
            start=start,
            end=end,
            location=location,
            description=description,
            assignment_type=assignment_type,
            class_name=class_name,
            status=status,
            priority=priority,
        )

    def update_event_tool(
        event_id: int,
        title: str | None = None,
        due_iso: str | None = None,
        location: str | None = None,
        description: str | None = None,
        assignment_type: str | None = None,
        class_name: str | None = None,
        status: str | None = None,
        priority: str | None = None,
    ) -> str:
        """Update an existing calendar task/event for the current user.

        Treat "due_iso" (if provided) as the single due date & time.
        Use after you are confident about which event should be changed.
        You can call list_events first to inspect events.
        """

        if due_iso:
            due = datetime.fromisoformat(due_iso)
            start = due
            end = due
        else:
            start = None
            end = None
        return db_update_event(
            db,
            user_id,
            event_id=event_id,
            title=title,
            start=start,
            end=end,
            location=location,
            description=description,
            assignment_type=assignment_type,
            class_name=class_name,
            status=status,
            priority=priority,
        )

    def delete_event_tool(event_id: int) -> str:
        """Delete one calendar event for the current user.

        Use only after confirming which event to delete.
        """

        return db_delete_event(db, user_id, event_id=event_id)

    def list_events_tool(
        date_iso: str | None = None,
        title_query: str | None = None,
    ) -> str:
        """List this user's calendar events.

        Use this to resolve ambiguity (e.g. "my meeting tomorrow") by
        listing candidate events before updating or deleting.
        """

        date: datetime | None = None
        if date_iso:
            lower = date_iso.strip().lower()
            try:
                if lower == "today":
                    date = datetime.now()
                elif lower == "tomorrow":
                    date = datetime.now() + timedelta(days=1)
                else:
                    # Try strict ISO format; if it fails, fall back to no date filter.
                    date = datetime.fromisoformat(date_iso)
            except ValueError:
                # If the date can't be parsed, ignore the date filter and still list events.
                date = None
        events = db_list_events(db, user_id, date=date, title_query=title_query)
        if not events:
            return "No events found."
        lines = []
        for e in events:
            lines.append(
                f"id={e.id} | {e.title} | {e.start.isoformat()} - {e.end.isoformat()} | {e.location or ''}"
            )
        return "\n".join(lines)

    create_event = StructuredTool.from_function(
        func=create_event_tool,
        name="create_event",
        description=create_event_tool.__doc__ or "Create an event.",
    )
    update_event = StructuredTool.from_function(
        func=update_event_tool,
        name="update_event",
        description=update_event_tool.__doc__ or "Update an event.",
    )
    delete_event = StructuredTool.from_function(
        func=delete_event_tool,
        name="delete_event",
        description=delete_event_tool.__doc__ or "Delete an event.",
    )
    list_events = StructuredTool.from_function(
        func=list_events_tool,
        name="list_events",
        description=list_events_tool.__doc__ or "List events.",
    )

    return [create_event, update_event, delete_event, list_events]


# --- Agent factory ---


def get_calendar_agent(db: Session, user_id: int):
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

    tools = make_calendar_tools(db, user_id)

    now_iso = datetime.now().isoformat()

    system_prompt = (
        f"Current datetime (ISO): {now_iso}. "
        "You are a helpful calendar assistant that manages school tasks/events. "
        "Treat every calendar entry as a single due-date task (no separate start vs. end); "
        "internally the system may store start/end, but you should think in terms of one due date & time. "
        "You manage events only for the currently authenticated user and must never access other users' calendars. "
        "When suggesting or creating titles, always use this template: \"<Subject> - <Assignment Type>: <Short Description>\". "
        "Examples of good titles: \"English - Homework: Read Chapter 3\", \"Math - Exam: Midterm 1\", \"Biology - Lab: Photosynthesis experiment report\". "
        "Prefer concise but specific descriptions that clarify what must be done. "
        "Ask clarifying questions when dates, times, or which task to modify are ambiguous. "
        "Use list_events to inspect events (and show IDs) before updating or deleting them. "
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent
