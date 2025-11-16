from datetime import datetime, timedelta
from typing import List, Dict, Optional, cast
import json

from pymongo.database import Database

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool

from . import models
from .database import get_next_id
from .llm_providers import get_llm


# --- DB helper functions ---


def db_create_event(
    db: Database,
    user_id: int,
    *,
    title: str,
    due: datetime,
    location: str | None = None,
    description: str | None = None,
    assignment_type: str | None = None,
    class_name: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> str:
    events = db["events"]
    event_id = get_next_id("events")
    now = datetime.utcnow()
    event: models.Event = {
        "_id": event_id,
        "id": event_id,
        "title": title,
        "due": due,
        "location": location,
        "description": description,
        "assignment_type": assignment_type,
        "class_name": class_name,
        "status": status,
        "priority": priority,
        "owner_id": user_id,
        "created_at": now,
        "updated_at": now,
    }
    events.insert_one(event)
    return f"Created event id={event_id} titled '{title}' due {due}"


def db_update_event(
    db: Database,
    user_id: int,
    *,
    event_id: int,
    title: str | None = None,
    due: datetime | None = None,
    location: str | None = None,
    description: str | None = None,
    assignment_type: str | None = None,
    class_name: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> str:
    events = db["events"]
    event = events.find_one({"id": event_id, "owner_id": user_id})
    if not event:
        return "Event not found for this user."

    update_fields: Dict[str, object] = {}
    if title is not None:
        update_fields["title"] = title
    if due is not None:
        update_fields["due"] = due
    if location is not None:
        update_fields["location"] = location
    if description is not None:
        update_fields["description"] = description
    if assignment_type is not None:
        update_fields["assignment_type"] = assignment_type
    if class_name is not None:
        update_fields["class_name"] = class_name
    if status is not None:
        update_fields["status"] = status
    if priority is not None:
        update_fields["priority"] = priority

    if update_fields:
        update_fields["updated_at"] = datetime.utcnow()
        events.update_one({"_id": event["_id"]}, {"$set": update_fields})
        event.update(update_fields)

    return f"Updated event id={event_id} titled '{event.get('title', title or '')}'"


def db_delete_event(db: Database, user_id: int, *, event_id: int) -> str:
    events = db["events"]
    event = events.find_one({"id": event_id, "owner_id": user_id})
    if not event:
        return "Event not found for this user."
    events.delete_one({"_id": event["_id"]})
    return f"Deleted event id={event_id}"


def db_list_events(
    db: Database,
    user_id: int,
    *,
    date: datetime | None = None,
    title_query: str | None = None,
) -> List[models.Event]:
    query: Dict[str, object] = {"owner_id": user_id}
    if date is not None:
        query["due"] = {
            "$gte": date.replace(hour=0, minute=0, second=0, microsecond=0),
            "$lte": date.replace(hour=23, minute=59, second=59, microsecond=999999),
        }
    if title_query:
        query["title"] = {"$regex": title_query, "$options": "i"}

    events = db["events"].find(query).sort("due", 1)
    return [cast(models.Event, e) for e in events]


# --- Tool factory ---


def make_calendar_tools(
    db: Database,
    user_id: int,
    conversation_uuid: str | None = None,
    message_index: int | None = None,
):
    """Create per-user tools so the LLM can never access other users' events.

    Backed by MongoDB collections; conversation_uuid and message_index (if provided)
    are attached to tool call logs.
    """

    classes_col = db["classes"]
    events_col = db["events"]
    logs_col = db["tool_call_logs"]

    def _ensure_default_class_name() -> str:
        default_class = classes_col.find_one({"owner_id": user_id, "name": "Default"})
        if not default_class:
            class_id = get_next_id("classes")
            now = datetime.utcnow()
            cls: models.Class = {
                "_id": class_id,
                "id": class_id,
                "name": "Default",
                "owner_id": user_id,
                "created_at": now,
                "updated_at": now,
            }
            classes_col.insert_one(cls)
        return "Default"

    def _validate_class_name(name: str | None) -> str:
        if name is None:
            return _ensure_default_class_name()
        existing = classes_col.find_one({"owner_id": user_id, "name": name})
        if not existing:
            return "INVALID_CLASS"
        return name

    def _log_tool_call(tool_name: str, arguments: dict, result: str) -> None:
        log_id = get_next_id("tool_call_logs")
        now = datetime.utcnow()
        log: models.ToolCallLog = {
            "_id": log_id,
            "id": log_id,
            "tool_name": tool_name,
            "arguments": json.dumps(arguments, default=str),
            "result": result,
            "conversation_uuid": conversation_uuid,
            "message_index": message_index,
            "owner_id": user_id,
            "created_at": now,
        }
        logs_col.insert_one(log)

    def list_classes_tool() -> str:
        """List the user's classes by name. Use this before assigning a class_name to a task."""

        classes = list(classes_col.find({"owner_id": user_id}).sort("name", 1))
        if not classes:
            result = "No classes found for this user."
            _log_tool_call("list_classes", {}, result)
            return result
        result = "\n".join(f"id={c['id']} | {c['name']}" for c in classes)
        _log_tool_call("list_classes", {}, result)
        return result

    def create_class_tool(name: str) -> str:
        """Create a new class for this user (e.g. "Math", "Biology").

        Use this if you need a class that does not yet exist before creating a task.
        """

        existing = classes_col.find_one({"owner_id": user_id, "name": name})
        if existing:
            result = f"Class '{name}' already exists."
            _log_tool_call("create_class", {"name": name}, result)
            return result

        class_id = get_next_id("classes")
        now = datetime.utcnow()
        new_class: models.Class = {
            "_id": class_id,
            "id": class_id,
            "name": name,
            "owner_id": user_id,
            "created_at": now,
            "updated_at": now,
        }
        classes_col.insert_one(new_class)
        result = f"Created class id={class_id} name='{name}'"
        _log_tool_call("create_class", {"name": name}, result)
        return result

    def rename_class_tool(old_name: str, new_name: str) -> str:
        """Rename an existing class for this user and update all events that use it.

        Use this when the subject name changes (e.g. "Algebra" to "Math").
        """

        cls = classes_col.find_one({"owner_id": user_id, "name": old_name})
        if not cls:
            result = f"Class '{old_name}' does not exist for this user."
            _log_tool_call(
                "rename_class",
                {"old_name": old_name, "new_name": new_name},
                result,
            )
            return result

        existing_new = classes_col.find_one({"owner_id": user_id, "name": new_name})
        if existing_new:
            result = f"Class '{new_name}' already exists for this user."
            _log_tool_call(
                "rename_class",
                {"old_name": old_name, "new_name": new_name},
                result,
            )
            return result

        classes_col.update_one(
            {"_id": cls["_id"]},
            {"$set": {"name": new_name, "updated_at": datetime.utcnow()}},
        )
        events_col.update_many(
            {"owner_id": user_id, "class_name": old_name},
            {"$set": {"class_name": new_name}},
        )

        result = f"Renamed class '{old_name}' to '{new_name}' and updated related events."
        _log_tool_call(
            "rename_class",
            {"old_name": old_name, "new_name": new_name},
            result,
        )
        return result

    def delete_class_tool(name: str) -> str:
        """Delete a class for this user and all events that belong to that class.

        Use with care. This will permanently remove associated events.
        """

        cls = classes_col.find_one({"owner_id": user_id, "name": name})
        if not cls:
            result = f"Class '{name}' does not exist for this user."
            _log_tool_call("delete_class", {"name": name}, result)
            return result

        events_col.delete_many({"owner_id": user_id, "class_name": name})
        classes_col.delete_one({"_id": cls["_id"]})

        result = f"Deleted class '{name}' and all related events."
        _log_tool_call("delete_class", {"name": name}, result)
        return result

    def create_event_tool(
        title: str,
        due_iso: str | None = None,
        location: str | None = None,
        description: str | None = None,
        assignment_type: str | None = None,
        class_name: str | None = None,
        status: str | None = None,
        priority: str | None = None,
    ) -> str:
        """Create a new calendar task/event for the current user.

        Treat "due_iso" as the single due date & time (no separate start/end).
        Always set status (default to "pending") and a reasonable priority if not provided.
        The class_name MUST be one of the user's classes; if you need a new class,
        call create_class_tool first.
        """

        if not due_iso:
            msg = (
                "Missing due_iso for create_event. "
                "Ask the user for a specific due date and time first, "
                "then call this tool again with due_iso as an ISO-8601 string."
            )
            _log_tool_call(
                "create_event",
                {
                    "title": title,
                    "due_iso": due_iso,
                    "location": location,
                    "description": description,
                    "assignment_type": assignment_type,
                    "class_name": class_name,
                    "status": status,
                    "priority": priority,
                },
                msg,
            )
            return msg

        due = datetime.fromisoformat(due_iso)
        validated_class = _validate_class_name(class_name)
        if validated_class == "INVALID_CLASS":
            msg = (
                "Class name does not exist for this user. "
                "Call list_classes_tool to inspect existing classes and "
                "create_class_tool to create a new one before creating the event."
            )
            _log_tool_call(
                "create_event",
                {
                    "title": title,
                    "due_iso": due_iso,
                    "location": location,
                    "description": description,
                    "assignment_type": assignment_type,
                    "class_name": class_name,
                    "status": status,
                    "priority": priority,
                },
                msg,
            )
            return msg

        final_status = status or "pending"
        final_priority = priority or "normal"

        result = db_create_event(
            db,
            user_id,
            title=title,
            due=due,
            location=location,
            description=description,
            assignment_type=assignment_type,
            class_name=validated_class,
            status=final_status,
            priority=final_priority,
        )

        _log_tool_call(
            "create_event",
            {
                "title": title,
                "due_iso": due_iso,
                "location": location,
                "description": description,
                "assignment_type": assignment_type,
                "class_name": validated_class,
                "status": final_status,
                "priority": final_priority,
            },
            result,
        )
        return result

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
        else:
            due = None

        validated_class: str | None
        if class_name is not None:
            validated_class = _validate_class_name(class_name)
            if validated_class == "INVALID_CLASS":
                msg = (
                    "Class name does not exist for this user. "
                    "Call list_classes_tool to inspect existing classes and "
                    "create_class_tool to create a new one before updating the event."
                )
                _log_tool_call(
                    "update_event",
                    {
                        "event_id": event_id,
                        "title": title,
                        "due_iso": due_iso,
                        "location": location,
                        "description": description,
                        "assignment_type": assignment_type,
                        "class_name": class_name,
                        "status": status,
                        "priority": priority,
                    },
                    msg,
                )
                return msg
        else:
            validated_class = None

        result = db_update_event(
            db,
            user_id,
            event_id=event_id,
            title=title,
            due=due,
            location=location,
            description=description,
            assignment_type=assignment_type,
            class_name=validated_class,
            status=status,
            priority=priority,
        )

        _log_tool_call(
            "update_event",
            {
                "event_id": event_id,
                "title": title,
                "due_iso": due_iso,
                "location": location,
                "description": description,
                "assignment_type": assignment_type,
                "class_name": validated_class,
                "status": status,
                "priority": priority,
            },
            result,
        )
        return result

    def delete_event_tool(event_id: int) -> str:
        """Delete one calendar event for the current user.

        Use only after confirming which event to delete.
        """

        result = db_delete_event(db, user_id, event_id=event_id)
        _log_tool_call("delete_event", {"event_id": event_id}, result)
        return result

    def list_events_tool(
        date_iso: str | None = None,
        title_query: str | None = None,
    ) -> str:
        """List this user's calendar events.

        Use this to resolve ambiguity (e.g. "my meeting tomorrow") by
        listing candidate events before updating or deleting.
        """

        date: datetime | None = None
        events: List[models.Event]
        if date_iso:
            lower = date_iso.strip().lower()
            now = datetime.now()
            # Handle common relative phrases like "this week" and "next week".
            if lower in {"this week", "thisweek", "this_week"}:
                # Start of the current week (Monday 00:00)
                week_start = now - timedelta(days=now.weekday())
                week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
                week_end = week_start + timedelta(days=7)
                week_end = week_end.replace(hour=23, minute=59, second=59, microsecond=999999)
                query: Dict[str, object] = {
                    "owner_id": user_id,
                    "due": {"$gte": week_start, "$lte": week_end},
                }
                cursor = events_col.find(query).sort("due", 1)
                events = [cast(models.Event, e) for e in cursor]
            elif lower in {"next week", "nextweek", "next_week"}:
                # Start of next week (Monday 00:00 of the following week)
                this_week_start = now - timedelta(days=now.weekday())
                next_week_start = this_week_start + timedelta(days=7)
                next_week_start = next_week_start.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                next_week_end = next_week_start + timedelta(days=7)
                next_week_end = next_week_end.replace(
                    hour=23, minute=59, second=59, microsecond=999999
                )
                query = {
                    "owner_id": user_id,
                    "due": {"$gte": next_week_start, "$lte": next_week_end},
                }
                cursor = events_col.find(query).sort("due", 1)
                events = [cast(models.Event, e) for e in cursor]
            else:
                try:
                    if lower == "today":
                        date = now
                    elif lower == "tomorrow":
                        date = now + timedelta(days=1)
                    else:
                        # Try strict ISO format; if it fails, fall back to no date filter.
                        date = datetime.fromisoformat(date_iso)
                except ValueError:
                    # If the date can't be parsed, ignore the date filter and still list events.
                    date = None
                events = db_list_events(db, user_id, date=date, title_query=title_query)
        else:
            events = db_list_events(db, user_id, date=None, title_query=title_query)
        if not events:
            result = "No events found."
            _log_tool_call(
                "list_events",
                {"date_iso": date_iso, "title_query": title_query},
                result,
            )
            return result
        lines = []
        for e in events:
            lines.append(
                f"id={e['id']} | {e['title']} | {e['due'].isoformat()} | {e.get('location') or ''}"
            )
        result = "\n".join(lines)
        _log_tool_call(
            "list_events",
            {"date_iso": date_iso, "title_query": title_query},
            result,
        )
        return result

    list_classes = StructuredTool.from_function(
        func=list_classes_tool,
        name="list_classes",
        description=list_classes_tool.__doc__ or "List classes.",
    )
    create_class = StructuredTool.from_function(
        func=create_class_tool,
        name="create_class",
        description=create_class_tool.__doc__ or "Create a class.",
    )
    rename_class = StructuredTool.from_function(
        func=rename_class_tool,
        name="rename_class",
        description=rename_class_tool.__doc__ or "Rename a class.",
    )
    delete_class = StructuredTool.from_function(
        func=delete_class_tool,
        name="delete_class",
        description=delete_class_tool.__doc__ or "Delete a class.",
    )
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

    return [
        list_classes,
        create_class,
        rename_class,
        delete_class,
        create_event,
        update_event,
        delete_event,
        list_events,
    ]


# --- Agent factory ---


def get_calendar_agent(
    db: Database,
    user_id: int,
    conversation_uuid: str | None = None,
    message_index: int | None = None,
):
    llm = get_llm()

    tools = make_calendar_tools(
        db,
        user_id,
        conversation_uuid=conversation_uuid,
        message_index=message_index,
    )

    now_iso = datetime.now().isoformat()

    system_prompt = f"""Current datetime (ISO): {now_iso}.
You are a warm, calming, and helpful school-task assistant.
You manage tasks only for the currently authenticated user.

OVERALL STYLE
- Keep the tone friendly, relaxed, and reassuring—not formal or robotic.
- Be concise but supportive, like a helpful classmate who’s good at organizing.

GENERAL PRINCIPLES
- Default to TAKING ACTION instead of asking lots of questions.
- If you have enough information to create or update a task, just do it.
- Only ask short, focused follow‑ups when something important is unclear.
- You may suggest optional improvements AFTER creating a task (e.g. ask if they
  want to add a description or adjust priority), but never block creation on
  purely optional details.

TASK CREATION (VERY IMPORTANT)
- The user often just wants the task created quickly.
- When they describe a task ("I have a conference at 4pm on Wednesday" or
  "I have math homework due tomorrow"), you should:
  1) Infer a clear, natural title yourself.
  2) Infer the class when it is obvious from context.
  3) Infer a reasonable status and priority.
  4) Create the task with sensible defaults.

When to ASK for a title
- Only ask the user to name the task if:
  - The request is extremely vague (no clear action), OR
  - The user explicitly says they want to choose the title.
- Otherwise, YOU create the title.

TITLES (HUMAN-FRIENDLY SENTENCES)
- Always create titles in a natural "to-do" style, using the user’s words
  where helpful:
  - "Attend conference with Bank of America"
  - "Finish essay for English"
  - "Study for the biology quiz"
  - "Complete homework 5 for Math"
- Action first, class second.
- Friendly, simple, and readable.

CLASSES
- Every task needs one class.
- If the user clearly indicates the subject, ASSIGN THAT CLASS without asking:
  - "math homework" → use the Math class if it exists.
  - "biology quiz" → use the Biology class if it exists.
- Use list_classes only when you truly need to see what exists.
- Ask which class to use ONLY when it is genuinely ambiguous and cannot be
  safely inferred.
- You may create a new class if the user clearly wants one with that name, or
  explicitly agrees.

ASSIGNMENT TYPES
- Normalize to: Homework, Reading, Lab, Project, Paper, Quiz, Exam,
  Presentation, etc., when appropriate.

PRIORITY
- If the user explicitly gives a priority or urgency, respect it.
- Otherwise, infer a reasonable default:
  - High → exams, big projects, next-day deadlines, very urgent language.
  - Medium → normal assignments.
  - Low → long-term or casual work.
- Do NOT ask for priority before creating the task. Create the task first, then
  you may briefly ask if they want to adjust it.

DESCRIPTION
- Descriptions are optional. Do NOT block task creation on description.
- After creating a task, you may gently ask if they want to add helpful details
  (instructions, materials, etc.).

DUE DATE & TIME
- Tasks always need a due date.
- If the user gives a date and time, use it directly.
- If the user gives only a date, default to 11:59 PM unless context suggests
  otherwise.
- If the user gives only a relative reference ("next Wednesday", "tomorrow"),
  resolve it to an exact date using the current datetime in this prompt.
- Only ask a follow‑up about timing when the deadline is truly unclear.

UPDATING OR DELETING EXISTING TASKS
- Use list_events to show candidates when there is any ambiguity about which
  task to update or delete.

GOAL
- Make organizing tasks feel easy, fast, and low‑stress.
- Minimize back‑and‑forth questions.
- Help the user stay on track with clear titles, helpful details, and minimal
  friction, while staying warm and encouraging.
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent
