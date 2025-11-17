from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, cast
import json

from pymongo.database import Database

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool

from . import models
from .database import get_next_id
from .llm_providers import get_llm
from .indexing import search_syllabus_chunks


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
    syllabus_chunks_col = db["syllabus_chunks"]

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

    def search_syllabus_tool(
        class_name: str,
        query: str,
        top_k: int = 5,
    ) -> str:
        """Search this user's syllabus content for a given class using semantic similarity.

        Use this when you need details from the class syllabus (e.g. exam rules,
        grading breakdown, assignment policies, or schedule) instead of asking
        the user to paste them. Always specify the class_name so the correct
        syllabus is used.
        """

        cls = classes_col.find_one({"owner_id": user_id, "name": class_name})
        if not cls:
            result = f"Class '{class_name}' does not exist for this user."
            _log_tool_call(
                "search_syllabus",
                {"class_name": class_name, "query": query, "top_k": top_k},
                result,
            )
            return result

        # Find syllabus for this class (if any)
        syllabi_col = db["class_syllabi"]
        syllabus = syllabi_col.find_one(
            {"owner_id": user_id, "class_id": cls["id"]}
        )
        syllabus_id = syllabus["id"] if syllabus else None

        results = search_syllabus_chunks(
            db,
            owner_id=user_id,
            class_id=cls["id"],
            syllabus_id=syllabus_id,
            query=query,
            top_k=top_k,
        )

        if not results:
            result = "No matching syllabus chunks found."
            _log_tool_call(
                "search_syllabus",
                {"class_name": class_name, "query": query, "top_k": top_k},
                result,
            )
            return result

        lines: List[str] = []
        for doc in results:
            score = doc.get("score", 0.0)
            text = str(doc.get("text", "")).strip()
            lines.append(f"score={score:.3f} | {text}")

        result = "\n".join(lines)
        _log_tool_call(
            "search_syllabus",
            {"class_name": class_name, "query": query, "top_k": top_k},
            result,
        )
        return result

    def search_all_syllabi_tool(
        query: str,
        top_k: int = 5,
    ) -> str:
        """Search this user's syllabus content across ALL classes using semantic similarity.

        Use this when the user does not clearly specify a class name but asks
        about something that might be in any syllabus (e.g. "When is the unit 3
        exam about environmental science?"). The results will include which
        class each chunk came from so you can explain that back to the user.
        """

        results = search_syllabus_chunks(
            db,
            owner_id=user_id,
            class_id=None,
            syllabus_id=None,
            query=query,
            top_k=top_k,
        )

        if not results:
            result = "No matching syllabus chunks found across any classes."
            _log_tool_call(
                "search_all_syllabi",
                {"query": query, "top_k": top_k},
                result,
            )
            return result

        # Preload class names for pretty formatting
        class_ids = {doc.get("class_id") for doc in results if "class_id" in doc}
        class_map: Dict[int, str] = {}
        if class_ids:
            for cls_doc in classes_col.find(
                {"owner_id": user_id, "id": {"$in": list(class_ids)}}
            ):
                class_map[cls_doc["id"]] = cls_doc.get("name", str(cls_doc["id"]))

        lines: List[str] = []
        for doc in results:
            score = doc.get("score", 0.0)
            text = str(doc.get("text", "")).strip()
            cid = doc.get("class_id")
            cname = class_map.get(cid, f"class_id={cid}") if cid is not None else "unknown class"
            lines.append(f"class={cname} | score={score:.3f} | {text}")

        result = "\n".join(lines)
        _log_tool_call(
            "search_all_syllabi",
            {"query": query, "top_k": top_k},
            result,
        )
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
    search_syllabus = StructuredTool.from_function(
        func=search_syllabus_tool,
        name="search_syllabus",
        description=search_syllabus_tool.__doc__ or "Search syllabus content.",
    )
    search_all_syllabi = StructuredTool.from_function(
        func=search_all_syllabi_tool,
        name="search_all_syllabi",
        description=search_all_syllabi_tool.__doc__
        or "Search syllabus content across all classes.",
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
        search_syllabus,
        search_all_syllabi,
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

    system_prompt = f"""Current datetime (ISO): {now_iso}

You’re the calm, capable friend who always remembers the homework so the user doesn’t have to.  
You manage only the authenticated user’s school tasks.

VIBE
- Speak like a relaxed classmate who’s good at organizing: short, warm, never robotic.  
- Default to DOING the thing first; chat only when you must.

WHAT YOU DO
1. Hear the request.  
2. If you can make a sensible task, CREATE IT immediately.  
3. Then—only if it adds value—offer one friendly follow-up: “Want to add a note?” or “Should we move the due date earlier?”  
4. Never volunteer extras the user didn’t ask for (reminders, calendar invites, etc.).

TASK CREATION
- User says: “bio quiz friday” → you create:  
  Title: “Study for biology quiz” | Class: Biology | Due: Friday 11:59 PM | Priority: Medium  
- Make titles natural “to-do” sentences; keep the user’s own words when they help.  
- Infer class, priority, and status from context; only ask if it’s genuinely murky.  
- Due dates: absolute > relative; default time is 11:59 PM unless context says otherwise.

CLASSES
- If the subject is obvious, pick it.  
- If you need the list, call list_classes quietly—don’t bother the user.  
- Ask only when you truly can’t tell.

SYLLABUS LOOK-UPS
- When the user asks about exams, policies, weights, etc., search their syllabus first (search_syllabus or search_all_syllabi).  
- Tell them which class the answer came from so they know.

DESCRIPTIONS & PRIORITY
- Optional. Create the task, then gently ask: “Add any details?” or “Bump priority to High?”  
- Do not block creation for these.

UPDATES / DELETES
- If there’s any doubt which task, show a tiny numbered list and let them pick.

REMINDERS
- You can’t set alarms.  
- Only talk reminders when the user brings it up: explain briefly, suggest a time, remind them to set it on their own device.

GOAL
- Make task-adding feel like tossing a backpack onto the couch: effortless, done, no second thought.  
- One warm message, one tidy task, then shut up unless they want more.

EMOTION RADAR
- If the user shares any feeling word (“scared”, “stressed”, “overwhelmed”, “panicked”, “anxious”, “freaking out”, etc.):
  1. Pause the workflow.
  2. Reply with ONE short, warm sentence that names the feeling and offers calm.
     Examples:  
     “I hear you—tests can feel scary. You’ve got this.”  
     “Totally get the stress; one step at a time.”
  3. Then immediately create the task (no extra questions unless they’re useful).
- Never launch into advice or therapy; just acknowledge, then act.
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


def run_syllabus_agent(
    db: Database,
    *,
    user_id: int,
    class_name: str,
    syllabus_text: str,
) -> Optional[str]:
    if not syllabus_text.strip():
        return None

    llm = get_llm()
    now_iso = datetime.now().isoformat()

    system_prompt = (
        "You are an assistant that reads full course syllabi and extracts a clear, detailed "
        "schedule of assignments, exams, quizzes, and other important dates.\n\n"
        "You must respond ONLY with valid JSON using this exact schema (no comments, no extra keys):\n"
        "{\n"
        "  \"summary\": \"string\",\n"
        "  \"events\": [\n"
        "    {\n"
        "      \"title\": \"string\",\n"
        "      \"due_iso\": \"string\",\n"
        "      \"location\": \"string or null\",\n"
        "      \"description\": \"string or null\",\n"
        "      \"assignment_type\": \"string or null\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "SUMMARY REQUIREMENTS\n"
        "- 3-8 sentences.\n"
        "- Mention overall course structure, major graded components (projects, exams, quizzes),\n"
        "  and how the timeline is organized (e.g., weekly topics, midterm timing, final project).\n\n"
        "EVENT REQUIREMENTS\n"
        "- Include one event for EACH clearly dated major graded item: exams, midterms, finals,\n"
        "  projects, papers, presentations, or important deadlines.\n"
        "- Optional: include recurring weekly events only if they are explicitly scheduled.\n"
        "- Use ISO-8601 for due_iso. If the syllabus only gives a date (YYYY-MM-DD), set the time\n"
        "  to 23:59.\n"
        "- Examples of valid due_iso: \"2025-10-03T23:59:00\", \"2025-11-15T09:00:00-05:00\".\n\n"
        "IMPORTANT\n"
        "- Output MUST be valid JSON: no trailing commas, no comments, no explanation outside the JSON.\n"
        "- If you are unsure of an exact date, do not invent one; skip that event instead."
    )

    user_prompt = (
        "Current datetime (ISO): "
        + now_iso
        + "\nClass name: "
        + class_name
        + "\n\nFull syllabus text follows:\n\n"
        + syllabus_text
    )

    message = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    content: Any = getattr(message, "content", None)
    if not content:
        return None
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        content_text = "".join(parts).strip()
    elif isinstance(content, str):
        content_text = content.strip()
    else:
        content_text = str(content).strip()

    if not content_text:
        return None

    # Try to be robust if the model accidentally adds text around the JSON
    json_text = content_text
    first_brace = content_text.find("{")
    last_brace = content_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_text = content_text[first_brace : last_brace + 1]

    try:
        payload = json.loads(json_text)
    except Exception:
        return None

    summary = payload.get("summary")
    events_data = payload.get("events") or []

    if isinstance(events_data, list):
        for item in events_data:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            due_iso = item.get("due_iso")
            if not isinstance(title, str) or not title.strip():
                continue
            if not isinstance(due_iso, str) or not due_iso.strip():
                continue
            normalized_due = due_iso.strip()
            # Handle common variants like a bare date or Z-terminated timestamp
            if len(normalized_due) == 10 and normalized_due[4] == "-" and normalized_due[7] == "-":
                normalized_due = normalized_due + "T23:59:00"
            if normalized_due.endswith("Z"):
                normalized_due = normalized_due[:-1] + "+00:00"
            try:
                due_dt = datetime.fromisoformat(normalized_due)
            except Exception:
                continue
            location = item.get("location")
            if location is not None and not isinstance(location, str):
                location = None
            description = item.get("description")
            if description is not None and not isinstance(description, str):
                description = None
            assignment_type = item.get("assignment_type")
            if assignment_type is not None and not isinstance(assignment_type, str):
                assignment_type = None

            db_create_event(
                db,
                user_id,
                title=title.strip(),
                due=due_dt,
                location=location,
                description=description,
                assignment_type=assignment_type,
                class_name=class_name,
                status="pending",
                priority="normal",
            )

    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return None
