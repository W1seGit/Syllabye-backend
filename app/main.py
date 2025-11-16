from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict
from datetime import datetime, date
from dotenv import load_dotenv

from .database import Base, engine, get_db
from . import models, schemas
from .auth import get_current_user, get_password_hash, authenticate_user, create_access_token
from .agents import get_calendar_agent


# Load environment variables from the project .env file (so OPENAI_API_KEY is available).
load_dotenv()

# Simple in-memory short-term conversation history per user (not persistent).
USER_HISTORY: Dict[int, List[dict]] = {}
MAX_HISTORY_TURNS = 10  # number of (user/assistant) messages to keep

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Calendar AI Backend", version="0.1.0")

# Allow browser-based frontends (like a local chat.html) to call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth endpoints ---


@app.post("/auth/register", response_model=schemas.UserOut)
def register_user(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.username == user_in.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
    user = models.User(
        username=user_in.username,
        hashed_password=get_password_hash(user_in.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    default_class = db.query(models.Class).filter(
        models.Class.owner_id == user.id,
        models.Class.name == "Default",
    ).first()
    if not default_class:
        default_class = models.Class(name="Default", owner_id=user.id)
        db.add(default_class)
        db.commit()

    return user


@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token({"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


# --- Chat endpoint using LangChain agent ---


@app.post("/chat", response_model=schemas.ChatResponse)
def chat(
    payload: schemas.ChatRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Chat endpoint that routes user messages through the LangChain calendar agent.

    The agent is scoped to the authenticated user and can add, update, delete,
    and list events only for that user.
    """

    db_history = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.owner_id == current_user.id)
        .order_by(models.ChatMessage.created_at.asc())
        .all()
    )

    history_messages = [
        {"role": m.role, "content": m.content} for m in db_history
    ]
    if len(history_messages) > MAX_HISTORY_TURNS:
        history_messages = history_messages[-MAX_HISTORY_TURNS:]

    messages_input = history_messages + [
        {"role": "user", "content": payload.message},
    ]

    agent = get_calendar_agent(db=db, user_id=current_user.id)

    result = agent.invoke({"messages": messages_input})

    messages = result.get("messages") or []
    if not messages:
        raise HTTPException(status_code=500, detail="Agent returned no messages")

    last_message = messages[-1]
    if hasattr(last_message, "content"):
        reply_text = last_message.content
    else:
        reply_text = last_message.get("content", "")

    user_msg = models.ChatMessage(
        role="user",
        content=payload.message,
        owner_id=current_user.id,
    )
    assistant_msg = models.ChatMessage(
        role="assistant",
        content=reply_text,
        owner_id=current_user.id,
    )
    db.add(user_msg)
    db.add(assistant_msg)
    db.commit()

    return schemas.ChatResponse(reply=reply_text)


# --- Class endpoints (per-user) ---


@app.get("/classes", response_model=List[schemas.ClassOut])
def list_classes(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    classes = (
        db.query(models.Class)
        .filter(models.Class.owner_id == current_user.id)
        .order_by(models.Class.name.asc())
        .all()
    )
    return [
        schemas.ClassOut(id=c.id, name=c.name)
        for c in classes
    ]


@app.post("/classes", response_model=schemas.ClassOut)
def create_class(
    class_in: schemas.ClassCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    existing = (
        db.query(models.Class)
        .filter(
            models.Class.owner_id == current_user.id,
            models.Class.name == class_in.name,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Class '{class_in.name}' already exists for this user.",
        )

    new_class = models.Class(name=class_in.name, owner_id=current_user.id)
    db.add(new_class)
    db.commit()
    db.refresh(new_class)
    return schemas.ClassOut(id=new_class.id, name=new_class.name)


# --- Event CRUD endpoints (per-user) ---


@app.post("/events", response_model=schemas.EventOut)
def create_event(
    event_in: schemas.EventCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    class_name = event_in.class_name
    if class_name is None:
        default_class = db.query(models.Class).filter(
            models.Class.owner_id == current_user.id,
            models.Class.name == "Default",
        ).first()
        if not default_class:
            default_class = models.Class(name="Default", owner_id=current_user.id)
            db.add(default_class)
            db.commit()
        class_name = "Default"
    else:
        existing_class = db.query(models.Class).filter(
            models.Class.owner_id == current_user.id,
            models.Class.name == class_name,
        ).first()
        if not existing_class:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Class '{class_name}' does not exist for this user.",
            )

    due = event_in.due
    status_value = event_in.status or "pending"

    event = models.Event(
        title=event_in.title,
        start=due,
        end=due,
        location=event_in.location,
        description=event_in.description,
        assignment_type=event_in.assignment_type,
        class_name=class_name,
        status=status_value,
        priority=event_in.priority,
        owner_id=current_user.id,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    return schemas.EventOut(
        id=event.id,
        title=event.title,
        due=event.start,
        location=event.location,
        description=event.description,
        assignment_type=event.assignment_type,
        class_name=event.class_name,
        status=event.status,
        priority=event.priority,
    )


@app.get("/events", response_model=List[schemas.EventOut])
def list_events(
    date_filter: date | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    q = db.query(models.Event).filter(models.Event.owner_id == current_user.id)
    if date_filter is not None:
        start_dt = datetime.combine(date_filter, datetime.min.time())
        end_dt = datetime.combine(date_filter, datetime.max.time())
        q = q.filter(models.Event.start.between(start_dt, end_dt))
    events = q.order_by(models.Event.start.asc()).all()
    return [
        schemas.EventOut(
            id=e.id,
            title=e.title,
            due=e.start,
            location=e.location,
            description=e.description,
            assignment_type=e.assignment_type,
            class_name=e.class_name,
            status=e.status,
            priority=e.priority,
        )
        for e in events
    ]


@app.get("/events/{event_id}", response_model=schemas.EventOut)
def get_event(
    event_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    event = (
        db.query(models.Event)
        .filter(models.Event.id == event_id, models.Event.owner_id == current_user.id)
        .first()
    )
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return schemas.EventOut(
        id=event.id,
        title=event.title,
        due=event.start,
        location=event.location,
        description=event.description,
        assignment_type=event.assignment_type,
        class_name=event.class_name,
        status=event.status,
        priority=event.priority,
    )


@app.patch("/events/{event_id}", response_model=schemas.EventOut)
def update_event(
    event_id: int,
    event_in: schemas.EventUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    event = (
        db.query(models.Event)
        .filter(models.Event.id == event_id, models.Event.owner_id == current_user.id)
        .first()
    )
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    if event_in.title is not None:
        event.title = event_in.title
    if event_in.due is not None:
        event.start = event_in.due
        event.end = event_in.due
    if event_in.location is not None:
        event.location = event_in.location
    if event_in.description is not None:
        event.description = event_in.description
    if event_in.assignment_type is not None:
        event.assignment_type = event_in.assignment_type
    if event_in.class_name is not None:
        existing_class = db.query(models.Class).filter(
            models.Class.owner_id == current_user.id,
            models.Class.name == event_in.class_name,
        ).first()
        if not existing_class:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Class '{event_in.class_name}' does not exist for this user.",
            )
        event.class_name = event_in.class_name
    if event_in.status is not None:
        event.status = event_in.status
    if event_in.priority is not None:
        event.priority = event_in.priority

    db.commit()
    db.refresh(event)
    return schemas.EventOut(
        id=event.id,
        title=event.title,
        due=event.start,
        location=event.location,
        description=event.description,
        assignment_type=event.assignment_type,
        class_name=event.class_name,
        status=event.status,
        priority=event.priority,
    )


@app.delete("/events/{event_id}")
def delete_event(
    event_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    event = (
        db.query(models.Event)
        .filter(models.Event.id == event_id, models.Event.owner_id == current_user.id)
        .first()
    )
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    db.delete(event)
    db.commit()
    return {"detail": "Event deleted"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
