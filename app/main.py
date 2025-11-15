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

    history = USER_HISTORY.get(current_user.id, [])

    messages_input = history + [
        {"role": "user", "content": payload.message},
    ]

    agent = get_calendar_agent(db=db, user_id=current_user.id)

    result = agent.invoke({"messages": messages_input})

    # The v1 agent returns a dict with "messages"; last one is the assistant reply
    messages = result.get("messages") or []
    if not messages:
        raise HTTPException(status_code=500, detail="Agent returned no messages")

    last_message = messages[-1]
    # last_message may be a BaseMessage object or dict depending on version
    if hasattr(last_message, "content"):
        reply_text = last_message.content
    else:
        reply_text = last_message.get("content", "")

    history.append({"role": "user", "content": payload.message})
    history.append({"role": "assistant", "content": reply_text})
    if len(history) > MAX_HISTORY_TURNS:
        history = history[-MAX_HISTORY_TURNS:]
    USER_HISTORY[current_user.id] = history

    return schemas.ChatResponse(reply=reply_text)


# --- Event CRUD endpoints (per-user) ---


@app.post("/events", response_model=schemas.EventOut)
def create_event(
    event_in: schemas.EventCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    event = models.Event(
        title=event_in.title,
        start=event_in.start,
        end=event_in.end,
        location=event_in.location,
        description=event_in.description,
        assignment_type=event_in.assignment_type,
        class_name=event_in.class_name,
        status=event_in.status,
        priority=event_in.priority,
        owner_id=current_user.id,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


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
    return events


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
    return event


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
    if event_in.start is not None:
        event.start = event_in.start
    if event_in.end is not None:
        event.end = event_in.end
    if event_in.location is not None:
        event.location = event_in.location
    if event_in.description is not None:
        event.description = event_in.description
    if event_in.assignment_type is not None:
        event.assignment_type = event_in.assignment_type
    if event_in.class_name is not None:
        event.class_name = event_in.class_name
    if event_in.status is not None:
        event.status = event_in.status
    if event_in.priority is not None:
        event.priority = event_in.priority

    db.commit()
    db.refresh(event)
    return event


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
