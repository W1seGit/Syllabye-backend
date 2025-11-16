from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict
from datetime import datetime, date
from dotenv import load_dotenv
import os


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
    expose_headers=["X-Conversation-UUID"],
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


@app.post("/chat")
def chat(
    payload: schemas.ChatRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Streaming chat endpoint using the same calendar agent.

    Streams the assistant reply incrementally while still persisting
    the final messages and conversation state when complete.
    """

    conversation: models.Conversation | None = None
    if payload.conversation_uuid:
        conversation = (
            db.query(models.Conversation)
            .filter(
                models.Conversation.owner_id == current_user.id,
                models.Conversation.uuid == payload.conversation_uuid,
            )
            .first()
        )
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found for this user.",
            )
    else:
        conversation = models.Conversation(owner_id=current_user.id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    db_history = (
        db.query(models.ChatMessage)
        .filter(
            models.ChatMessage.owner_id == current_user.id,
            models.ChatMessage.conversation_id == conversation.id,
        )
        .order_by(models.ChatMessage.message_index.asc())
        .all()
    )

    history_messages = [
        {"role": m.role, "content": m.content} for m in db_history
    ]
    if len(history_messages) > MAX_HISTORY_TURNS:
        history_messages = history_messages[-MAX_HISTORY_TURNS:]

    next_index = 1
    if db_history:
        next_index = db_history[-1].message_index + 1

    messages_input = history_messages + [
        {"role": "user", "content": payload.message},
    ]

    agent = get_calendar_agent(
        db=db,
        user_id=current_user.id,
        conversation_uuid=conversation.uuid,
        message_index=next_index,
    )

    # Call the agent once to get the full reply, then stream it to the client.
    result = agent.invoke({"messages": messages_input})
    messages = result.get("messages") or []
    if not messages:
        raise HTTPException(status_code=500, detail="Agent returned no messages")

    last_message = messages[-1]
    if hasattr(last_message, "content"):
        reply_text = last_message.content
    else:
        reply_text = last_message.get("content", "")

    def event_stream():
        try:
            # Stream the reply in small chunks so the UI can update incrementally.
            chunk_size = 128
            for i in range(0, len(reply_text), chunk_size):
                yield reply_text[i : i + chunk_size]
        finally:
            # Persist messages once the stream ends (successfully or not)
            user_msg = models.ChatMessage(
                role="user",
                content=payload.message,
                message_index=next_index,
                conversation_id=conversation.id,
                conversation_uuid=conversation.uuid,
                owner_id=current_user.id,
            )
            assistant_msg = models.ChatMessage(
                role="assistant",
                content=reply_text,
                message_index=next_index + 1,
                conversation_id=conversation.id,
                conversation_uuid=conversation.uuid,
                owner_id=current_user.id,
            )
            db.add(user_msg)
            db.add(assistant_msg)
            db.commit()

    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={"X-Conversation-UUID": conversation.uuid},
    )


@app.post("/chat/experimental-stream")
async def chat_experimental_stream(
    payload: schemas.ChatRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Experimental true streaming chat endpoint using agent streaming events.

    This keeps all agent tooling and DB logging but relies on LangChain's
    streaming event API, which may change in future versions.
    """

    conversation: models.Conversation | None = None
    if payload.conversation_uuid:
        conversation = (
            db.query(models.Conversation)
            .filter(
                models.Conversation.owner_id == current_user.id,
                models.Conversation.uuid == payload.conversation_uuid,
            )
            .first()
        )
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found for this user.",
            )
    else:
        conversation = models.Conversation(owner_id=current_user.id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    db_history = (
        db.query(models.ChatMessage)
        .filter(
            models.ChatMessage.owner_id == current_user.id,
            models.ChatMessage.conversation_id == conversation.id,
        )
        .order_by(models.ChatMessage.message_index.asc())
        .all()
    )

    history_messages = [
        {"role": m.role, "content": m.content} for m in db_history
    ]
    if len(history_messages) > MAX_HISTORY_TURNS:
        history_messages = history_messages[-MAX_HISTORY_TURNS:]

    next_index = 1
    if db_history:
        next_index = db_history[-1].message_index + 1

    messages_input = history_messages + [
        {"role": "user", "content": payload.message},
    ]

    agent = get_calendar_agent(
        db=db,
        user_id=current_user.id,
        conversation_uuid=conversation.uuid,
        message_index=next_index,
    )

    async def event_stream():
        full_reply = ""
        try:
            # NOTE: This uses LangChain's experimental streaming event API.
            async for event in agent.astream_events({"messages": messages_input}):
                event_type = event.get("event")
                if event_type != "on_chat_model_stream":
                    continue
                data = event.get("data") or {}
                chunk = data.get("chunk")
                if chunk is None:
                    continue
                if hasattr(chunk, "content"):
                    text = chunk.content
                else:
                    text = chunk.get("content", "")
                if not text:
                    continue
                full_reply_local = full_reply + text
                # send delta straight to client
                yield text
                full_reply = full_reply_local
        finally:
            user_msg = models.ChatMessage(
                role="user",
                content=payload.message,
                message_index=next_index,
                conversation_id=conversation.id,
                conversation_uuid=conversation.uuid,
                owner_id=current_user.id,
            )
            assistant_msg = models.ChatMessage(
                role="assistant",
                content=full_reply,
                message_index=next_index + 1,
                conversation_id=conversation.id,
                conversation_uuid=conversation.uuid,
                owner_id=current_user.id,
            )
            db.add(user_msg)
            db.add(assistant_msg)
            db.commit()

    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={"X-Conversation-UUID": conversation.uuid},
    )


@app.get("/conversations", response_model=List[schemas.ConversationOut])
def list_conversations(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    conversations = (
        db.query(models.Conversation)
        .filter(models.Conversation.owner_id == current_user.id)
        .order_by(models.Conversation.created_at.desc())
        .all()
    )
    return [
        schemas.ConversationOut(uuid=c.uuid, name=c.name)
        for c in conversations
    ]


@app.get("/conversations/{conversation_uuid}/messages", response_model=List[schemas.ChatMessageOut])
def get_conversation_messages(
    conversation_uuid: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    conversation = (
        db.query(models.Conversation)
        .filter(
            models.Conversation.owner_id == current_user.id,
            models.Conversation.uuid == conversation_uuid,
        )
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        db.query(models.ChatMessage)
        .filter(
            models.ChatMessage.owner_id == current_user.id,
            models.ChatMessage.conversation_id == conversation.id,
        )
        .order_by(models.ChatMessage.message_index.asc())
        .all()
    )
    return [
        schemas.ChatMessageOut(
            role=m.role,
            content=m.content,
            message_index=m.message_index,
        )
        for m in messages
    ]


@app.patch("/conversations/{conversation_uuid}", response_model=schemas.ConversationOut)
def update_conversation_name(
    conversation_uuid: str,
    name: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    conversation = (
        db.query(models.Conversation)
        .filter(
            models.Conversation.owner_id == current_user.id,
            models.Conversation.uuid == conversation_uuid,
        )
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation.name = name or "New Chat"
    db.commit()
    db.refresh(conversation)

    return schemas.ConversationOut(uuid=conversation.uuid, name=conversation.name)


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


@app.patch("/classes/{class_id}", response_model=schemas.ClassOut)
def update_class(
    class_id: int,
    class_in: schemas.ClassCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    cls = (
        db.query(models.Class)
        .filter(
            models.Class.id == class_id,
            models.Class.owner_id == current_user.id,
        )
        .first()
    )
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    existing = (
        db.query(models.Class)
        .filter(
            models.Class.owner_id == current_user.id,
            models.Class.name == class_in.name,
            models.Class.id != class_id,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Class '{class_in.name}' already exists for this user.",
        )

    old_name = cls.name
    cls.name = class_in.name
    db.commit()

    db.query(models.Event).filter(
        models.Event.owner_id == current_user.id,
        models.Event.class_name == old_name,
    ).update({models.Event.class_name: class_in.name})
    db.commit()

    return schemas.ClassOut(id=cls.id, name=cls.name)


@app.delete("/classes/{class_id}")
def delete_class(
    class_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    cls = (
        db.query(models.Class)
        .filter(
            models.Class.id == class_id,
            models.Class.owner_id == current_user.id,
        )
        .first()
    )
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    db.query(models.Event).filter(
        models.Event.owner_id == current_user.id,
        models.Event.class_name == cls.name,
    ).delete(synchronize_session=False)
    db.delete(cls)
    db.commit()

    return {"detail": "Class and related events deleted"}


SYLLABUS_STORAGE_DIR = "syllabus_files"


def _ensure_syllabus_storage_dir() -> None:
    if not os.path.exists(SYLLABUS_STORAGE_DIR):
        os.makedirs(SYLLABUS_STORAGE_DIR, exist_ok=True)


def _get_or_create_syllabus(db: Session, current_user: models.User, class_id: int) -> models.ClassSyllabus:
    cls = (
        db.query(models.Class)
        .filter(models.Class.id == class_id, models.Class.owner_id == current_user.id)
        .first()
    )
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    syllabus = (
        db.query(models.ClassSyllabus)
        .filter(
            models.ClassSyllabus.class_id == class_id,
            models.ClassSyllabus.owner_id == current_user.id,
        )
        .first()
    )
    if not syllabus:
        syllabus = models.ClassSyllabus(
            class_id=class_id,
            owner_id=current_user.id,
            text=None,
            pdf_path=None,
        )
        db.add(syllabus)
        db.commit()
        db.refresh(syllabus)
    return syllabus


@app.get("/classes/{class_id}/syllabus", response_model=schemas.ClassSyllabusOut)
def get_class_syllabus(
    class_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    syllabus = _get_or_create_syllabus(db, current_user, class_id)
    return syllabus


@app.put("/classes/{class_id}/syllabus/text", response_model=schemas.ClassSyllabusOut)
def update_class_syllabus_text(
    class_id: int,
    payload: schemas.ClassSyllabusTextUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    syllabus = _get_or_create_syllabus(db, current_user, class_id)
    syllabus.text = payload.text
    db.commit()
    db.refresh(syllabus)
    return syllabus


@app.post("/classes/{class_id}/syllabus/pdf", response_model=schemas.ClassSyllabusOut)
def upload_class_syllabus_pdf(
    class_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if file.content_type not in ["application/pdf"]:
        raise HTTPException(status_code=400, detail="Only PDF files are allowed for syllabus pdf")

    syllabus = _get_or_create_syllabus(db, current_user, class_id)

    _ensure_syllabus_storage_dir()
    user_dir = os.path.join(SYLLABUS_STORAGE_DIR, f"user_{current_user.id}")
    class_dir = os.path.join(user_dir, f"class_{class_id}")
    os.makedirs(class_dir, exist_ok=True)

    if syllabus.pdf_path:
        old_path = syllabus.pdf_path
        try:
            if os.path.exists(old_path):
                os.remove(old_path)
        except OSError:
            pass

    filename = f"syllabus_{class_id}.pdf"
    dest_path = os.path.join(class_dir, filename)

    with open(dest_path, "wb") as out_file:
        content = file.file.read()
        out_file.write(content)

    syllabus.pdf_path = dest_path
    db.commit()
    db.refresh(syllabus)
    return syllabus


@app.delete("/classes/{class_id}/syllabus/pdf", response_model=schemas.ClassSyllabusOut)
def delete_class_syllabus_pdf(
    class_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    syllabus = _get_or_create_syllabus(db, current_user, class_id)
    if syllabus.pdf_path:
        try:
            if os.path.exists(syllabus.pdf_path):
                os.remove(syllabus.pdf_path)
        except OSError:
            pass
        syllabus.pdf_path = None
        db.commit()
        db.refresh(syllabus)
    return syllabus


@app.post("/classes/{class_id}/syllabus/images", response_model=schemas.ClassSyllabusOut)
def upload_class_syllabus_images(
    class_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    for f in files:
        if f.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Only image files are allowed for syllabus images")

    syllabus = _get_or_create_syllabus(db, current_user, class_id)

    _ensure_syllabus_storage_dir()
    user_dir = os.path.join(SYLLABUS_STORAGE_DIR, f"user_{current_user.id}")
    class_dir = os.path.join(user_dir, f"class_{class_id}")
    images_dir = os.path.join(class_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for f in files:
        ext = os.path.splitext(f.filename or "")[1] or ".png"
        filename = f"img_{datetime.utcnow().timestamp()}_{f.filename}"
        filename = filename.replace(" ", "_")
        dest_path = os.path.join(images_dir, filename)
        with open(dest_path, "wb") as out_file:
            content = f.file.read()
            out_file.write(content)

        img = models.ClassSyllabusImage(
            file_path=dest_path,
            syllabus_id=syllabus.id,
        )
        db.add(img)

    db.commit()
    db.refresh(syllabus)
    return syllabus


@app.delete("/classes/{class_id}/syllabus/images/{image_id}", response_model=schemas.ClassSyllabusOut)
def delete_class_syllabus_image(
    class_id: int,
    image_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    syllabus = _get_or_create_syllabus(db, current_user, class_id)

    img = (
        db.query(models.ClassSyllabusImage)
        .join(models.ClassSyllabus)
        .filter(
            models.ClassSyllabus.id == syllabus.id,
            models.ClassSyllabus.owner_id == current_user.id,
            models.ClassSyllabusImage.id == image_id,
        )
        .first()
    )
    if not img:
        raise HTTPException(status_code=404, detail="Syllabus image not found")

    try:
        if os.path.exists(img.file_path):
            os.remove(img.file_path)
    except OSError:
        pass

    db.delete(img)
    db.commit()
    db.refresh(syllabus)
    return syllabus


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
        due=due,
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
        due=event.due,
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
        q = q.filter(models.Event.due.between(start_dt, end_dt))
    events = q.order_by(models.Event.due.asc()).all()
    return [
        schemas.EventOut(
            id=e.id,
            title=e.title,
            due=e.due,
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
        due=event.due,
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
        event.due = event_in.due
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
        due=event.due,
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
