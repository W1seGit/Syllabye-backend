from datetime import datetime, date
from typing import Dict, List, Optional, cast
import os

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pymongo.database import Database
from dotenv import load_dotenv

from .database import get_db, get_next_id
from . import models, schemas
from .indexing import index_syllabus_text, index_syllabus_pdf, index_syllabus_images
from .auth import (
    get_current_user,
    get_password_hash,
    authenticate_user,
    create_access_token,
    get_user_by_username,
    SECRET_KEY,
    ALGORITHM,
)
from .agents import get_calendar_agent, run_syllabus_agent
from jose import JWTError, jwt


load_dotenv()

app = FastAPI(title="Calendar AI Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-UUID"],
)


SYLLABUS_STORAGE_DIR = "syllabus_files"


def _ensure_syllabus_storage_dir() -> None:
    if not os.path.exists(SYLLABUS_STORAGE_DIR):
        os.makedirs(SYLLABUS_STORAGE_DIR, exist_ok=True)


def _get_user_id(current_user: models.User) -> int:
    return int(current_user["id"])


def _get_or_create_conversation(
    db: Database,
    user_id: int,
    conversation_uuid: Optional[str],
) -> models.Conversation:
    conversations = db["conversations"]
    if conversation_uuid:
        conv = conversations.find_one({"owner_id": user_id, "uuid": conversation_uuid})
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found for this user.",
            )
        return cast(models.Conversation, conv)

    conv_id = get_next_id("conversations")
    now = datetime.utcnow()
    conv_doc: models.Conversation = {
        "_id": conv_id,
        "id": conv_id,
        "uuid": models.new_conversation_uuid(),
        "name": "New Chat",
        "owner_id": user_id,
        "created_at": now,
    }
    conversations.insert_one(conv_doc)
    return conv_doc


def _get_chat_history(
    db: Database,
    user_id: int,
    conversation_id: int,
) -> List[models.ChatMessage]:
    messages_col = db["chat_messages"]
    cursor = messages_col.find(
        {"owner_id": user_id, "conversation_id": conversation_id}
    ).sort("message_index", 1)
    return [cast(models.ChatMessage, m) for m in cursor]


def _syllabus_to_response(db: Database, syllabus: models.ClassSyllabus) -> dict:
    images_col = db["class_syllabus_images"]
    images = list(images_col.find({"syllabus_id": syllabus["id"]}))
    return {
        "id": syllabus["id"],
        "class_id": syllabus["class_id"],
        "text": syllabus.get("text"),
        "pdf_path": syllabus.get("pdf_path"),
        "summary": syllabus.get("summary"),
        "images": [
            {"id": img["id"], "file_path": img["file_path"]}
            for img in images
        ],
    }


def _run_syllabus_ai(
    db: Database,
    *,
    user_id: int,
    class_id: int,
    syllabus: models.ClassSyllabus,
) -> None:
    classes = db["classes"]
    cls = classes.find_one({"id": class_id, "owner_id": user_id})
    if not cls:
        return

    syllabus_text: str | None = None
    raw_text = syllabus.get("text")
    if isinstance(raw_text, str) and raw_text.strip():
        syllabus_text = raw_text
    else:
        chunks_col = db["syllabus_chunks"]
        cursor = chunks_col.find(
            {
                "owner_id": user_id,
                "class_id": class_id,
                "syllabus_id": syllabus["id"],
            }
        ).sort("chunk_index", 1)
        parts: list[str] = []
        total_len = 0
        for doc in cursor:
            text = str(doc.get("text", ""))
            if not text.strip():
                continue
            parts.append(text)
            total_len += len(text)
            if total_len > 20000:
                break
        if parts:
            syllabus_text = "\n\n".join(parts)

    if not syllabus_text:
        return

    try:
        summary = run_syllabus_agent(
            db=db,
            user_id=user_id,
            class_name=str(cls.get("name", "")),
            syllabus_text=syllabus_text,
            syllabus_id=syllabus.get("id"),
        )
    except Exception:
        return

    if not summary:
        return

    syllabi = db["class_syllabi"]
    now = datetime.utcnow()
    syllabi.update_one(
        {"_id": syllabus["_id"]},
        {"$set": {"summary": summary, "updated_at": now}},
    )
    syllabus["summary"] = summary
    syllabus["updated_at"] = now


def _get_or_create_syllabus(
    db: Database,
    current_user: models.User,
    class_id: int,
) -> models.ClassSyllabus:
    user_id = _get_user_id(current_user)
    classes = db["classes"]
    cls = classes.find_one({"id": class_id, "owner_id": user_id})
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    syllabi = db["class_syllabi"]
    syllabus = syllabi.find_one({"class_id": class_id, "owner_id": user_id})
    if syllabus:
        return cast(models.ClassSyllabus, syllabus)

    syllabus_id = get_next_id("class_syllabi")
    now = datetime.utcnow()
    syllabus_doc: models.ClassSyllabus = {
        "_id": syllabus_id,
        "id": syllabus_id,
        "class_id": class_id,
        "owner_id": user_id,
        "text": None,
        "pdf_path": None,
        "created_at": now,
        "updated_at": now,
    }
    syllabi.insert_one(syllabus_doc)
    return syllabus_doc


# --- Auth endpoints ---


@app.post("/auth/register", response_model=schemas.UserOut)
def register_user(user_in: schemas.UserCreate, db: Database = Depends(get_db)):
    users = db["users"]
    existing = users.find_one({"username": user_in.username})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    user_id = get_next_id("users")
    user_doc: models.User = {
        "_id": user_id,
        "id": user_id,
        "username": user_in.username,
        "hashed_password": get_password_hash(user_in.password),
    }
    users.insert_one(user_doc)

    classes = db["classes"]
    default_class = classes.find_one({"owner_id": user_id, "name": "Default"})
    if not default_class:
        class_id = get_next_id("classes")
        now = datetime.utcnow()
        classes.insert_one(
            {
                "_id": class_id,
                "id": class_id,
                "name": "Default",
                "owner_id": user_id,
                "created_at": now,
                "updated_at": now,
            }
        )

    return {"id": user_doc["id"], "username": user_doc["username"]}


@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Database = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token({"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


# --- Chat endpoints ---


@app.post("/chat")
async def chat(
    payload: schemas.ChatRequest,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Streaming chat endpoint using agent streaming events."""

    user_id = _get_user_id(current_user)
    conversation = _get_or_create_conversation(db, user_id, payload.conversation_uuid)

    db_history = _get_chat_history(db, user_id, conversation["id"])
    history_messages = [{"role": m["role"], "content": m["content"]} for m in db_history]
    if len(history_messages) > 10:
        history_messages = history_messages[-10:]

    next_index = db_history[-1]["message_index"] + 1 if db_history else 1

    messages_input = history_messages + [
        {"role": "user", "content": payload.message},
    ]

    agent = get_calendar_agent(
        db=db,
        user_id=user_id,
        conversation_uuid=conversation["uuid"],
        message_index=next_index,
    )

    async def event_stream():
        full_reply = ""
        try:
            async for event in agent.astream_events({"messages": messages_input}):
                if event.get("event") != "on_chat_model_stream":
                    continue
                data = event.get("data") or {}
                chunk = data.get("chunk")
                if chunk is None:
                    continue
                # `chunk` is typically an AIMessageChunk / ChatGenerationChunk;
                # use its `content` attribute directly.
                text = getattr(chunk, "content", "") or ""
                if not text:
                    continue
                full_reply += text
                yield text
        finally:
            messages_col = db["chat_messages"]
            now = datetime.utcnow()
            user_msg_id = get_next_id("chat_messages")
            assistant_msg_id = get_next_id("chat_messages")
            messages_col.insert_many(
                [
                    {
                        "_id": user_msg_id,
                        "id": user_msg_id,
                        "role": "user",
                        "content": payload.message,
                        "message_index": next_index,
                        "conversation_id": conversation["id"],
                        "conversation_uuid": conversation["uuid"],
                        "owner_id": user_id,
                        "created_at": now,
                    },
                    {
                        "_id": assistant_msg_id,
                        "id": assistant_msg_id,
                        "role": "assistant",
                        "content": full_reply,
                        "message_index": next_index + 1,
                        "conversation_id": conversation["id"],
                        "conversation_uuid": conversation["uuid"],
                        "owner_id": user_id,
                        "created_at": now,
                    },
                ]
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={"X-Conversation-UUID": conversation["uuid"]},
    )


@app.websocket("/chat/ws")
async def chat_ws(websocket: WebSocket, db: Database = Depends(get_db)):
    await websocket.accept()

    token = websocket.query_params.get("token")
    conversation_uuid = websocket.query_params.get("conversation_uuid")

    if not token:
        await websocket.close(code=4401)
        return

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise JWTError()
    except JWTError:
        await websocket.close(code=4401)
        return

    current_user = get_user_by_username(db, username=username)
    if not current_user:
        await websocket.close(code=4401)
        return

    user_id = _get_user_id(current_user)

    try:
        msg_data = await websocket.receive_json()
    except WebSocketDisconnect:
        await websocket.close()
        return
    except Exception:
        await websocket.close(code=4400)
        return

    message_text = msg_data.get("message")
    incoming_conversation_uuid = msg_data.get("conversation_uuid") or conversation_uuid
    if not isinstance(message_text, str) or not message_text.strip():
        await websocket.close(code=4400)
        return

    conversation = _get_or_create_conversation(db, user_id, incoming_conversation_uuid)

    db_history = _get_chat_history(db, user_id, conversation["id"])
    history_messages = [{"role": m["role"], "content": m["content"]} for m in db_history]
    if len(history_messages) > 10:
        history_messages = history_messages[-10:]

    next_index = db_history[-1]["message_index"] + 1 if db_history else 1

    messages_input = history_messages + [
        {"role": "user", "content": message_text},
    ]

    agent = get_calendar_agent(
        db=db,
        user_id=user_id,
        conversation_uuid=conversation["uuid"],
        message_index=next_index,
    )

    await websocket.send_json({"type": "meta", "conversation_uuid": conversation["uuid"]})

    full_reply = ""
    try:
        async for event in agent.astream_events({"messages": messages_input}):
            if event.get("event") != "on_chat_model_stream":
                continue
            data = event.get("data") or {}
            chunk = data.get("chunk")
            if chunk is None:
                continue
            text = getattr(chunk, "content", "") or ""
            if not text:
                continue
            full_reply += text
            await websocket.send_text(text)
    except WebSocketDisconnect:
        pass
    finally:
        messages_col = db["chat_messages"]
        now = datetime.utcnow()
        user_msg_id = get_next_id("chat_messages")
        assistant_msg_id = get_next_id("chat_messages")
        messages_col.insert_many(
            [
                {
                    "_id": user_msg_id,
                    "id": user_msg_id,
                    "role": "user",
                    "content": message_text,
                    "message_index": next_index,
                    "conversation_id": conversation["id"],
                    "conversation_uuid": conversation["uuid"],
                    "owner_id": user_id,
                    "created_at": now,
                },
                {
                    "_id": assistant_msg_id,
                    "id": assistant_msg_id,
                    "role": "assistant",
                    "content": full_reply,
                    "message_index": next_index + 1,
                    "conversation_id": conversation["id"],
                    "conversation_uuid": conversation["uuid"],
                    "owner_id": user_id,
                    "created_at": now,
                },
            ]
        )

    try:
        await websocket.send_json({"type": "done"})
    except Exception:
        pass
    await websocket.close()


@app.get("/conversations", response_model=List[schemas.ConversationOut])
def list_conversations(
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    conversations = db["conversations"].find({"owner_id": user_id}).sort("created_at", -1)
    return [
        {"uuid": c["uuid"], "name": c.get("name", "New Chat")}
        for c in conversations
    ]


@app.get("/conversations/{conversation_uuid}/messages", response_model=List[schemas.ChatMessageOut])
def get_conversation_messages(
    conversation_uuid: str,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    conv = db["conversations"].find_one({"owner_id": user_id, "uuid": conversation_uuid})
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = db["chat_messages"].find(
        {"owner_id": user_id, "conversation_id": conv["id"]}
    ).sort("message_index", 1)
    return [
        {"role": m["role"], "content": m["content"], "message_index": m["message_index"]}
        for m in messages
    ]


@app.patch("/conversations/{conversation_uuid}", response_model=schemas.ConversationOut)
def update_conversation_name(
    conversation_uuid: str,
    name: str,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    conversations = db["conversations"]
    conv = conversations.find_one({"owner_id": user_id, "uuid": conversation_uuid})
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    new_name = name or "New Chat"
    conversations.update_one(
        {"_id": conv["_id"]},
        {"$set": {"name": new_name}},
    )
    return {"uuid": conv["uuid"], "name": new_name}


# --- Class endpoints ---


@app.get("/classes", response_model=List[schemas.ClassOut])
def list_classes(
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    classes = db["classes"].find({"owner_id": user_id}).sort("name", 1)
    return [
        {"id": c["id"], "name": c["name"]}
        for c in classes
    ]


@app.post("/classes", response_model=schemas.ClassOut)
def create_class(
    class_in: schemas.ClassCreate,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    classes = db["classes"]
    existing = classes.find_one({"owner_id": user_id, "name": class_in.name})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Class '{class_in.name}' already exists for this user.",
        )

    class_id = get_next_id("classes")
    now = datetime.utcnow()
    cls: models.Class = {
        "_id": class_id,
        "id": class_id,
        "name": class_in.name,
        "owner_id": user_id,
        "created_at": now,
        "updated_at": now,
    }
    classes.insert_one(cls)
    return {"id": cls["id"], "name": cls["name"]}


@app.patch("/classes/{class_id}", response_model=schemas.ClassOut)
def update_class(
    class_id: int,
    class_in: schemas.ClassCreate,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    classes = db["classes"]
    cls = classes.find_one({"id": class_id, "owner_id": user_id})
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    existing = classes.find_one(
        {"owner_id": user_id, "name": class_in.name, "id": {"$ne": class_id}}
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Class '{class_in.name}' already exists for this user.",
        )

    old_name = cls["name"]
    new_name = class_in.name
    now = datetime.utcnow()
    classes.update_one(
        {"_id": cls["_id"]},
        {"$set": {"name": new_name, "updated_at": now}},
    )

    events = db["events"]
    events.update_many(
        {"owner_id": user_id, "class_name": old_name},
        {"$set": {"class_name": new_name}},
    )

    return {"id": class_id, "name": new_name}


@app.delete("/classes/{class_id}")
def delete_class(
    class_id: int,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    classes = db["classes"]
    cls = classes.find_one({"id": class_id, "owner_id": user_id})
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    events = db["events"]
    events.delete_many({"owner_id": user_id, "class_name": cls["name"]})
    classes.delete_one({"_id": cls["_id"]})
    return {"detail": "Class and related events deleted"}


# --- Syllabus endpoints ---


@app.get("/classes/{class_id}/syllabus", response_model=schemas.ClassSyllabusOut)
def get_class_syllabus(
    class_id: int,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    syllabus = _get_or_create_syllabus(db, current_user, class_id)
    return _syllabus_to_response(db, syllabus)


@app.put("/classes/{class_id}/syllabus/text", response_model=schemas.ClassSyllabusOut)
def update_class_syllabus_text(
    class_id: int,
    payload: schemas.ClassSyllabusTextUpdate,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    syllabus = _get_or_create_syllabus(db, current_user, class_id)
    syllabi = db["class_syllabi"]
    now = datetime.utcnow()
    syllabi.update_one(
        {"_id": syllabus["_id"]},
        {"$set": {"text": payload.text, "updated_at": now}},
    )
    syllabus["text"] = payload.text
    syllabus["updated_at"] = now

    # Index updated text into vectors
    user_id = _get_user_id(current_user)
    index_syllabus_text(
        db,
        owner_id=user_id,
        class_id=class_id,
        syllabus_id=syllabus["id"],
        text=payload.text,
    )

    _run_syllabus_ai(
        db,
        user_id=user_id,
        class_id=class_id,
        syllabus=syllabus,
    )
    return _syllabus_to_response(db, syllabus)


@app.post("/classes/{class_id}/syllabus/pdf", response_model=schemas.ClassSyllabusOut)
def upload_class_syllabus_pdf(
    class_id: int,
    file: UploadFile = File(...),
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if file.content_type not in ["application/pdf"]:
        raise HTTPException(status_code=400, detail="Only PDF files are allowed for syllabus pdf")

    syllabus = _get_or_create_syllabus(db, current_user, class_id)

    _ensure_syllabus_storage_dir()
    user_dir = os.path.join(SYLLABUS_STORAGE_DIR, f"user_{_get_user_id(current_user)}")
    class_dir = os.path.join(user_dir, f"class_{class_id}")
    os.makedirs(class_dir, exist_ok=True)

    if syllabus.get("pdf_path"):
        old_path = syllabus["pdf_path"]
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

    syllabi = db["class_syllabi"]
    now = datetime.utcnow()
    syllabi.update_one(
        {"_id": syllabus["_id"]},
        {"$set": {"pdf_path": dest_path, "updated_at": now}},
    )
    syllabus["pdf_path"] = dest_path
    syllabus["updated_at"] = now

    # Index PDF content into vectors
    user_id = _get_user_id(current_user)
    index_syllabus_pdf(
        db,
        owner_id=user_id,
        class_id=class_id,
        syllabus_id=syllabus["id"],
        pdf_path=dest_path,
    )

    _run_syllabus_ai(
        db,
        user_id=user_id,
        class_id=class_id,
        syllabus=syllabus,
    )
    return _syllabus_to_response(db, syllabus)


@app.delete("/classes/{class_id}/syllabus/pdf", response_model=schemas.ClassSyllabusOut)
def delete_class_syllabus_pdf(
    class_id: int,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    syllabus = _get_or_create_syllabus(db, current_user, class_id)
    pdf_path = syllabus.get("pdf_path")
    if pdf_path:
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
        except OSError:
            pass
        syllabi = db["class_syllabi"]
        now = datetime.utcnow()
        syllabi.update_one(
            {"_id": syllabus["_id"]},
            {"$set": {"pdf_path": None, "updated_at": now}},
        )
        syllabus["pdf_path"] = None
        syllabus["updated_at"] = now
    return _syllabus_to_response(db, syllabus)


@app.post("/classes/{class_id}/syllabus/images", response_model=schemas.ClassSyllabusOut)
def upload_class_syllabus_images(
    class_id: int,
    files: List[UploadFile] = File(...),
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    for f in files:
        if f.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Only image files are allowed for syllabus images")

    syllabus = _get_or_create_syllabus(db, current_user, class_id)

    _ensure_syllabus_storage_dir()
    user_dir = os.path.join(SYLLABUS_STORAGE_DIR, f"user_{_get_user_id(current_user)}")
    class_dir = os.path.join(user_dir, f"class_{class_id}")
    images_dir = os.path.join(class_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    images_col = db["class_syllabus_images"]
    now = datetime.utcnow()

    image_paths: List[str] = []

    for f in files:
        ext = os.path.splitext(f.filename or "")[1] or ".png"
        filename = f"img_{datetime.utcnow().timestamp()}_{f.filename}"
        filename = filename.replace(" ", "_")
        dest_path = os.path.join(images_dir, filename)
        with open(dest_path, "wb") as out_file:
            content = f.file.read()
            out_file.write(content)

        image_paths.append(dest_path)

        img_id = get_next_id("class_syllabus_images")
        img: models.ClassSyllabusImage = {
            "_id": img_id,
            "id": img_id,
            "file_path": dest_path,
            "syllabus_id": syllabus["id"],
            "created_at": now,
        }
        images_col.insert_one(img)

    syllabi = db["class_syllabi"]
    syllabi.update_one(
        {"_id": syllabus["_id"]},
        {"$set": {"updated_at": datetime.utcnow()}},
    )

    if image_paths:
        user_id = _get_user_id(current_user)
        index_syllabus_images(
            db,
            owner_id=user_id,
            class_id=class_id,
            syllabus_id=syllabus["id"],
            image_paths=image_paths,
        )

        _run_syllabus_ai(
            db,
            user_id=user_id,
            class_id=class_id,
            syllabus=syllabus,
        )

    updated = syllabi.find_one({"_id": syllabus["_id"]})
    return _syllabus_to_response(db, cast(models.ClassSyllabus, updated))


@app.delete("/classes/{class_id}/syllabus/images/{image_id}", response_model=schemas.ClassSyllabusOut)
def delete_class_syllabus_image(
    class_id: int,
    image_id: int,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    syllabus = _get_or_create_syllabus(db, current_user, class_id)

    images_col = db["class_syllabus_images"]
    img = images_col.find_one({"id": image_id, "syllabus_id": syllabus["id"]})
    if not img:
        raise HTTPException(status_code=404, detail="Syllabus image not found")

    try:
        if os.path.exists(img["file_path"]):
            os.remove(img["file_path"])
    except OSError:
        pass

    images_col.delete_one({"_id": img["_id"]})

    syllabi = db["class_syllabi"]
    syllabi.update_one(
        {"_id": syllabus["_id"]},
        {"$set": {"updated_at": datetime.utcnow()}},
    )
    updated = syllabi.find_one({"_id": syllabus["_id"]})
    return _syllabus_to_response(db, cast(models.ClassSyllabus, updated))


# --- Event endpoints ---


@app.post("/events", response_model=schemas.EventOut)
def create_event(
    event_in: schemas.EventCreate,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    classes = db["classes"]
    events = db["events"]

    class_name = event_in.class_name
    if class_name is None:
        default_class = classes.find_one({"owner_id": user_id, "name": "Default"})
        if not default_class:
            class_id = get_next_id("classes")
            now = datetime.utcnow()
            default_class = {
                "_id": class_id,
                "id": class_id,
                "name": "Default",
                "owner_id": user_id,
                "created_at": now,
                "updated_at": now,
            }
            classes.insert_one(default_class)
        class_name = "Default"
    else:
        existing_class = classes.find_one({"owner_id": user_id, "name": class_name})
        if not existing_class:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Class '{class_name}' does not exist for this user.",
            )

    status_value = event_in.status or "pending"
    event_id = get_next_id("events")
    now = datetime.utcnow()
    event_doc: models.Event = {
        "_id": event_id,
        "id": event_id,
        "title": event_in.title,
        "due": event_in.due,
        "location": event_in.location,
        "description": event_in.description,
        "assignment_type": event_in.assignment_type,
        "class_name": class_name,
        "status": status_value,
        "priority": event_in.priority,
        "syllabus_id": None,
        "source": "user",
        "owner_id": user_id,
        "created_at": now,
        "updated_at": now,
    }
    events.insert_one(event_doc)

    return {
        "id": event_doc["id"],
        "title": event_doc["title"],
        "due": event_doc["due"],
        "location": event_doc.get("location"),
        "description": event_doc.get("description"),
        "assignment_type": event_doc.get("assignment_type"),
        "class_name": event_doc.get("class_name"),
        "status": event_doc.get("status"),
        "priority": event_doc.get("priority"),
        "syllabus_id": event_doc.get("syllabus_id"),
        "source": event_doc.get("source"),
    }


@app.get("/events", response_model=List[schemas.EventOut])
def list_events(
    date_filter: Optional[date] = None,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    events_col = db["events"]

    query: Dict = {"owner_id": user_id}
    if date_filter is not None:
        start_dt = datetime.combine(date_filter, datetime.min.time())
        end_dt = datetime.combine(date_filter, datetime.max.time())
        query["due"] = {"$gte": start_dt, "$lte": end_dt}

    events = events_col.find(query).sort("due", 1)
    return [
        {
            "id": e["id"],
            "title": e["title"],
            "due": e["due"],
            "location": e.get("location"),
            "description": e.get("description"),
            "assignment_type": e.get("assignment_type"),
            "class_name": e.get("class_name"),
            "status": e.get("status"),
            "priority": e.get("priority"),
            "syllabus_id": e.get("syllabus_id"),
            "source": e.get("source"),
        }
        for e in events
    ]


@app.get("/events/{event_id}", response_model=schemas.EventOut)
def get_event(
    event_id: int,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    events = db["events"]
    event = events.find_one({"id": event_id, "owner_id": user_id})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    return {
        "id": event["id"],
        "title": event["title"],
        "due": event["due"],
        "location": event.get("location"),
        "description": event.get("description"),
        "assignment_type": event.get("assignment_type"),
        "class_name": event.get("class_name"),
        "status": event.get("status"),
        "priority": event.get("priority"),
        "syllabus_id": event.get("syllabus_id"),
        "source": event.get("source"),
    }


@app.patch("/events/{event_id}", response_model=schemas.EventOut)
def update_event(
    event_id: int,
    event_in: schemas.EventUpdate,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    events = db["events"]
    classes = db["classes"]

    event = events.find_one({"id": event_id, "owner_id": user_id})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    update_fields: Dict = {}
    if event_in.title is not None:
        update_fields["title"] = event_in.title
    if event_in.due is not None:
        update_fields["due"] = event_in.due
    if event_in.location is not None:
        update_fields["location"] = event_in.location
    if event_in.description is not None:
        update_fields["description"] = event_in.description
    if event_in.assignment_type is not None:
        update_fields["assignment_type"] = event_in.assignment_type
    if event_in.class_name is not None:
        existing_class = classes.find_one(
            {"owner_id": user_id, "name": event_in.class_name}
        )
        if not existing_class:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Class '{event_in.class_name}' does not exist for this user.",
            )
        update_fields["class_name"] = event_in.class_name
    if event_in.status is not None:
        update_fields["status"] = event_in.status
    if event_in.priority is not None:
        update_fields["priority"] = event_in.priority

    if update_fields:
        update_fields["updated_at"] = datetime.utcnow()
        events.update_one(
            {"_id": event["_id"]},
            {"$set": update_fields},
        )
        event.update(update_fields)

    return {
        "id": event["id"],
        "title": event["title"],
        "due": event["due"],
        "location": event.get("location"),
        "description": event.get("description"),
        "assignment_type": event.get("assignment_type"),
        "class_name": event.get("class_name"),
        "status": event.get("status"),
        "priority": event.get("priority"),
    }


@app.delete("/events/{event_id}")
def delete_event(
    event_id: int,
    db: Database = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    events = db["events"]
    event = events.find_one({"id": event_id, "owner_id": user_id})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    events.delete_one({"_id": event["_id"]})
    return {"detail": "Event deleted"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
