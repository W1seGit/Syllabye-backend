from datetime import datetime
from typing import TypedDict, Optional
import uuid


class User(TypedDict):
    id: int
    username: str
    hashed_password: str


class Class(TypedDict):
    id: int
    name: str
    owner_id: int
    created_at: datetime
    updated_at: datetime


class ClassSyllabus(TypedDict, total=False):
    id: int
    text: Optional[str]
    pdf_path: Optional[str]
    class_id: int
    owner_id: int
    created_at: datetime
    updated_at: datetime


class ClassSyllabusImage(TypedDict):
    id: int
    file_path: str
    syllabus_id: int
    created_at: datetime


class Event(TypedDict, total=False):
    id: int
    title: str
    due: datetime
    location: Optional[str]
    description: Optional[str]
    assignment_type: Optional[str]
    class_name: Optional[str]
    status: Optional[str]
    priority: Optional[str]
    owner_id: int
    created_at: datetime
    updated_at: datetime


class Conversation(TypedDict):
    id: int
    uuid: str
    name: str
    owner_id: int
    created_at: datetime


class ChatMessage(TypedDict):
    id: int
    role: str
    content: str
    message_index: int
    conversation_id: int
    conversation_uuid: str
    owner_id: int
    created_at: datetime


class ToolCallLog(TypedDict, total=False):
    id: int
    tool_name: str
    arguments: Optional[str]
    result: Optional[str]
    conversation_uuid: Optional[str]
    message_index: Optional[int]
    owner_id: int
    created_at: datetime


def new_conversation_uuid() -> str:
    return str(uuid.uuid4())
