from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    events = relationship("Event", back_populates="owner", cascade="all, delete-orphan")
    classes = relationship("Class", back_populates="owner", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="owner", cascade="all, delete-orphan")
    tool_calls = relationship("ToolCallLog", back_populates="owner", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="owner", cascade="all, delete-orphan")
    syllabi = relationship("ClassSyllabus", back_populates="owner", cascade="all, delete-orphan")


class Class(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)

    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    owner = relationship("User", back_populates="classes")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    syllabus = relationship("ClassSyllabus", back_populates="class_", uselist=False, cascade="all, delete-orphan")


class ClassSyllabus(Base):
    __tablename__ = "class_syllabi"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=True)
    pdf_path = Column(String, nullable=True)

    class_id = Column(Integer, ForeignKey("classes.id", ondelete="CASCADE"), nullable=False, index=True)
    class_ = relationship("Class", back_populates="syllabus")

    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    owner = relationship("User", back_populates="syllabi")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    images = relationship("ClassSyllabusImage", back_populates="syllabus", cascade="all, delete-orphan")


class ClassSyllabusImage(Base):
    __tablename__ = "class_syllabus_images"

    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String, nullable=False)

    syllabus_id = Column(Integer, ForeignKey("class_syllabi.id", ondelete="CASCADE"), nullable=False, index=True)
    syllabus = relationship("ClassSyllabus", back_populates="images")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    due = Column(DateTime, nullable=False)
    location = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    assignment_type = Column(String, nullable=True)
    class_name = Column(String, nullable=True)
    status = Column(String, nullable=True)
    priority = Column(String, nullable=True)

    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    owner = relationship("User", back_populates="events")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()), nullable=False)
    name = Column(String, nullable=False, default="New Chat")

    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    owner = relationship("User", back_populates="conversations")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    message_index = Column(Integer, nullable=False)

    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    conversation = relationship("Conversation", back_populates="messages")
    conversation_uuid = Column(String, nullable=False, index=True)

    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    owner = relationship("User", back_populates="chat_messages")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ToolCallLog(Base):
    __tablename__ = "tool_call_logs"

    id = Column(Integer, primary_key=True, index=True)
    tool_name = Column(String, nullable=False)
    arguments = Column(Text, nullable=True)
    result = Column(Text, nullable=True)

    conversation_uuid = Column(String, nullable=True, index=True)
    message_index = Column(Integer, nullable=True)

    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    owner = relationship("User", back_populates="tool_calls")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
