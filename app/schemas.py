from datetime import datetime
from typing import Optional

from pydantic import BaseModel, constr


class UserCreate(BaseModel):
    username: constr(min_length=3, max_length=50)
    password: constr(min_length=6, max_length=128)


class UserLogin(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str

    class Config:
        orm_mode = True


class EventBase(BaseModel):
    title: str
    start: datetime
    end: datetime
    location: Optional[str] = None
    description: Optional[str] = None
    assignment_type: Optional[str] = None
    class_name: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None


class EventCreate(EventBase):
    pass


class EventUpdate(BaseModel):
    title: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    location: Optional[str] = None
    description: Optional[str] = None
    assignment_type: Optional[str] = None
    class_name: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None


class EventOut(EventBase):
    id: int

    class Config:
        orm_mode = True


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
