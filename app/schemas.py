from pydantic import BaseModel, EmailStr
from uuid import UUID
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class PaperOut(BaseModel):
    id: UUID
    title: str
    filename: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    question: str

class SourceOut(BaseModel):
    citation_number: int
    paper_title: str | None
    section_title: str | None
    page_number: int | None

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]

