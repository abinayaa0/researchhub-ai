import uuid
"""
UUID: Universally Unique Identifier unique id. generated to be unique.
used for primary key.
"""
from sqlalchemy import Column, String, Boolean, DateTime,ForeignKey
from sqlalchemy.dialects.postgresql import UUID #postgre's UUID column type.
from sqlalchemy.sql import func #func lets SQLAlchemy call sql functions
from database import Base #this is declarative base


class User(Base): # user sqlalchemy model creation
    __tablename__ = "users" 

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4) #
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Paper(Base):
    __tablename__ = "papers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    status = Column(String, default="uploaded")
    qdrant_paper_id = Column(String, nullable=True, index=True)  # NEW — links to Qdrant's paper_id payload
    created_at = Column(DateTime(timezone=True), server_default=func.now())

"""
server_default:
the database generates the value no the client

"""
