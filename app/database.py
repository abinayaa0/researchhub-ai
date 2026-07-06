import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL")

#engine is responsible for connecting to the db.
engine = create_engine(DATABASE_URL) #this creates the connection manager
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) 
Base = declarative_base() #creates a base class for SQLAlchemy models. any class inherits from this base is treated as ORM model and mapped to a database table.
"""
Bind engine means that every session created by the sessionlocal() should use this engine.



"""

"""
FastAPI dependency injection"""
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()