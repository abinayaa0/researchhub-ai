from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import inspect
import os
import uuid
import shutil
from fastapi import UploadFile,File,Form
from models import User,Paper

from database import Base, engine, get_db
from models import User
import schemas
import auth

app = FastAPI(title="ResearchHub AI") #creates a fastAPI application object.


#creates tables 
#metadata: info about your models/tables like a catalog
Base.metadata.create_all(bind=engine)

#reads/extracts the JWT from the authorization header and gives it to your function
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

#given a JWT who is the current user?
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = auth.decode_token(token)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", response_model=schemas.UserOut)
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=user_in.email, hashed_password=auth.hash_password(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    return schemas.Token(
        access_token=auth.create_access_token(str(user.id)),
        refresh_token=auth.create_refresh_token(str(user.id)),
    )


@app.post("/auth/refresh", response_model=schemas.Token)
def refresh(refresh_token: str, db: Session = Depends(get_db)):
    payload = auth.decode_token(refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return schemas.Token(
        access_token=auth.create_access_token(str(user.id)),
        refresh_token=auth.create_refresh_token(str(user.id)),
    )


@app.get("/auth/me", response_model=schemas.UserOut)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

STORAGE_DIR = "storage/papers"
os.makedirs(STORAGE_DIR, exist_ok=True)


@app.post("/papers/upload", response_model=schemas.PaperOut)
def upload_paper(
    title: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_id = str(uuid.uuid4())
    storage_path = os.path.join(STORAGE_DIR, f"{file_id}.pdf")

    with open(storage_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    paper = Paper(
        user_id=current_user.id,
        title=title,
        filename=file.filename,
        storage_path=storage_path,
        status="uploaded",
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)
    return paper


@app.get("/papers", response_model=list[schemas.PaperOut])
def list_papers(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return db.query(Paper).filter(Paper.user_id == current_user.id).all()