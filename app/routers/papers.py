import os
import uuid
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from database import get_db
from models import User, Paper
import schemas
from routers.auth import get_current_user

router = APIRouter(prefix="/papers", tags=["papers"])

STORAGE_DIR = "storage/papers"
os.makedirs(STORAGE_DIR, exist_ok=True)


@router.post("/upload", response_model=schemas.PaperOut)
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


@router.get("", response_model=list[schemas.PaperOut])
def list_papers(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return db.query(Paper).filter(Paper.user_id == current_user.id).all()