from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from models import User, Paper
import schemas
from routers.auth import get_current_user
import rag_pipeline

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=schemas.QueryResponse)
def query_papers(
    request: schemas.QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_papers = db.query(Paper).filter(Paper.user_id == current_user.id).all()
    paper_ids = [p.qdrant_paper_id for p in user_papers if p.qdrant_paper_id]

    result = rag_pipeline.answer_query(request.question, paper_ids)
    return result