from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
"""
APIRouter: creates a group of related routes
Depends: FastAPI dependency injection.
HTTPException: returns API errors
status: HTTP status codes like 401, 404
"""

from database import get_db  #fastApi dependency engine.
from models import User
import schemas
import security

router = APIRouter(prefix="/auth", tags=["auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
"""
FastAPI helper that extracts the Bearer token from the Authorization header.
looks for authorization header, ensures it has bearer scheme and extracts teh token for the code

"""

"""
for all these functions authorizarion
pydantic validation schema
as well as database validation: db validation , code checks whether the data makes sense in the database.

"""
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = security.decode_token(token)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
    


@router.post("/register", response_model=schemas.UserOut)
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=user_in.email, hashed_password=security.hash_password(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    return schemas.Token(
        access_token=security.create_access_token(str(user.id)),
        refresh_token=security.create_refresh_token(str(user.id)),
    )




@router.post("/refresh", response_model=schemas.Token)
def refresh(refresh_token: str, db: Session = Depends(get_db)):
    payload = security.decode_token(refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return schemas.Token(
        access_token=security.create_access_token(str(user.id)),
        refresh_token=security.create_refresh_token(str(user.id)),
    )


@router.get("/me", response_model=schemas.UserOut)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
"""
There are two places where pydantic is used:
Request validation (Input)
examples: /register etc
Response validation(Output)

the entire declaration @router path+ the function that handles it is hte route
endpoint: http method+ path , it is what the client actually calls.


"""