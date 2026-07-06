import os
from datetime import datetime, timedelta  #yimedelta means a duration of time
from passlib.context import CryptContext  #thilibrary performs bcrypt hashing password verification
from jose import jwt, JWTError  #this library handles JWTs. jwt.encode(..) jwt.decode(...)

"""
passlib: python library for handling passwords securely. cna hash pwds, verify pwds,supports many hashing algos
context:module (python file) inside the passlib package.
CryptContext: is a class that has bcrypt hashing password verifications.

jose:another python library that creates verifies and decodes JWTs
"""



#CONFIGURATIONS:
"""
HS256 takes the header+payload+secret_key and produces a signature.
"""
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-this") #the value comes from os only known to the server.
ALGORITHM = "HS256"  
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


"""
CryptContext: creates a cryptcontext object.
we store it in pwd_context. The object now has two important methods:
pwd_context.hash(...)
pwd_context.verify(...)
bcrypt: hashing method used for password hashing


"""
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


"""
hashes the password using bcrypt.
Why do we create our own hash_password()
 function instead of calling pwd_context.hash()?
"""

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

"""
during registeration: 
reads the stored hash extracts salt from it hashed the entered password using the same salt.
compares hte results 
and returns
"""

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


"""
creating JWT token
data is the payload

"""
"""
this demonstrates software engineering principle: code reuse 
also called as DRY principle.
(Dont repeat yourself)

"""
def create_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + expires_delta
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_access_token(user_id: str) -> str:
    return create_token({"sub": user_id, "type": "access"}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

def create_refresh_token(user_id: str) -> str:
    return create_token({"sub": user_id, "type": "refresh"}, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))

"""
this below function used whenever a client sends a JWT back to your sever.
verifies the JWT signature using signature_key and chceks ehether hte tokoen has expired
extracts and returns the payload.
incase of error(like token not found etc) return none
if it is valid hten it returns its payload

"""
def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None