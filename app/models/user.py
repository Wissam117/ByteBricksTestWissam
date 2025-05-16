# app/models/user.py

from pydantic import BaseModel, EmailStr
from typing import Optional

class UserBase(BaseModel):
    email: EmailStr
    
class UserCreate(UserBase):
    password: str
    
class UserInDB(UserBase):
    id: str
    hashed_password: str
    
class User(UserBase):
    id: str
    
class Token(BaseModel):
    access_token: str
    token_type: str
    
class TokenPayload(BaseModel):
    sub: Optional[str] = None