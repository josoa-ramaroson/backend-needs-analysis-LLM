from pydantic import BaseModel

class SignInRequest(BaseModel):
    uid: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    user: dict | None = None
    access_token: str | None = None
