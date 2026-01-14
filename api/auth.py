from fastapi import APIRouter, HTTPException
from services.user_service import UserService
from models.auth import SignInRequest, AuthResponse
from utils.jwt import create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])
user_service = UserService()

@router.post("/sign_in", response_model=AuthResponse)
async def sign_in(payload: SignInRequest):
    if not user_service.authenticate_user(payload.dict()):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(payload.uid)
    user = user_service.get_user(payload.uid)

    return AuthResponse(
        success=True,
        user=user,
        access_token=token
    )

@router.post("/sign_up", response_model=AuthResponse, status_code=201)
async def sign_up(payload: SignUpRequest):
    """
    Register a new user and return a JWT.
    """
    try:
        user_service.register_user(payload.dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch created user
    user = user_service.get_user(payload.uid)

    token = create_access_token(user["uid"])

    return AuthResponse(
        success=True,
        user=user,
        access_token=token
    )