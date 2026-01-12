from fastapi import APIRouter
from services.user_service import UserService

router = APIRouter(prefix="/auth", tags=["Auth"])
user_service = UserService()


@router.post("/sign_in", response_model=ChatResponse)
async def sign_in():
    user_service.