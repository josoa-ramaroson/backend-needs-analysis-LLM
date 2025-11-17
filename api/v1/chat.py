from fastapi import APIRouter
from models.chat import ChatRequest, ChatResponse
from services.chat_service import generate_answer

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest):
    response = generate_answer(payload.message)
    return ChatResponse(response=response)
