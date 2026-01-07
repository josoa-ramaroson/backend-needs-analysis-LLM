from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    model_id: str

class ChatResponse(BaseModel):
    response: str
    model_id: str
    file_url: str = ""
    uid: str
    timestamp: str
