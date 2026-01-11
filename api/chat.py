import json
import logging
import os

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from models.chat import ChatResponse
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from services.ollama_model_service import OllamaModelService
from services.rag_service import RAGService
from services.file_service import FileService, FileServiceError
from services.tiny_db_service import TinyDBService
logger = logging.getLogger(__name__)

# --- configuration / services (kept simple) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DB_FILE_NAME = "messages.json"
MESSAGE_COLLECTION_NAME = "bot_messages"
FILE_DIR = "static/docs"

file_service = FileService(FILE_DIR)

# instantiate simple model services (for a production app prefer startup events)
llama_service = OllamaModelService(OLLAMA_URL, "llama3.2")
fine_tuned_llama = OllamaModelService(OLLAMA_URL, "llama3.2-3b-finetuned")
rag_service = RAGService(fine_tuned_llama)

chat_db_service = TinyDBService(DB_FILE_NAME, MESSAGE_COLLECTION_NAME)
model_list_service = {
    "llama3.2-3B": llama_service,
    "finetuned-llama3.2-3B-Instruct": fine_tuned_llama,
    "RAG-pipeline": rag_service,
}

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.get("/history", response_model=List[ChatResponse])
async def get_history():
    """
    Returns the chat message history.
    - limit: maximum number of messages (default 100)
    """
    try:
        records = chat_db_service.query()
     
        return [ChatResponse(
            response=json.dumps(r.get("response"), ensure_ascii=False),
            model_id=r.get("model_id"),
            file_url=r.get("file_url"),
            uid=r.get("uid"),
            timestamp=r.get("timestamp")
            ) for r in records]

    except Exception as e:
        # Log for debugging
        logger.exception("Error while retrieving chat history: %s", e)
        # Return HTTP 500 with a clear message
        raise HTTPException(status_code=500, detail="Unable to retrieve chat history")

@router.post("/extract_requirements", response_model=ChatResponse)
async def extract_requirements(
    model_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Receive a file (PDF/DOCX/TXT), extract text, call the selected model service,
    and return the model JSON response (serialized as a string) together with model_id and file_url.

    Minimal and synchronous-safe: long/blocking model calls are executed in a threadpool.
    """
    uid = "2b0b7308d82647b0b4e"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # validate incoming file
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # read, extract and save file (async operation)
    try:
        text, saved_path = await file_service.read_and_save(file)
        
    except FileServiceError as exc:
        logger.exception("FileService error while reading/saving file")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error while reading/saving file")
        raise HTTPException(status_code=500, detail=f"Unexpected error reading file: {exc}")

    if not text:
        raise HTTPException(status_code=400, detail="Uploaded file contains no extractable text.")

    logger.info("Extracted text : ", text[:100])
    # validate model id
    if model_id not in model_list_service:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' not found.")

    model_service = model_list_service.get(model_id)
    if model_service is None:
        raise HTTPException(status_code=500, detail="Model service is not available.")

    # call the (potentially blocking) sync model method in a thread pool
    try:
        logger.info(f"Starting Request to model {model_id}...")
        print(f"Starting Request to model {model_id}...")
        # result = [
        # {
        #     'id': 'REQ-001',
        #     'type': 'non_functional',
        #     'title': 'Data encryption',
        #     'description': 'All stored patient data must be encrypted using AES-256.'
        # }
        # ]

        result =  model_service.getRequierementAsJson(text)
    except Exception as exc:
        logger.exception("Error while calling model service")
        raise HTTPException(status_code=500, detail=f"Model call failed: {exc}")

    # serialize result to JSON string (ensure unicode preserved)
    try:
        #response_str = "[{\"id\": \"REQ-001\", \"type\": \"non_functional\", \"title\": \"Data encryption\", \"description\": \"All stored patient data must be encrypted using AES-256.\"}]"
        response_str = json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.exception("Error serializing model result to JSON")
        raise HTTPException(status_code=500, detail=f"Result serialization failed: {exc}")
     
    # build a user-facing file_url (returning saved path here; adapt if you have a static server)
    file_url = str(saved_path)
    chat_db_service.store({
        "response": result,
        "model_id": model_id,
        "file_url": file_url,
        "uid": uid,
        "timestamp": timestamp
    })
    return ChatResponse(response=response_str, model_id=model_id, file_url=file_url, uid=uid, timestamp=timestamp)


@router.get("/models")
async def get_models():
    """
    Return a list of available model IDs.
    """
    return {"available_models": list(model_list_service.keys())}
