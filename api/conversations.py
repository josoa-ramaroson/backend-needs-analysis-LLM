from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from services.conversations_service import ConversationsService
from dependencies.auth import get_current_user

router = APIRouter(prefix="/conversations", tags=["Conversations"])
conversation_service = ConversationsService()

@router.post("/", status_code=201)
def create_conversation(
    conversation_name: str,
    current_user: dict = Depends(get_current_user)
):
    cid = conversation_service.create_conversation(
        uid=current_user["uid"],
        conversation_name=conversation_name
    )

    return {
        "success": True,
        "cid": cid
    }

@router.get("/", response_model=List[dict])
def get_my_conversations(current_user: dict = Depends(get_current_user)):
    return conversation_service.get_conversations(
        filters={"uid": current_user["uid"]}
    )

@router.put("/{cid}")
def update_conversation(
    cid: str,
    updates: dict,
    current_user: dict = Depends(get_current_user)
):
    conversations = conversation_service.get_conversations(
        {"cid": cid, "uid": current_user["uid"]}
    )

    if not conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    updated = conversation_service.update_conversation(cid, updates)

    return {
        "success": updated > 0,
        "updated": updated
    }

@router.delete("/{cid}", status_code=204)
def delete_conversation(
    cid: str,
    current_user: dict = Depends(get_current_user)
):
    conversations = conversation_service.get_conversations(
        {"cid": cid, "uid": current_user["uid"]}
    )

    if not conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation_service.delete_conversation(cid)
