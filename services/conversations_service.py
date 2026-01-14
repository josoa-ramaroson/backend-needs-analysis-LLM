from services.tiny_db_service import TinyDBService
from utils.generate_id import generate_id
from typing import Dict, Any, List, Optional
from datetime import datetime


class ConversationsService:
    COLLECTION_NAME = "conversations"

    def __init__(self):
        self.db_service = TinyDBService(document_collection_name=self.COLLECTION_NAME)

    def create_conversation(self, uid: str, conversation_name: str) -> str:
        """
        Create a new conversation.
        Returns the generated conversation ID (cid).
        """
        started_from = datetime.now().isoformat()

        cid = generate_id()
        conversation_data = {
            "cid": cid,
            "uid": uid,
            "conversation_name": conversation_name,
            "started_from": started_from
        }
        self.db_service.store(conversation_data)
        return cid

    def get_conversations(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get conversations. Optionally filter by fields (e.g., uid).
        """
        return self.db_service.query(filters)

    def update_conversation(self, cid: str, updates: Dict[str, Any]) -> int:
        """
        Update conversation fields by cid.
        Returns the number of updated documents.
        """
        return self.db_service.update({"cid": cid}, updates)

    def delete_conversation(self, cid: str) -> int:
        """
        Delete a conversation by cid.
        Returns the number of deleted documents.
        """
        return self.db_service.delete({"cid": cid})
