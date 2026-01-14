from services.tiny_db_service import TinyDBService
from utils.generate_id import generate_id
from typing import Dict, Any, List, Optional

class MessageService:
    COLLECTION_NAME = "messages"

    def __init__(self):
        self.db_service = TinyDBService(document_collection_name=self.COLLECTION_NAME)

    def add_message(self, cid: str, content: List[Dict[str, Any]]) -> str:
        """
        Add a new message to a conversation.
        - content is a list of dicts: [{filename, requirements, timestamp, model_id}, ...]
        Returns the generated message ID (mid).
        """
        mid = generate_id()
        message_data = {
            "mid": mid,
            "cid": cid,
            "content": content
        }
        self.db_service.store(message_data)
        return mid

    def get_messages(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get messages. Optionally filter by fields (e.g., cid or mid).
        """
        return self.db_service.query(filters)

    def update_message(self, mid: str, updates: Dict[str, Any]) -> int:
        """
        Update message fields by mid.
        Returns the number of updated documents.
        """
        return self.db_service.update({"mid": mid}, updates)

    def delete_message(self, mid: str) -> int:
        """
        Delete a message by mid.
        Returns the number of deleted documents.
        """
        return self.db_service.delete({"mid": mid})
