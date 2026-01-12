from tinydb import TinyDB, Query
from typing import Any, Dict, List, Optional


class TinyDBService:
    """
    Very simple TinyDB-backed service for storing and querying documents.
    - db_path: path to the TinyDB JSON file (default: data.json)
    - document_collection_name: name of the TinyDB table/collection
    """

    def __init__(self, db_path: str = "data.json", document_collection_name: str = "default"):
        self.db = TinyDB(db_path)
        self.table = self.db.table(document_collection_name)
        self.document_collection_name = document_collection_name

    def store(self, document: Dict[str, Any]) -> int:
        """
        Insert a document into the collection.
        Returns the TinyDB doc_id of the inserted document.
        """
        return self.table.insert(document)

    def query(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query documents.
        - If filters is None or empty, return all documents.
        - filters should be a dict of {field: value} which will be AND-ed.
        Example: query({"type": "functional", "priority": "must"})
        """
        if not filters:
            return self.table.all()

        q = Query()
        condition = None
        for key, value in filters.items():
            expr = (q[key] == value)
            condition = expr if condition is None else condition & expr

        return self.table.search(condition)
    
    def update_field(
        self,
        filters: Dict[str, Any],
        field: str,
        value: Any
    ) -> int:
        """
        Update a single field for documents matching filters.
        Returns number of updated documents.
        """
        q = Query()
        condition = None
        for k, v in filters.items():
            expr = (q[k] == v)
            condition = expr if condition is None else condition & expr

        return self.table.update({field: value}, condition)

    def update(
        self,
        filters: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> int:
        """
        Update multiple fields for documents matching filters.
        Returns number of updated documents.
        """
        q = Query()
        condition = None
        for k, v in filters.items():
            expr = (q[k] == v)
            condition = expr if condition is None else condition & expr

        return self.table.update(updates, condition)

    def delete(self, filters: Dict[str, Any]) -> int:
        """
        Delete documents matching filters.
        Returns number of deleted documents.
        """
        q = Query()
        condition = None
        for k, v in filters.items():
            expr = (q[k] == v)
            condition = expr if condition is None else condition & expr

        return self.table.remove(condition)
        
    def delete_by_id(self, doc_id: int) -> bool:
        """
        Delete a document by its TinyDB doc_id.
        Returns True if deleted.
        """
        return bool(self.table.remove(doc_ids=[doc_id]))
