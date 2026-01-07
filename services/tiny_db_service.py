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
