from services.tiny_db_service import TinyDBService
from utils.generate_id import generate_id

from typing import Dict, Any
import bcrypt


class UserService:
    COLLECTION_NAME = "users"

    def __init__(self):
        self.db_service = TinyDBService(document_collection_name=self.COLLECTION_NAME)

    def hash_password(self, plain_password: str) -> str:
        """
        Hash a password using bcrypt.
        Returns a UTF-8 string safe to store in DB.
        """
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def verify_password(self, plain_password: str, stored_hash: str) -> bool:
        """
        Verify a password against the stored bcrypt hash.
        """
        return bcrypt.checkpw(plain_password.encode("utf-8"), stored_hash.encode("utf-8"))

    def register_user(self, user_data: Dict[str, str]):
        """
        Register a new user with hashed password and generated UID.
        Raises ValueError if required fields are missing.
        """
        required_fields = ["username", "password"]
        for field in required_fields:
            if field not in user_data:
                raise ValueError(f"Missing required attribute: {field}")

        # Hash password and generate UID
        hashed_password = self.hash_password(user_data["password"])
        user_data["password"] = hashed_password
        user_data["uid"] = generate_id()

        # Store in DB
        self.db_service.store(user_data)

    def authenticate_user(self, user_data: Dict[str, str]) -> bool:
        """
        Authenticate a user by uid and password.
        Returns True if successful, False otherwise.
        """
        if "uid" not in user_data or "password" not in user_data:
            raise ValueError("Missing required attribute: uid or password")

        registered_data = self.db_service.query({"uid": user_data["uid"]})
        if not registered_data:
            return False

        stored_hash = registered_data[0].get("password")
        return self.verify_password(user_data["password"], stored_hash)

    def get_user(self, uid: str) -> Dict[str, Any] | None:
        """
        Get user information by UID.
        Password is excluded from the returned data.
        Returns None if user does not exist.
        """
        result = self.db_service.query({"uid": uid})

        if not result:
            return None

        user = result[0].copy()
        user.pop("password", None)  # never expose password

        return user

    def delete_user(self, user_id: str) -> int:
        """
        Delete a user by UID.
        Returns the number of deleted documents.
        """
        return self.db_service.delete({"uid": user_id})
