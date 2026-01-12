import secrets

def generate_id(prefix: str = "") -> str:
    generated_id = secrets.token_urlsafe(16)
    return f"{prefix}_{generated_id}" if prefix else generated_id