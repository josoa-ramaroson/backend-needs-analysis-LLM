import os

UPLOAD_DIR = "static/uploads"

def save_file(content, filename: str):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(content)

    return filepath
