from fastapi import APIRouter, UploadFile, File
from services.file_service import save_file

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    path = save_file(content, file.filename)
    return {"filename": file.filename, "path": path}
