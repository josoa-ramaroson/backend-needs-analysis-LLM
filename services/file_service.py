
import io
import os
import uuid
import logging
from pathlib import Path
from typing import Tuple, Optional, Set

import fitz  # PyMuPDF
from docx import Document
from fastapi import UploadFile

logger = logging.getLogger(__name__)


# -------------------------
# Exceptions
# -------------------------
class FileServiceError(Exception):
    """Base exception for file service errors."""
    pass

class FileReadError(FileServiceError):
    """Raised when the uploaded file cannot be read."""
    pass

class UnsupportedFileTypeError(FileServiceError):
    """Raised when the file extension is not supported."""
    pass

class ExtractionError(FileServiceError):
    """Raised when text extraction fails for a supported file type."""
    pass

class FileSaveError(FileServiceError):
    """Raised when saving the uploaded file to disk fails."""
    pass


# -------------------------
# FileService
# -------------------------
class FileService:
    """
    Service responsible for reading uploaded files, extracting text, and saving the raw bytes to disk.

    Usage:
        file_service = FileService(file_dir="static/docs")
        text, saved_path = await file_service.read_and_save(uploaded_file)
    """

    SUPPORTED_EXTS: Set[str] = {"pdf", "docx", "doc", "txt"}

    def __init__(self, file_dir: str = "static/docs") -> None:
        """
        Initialize FileService.

        Args:
            file_dir: Directory where uploaded files will be saved.
        """
        self.file_dir = Path(file_dir)

    async def read_and_save(self, file: UploadFile) -> Tuple[str, str]:
        """
        Read an UploadFile, extract text according to file type, save the raw bytes
        under a unique filename, and return (extracted_text, saved_path_str).

        Raises:
            FileReadError: if reading the uploaded file fails.
            UnsupportedFileTypeError: if extension is unsupported.
            ExtractionError: if text extraction fails.
            FileSaveError: if writing bytes to disk fails.
        """
        # Original filename (as uploaded by client)
        original_filename = file.filename or ""
        ext = (
            original_filename.lower().rsplit(".", 1)[-1]
            if "." in original_filename
            else ""
        )

        # Read file bytes (async)
        try:
            content_bytes = await file.read()
        except Exception as exc:
            logger.exception("Failed to read uploaded file")
            raise FileReadError(f"Failed to read uploaded file: {exc}") from exc
        finally:
            # Always try to close the UploadFile
            try:
                await file.close()
            except Exception:
                # best-effort close; log at debug
                logger.debug("Failed to close UploadFile (ignored)")

        # Normalize extension and check supported types
        ext = ext.lower()
        if ext not in self.SUPPORTED_EXTS:
            raise UnsupportedFileTypeError(f"Only PDF, DOCX/DOC and TXT are supported. Got: '{ext}'")

        # Extract text according to extension
        text: str = ""
        if ext == "pdf":
            try:
                # use PyMuPDF to extract text
                pdf = fitz.open(stream=content_bytes, filetype="pdf")
                parts = []
                for page in pdf:
                    parts.append(page.get_text("text") or "")
                text = "\n".join(parts).strip()
                pdf.close()
            except Exception as exc:
                logger.exception("PDF text extraction failed")
                raise ExtractionError(f"PDF text extraction failed: {exc}") from exc

        elif ext in ("docx", "doc"):
            try:
                bio = io.BytesIO(content_bytes)
                doc = Document(bio)
                paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
                text = "\n".join(paragraphs).strip()
            except Exception as exc:
                logger.exception("DOCX text extraction failed")
                raise ExtractionError(f"DOCX text extraction failed: {exc}") from exc

        elif ext == "txt":
            try:
                # Try common encodings
                try:
                    text = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        text = content_bytes.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        text = content_bytes.decode("latin-1")
                text = text.strip()
            except Exception as exc:
                logger.exception("TXT text extraction failed")
                raise ExtractionError(f"TXT text extraction failed: {exc}") from exc
        else:
            # This branch should not be hit because of earlier check.
            raise UnsupportedFileTypeError(f"Unsupported file extension: {ext}")

        # Ensure directory exists and save the original bytes under a unique name
        try:
            self.file_dir.mkdir(parents=True, exist_ok=True)
            unique_id = uuid.uuid4().hex
            saved_filename = f"{unique_id}.{ext}" if ext else unique_id
            saved_path = self.file_dir / saved_filename
            saved_path.write_bytes(content_bytes)
            logger.info("Uploaded file saved to %s (original: %s)", saved_path, original_filename)
        except Exception as exc:
            logger.exception("Failed to save uploaded file to disk")
            raise FileSaveError(f"Failed to save uploaded file: {exc}") from exc

        
        return text, str(saved_path)
