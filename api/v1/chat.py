from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from models.chat import ChatResponse
from services.model_list_service import ModelListService
import json
import io
import logging
import uuid
from pathlib import Path

# libs pour extraction
import fitz  #
from docx import Document

logger = logging.getLogger(__name__)

model_list_service = ModelListService()
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/upload", response_model=ChatResponse)
async def upload(
    model_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Reçoit un fichier (PDF ou DOCX), extrait le texte, appelle le model service
    et renvoie la réponse JSON (sous forme de chaîne) et model_id.
    """
    # validation fichier
    if not file.filename:
        raise HTTPException(status_code=400, detail="Aucun fichier envoyé.")

    filename = file.filename
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    # Lire le contenu du fichier (bytes)
    try:
        content_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Impossible de lire le fichier : {e}")
    finally:
        try:
            await file.close()
        except Exception:
            pass

    # Extraire texte selon l'extension
    text = ""
    if ext == "pdf":
        try:
            pdf = fitz.open(stream=content_bytes, filetype="pdf")
            parts = []
            for page in pdf:
                parts.append(page.get_text())  # get_text() renvoie le texte de la page
            text = "\n".join(parts).strip()
            pdf.close()
        except Exception as e:
            logger.exception("Erreur extraction PDF")
            raise HTTPException(status_code=500, detail=f"Échec d'extraction du PDF : {e}")
    elif ext in ("docx",):
        try:
            bio = io.BytesIO(content_bytes)
            doc = Document(bio)
            paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
            text = "\n".join(paragraphs).strip()
        except Exception as e:
            logger.exception("Erreur extraction DOCX")
            raise HTTPException(status_code=500, detail=f"Échec d'extraction du DOCX : {e}")
    elif ext == "txt":
        try:
            # Essayer plusieurs encodages courants
            try:
                text = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = content_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = content_bytes.decode("latin-1")  # fallback
            text = text.strip()
        except Exception as e:
            logger.exception("Erreur extraction TXT")
            raise HTTPException(status_code=500, detail=f"Échec d'extraction du fichier texte : {e}")
    else:
        raise HTTPException(status_code=400, detail="Type de fichier non supporté. Seuls PDF et DOCX sont acceptés.")

    if not text:
        raise HTTPException(status_code=400, detail="Le document ne contient pas de texte exploitable.")
    saved_filename = filename
     # --- Sauvegarder le fichier reçu sous ./static/docs/<uuid>.<ext> ---
   
    try:
        docs_dir = Path("./static/docs")
        docs_dir.mkdir(parents=True, exist_ok=True)

        unique_id = uuid.uuid4().hex  # id unique
        saved_filename = f"{unique_id}.{ext}" if ext else unique_id
        saved_path = docs_dir / saved_filename

        # écrire les bytes lus précédemment
        saved_path.write_bytes(content_bytes)

        # debug / log (optionnel)
        logger.info("Fichier sauvegardé dans %s", saved_path)
    except Exception as e:
        logger.exception("Erreur lors de la sauvegarde du fichier")
        raise HTTPException(status_code=500, detail=f"Impossible de sauvegarder le fichier : {e}")
   
    # Appeler le service de modèle pour extraire les exigences
    try:
        result = model_list_service.extract_requierement_from_model(model_id, paragraph=text)
    except Exception as e:
        logger.exception("Erreur appel ModelListService")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au modèle : {e}")

    # Vérifier le résultat renvoyé par le service
    if not result.get("ok", False):
        # Erreur métier (ex : modèle non trouvé, erreur de chargement, etc.)
        raise HTTPException(status_code=400, detail=result.get("error", "Extraction échouée"))

    # Sérialiser data (liste d'exigences) en chaîne JSON
    try:
        response_str = json.dumps(result.get("data", []), ensure_ascii=False)
    except Exception as e:
        logger.exception("Erreur de sérialisation JSON")
        raise HTTPException(status_code=500, detail=f"Impossible de sérialiser la réponse : {e}")

    return ChatResponse(response=response_str, model_id=model_id, saved_filename=saved_filename)

@router.get("/models")
async def get_models():
    """
    Retourne la liste des modèles instanciés/registrés.
    """
    models = model_list_service.list_models()
    return {"available_models": models}