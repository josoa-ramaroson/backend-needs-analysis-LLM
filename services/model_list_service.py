# services/model_list_service.py
from typing import Dict, Any, List, Optional
from services.model_service import ModelService, ModelServiceError
from services.llama_model_service import LlamaModelService
import logging

logger = logging.getLogger(__name__)


class ModelListService:
    """
    Service to manage instantiated ModelService implementations.
    It keeps instances in memory and provides helpers to call them.
    """

    def __init__(self) -> None:
        # mapping model_id -> ModelService instance
        self.model_list: Dict[str, ModelService] = {}

        # Instantiate default Llama model service using the local path
        default_id = "llama-3.2-3B-Instruct"
        try:
            llama_service = LlamaModelService(model_id="./static/models/Llama-3.2-3B-Instruct", device="cpu")
            # do not call load() here automatically if you prefer lazy loading;
            # we keep the instance ready and let extract method call load() on demand.
            self.model_list[default_id] = llama_service
        except Exception as e:
            logger.exception("Failed to initialize default LlamaModelService: %s", e)

    # -----------------------
    # Model management
    # -----------------------
    def add_model(self, model_id: str, service: ModelService) -> None:
        """Register a ModelService instance for a given model_id."""
        self.model_list[model_id] = service
        logger.info("Model added: %s", model_id)

    def remove_model(self, model_id: str) -> bool:
        """Remove a registered model. Returns True if removed, False if not found."""
        if model_id in self.model_list:
            # attempt to unload resources if possible
            try:
                svc = self.model_list[model_id]
                try:
                    svc.unload()
                except Exception:
                    logger.debug("Error while unloading model %s", model_id, exc_info=True)
            finally:
                del self.model_list[model_id]
            logger.info("Model removed: %s", model_id)
            return True
        return False

    def list_models(self) -> List[str]:
        """Return the list of registered model IDs."""
        return list(self.model_list.keys())

    def get_model_instance(self, model_id: str) -> Optional[ModelService]:
        """Return the ModelService instance or None if not registered."""
        return self.model_list.get(model_id)

    # -----------------------
    # Extraction wrapper
    # -----------------------
    def extract_requierement_from_model(
        self,
        model_id: str,
        paragraph: str,
        retries: int = 1,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Call the given model to extract requirements from `paragraph`.

        Returns a structured dict:
        {
          "model_id": model_id,
          "ok": True/False,
          "data": <list of exigences> (if ok),
          "error": "<message>" (if not ok)
        }
        """
        if model_id not in self.model_list:
            return {
                "model_id": model_id,
                "ok": False,
                "error": f"Model '{model_id}' not found",
            }

        svc = self.model_list[model_id]

        # Ensure model is loaded (lazy load)
        try:
            if not getattr(svc, "_is_loaded", False):
                svc.load()
        except Exception as e:
            logger.exception("Failed to load model %s: %s", model_id, e)
            return {
                "model_id": model_id,
                "ok": False,
                "error": f"Failed to load model '{model_id}': {e}",
            }

        # Call the extract pipeline from the ModelService
        try:
            result = svc.extract_requirements(paragraph, retries=retries, strict=strict)
            return {
                "model_id": model_id,
                "ok": True,
                "data": result,
            }
        except ModelServiceError as me:
            logger.exception("ModelServiceError while extracting with %s: %s", model_id, me)
            return {
                "model_id": model_id,
                "ok": False,
                "error": str(me),
            }
        except Exception as e:
            logger.exception("Unexpected error while extracting with %s: %s", model_id, e)
            return {
                "model_id": model_id,
                "ok": False,
                "error": f"Unexpected error: {e}",
            }
