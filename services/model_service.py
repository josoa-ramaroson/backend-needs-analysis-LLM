from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import re
import logging

logger = logging.getLogger(__name__)


class ModelServiceError(Exception):
    """Erreur générique pour ModelService."""
    pass


class JSONParseError(ModelServiceError):
    """Erreur levée quand on ne parvient pas à parser du JSON."""
    pass


class SchemaValidationError(ModelServiceError):
    """Erreur levée quand un item ne respecte pas le schéma attendu."""
    pass


class ModelService(ABC):
    """
    Classe abstraite qui définit le contrat des services de modèles (LLM).
    Hériter et implémenter : load(), unload(), generate().
    """

    PROMPT_TEMPLATE = (
        "Tu es un agent d’extraction d’exigences. Règles :\n"
        "1. Une exigence = besoin explicite ou implicite, fonctionnel ou non.\n"
        "2. Décomposer les exigences (et implicites) si plusieurs dans un même paragraphe.\n"
        "3. Reformuler clairement chaque exigence principale.\n"
        "4. Fournir une courte description textuelle expliquant l'exigence.\n"
        "5. Tu dois classifier chaque exigence en deux catégories :\n"
        "   - fonctionnelle\n"
        "   - non_fonctionnelle\n"
        "6. La sortie DOIT être un JSON strictement valide, contenant une liste d'exigences. Chaque exigence doit contenir exactement les clés :\n"
        "   exigence, description, type\n"
        "   où type ∈ {{\"fonctionnelle\", \"non_fonctionnelle\"}}\n\n"
        "IMPORTANT : **Tu DOIS répondre uniquement par un objet JSON valide**, sans texte additionnel.\n"
        "IMPORTANT : **Tu DOIS extraire toutes les exigences**, et tu peux retourner plusieurs exigences principales.\n\n"
        "Texte à analyser : {paragraph}\n"
    )

    # schéma minimal attendu pour chaque exigence
    EXPECTED_KEYS = {"exigence", "description", "type"}
    ALLOWED_TYPES = {"fonctionnelle", "non_fonctionnelle"}

    def __init__(self, model_id: str, device: str = "cpu"):
        """
        :param model_id: identifiant ou chemin du modèle
        :param device: 'cpu' ou 'cuda' (ou autre, selon implémentation)
        """
        self.model_id = model_id
        self.device = device
        self._is_loaded = False

    # ---- Méthodes abstraites à implémenter par chaque service ----
    @abstractmethod
    def load(self) -> None:
        """
        Charger le modèle en mémoire / initialiser les ressources.
        Doit définir self._is_loaded = True si succès.
        """
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """
        Libérer les ressources du modèle (GPU / fichiers / processus).
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0, **kwargs) -> str:
        """
        Générer la sortie textuelle du modèle à partir d'un prompt.
        Retourne toujours une chaîne (raw text).
        Implémentations possibles : transformers, vLLM, remote API...
        """
        raise NotImplementedError

    # ---- Helpers concrets réutilisables ----
    def render_prompt(self, paragraph: str) -> str:
        """Rendre le prompt final à partir du template."""
        return self.PROMPT_TEMPLATE.format(paragraph=paragraph)

    @staticmethod
    def extract_json_block(raw: str) -> Optional[str]:
        """
        Tenter d'extraire le premier bloc JSON trouvable dans la sortie brute.
        Retourne la chaîne JSON ou None si non trouvée.
        """
        # Recherche du premier '{' ... '}' correspondant. On utilise un parse simple.
        match = re.search(r"(\{(?:.|\n)*\})", raw)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def safe_json_load(s: str) -> Any:
        """
        Charger JSON en gérant les erreurs avec messages clairs.
        """
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            raise JSONParseError(f"JSON decode error: {e.msg} at pos {e.pos}")

    def validate_item_schema(self, item: Dict[str, Any]) -> None:
        """
        Valide un item d'exigence basique :
         - contient exactement les clés attendues
         - 'type' appartient aux valeurs autorisées
        Lève SchemaValidationError si invalide.
        """
        if not isinstance(item, dict):
            raise SchemaValidationError("Item is not a JSON object")

        keys = set(item.keys())
        if keys != self.EXPECTED_KEYS:
            raise SchemaValidationError(f"Item keys invalid. Expected {self.EXPECTED_KEYS}, got {keys}")

        if not isinstance(item["exigence"], str) or not item["exigence"].strip():
            raise SchemaValidationError("exigence must be a non-empty string")

        if not isinstance(item["description"], str):
            raise SchemaValidationError("description must be a string")

        if item["type"] not in self.ALLOWED_TYPES:
            raise SchemaValidationError(f"type must be one of {self.ALLOWED_TYPES}")

    # ---- Méthode de haut niveau fournie ----
    def extract_requirements(self, paragraph: str, retries: int = 1, strict: bool = True) -> List[Dict[str, Any]]:
        """
        Pipeline de génération + parsing + validation.

        :param paragraph: texte à analyser
        :param retries: nombre de réessais si JSON invalide (par ex. rappeler generate avec instructions de correction)
        :param strict: si True, lève une exception si validation échoue; sinon retourne ce qui a pu être parsé
        :return: liste d'exigences (list[dict])
        """
        if not self._is_loaded:
            raise ModelServiceError("Model not loaded. Call load() before extract_requirements().")

        prompt = self.render_prompt(paragraph)
        raw = self.generate(prompt)

        # 1) essai direct parse
        try:
            parsed = self.safe_json_load(raw)
        except JSONParseError:
            # 2) essayer d'extraire un bloc JSON et parser
            json_block = self.extract_json_block(raw)
            if json_block:
                try:
                    parsed = self.safe_json_load(json_block)
                except JSONParseError as e:
                    logger.warning("Échec du parse après extraction de bloc JSON: %s", e)
                    parsed = None
            else:
                parsed = None

        # 3) si parsing impossible -> retries (demander au modèle de corriger la sortie)
        attempt = 0
        while parsed is None and attempt < retries:
            attempt += 1
            logger.info("Retrying generation to obtain valid JSON (attempt %d/%d)", attempt, retries)
            # On demande explicitement au modèle de renvoyer uniquement du JSON
            repair_prompt = (
                "Votre dernière réponse n'était pas un JSON valide. "
                "Réponds uniquement par un JSON strictement valide qui correspond au schéma : "
                f"{list(self.EXPECTED_KEYS)}. "
                "Corrige uniquement le JSON, sans autre texte.\n\n"
                f"Texte à analyser : {paragraph}\n"
            )
            raw_retry = self.generate(repair_prompt)
            # try parse retry
            try:
                parsed = self.safe_json_load(raw_retry)
            except JSONParseError:
                json_block = self.extract_json_block(raw_retry)
                if json_block:
                    try:
                        parsed = self.safe_json_load(json_block)
                    except JSONParseError:
                        parsed = None
                else:
                    parsed = None

        if parsed is None:
            msg = "Impossible d'obtenir un JSON valide depuis le modèle."
            if strict:
                raise JSONParseError(msg)
            else:
                logger.warning(msg)
                # retourner ce qu'on a (vide)
                return []

        # 4) normalize parsed -> should be list of exigences
        if isinstance(parsed, dict):
            # si le modèle renvoie {"exigences": [...] } ou similaire
            if "exigences" in parsed and isinstance(parsed["exigences"], list):
                items = parsed["exigences"]
            else:
                # peut-être un seul item (dict) -> wrap
                items = [parsed]
        elif isinstance(parsed, list):
            items = parsed
        else:
            if strict:
                raise SchemaValidationError("Parsed JSON has unexpected top-level type")
            else:
                return []

        # 5) validate each item
        validated_items: List[Dict[str, Any]] = []
        for item in items:
            try:
                self.validate_item_schema(item)
                validated_items.append(item)
            except SchemaValidationError as e:
                logger.warning("Item schema invalid: %s; item=%s", e, item)
                if strict:
                    raise

        return validated_items
