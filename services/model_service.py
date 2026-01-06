from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import re
import logging

logger = logging.getLogger(__name__)


class ModelService(ABC):
    """
    Classe abstraite qui définit le contrat des services de modèles (LLM).
    Hériter et implémenter : getRequirementAsJSON(), extract().
    """
    # regex to accept model output lines to be parsed

    REQ_LINE_RE = re.compile(
        r"^\s*(REQ-[0-9A-Za-z_]+)\s*\|\s*(functional|non_functional)\s*\|\s*([^\|]{1,200})\s*\|\s*([^\|]{1,1000})\s*$",
        re.IGNORECASE,
    )

    PROMPT_TEMPLATE = """
You are a Requirements Extraction Agent.

GOAL:
Extract every explicit or implicit requirement from the Text block below.
A requirement can be functional or non-functional.

OUTPUT FORMAT (strict):
Each requirement must be on its own line and follow this exact format:
REQ-XXX | functional | <title> | <description>
or
REQ-XXX | non_functional | <title> | <description>

Rules & constraints:
1. Output EXACTLY one requirement per line.
2. Use numeric IDs with 3 digits, e.g. REQ-001, REQ-002, ...
   (If the model produces placeholders like REQ-XXX, they will be renumbered by the caller.)
3. The <title> must be a short, single sentence summary (no pipes '|' inside).
4. The <description> must expand the title in one or two sentences (no pipes '|' inside).
5. Do NOT output any extra text, explanations, bullets, code blocks, or metadata.
6. Do NOT repeat or echo the input Text.
7. Answer in the SAME LANGUAGE as the Text.

Example lines (do not include these in the output, they are only example formats):
REQ-001 | functional | Authentificagion | Users must authenticate using email and password.
REQ-002 | non_functional | Data encryption | All stored patient data must be encrypted using AES-256.

TEXT:
{document}

INSTRUCTION: Extract ALL requirements and return ONLY the requirements list (one requirement per line) using the EXACT format above. Return an empty string if there is no requirement.
""".strip()
    def extract(self, document: str) -> str:
        """
        Comportement par défaut : renvoie le document inchangé.
        Surcharger cette méthode dans une sous-classe pour appeler un LLM ou tout autre
        pipeline d'extraction qui doit renvoyer un texte multiligne (une ligne = une requirement).
        """
        return document

    def getRequierementAsJson(self, document: str) -> List[Dict[str, str]]:
        """
        Convertit les exigences extraites en une liste de dictionnaires:
        [
        {"id": "REQ-001", "type": "functional", "title": "...", "description": "..."},
        ...
        ]

        Cette méthode s'appuie sur self.extract(document) qui retourne les lignes
        déjà renumérotées au format strict. Si certaines lignes ne correspondent
        pas au regex strict, on tente une récupération simple via split("|").
        """
        try:
            extracted_text = self.extract(document)
        except Exception as e:
            logger.exception("Erreur lors de l'extraction des requirements: %s", e)
            raise

        results: List[Dict[str, str]] = []

        for line in extracted_text.splitlines():
            line = line.strip()
            if not line:
                continue

            m = getattr(self, "REQ_LINE_RE", None)
            parsed = None

            if m:
                match = m.match(line)
                if match:
                    req_id = match.group(1).upper()
                    req_type = match.group(2).lower()
                    title = match.group(3).strip()
                    description = match.group(4).strip()
                    parsed = (req_id, req_type, title, description)

            if parsed is None:
                # Fallback: try to split by '|' and recover fields
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4 and parts[0].upper().startswith("REQ"):
                    req_id = parts[0].upper()
                    req_type = parts[1].lower()
                    title = parts[2]
                    description = parts[3]
                    parsed = (req_id, req_type, title, description)
                else:
                    # ignore non-conforming line
                    logger.debug("Ignored non-conforming line while parsing to JSON: %s", line[:200])
                    continue

            req_id, req_type, title, description = parsed
            # Normaliser le type si besoin
            if req_type in ("non-functional", "non functional"):
                req_type = "non_functional"
            elif req_type == "non_functional":
                pass
            else:
                req_type = "functional" if "functional" in req_type else req_type

            results.append({
                "id": req_id,
                "type": req_type,
                "title": title,
                "description": description
            })

        return results