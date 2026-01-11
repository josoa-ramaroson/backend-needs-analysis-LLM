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

    PROMPT_TEMPLATE = """TASK: is to extract all functional and non-functional requirements from a document and reformulate them clearly, concisely, and accurately. The text below is part of a discussion or a document about a project.
IMPORTANT: the provided TEXT may be split into short sentences (the document has been split by sentence). You must consider adjacent sentences together when necessary to detect implicit or multi-sentence requirements.

GOAL:
Extract every explicit or implicit requirement from the TEXT block below.
A requirement can be functional or non-functional.

OUTPUT FORMAT (strict):
Each requirement must be on its own line and follow this exact format:
<IDS> | functional | <title> | <description>
or
<IDS> | non_functional | <title> | <description>
Rules & constraints (must be followed exactly):
1. Output EXACTLY one requirement per line. Do NOT add any surrounding text, headings, code fences, or explanation.
2. Use numeric <IDS> with 3 digits, starting at REQ-001 and incrementing by 1 (REQ-001, REQ-002, ...).
3. The <title> must be a short, single sentence (no '|' characters). Keep it concise (6–12 words recommended). Do NOT include trailing periods.
4. The <description> should be a short paragraph (1–2 sentences) clarifying the requirement.
5. Keep single spaces around the pipe separator: `"REQ-001 | functional | Title | Description"`.
6. Answer in the SAME LANGUAGE as the TEXT.
7. Merge duplicates: if the same requirement appears multiple times, return it only once.
8. If a requirement is implicit, infer it but mark it as functional or non_functional according to intent (do not invent unrelated features).
9. Write the <title> and <description> in the SAME LANGUAGE as the TEXT. 
   Example (English): write the title and description in English. 
   If the text is French, write the title and description in French.
10. Do NOT return anything other than the lines described above. Do not add any explanation.
Examples of valid lines (for format only — do not include these lines in output):
REQ-001 | functional | User login | Users must authenticate using email and password.
REQ-002 | non_functional | Data encryption | All stored patient data must be encrypted using AES-256.
REQ-003 | functional | Mobile application version | The system must provide a mobile application version compatible with Android and iOS devices.
INSTRUCTION: Extract ALL requirements and return ONLY the requirements list using the EXACT format above.
TEXT:
{document}
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
            logger.exception("Error during model call: %s", e)
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