
from typing import Any, List, Set, Dict
from services.model_service import ModelService
from services.model_service_error import ModelServiceError
import logging
import requests
import re

logger = logging.getLogger(__name__)


class OllamaModelService(ModelService):
    """
    Service for extracting requirements using a local Ollama model.
    """

  
    def __init__(self, ollama_url: str, model_id) -> None:
        self.OLLAMA_URL = ollama_url
        self.MODEL_ID = model_id

    def _call(self, prompt: str) -> str:
        """
        Call the Ollama REST API and return the raw response text.
        Raises ModelServiceError on non-200 or parsing problems.
        """
        payload = {
            "model": self.MODEL_ID,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.2
            }
        }

        headers = {"Content-Type": "application/json"}

        try:
            logger.debug("Calling Ollama at %s", self.OLLAMA_URL)
            resp = requests.post(self.OLLAMA_URL, json=payload, headers=headers)
        except requests.RequestException as e:
            logger.exception("Failed to call Ollama: %s", e)
            raise ModelServiceError(f"Ollama request failed: {e}") from e

        if resp.status_code != 200:
            # Try to include response text for debugging
            logger.error("Ollama returned status %s: %s", resp.status_code, resp.text[:1000])
            raise ModelServiceError(f"Ollama returned status {resp.status_code}: {resp.text}")

        # Ollama often returns JSON with a 'response' field; fallback to raw text.
        try:
            data = resp.json()
            # common key 'response' used in examples
            if isinstance(data, dict) and "response" in data:
                return data["response"]
            # Some versions might return {'id':..., 'output':[{'type':'message','content':...}]}, try to find text
            if isinstance(data, dict) and "output" in data:
                # try to extract textual content
                outputs = data.get("output", [])
                if outputs and isinstance(outputs, list):
                    texts = []
                    for o in outputs:
                        if isinstance(o, dict) and "content" in o:
                            c = o["content"]
                            if isinstance(c, list):
                                for part in c:
                                    if isinstance(part, dict) and part.get("type") == "message":
                                        texts.append(part.get("text") or part.get("content") or "")
                            else:
                                texts.append(str(c))
                    if texts:
                        return "\n".join(texts)
            # fallback to entire json as string
            return resp.text
        except ValueError:
            # not json
            return resp.text

    def extract(self, document: str) -> str:
        """
        Extract requirements from `document`.

        - Split the document into chunks of 100 lines (to avoid too-large prompts).
        - Call Ollama for each chunk.
        - Parse returned lines that match the required format.
        - De-duplicate and renumber REQ-001, REQ-002, ... in the final result.
        - Returns the final requirements as a single string (one line per requirement).
        """
        if not document or not document.strip():
            raise ModelServiceError("Empty document provided for extraction.")

        lines = document.splitlines()
        chunk_size = 100
        chunks: List[str] = []
        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i : i + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)

        collected_reqs: List[tuple] = []  # list of tuples (type, title, desc)
        seen_signatures: Set[str] = set()

        for idx, chunk in enumerate(chunks, start=1):
            prompt = self.PROMPT_TEMPLATE.format(document=chunk)
            logger.debug("Processing chunk %d/%d (len=%d lines)", idx, len(chunks), len(chunk.splitlines()))

            raw = self._call(prompt)
            logger.debug("Raw model output (first 500 chars): %s", raw[:500])

            # Split into lines and parse requirement lines
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue

                # try to parse strict format
                m = self.REQ_LINE_RE.match(line)
                if m:
                    # group refs:
                    # id_raw = m.group(1)  # might be REQ-XXX or REQ-123
                    req_type = m.group(2).lower()
                    title = m.group(3).strip()
                    desc = m.group(4).strip()

                    # signature to dedupe: type + title + desc
                    sig = f"{req_type}||{title}||{desc}"
                    if sig in seen_signatures:
                        logger.debug("Duplicate requirement ignored: %s", title)
                        continue
                    seen_signatures.add(sig)
                    collected_reqs.append((req_type, title, desc))
                else:
                    # If line doesn't strictly match, try to salvage by splitting on '|'
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 4 and parts[0].upper().startswith("REQ"):
                        # take parts[1], parts[2], parts[3]
                        req_type = parts[1].lower()
                        title = parts[2]
                        desc = parts[3]
                        if req_type in ("functional", "non_functional"):
                            sig = f"{req_type}||{title}||{desc}"
                            if sig in seen_signatures:
                                continue
                            seen_signatures.add(sig)
                            collected_reqs.append((req_type, title, desc))
                        else:
                            logger.debug("Ignored line with invalid type: %s", line)
                    else:
                        # ignore lines that are not requirements to avoid echoing input
                        logger.debug("Ignored non-REQ line: %s", line[:200])

        if not collected_reqs:
            raise ModelServiceError("No requirements extracted from the document.")

        # Renumber and format as required: REQ-001 | functional | title | desc
        output_lines: List[str] = []
        for i, (req_type, title, desc) in enumerate(collected_reqs, start=1):
            req_id = f"REQ-{i:03d}"
            output_lines.append(f"{req_id} | {req_type} | {title} | {desc}")

        result = "\n".join(output_lines)
        logger.info("Extracted %d requirements.", len(output_lines))
        return result
    