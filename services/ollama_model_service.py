
from typing import Any, List, Set, Dict
from services.model_service import ModelService
from services.model_service_error import ModelServiceError
import logging
import requests
import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


def clean_sentence_simple(sentence: str) -> str:
    """
    Very small sanitizer:
     - remove CR (\r)
     - replace tabs with single space
     - collapse sequences of 3+ dots into '...'
     - remove most junk characters (keeps word chars, whitespace and common punctuation)
     - collapse multiple newlines into one newline
     - collapse multiple spaces into one space
     - strip leading/trailing whitespace on each line and overall
    """
    if not sentence:
        return ""

    # 1) Basic normalisation of control chars
    s = sentence.replace("\r", "")
    s = s.replace("\t", " ")

    # 2) Collapse long sequences of dots (3 or more) into '...'
    s = re.sub(r"\.{3,}", ".", s)

    # 3) Remove undesirable characters.
    # Keep:
    #   \w   : unicode word characters (letters incl. accents, digits, underscore)
    #   \s   : whitespace (spaces, tabs already converted, newlines)
    #   common punctuation: . , ? ! ; : ' " - ( ) [ ] { } / % @ &
    s = re.sub(r"[^\w\s\.,\?\!\;\:\'\"\-\(\)\[\]\{\}\/%@&]", "", s)

    # 4) Collapse multiple newlines into a single newline
    s = re.sub(r"\n{2,}", "\n", s)

    # 5) Collapse multiple spaces into one
    s = re.sub(r"[ ]{2,}", " ", s)

    # 6) Trim spaces at start/end of each line, and overall
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = s.strip()

    return s
# char_length 15 000 is an approximation of characters equivalent to 4096 token
def chunk_by_sentences(document: str, char_length: int = 18000):
    chunks = []
    current_chunk = ""
    sentences = document.split(".")

    for sentence in sentences:
        sentence = clean_sentence_simple(sentence)
        if not sentence:
            continue

        sentence_with_dot = sentence + "."

        # 🔹 3) Gestion de la taille des chunks
        if len(current_chunk) + len(sentence_with_dot) > char_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                logger.debug("New chunk created (%d chars)", len(current_chunk))
            current_chunk = sentence_with_dot
        else:
            current_chunk += " " + sentence_with_dot if current_chunk else sentence_with_dot

    if current_chunk:
        chunks.append(current_chunk.strip())

    #logging info
    logger.warn(f"Length of chunking : {len(chunks)}")
    return chunks



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
            "system": "You are a project manager. Always answer in plain text. Do not use Markdown, lists, bullet points, or formatting.",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
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

        chunks = chunk_by_sentences(document)
        #print(chunks)
        collected_reqs: List[tuple] = []  # list of tuples (type, title, desc)
        seen_signatures: Set[str] = set()

        for idx, chunk in enumerate(chunks, start=1):
            prompt = self.PROMPT_TEMPLATE.format(document=chunk)
            logger.debug("Processing chunk %d/%d (len=%d lines)", idx, len(chunks), len(chunk.splitlines()))

            raw = self._call(prompt)
            #print("raw: ", raw)
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
    