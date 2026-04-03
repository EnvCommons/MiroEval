import json
import re
from pathlib import Path
from typing import Optional


def _find_data_file(filename: str) -> Optional[Path]:
    """Find a data file in production or local paths."""
    candidates = [
        Path("/orwd_data") / filename,
        Path(__file__).parent / "data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _find_attachments_dir() -> Optional[Path]:
    """Find the multimodal-attachments directory."""
    candidates = [
        Path("/orwd_data/multimodal-attachments"),
        Path(__file__).parent / "data" / "multimodal-attachments",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return None


def load_tasks() -> tuple[list[dict], dict[str, dict]]:
    """
    Load all MiroEval queries (text + multimodal) into a single 'test' split.

    Returns:
        tasks: list of user-facing task specs
        full_tasks: dict keyed by id with full annotation data
    """
    tasks = []
    full_tasks = {}

    # Load text queries (70)
    text_path = _find_data_file("mirobench_text.json")
    if text_path is None:
        raise FileNotFoundError("mirobench_text.json not found")
    with open(text_path, "r", encoding="utf-8") as f:
        text_raw = json.load(f)

    for item in text_raw:
        task_id = str(item["id"])
        annotation = item.get("annotation", {})
        task_spec = {
            "id": task_id,
            "query": item["rewritten_query"],
            "domain": annotation.get("domain", "other"),
            "language": annotation.get("language", "en"),
            "files": [],
        }
        tasks.append(task_spec)
        full_tasks[task_id] = {
            **task_spec,
            "chat_id": item.get("chat_id", ""),
            "annotation": annotation,
        }

    # Load multimodal queries (30)
    mm_path = _find_data_file("mirobench_multimodal.json")
    if mm_path is not None:
        with open(mm_path, "r", encoding="utf-8") as f:
            mm_raw = json.load(f)

        for item in mm_raw:
            task_id = str(item["id"])
            annotation = item.get("annotation", {})
            files = item.get("files", [])
            task_spec = {
                "id": task_id,
                "query": item["rewritten_query"],
                "domain": annotation.get("domain", "other"),
                "language": annotation.get("language", "en"),
                "files": [
                    {
                        "filename": f["filename"],
                        "type": f.get("type", "unknown"),
                        "dir": f.get("dir", ""),
                    }
                    for f in files
                ],
            }
            tasks.append(task_spec)
            full_tasks[task_id] = {
                **task_spec,
                "chat_id": item.get("chat_id", ""),
                "annotation": annotation,
            }

    return tasks, full_tasks


def resolve_attachment_path(task_id: str, filename: str) -> Optional[Path]:
    """Resolve the full path to an attachment file."""
    attachments_dir = _find_attachments_dir()
    if attachments_dir is None:
        return None
    path = attachments_dir / task_id / filename
    if path.exists():
        return path
    return None


def extract_json_from_response(text: str) -> Optional[str]:
    """
    Extract JSON from LLM response. Tries patterns in order:
    1. <json_output>...</json_output> tags
    2. ```json ... ``` fenced blocks
    3. ``` ... ``` generic fenced blocks
    4. Raw [...] array
    5. Raw {...} object

    Faithful to original MiroEval base_evaluator.py extraction logic.
    """
    if not text:
        return None

    # 1. <json_output> tags
    match = re.search(r"<json_output>\s*(.*?)\s*</json_output>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. ```json fenced blocks
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. ``` generic fenced blocks
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content.startswith(("[", "{")):
            return content

    # 4. Raw [...] array
    match = re.search(r"(\[[\s\S]*\])", text)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # 5. Raw {...} object
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def extract_json_from_analysis_output(text: str) -> Optional[str]:
    """
    Extract JSON from LLM output that may contain <analysis> and <json_output> sections.
    Tries <json_output> tags first, then falls back to extract_json_from_response.

    Faithful to original MiroEval base_evaluator.py.
    """
    if not text:
        return None

    # Try <json_output> tags first (most reliable)
    match = re.search(r"<json_output>\s*(.*?)\s*</json_output>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback to general extraction
    return extract_json_from_response(text)
