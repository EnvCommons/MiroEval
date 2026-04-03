import json
import re
from pathlib import Path
from typing import Optional


def load_tasks() -> tuple[list[dict], dict[str, dict]]:
    """
    Load MiroEval text queries.

    Tries /orwd_data/mirobench_text.json first (production),
    falls back to local data/mirobench_text.json (development).

    Returns:
        tasks: list of user-facing task specs (id, query, domain, language)
        full_tasks: dict keyed by id with full annotation data
    """
    candidates = [
        Path("/orwd_data/mirobench_text.json"),
        Path(__file__).parent / "data" / "mirobench_text.json",
    ]

    raw = None
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            break

    if raw is None:
        raise FileNotFoundError(
            f"mirobench_text.json not found. Searched: {[str(p) for p in candidates]}"
        )

    tasks = []
    full_tasks = {}

    for item in raw:
        task_id = str(item["id"])
        annotation = item.get("annotation", {})

        task_spec = {
            "id": task_id,
            "query": item["rewritten_query"],
            "domain": annotation.get("domain", "other"),
            "language": annotation.get("language", "en"),
        }
        tasks.append(task_spec)

        full_tasks[task_id] = {
            **task_spec,
            "chat_id": item.get("chat_id", ""),
            "annotation": annotation,
        }

    return tasks, full_tasks


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
