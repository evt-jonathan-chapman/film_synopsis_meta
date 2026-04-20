from dataclasses import dataclass
from typing import Callable, Any
import re
import json


@dataclass
class ExtractionTask:
    name: str
    system_prompt: str
    temperature: float = 0.0
    max_tokens: int = 256
    top_p: float = 1.0
    repeat_penalty: float = 1.1
    # Optional post-processor: takes parsed JSON dict, returns cleaned dict
    postprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None


def extract_json(text: str) -> dict:
    """Extract the first JSON object from a model response.

    Raises:
        ValueError if no JSON object is found or it cannot be parsed.
    """
    text = text.strip()

    # Handle ```json ... ``` or ``` ... ``` wrappers
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            inner = inner.lstrip()
            if inner.lower().startswith("json"):
                inner = inner[4:].lstrip("\n\r ")
            text = inner.strip()

    # Fallback: grab first {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {repr(text[:200])}")

    json_str = match.group(0)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Re-wrap as ValueError with context so callers can handle uniformly
        snippet = json_str[:200]
        raise ValueError(
            f"Failed to parse JSON from model output: {e}. Snippet: {snippet!r}"
        ) from e


def flatten_extraction(task_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Flatten per-task payloads into a single film-level dict.

    task_results: {task_name: {key: value, ...}, ...}

    Returns:
        {key: value, ...} with minimal collision handling:
        - if a key appears in multiple tasks with the same value: keep it once
        - if it appears with different values: store subsequent ones as
          {f"{task_name}__{key}": value}
    """
    flat: dict[str, Any] = {}

    for task_name, payload in task_results.items():
        if not isinstance(payload, dict):
            continue

        for k, v in payload.items():
            if k not in flat:
                flat[k] = v
            else:
                # same value? ignore duplicate
                if flat[k] == v:
                    continue
                # different value: namespace it
                alt_key = f"{task_name}__{k}"
                # avoid overwriting an existing alt_key; crude but safe
                idx = 2
                while alt_key in flat:
                    alt_key = f"{task_name}__{k}__{idx}"
                    idx += 1
                flat[alt_key] = v

    return flat
