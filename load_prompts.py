from pathlib import Path

import yaml

from synopses.extraction import ExtractionTask


def load_tasks_from_yaml(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))

    tasks = {}
    for name, cfg in data["tasks"].items():
        if not cfg.get("enabled", True):
            continue

        task = ExtractionTask(
            name=name,
            system_prompt=cfg["system_prompt"],
            temperature=cfg.get("temperature", 0.0),
            top_p=cfg.get("top_p", 1.0),
            max_tokens=cfg.get("max_tokens", 256),
            repeat_penalty=cfg.get("repeat_penalty", 1.1),
            postprocess=cfg.get("postprocess"),
        )
        tasks[name] = task

    return tasks
