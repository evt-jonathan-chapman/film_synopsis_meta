import re

EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_DATA_ROWS = EXCEL_MAX_ROWS - 1  # leave 1 row for header


def safe_sheet_name(name: str, used: set[str]) -> str:
    """
    Excel sheet rules:
      - max 31 chars
      - cannot contain: : \\ / ? * [ ]
      - cannot be blank
      - must be unique in workbook
    """
    name = "" if name is None else str(name)
    name = re.sub(r"[:\\/?*\[\]]", " ", name).strip()
    name = re.sub(r"\s+", " ", name)
    if not name:
        name = "Sheet"

    base = name[:31]
    candidate = base
    i = 2
    while candidate in used:
        suffix = f"_{i}"
        candidate = (base[: 31 - len(suffix)] + suffix) if len(base) + len(suffix) > 31 else (base + suffix)
        i += 1

    used.add(candidate)
    return candidate


def safe_sheet_name_with_suffix(base_title: str, part: int, used: set[str]) -> str:
    """
    part=1 => use base title (no suffix) if available
    part>=2 => suffix as _2, _3, ...
    Ensures Excel-safe and unique.
    """
    base_title = "" if base_title is None else str(base_title)
    base_title = re.sub(r"[:\\/?*\[\]]", " ", base_title).strip()
    base_title = re.sub(r"\s+", " ", base_title) or "Sheet"

    suffix = "" if part == 1 else f"_{part}"
    # Reserve space for suffix within 31 chars
    trimmed = base_title[: 31 - len(suffix)]
    return safe_sheet_name(trimmed + suffix, used)
