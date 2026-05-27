"""Filesystem skill loader.

Walks a directory recursively; for every ``.md`` file, parses the YAML
frontmatter and the markdown body and produces a :class:`Skill`.

Frontmatter is optional — files without one are loaded as fallback
"general" skills with the file stem as name and no intents (so they
won't fire automatically; the agent would need to explicitly recall
them, which is a Phase-3 concern).

Malformed files are logged + skipped rather than raising; one bad
template shouldn't take down the loader.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from backend.v.skills.model import Skill
from backend.v.utils.logging import get_logger

_log = get_logger("skills.loader")


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Return ``(frontmatter_dict, body)``.

    Accepts the standard ``---\\n...\\n---\\n<body>`` envelope; if no
    frontmatter is present, returns ``({}, text)``.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx == -1:
        return {}, text
    fm_text = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1 :]).lstrip("\n")
    try:
        fm = yaml.safe_load(fm_text) or {}
        if not isinstance(fm, dict):
            return {}, text
    except yaml.YAMLError as exc:
        _log.warning("skills.loader.frontmatter_parse_failed", error=str(exc))
        return {}, text
    return fm, body


def _parse_file(path: Path) -> Skill | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        _log.warning("skills.loader.read_failed", path=str(path), error=str(exc))
        return None
    fm, body = _split_frontmatter(raw)
    if not body.strip():
        _log.warning("skills.loader.empty_body", path=str(path))
        return None
    name = fm.get("name") or path.stem
    payload = {
        "kind": "markdown",
        "name": name,
        "description": fm.get("description", ""),
        "intents": list(fm.get("intents") or []),
        "priority": int(fm.get("priority", 3)),
        "channels": fm.get("channels"),
        "body": body,
        "source_path": str(path),
    }
    try:
        return Skill.model_validate(payload)
    except ValidationError as exc:
        _log.warning(
            "skills.loader.validation_failed",
            path=str(path),
            error=str(exc),
        )
        return None


def load_skills(directory: str | Path) -> list[Skill]:
    """Load every ``*.md`` file under ``directory`` (recursive).

    Returns the list of successfully-parsed skills, sorted by name for
    stable test ordering. Empty path, empty string, or non-existent path
    returns ``[]`` and logs once. Empty path is the documented way to
    disable the loader (see :class:`backend.v.configs.base.SkillSettings`).
    """
    if not directory:
        _log.info("skills.loader.disabled")
        return []
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        _log.info("skills.loader.no_directory", path=str(root))
        return []
    skills: list[Skill] = []
    for path in sorted(root.rglob("*.md")):
        skill = _parse_file(path)
        if skill is not None:
            skills.append(skill)
    _log.info("skills.loader.loaded", count=len(skills), root=str(root))
    return skills
