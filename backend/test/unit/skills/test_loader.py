"""Unit tests for ``backend.v.skills.loader`` and ``model``."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.v.skills.loader import _split_frontmatter, load_skills
from backend.v.skills.model import Skill


class TestSplitFrontmatter:
    def test_no_frontmatter(self) -> None:
        text = "# 标题\n\n纯 markdown 内容。"
        fm, body = _split_frontmatter(text)
        assert fm == {}
        assert body == text

    def test_minimal_frontmatter(self) -> None:
        text = "---\nname: x\n---\n\n# Body\n"
        fm, body = _split_frontmatter(text)
        assert fm == {"name": "x"}
        assert body.startswith("# Body")

    def test_frontmatter_with_lists(self) -> None:
        text = "---\nname: refund\nintents:\n  - 退款\n  - refund\npriority: 5\n---\n\nbody"
        fm, body = _split_frontmatter(text)
        assert fm["name"] == "refund"
        assert fm["intents"] == ["退款", "refund"]
        assert fm["priority"] == 5
        assert body == "body"

    def test_unterminated_frontmatter(self) -> None:
        # Opening --- but no closing --- → treat as no frontmatter.
        text = "---\nname: x\nthis was supposed to close but didn't"
        fm, body = _split_frontmatter(text)
        assert fm == {}
        assert body == text

    def test_invalid_yaml_falls_back(self) -> None:
        text = "---\n: : not valid : yaml\n---\nbody"
        fm, _ = _split_frontmatter(text)
        assert fm == {}


class TestSkillModel:
    def test_default_kind_markdown(self) -> None:
        s = Skill(name="x", body="content")
        assert s.kind == "markdown"
        assert s.priority == 3
        assert s.intents == []
        assert s.channels is None

    def test_priority_bounded(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            Skill(name="x", body="b", priority=0)
        with pytest.raises(Exception):  # noqa: B017
            Skill(name="x", body="b", priority=99)

    def test_applies_to_channel_default(self) -> None:
        s = Skill(name="x", body="b")
        assert s.applies_to_channel("wecom")
        assert s.applies_to_channel("feishu")

    def test_applies_to_channel_restricted(self) -> None:
        s = Skill(name="x", body="b", channels=["wecom"])
        assert s.applies_to_channel("wecom")
        assert not s.applies_to_channel("feishu")


class TestLoadSkills:
    def test_empty_directory(self, tmp_path: Path) -> None:
        # No .md files.
        skills = load_skills(tmp_path)
        assert skills == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        skills = load_skills(tmp_path / "missing")
        assert skills == []

    def test_empty_string_path(self) -> None:
        skills = load_skills("")
        assert skills == []

    def test_single_skill(self, tmp_path: Path) -> None:
        (tmp_path / "refund.md").write_text(
            "---\nname: refund\nintents: [退款, refund]\npriority: 5\n---\n\n"
            "# 退款流程\n\n1. 询问订单号\n2. ...",
            encoding="utf-8",
        )
        skills = load_skills(tmp_path)
        assert len(skills) == 1
        s = skills[0]
        assert s.name == "refund"
        assert "退款" in s.intents
        assert s.priority == 5
        assert "询问订单号" in s.body

    def test_recursive_walk(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("---\nname: a\n---\n\nbody A", encoding="utf-8")
        sub = tmp_path / "logistics"
        sub.mkdir()
        (sub / "delivery.md").write_text("---\nname: delivery\n---\n\nbody D", encoding="utf-8")
        skills = load_skills(tmp_path)
        names = sorted(s.name for s in skills)
        assert names == ["a", "delivery"]

    def test_no_frontmatter_uses_filename_stem(self, tmp_path: Path) -> None:
        (tmp_path / "general.md").write_text("just markdown", encoding="utf-8")
        skills = load_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "general"
        assert skills[0].intents == []

    def test_empty_body_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "empty.md").write_text("---\nname: empty\n---\n\n   \n", encoding="utf-8")
        skills = load_skills(tmp_path)
        assert skills == []

    def test_invalid_priority_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "bad.md").write_text(
            "---\nname: bad\npriority: 999\n---\n\nbody", encoding="utf-8"
        )
        skills = load_skills(tmp_path)
        assert skills == []  # Validation failed; logged + skipped.

    def test_source_path_recorded(self, tmp_path: Path) -> None:
        path = tmp_path / "x.md"
        path.write_text("---\nname: x\n---\n\nbody", encoding="utf-8")
        skills = load_skills(tmp_path)
        assert str(path) in skills[0].source_path
