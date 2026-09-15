"""Regression tests for profile-scoped skills_tool path resolution."""

import hashlib
import importlib
import json
from pathlib import Path


def _write_skill(root: Path, category: str, name: str, description: str) -> Path:
    skill_dir = root / "skills" / category / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"---\n\n"
        f"# {name}\n\n"
        f"Loaded from {description}.\n",
        encoding="utf-8",
    )
    return skill_dir


def _reload_skills_tool(import_home: Path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(import_home))
    import tools.skills_tool as skills_tool

    return importlib.reload(skills_tool)


def test_skill_view_uses_live_profile_home_after_module_import(tmp_path, monkeypatch):
    """skill_view should not stay pinned to HERMES_HOME from import time."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "orchestrator"
    _write_skill(default_home, "autonomous-ai-agents", "default-only", "default home")
    profile_skill_dir = _write_skill(
        profile_home,
        "software-development",
        "kanban-orchestrator-operations",
        "orchestrator profile",
    )

    skills_tool = _reload_skills_tool(default_home, monkeypatch)
    assert skills_tool.SKILLS_DIR == default_home / "skills"

    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    result = json.loads(
        skills_tool.skill_view("kanban-orchestrator-operations", preprocess=False)
    )

    assert result["success"] is True
    assert result["name"] == "kanban-orchestrator-operations"
    assert Path(result["skill_dir"]) == profile_skill_dir
    assert "orchestrator profile" in result["content"]


def test_raw_skill_identity_uses_live_profile_home_after_module_import(tmp_path, monkeypatch):
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "orchestrator"
    default_skill_dir = _write_skill(
        default_home, "software-development", "same-skill", "default home"
    )
    profile_skill_dir = _write_skill(
        profile_home, "software-development", "same-skill", "orchestrator profile"
    )
    default_bytes = (default_skill_dir / "SKILL.md").read_bytes()
    profile_bytes = (profile_skill_dir / "SKILL.md").read_bytes()
    assert default_bytes != profile_bytes

    skills_tool = _reload_skills_tool(default_home, monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    result = json.loads(
        skills_tool.skill_view(
            "same-skill", preprocess=False, raw_identity=True, raw_text=True
        )
    )

    assert result["success"] is True
    assert result["raw_text"].encode("utf-8") == profile_bytes
    assert result["raw_files"][0]["sha256"] == hashlib.sha256(profile_bytes).hexdigest()
    assert result["raw_files"][0]["sha256"] != hashlib.sha256(default_bytes).hexdigest()


def test_explicit_skills_dir_monkeypatch_still_wins(tmp_path, monkeypatch):
    """Existing tests can still override tools.skills_tool.SKILLS_DIR directly."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "orchestrator"
    patched_root = tmp_path / "patched"
    patched_skill_dir = _write_skill(
        patched_root,
        "software-development",
        "patched-skill",
        "patched skills dir",
    )
    _write_skill(
        profile_home,
        "software-development",
        "profile-skill",
        "orchestrator profile",
    )

    skills_tool = _reload_skills_tool(default_home, monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", patched_root / "skills")

    result = json.loads(skills_tool.skill_view("patched-skill", preprocess=False))

    assert result["success"] is True
    assert Path(result["skill_dir"]) == patched_skill_dir
