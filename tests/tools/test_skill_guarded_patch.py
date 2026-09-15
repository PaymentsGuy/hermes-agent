"""Closed validation and read-only resolution for guarded skill patches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


FIELDS = {
    "name",
    "action",
    "file_path",
    "old_string",
    "new_string",
    "replace_all",
    "expected_target_state",
    "expected_sha256",
    "expected_result_sha256",
    "authorization_scope_id",
    "authorization_scope_sha256",
    "sync_policy",
    "required_owner_class",
    "expected_profile_relative_skill_root",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _operation(before: bytes = b"before\r\n", after: bytes = b"after\r\n", **overrides):
    operation = {
        "name": "managed-skill",
        "action": "patch",
        "file_path": "SKILL.md",
        "old_string": "before",
        "new_string": "after",
        "replace_all": False,
        "expected_target_state": "present",
        "expected_sha256": _sha(before),
        "expected_result_sha256": _sha(after),
        "authorization_scope_id": "review.scope-1:patch",
        "authorization_scope_sha256": "a" * 64,
        "sync_policy": "suppress",
        "required_owner_class": "curator_managed",
        "expected_profile_relative_skill_root": "skills/managed-skill",
    }
    operation.update(overrides)
    return operation


@pytest.fixture
def guarded_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    skill = skills / "managed-skill"
    skill.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (skill / "SKILL.md").write_bytes(b"before\r\n")
    (skills / ".usage.json").write_text(
        json.dumps({"managed-skill": {"created_by": "agent", "pinned": False}}),
        encoding="utf-8",
    )
    return home


def test_validation_returns_detached_frozen_value():
    from tools.skill_guarded_patch import validate_guarded_patch_operation

    source = _operation()
    validated = validate_guarded_patch_operation(source)
    source["name"] = "changed"

    assert validated.name == "managed-skill"
    with pytest.raises(FrozenInstanceError):
        validated.name = "changed"


def test_validation_accepts_exact_text_limits_and_normal_unicode():
    from tools.skill_guarded_patch import validate_guarded_patch_operation

    validated = validate_guarded_patch_operation(
        _operation(old_string="☃" * 8000, new_string="e\u0301" * 6000)
    )

    assert len(validated.old_string) == 8000
    assert len(validated.new_string) == 12000


@pytest.mark.parametrize("field", sorted(FIELDS))
def test_validation_rejects_every_missing_field(field):
    from tools.skill_guarded_patch import GuardedPatchValidationError, validate_guarded_patch_operation

    operation = _operation()
    del operation[field]
    with pytest.raises(GuardedPatchValidationError, match="schema"):
        validate_guarded_patch_operation(operation)


def test_validation_rejects_unknown_field_and_non_mapping():
    from tools.skill_guarded_patch import GuardedPatchValidationError, validate_guarded_patch_operation

    with pytest.raises(GuardedPatchValidationError, match="mapping"):
        validate_guarded_patch_operation([])
    with pytest.raises(GuardedPatchValidationError, match="schema"):
        validate_guarded_patch_operation({**_operation(), "extra": True})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action", "edit"),
        ("replace_all", True),
        ("replace_all", 0),
        ("expected_target_state", "missing"),
        ("sync_policy", "push"),
        ("required_owner_class", "user"),
        ("name", "Uppercase"),
        ("name", "x" * 65),
        ("old_string", ""),
        ("old_string", "x" * 8001),
        ("new_string", "x" * 12001),
        ("old_string", "\ud800"),
        ("new_string", "\udfff"),
        ("expected_sha256", "A" * 64),
        ("expected_result_sha256", "0" * 63),
        ("authorization_scope_sha256", "g" * 64),
        ("authorization_scope_id", ""),
        ("authorization_scope_id", "x" * 129),
        ("authorization_scope_id", "has space"),
        ("authorization_scope_id", "snowman-☃"),
    ],
)
def test_validation_rejects_closed_constants_types_limits_and_unicode(field, value):
    from tools.skill_guarded_patch import GuardedPatchValidationError, validate_guarded_patch_operation

    with pytest.raises(GuardedPatchValidationError):
        validate_guarded_patch_operation(_operation(**{field: value}))


@pytest.mark.parametrize(
    "field",
    [
        "name",
        "action",
        "file_path",
        "old_string",
        "new_string",
        "expected_target_state",
        "expected_sha256",
        "expected_result_sha256",
        "authorization_scope_id",
        "authorization_scope_sha256",
        "sync_policy",
        "required_owner_class",
        "expected_profile_relative_skill_root",
    ],
)
def test_validation_rejects_non_string_fields(field):
    from tools.skill_guarded_patch import GuardedPatchValidationError, validate_guarded_patch_operation

    with pytest.raises(GuardedPatchValidationError):
        validate_guarded_patch_operation(_operation(**{field: 1}))


@pytest.mark.parametrize(
    "path",
    [
        "references/note.md",
        "references/nested/note.md",
        "references/é.md",
    ],
)
def test_validation_accepts_normalized_reference_paths(path):
    from tools.skill_guarded_patch import validate_guarded_patch_operation

    assert validate_guarded_patch_operation(_operation(file_path=path)).file_path == path


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/references/a.md",
        "references",
        "references/",
        "references//a.md",
        "references/./a.md",
        "references/../a.md",
        "references\\a.md",
        "scripts/a.py",
        "SKILL.md/child",
    ],
)
def test_validation_rejects_noncanonical_or_out_of_scope_target_paths(path):
    from tools.skill_guarded_patch import GuardedPatchValidationError, validate_guarded_patch_operation

    with pytest.raises(GuardedPatchValidationError):
        validate_guarded_patch_operation(_operation(file_path=path))


@pytest.mark.parametrize(
    "root",
    [
        "managed-skill",
        "/skills/managed-skill",
        "skills//managed-skill",
        "skills/./managed-skill",
        "skills/../managed-skill",
        "skills\\managed-skill",
        "skills/other",
    ],
)
def test_validation_rejects_invalid_expected_owner_root(root):
    from tools.skill_guarded_patch import GuardedPatchValidationError, validate_guarded_patch_operation

    with pytest.raises(GuardedPatchValidationError):
        validate_guarded_patch_operation(_operation(expected_profile_relative_skill_root=root))


def _resolve(operation):
    from tools.skill_guarded_patch import resolve_guarded_patch_target, validate_guarded_patch_operation

    return resolve_guarded_patch_target(validate_guarded_patch_operation(operation))


def test_resolution_requires_current_thread_lock(guarded_home):
    from tools.skill_guarded_patch import GuardedPatchResolutionError

    with pytest.raises(GuardedPatchResolutionError, match="lock"):
        _resolve(_operation())


def test_resolution_fails_if_active_profile_drifts_inside_lock(guarded_home, monkeypatch):
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    with skill_mutation_lock():
        monkeypatch.setenv("HERMES_HOME", str(guarded_home.parent / "other"))
        with pytest.raises(GuardedPatchResolutionError, match="profile"):
            _resolve(_operation())


def test_precondition_resolution_is_exact_immutable_and_performs_no_writes(guarded_home):
    from tools.skill_mutation_lock import skill_mutation_lock

    target = guarded_home / "skills" / "managed-skill" / "SKILL.md"
    usage = guarded_home / "skills" / ".usage.json"
    before_target = target.read_bytes()
    before_usage = usage.read_bytes()

    with skill_mutation_lock():
        resolved = _resolve(_operation())

    assert resolved.resolution == "precondition_met"
    assert dict(resolved.current_manifest) == {
        "path": "skills/managed-skill/SKILL.md",
        "state": "present",
        "byte_length": len(before_target),
        "sha256": _sha(before_target),
    }
    with pytest.raises(TypeError):
        resolved.current_manifest["state"] = "changed"
    assert resolved.current_bytes == before_target
    assert resolved.current_text == "before\r\n"
    assert resolved.expected_result_bytes == b"after\r\n"
    assert target.read_bytes() == before_target
    assert usage.read_bytes() == before_usage


def test_resolution_opens_the_target_snapshot_once(guarded_home, monkeypatch):
    from tools import skills_tool_plugin
    from tools.skill_mutation_lock import skill_mutation_lock

    target = guarded_home / "skills" / "managed-skill" / "SKILL.md"
    real_open = skills_tool_plugin.os.open
    opens = []

    def counted_open(path, flags):
        if Path(path) == target:
            opens.append(Path(path))
        return real_open(path, flags)

    monkeypatch.setattr(skills_tool_plugin.os, "open", counted_open)
    with skill_mutation_lock():
        resolved = _resolve(_operation())

    assert resolved.current_bytes == b"before\r\n"
    assert opens == [target]


def test_resolution_fails_closed_if_parent_is_swapped_to_external_symlink_before_open(
    guarded_home, tmp_path, monkeypatch
):
    from tools import skills_tool_plugin
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    skill = guarded_home / "skills" / "managed-skill"
    references = skill / "references"
    references.mkdir()
    target = references / "target.md"
    target.write_bytes(b"inside\r\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / "target.md"
    outside_bytes = b"outside\r\n"
    outside_target.write_bytes(outside_bytes)
    usage = guarded_home / "skills" / ".usage.json"
    before_usage = usage.read_bytes()
    moved_references = skill / "references-before-swap"
    real_open = skills_tool_plugin.os.open
    seam_fired = False

    def swap_parent_before_open(path, flags):
        nonlocal seam_fired
        if Path(path) == target:
            seam_fired = True
            references.rename(moved_references)
            references.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags)

    monkeypatch.setattr(skills_tool_plugin.os, "open", swap_parent_before_open)
    operation = _operation(
        outside_bytes,
        b"after\r\n",
        file_path="references/target.md",
        old_string="outside",
        new_string="after",
    )

    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError) as error:
        _resolve(operation)

    assert seam_fired is True
    assert str(error.value) == "guarded target could not be read securely"
    assert (moved_references / "target.md").read_bytes() == b"inside\r\n"
    assert outside_target.read_bytes() == outside_bytes
    assert usage.read_bytes() == before_usage


def test_reference_target_preserves_crlf_unicode_and_no_final_newline(guarded_home):
    from tools.skill_mutation_lock import skill_mutation_lock

    target = guarded_home / "skills" / "managed-skill" / "references" / "exact.md"
    target.parent.mkdir()
    before = "α\r\ne\u0301 no-final".encode("utf-8")
    after = "α\r\né no-final".encode("utf-8")
    target.write_bytes(before)
    op = _operation(
        before,
        after,
        file_path="references/exact.md",
        old_string="e\u0301",
        new_string="é",
    )
    with skill_mutation_lock():
        resolved = _resolve(op)

    assert resolved.current_bytes == before
    assert resolved.expected_result_bytes == after
    assert resolved.current_manifest["byte_length"] == len(before)


@pytest.mark.parametrize("text", ["nothing here", "before and before"])
def test_resolution_rejects_zero_or_multiple_literal_occurrences(guarded_home, text):
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    raw = text.encode()
    target = guarded_home / "skills" / "managed-skill" / "SKILL.md"
    target.write_bytes(raw)
    op = _operation(raw, b"unused", expected_result_sha256="b" * 64)
    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError, match="exactly once"):
        _resolve(op)


def test_resolution_rejects_wrong_expected_result_hash(guarded_home):
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError, match="result"):
        _resolve(_operation(expected_result_sha256="b" * 64))


def test_already_applied_does_not_require_old_string_to_remain(guarded_home):
    from tools.skill_mutation_lock import skill_mutation_lock

    target = guarded_home / "skills" / "managed-skill" / "SKILL.md"
    target.write_bytes(b"after\r\n")
    with skill_mutation_lock():
        resolved = _resolve(_operation())

    assert resolved.resolution == "already_applied"
    assert resolved.current_bytes == b"after\r\n"
    assert resolved.expected_result_bytes is None


def test_stale_state_returns_no_target(guarded_home):
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    (guarded_home / "skills" / "managed-skill" / "SKILL.md").write_bytes(b"third state")
    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError, match="stale"):
        _resolve(_operation())


def test_invalid_utf8_is_rejected_without_rendering_content(guarded_home):
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    raw = b"secret-prefix\xffsecret-suffix"
    (guarded_home / "skills" / "managed-skill" / "SKILL.md").write_bytes(raw)
    op = _operation(raw, b"unused")
    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError) as error:
        _resolve(op)
    assert "secret" not in str(error.value)
    assert "UTF-8" in str(error.value)


@pytest.mark.parametrize(
    ("record", "match"),
    [
        ({"created_by": None, "pinned": False}, "curator-managed"),
        ({"agent_created": False, "pinned": False}, "curator-managed"),
        ({"created_by": "agent", "pinned": True}, "pinned"),
        ({"created_by": "agent", "pinned": "false"}, "pinned"),
    ],
)
def test_owner_record_must_be_exactly_managed_and_unpinned(guarded_home, record, match):
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    usage = guarded_home / "skills" / ".usage.json"
    usage.write_text(json.dumps({"managed-skill": record}), encoding="utf-8")
    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError, match=match):
        _resolve(_operation())


def test_legacy_canonical_agent_created_marker_is_accepted(guarded_home):
    from tools.skill_mutation_lock import skill_mutation_lock

    usage = guarded_home / "skills" / ".usage.json"
    usage.write_text(
        json.dumps({"managed-skill": {"agent_created": True, "pinned": False}}),
        encoding="utf-8",
    )
    with skill_mutation_lock():
        assert _resolve(_operation()).resolution == "precondition_met"


@pytest.mark.parametrize("usage_text", ["{}", "{broken", "[]"])
def test_missing_or_corrupt_usage_fails_closed(guarded_home, usage_text):
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    (guarded_home / "skills" / ".usage.json").write_text(usage_text, encoding="utf-8")
    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError, match="ownership"):
        _resolve(_operation())


@pytest.mark.parametrize(
    ("predicate", "label"),
    [
        ("is_protected_builtin", "protected"),
        ("is_hub_installed", "hub-installed"),
        ("is_bundled", "bundled"),
    ],
)
def test_authoritative_owner_predicates_deny_and_are_exercised(
    guarded_home, monkeypatch, predicate, label
):
    from tools import skill_usage
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    calls = []
    monkeypatch.setattr(skill_usage, predicate, lambda name: calls.append(name) or True)
    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError, match=label):
        _resolve(_operation())
    assert calls == ["managed-skill"]


def test_plugin_owner_predicate_denies_and_is_exercised(guarded_home, monkeypatch):
    from tools import skill_guarded_patch
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    calls = []
    monkeypatch.setattr(
        skill_guarded_patch,
        "_find_plugin_skill",
        lambda name: calls.append(name) or Path("plugin/SKILL.md"),
    )
    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError, match="plugin"):
        _resolve(_operation())
    assert calls == ["managed-skill"]


def test_missing_and_external_only_skills_are_rejected(guarded_home, monkeypatch):
    from tools import skill_usage
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    calls = []
    monkeypatch.setattr(skill_usage, "_find_skill_dir", lambda name: calls.append(name) or None)
    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError, match="active profile"):
        _resolve(_operation())
    assert calls == ["managed-skill"]


def test_local_owner_resolving_outside_active_skills_root_is_rejected(
    guarded_home, tmp_path, monkeypatch
):
    from tools import skill_usage
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    outside = tmp_path / "external" / "managed-skill"
    outside.mkdir(parents=True)
    (outside / "SKILL.md").write_bytes(b"before\r\n")
    monkeypatch.setattr(skill_usage, "_find_skill_dir", lambda _name: outside)
    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError, match="outside"):
        _resolve(_operation())


def test_wrong_actual_or_declared_owner_root_is_rejected(guarded_home):
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    nested = guarded_home / "skills" / "category"
    (nested / "managed-skill").parent.mkdir(exist_ok=True)
    (guarded_home / "skills" / "managed-skill").rename(nested / "managed-skill")
    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError, match="root"):
        _resolve(_operation())


@pytest.mark.parametrize("kind", ["root", "component", "file"])
def test_symlink_root_component_or_file_is_rejected(guarded_home, tmp_path, kind):
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    skill = guarded_home / "skills" / "managed-skill"
    if kind == "root":
        real = tmp_path / "real-skill"
        skill.rename(real)
        try:
            skill.symlink_to(real, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
        op = _operation()
    else:
        references = skill / "references"
        references.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "target.md").write_bytes(b"before\r\n")
        try:
            if kind == "component":
                (references / "linked").symlink_to(outside, target_is_directory=True)
                path = "references/linked/target.md"
            else:
                (references / "target.md").symlink_to(outside / "target.md")
                path = "references/target.md"
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
        op = _operation(file_path=path)

    with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError, match="symlink"):
        _resolve(op)


def test_missing_and_non_regular_targets_are_rejected(guarded_home):
    from tools.skill_guarded_patch import GuardedPatchResolutionError
    from tools.skill_mutation_lock import skill_mutation_lock

    for path, match in (
        ("references/missing.md", "regular file"),
        ("references/directory", "read securely"),
    ):
        if path.endswith("directory"):
            (guarded_home / "skills" / "managed-skill" / path).mkdir(parents=True)
        with skill_mutation_lock(), pytest.raises(GuardedPatchResolutionError, match=match):
            _resolve(_operation(file_path=path))


def _approved_guarded_dispatch(operation, monkeypatch, *, receipt_scope=None):
    from tools.intrinsic_approval import (
        IntrinsicApprovalReceiptContext,
        _ApprovedDispatchCapability,
        bind_intrinsic_approval_receipt_context,
        dispatch_intrinsic_approved_tool,
    )
    from tools.registry import ToolRegistry, _canonical_json_mapping
    from tools.skill_manager_tool import _skill_manage_from

    arguments, encoded = _canonical_json_mapping(
        {"operations": [operation]}, label="test guarded arguments"
    )
    registry = ToolRegistry()
    scope = registry.current_scope_key()
    registry.register(
        name="skill_manage",
        toolset="skills",
        schema={"name": "skill_manage", "parameters": {"type": "object"}},
        handler=lambda args, **kwargs: _skill_manage_from(args),
    )
    context = type("Context", (), {"_tool_registration_registry": registry})()
    monkeypatch.setattr(
        registry, "_plugin_context_active_in_scope", lambda supplied, supplied_scope: True
    )
    capability = _ApprovedDispatchCapability(
        registry=registry,
        scope=scope,
        tool_name="skill_manage",
        arguments_sha256=_sha(encoded),
        arguments_json=encoded,
        approval_scope_sha256=receipt_scope or operation["authorization_scope_sha256"],
    )
    receipt = IntrinsicApprovalReceiptContext(
        receipt_id="receipt-1",
        profile_identity="default",
        profile_home="/profile",
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-1",
        tool_name="review-tool",
        original_args_sha256="1" * 64,
        preview_sha256="2" * 64,
        approval_scope_sha256=receipt_scope or operation["authorization_scope_sha256"],
        decision="approved",
        decided_at=1.0,
        expires_at=2.0,
    )
    with bind_intrinsic_approval_receipt_context(receipt, capability):
        return json.loads(dispatch_intrinsic_approved_tool(context, "skill_manage", arguments))


def _track_guarded_side_effects(monkeypatch):
    from agent import prompt_builder
    from tools import skill_manager_tool, skill_usage

    calls = {"bump": [], "sync": [], "clear": []}
    monkeypatch.setattr(
        skill_usage,
        "bump_patch",
        lambda *args, **kwargs: calls["bump"].append((args, kwargs)),
    )
    monkeypatch.setattr(
        skill_manager_tool,
        "_maybe_debounced_sync_push",
        lambda *args, **kwargs: calls["sync"].append((args, kwargs)),
    )
    monkeypatch.setattr(
        prompt_builder,
        "clear_skills_system_prompt_cache",
        lambda *args, **kwargs: calls["clear"].append((args, kwargs)),
    )
    return calls


def test_exact_approved_commit_reports_identity_manifests_and_suppresses_sync(
    guarded_home, monkeypatch
):
    from tools import skill_manager_tool

    target = guarded_home / "skills" / "managed-skill" / "SKILL.md"
    target.chmod(0o640)
    calls = _track_guarded_side_effects(monkeypatch)
    writes = []
    original_write = skill_manager_tool._guarded_write

    def counted_write(*args, **kwargs):
        writes.append((args, kwargs))
        return original_write(*args, **kwargs)

    monkeypatch.setattr(skill_manager_tool, "_guarded_write", counted_write)

    result = _approved_guarded_dispatch(_operation(), monkeypatch)

    assert result == {
        "success": True,
        "result_kind": "committed",
        "approval_receipt_id": "receipt-1",
        "authorization_scope_id": "review.scope-1:patch",
        "authorization_scope_sha256": "a" * 64,
        "required_owner_class": "curator_managed",
        "observed_owner_class": "curator_managed",
        "expected_profile_relative_skill_root": "skills/managed-skill",
        "observed_profile_relative_skill_root": "skills/managed-skill",
        "before_manifest": [{
            "path": "skills/managed-skill/SKILL.md",
            "state": "present",
            "byte_length": len(b"before\r\n"),
            "sha256": _sha(b"before\r\n"),
        }],
        "after_manifest": [{
            "path": "skills/managed-skill/SKILL.md",
            "state": "present",
            "byte_length": len(b"after\r\n"),
            "sha256": _sha(b"after\r\n"),
        }],
        "external_sync": "suppressed",
    }
    assert target.read_bytes() == b"after\r\n"
    assert target.stat().st_mode & 0o777 == 0o640
    assert len(writes) == 1
    assert len(calls["bump"]) == 1
    assert len(calls["clear"]) == 1
    assert calls["sync"] == []


def test_rejected_security_scan_restores_exact_original_bytes_mode_and_no_side_effects(
    guarded_home, monkeypatch
):
    from tools import skill_manager_tool

    target = guarded_home / "skills" / "managed-skill" / "SKILL.md"
    original = target.read_bytes()
    target.chmod(0o640)
    calls = _track_guarded_side_effects(monkeypatch)
    monkeypatch.setattr(
        skill_manager_tool,
        "_security_scan_skill",
        lambda _skill_dir: "sensitive rejected scan detail",
    )

    result = _approved_guarded_dispatch(_operation(), monkeypatch)

    assert result["success"] is False
    assert result["result_kind"] == "failed"
    assert result["error_code"] == "transaction_failed"
    assert result["error_summary"] == "Guarded patch transaction failed."
    assert "sensitive rejected scan detail" not in json.dumps(result)
    assert target.read_bytes() == original
    assert target.stat().st_mode & 0o777 == 0o640
    assert calls == {"bump": [], "sync": [], "clear": []}


@pytest.mark.parametrize(
    "operation",
    [
        _operation(expected_sha256="b" * 64),
        _operation(expected_result_sha256="b" * 64),
    ],
    ids=["stale", "wrong-computed-result"],
)
def test_stale_or_wrong_result_has_no_write_bump_or_sync(
    guarded_home, monkeypatch, operation
):
    from tools import skill_manager_tool

    target = guarded_home / "skills" / "managed-skill" / "SKILL.md"
    calls = _track_guarded_side_effects(monkeypatch)
    writes = []
    monkeypatch.setattr(
        skill_manager_tool,
        "_guarded_write",
        lambda *args, **kwargs: writes.append(True),
    )

    result = _approved_guarded_dispatch(operation, monkeypatch)

    assert result["success"] is False
    assert result["result_kind"] == "failed"
    assert target.read_bytes() == b"before\r\n"
    assert writes == []
    assert calls == {"bump": [], "sync": [], "clear": []}


def test_already_applied_rechecks_without_write_bump_or_sync(guarded_home, monkeypatch):
    from tools import skill_manager_tool

    target = guarded_home / "skills" / "managed-skill" / "SKILL.md"
    target.write_bytes(b"after\r\n")
    calls = _track_guarded_side_effects(monkeypatch)
    monkeypatch.setattr(
        skill_manager_tool,
        "_guarded_write",
        lambda *args, **kwargs: pytest.fail("already-applied path wrote"),
    )

    result = _approved_guarded_dispatch(_operation(), monkeypatch)

    assert result["success"] is True
    assert result["result_kind"] == "already_applied"
    assert result["before_manifest"] == result["after_manifest"]
    assert result["external_sync"] == "suppressed"
    assert target.read_bytes() == b"after\r\n"
    assert calls == {"bump": [], "sync": [], "clear": []}


def test_direct_or_wrong_scope_guarded_dispatch_denies_without_mutation(
    guarded_home, monkeypatch
):
    from tools import skill_manager_tool

    target = guarded_home / "skills" / "managed-skill" / "SKILL.md"
    calls = _track_guarded_side_effects(monkeypatch)
    direct = json.loads(skill_manager_tool._skill_manage_from({"operations": [_operation()]}))
    wrong_scope = _approved_guarded_dispatch(
        _operation(authorization_scope_sha256="b" * 64),
        monkeypatch,
        receipt_scope="a" * 64,
    )

    assert direct["success"] is False
    assert wrong_scope["success"] is False
    assert target.read_bytes() == b"before\r\n"
    assert calls == {"bump": [], "sync": [], "clear": []}


def test_guarded_write_approval_stages_exact_batch_and_replay_without_authority_denies(
    guarded_home, monkeypatch
):
    from tools import skill_manager_tool, write_approval

    target = guarded_home / "skills" / "managed-skill" / "SKILL.md"
    operation = _operation()
    staged = []
    calls = _track_guarded_side_effects(monkeypatch)
    monkeypatch.setattr(
        write_approval,
        "evaluate_gate",
        lambda subsystem: write_approval.GateDecision(stage=True, message="staged"),
    )
    monkeypatch.setattr(
        write_approval,
        "stage_write",
        lambda subsystem, payload, **kwargs: staged.append((subsystem, payload, kwargs)) or {"id": "p1"},
    )

    first = _approved_guarded_dispatch(operation, monkeypatch)

    assert first["staged"] is True
    assert staged == [(
        write_approval.SKILLS,
        {"action": "batch", "operations": [operation]},
        {
            "summary": "batch(1 ops: patch) on managed-skill",
            "origin": write_approval.current_origin(),
        },
    )]
    assert target.read_bytes() == b"before\r\n"

    replay = json.loads(skill_manager_tool.apply_skill_pending(staged[0][1]))
    assert replay["success"] is False
    assert target.read_bytes() == b"before\r\n"
    assert calls == {"bump": [], "sync": [], "clear": []}


@pytest.mark.parametrize("rollback_write_fails", [False, True], ids=["restored", "rollback-failed"])
def test_post_write_readback_failure_restores_only_target_or_reports_bounded_failure(
    guarded_home, monkeypatch, rollback_write_fails
):
    from tools import skill_guarded_patch

    target = guarded_home / "skills" / "managed-skill" / "SKILL.md"
    target.chmod(0o640)
    calls = _track_guarded_side_effects(monkeypatch)
    original_resolve = skill_guarded_patch.resolve_guarded_patch_target
    resolve_calls = []

    def fail_readback(operation):
        resolve_calls.append(True)
        if len(resolve_calls) == 3:
            raise skill_guarded_patch.GuardedPatchResolutionError("injected readback failure")
        return original_resolve(operation)

    monkeypatch.setattr(skill_guarded_patch, "resolve_guarded_patch_target", fail_readback)
    if rollback_write_fails:
        real_atomic = skill_guarded_patch.atomic_write_text

        def fail_rollback(path, content, **kwargs):
            if content == "before\r\n":
                raise OSError("injected rollback failure")
            return real_atomic(path, content, **kwargs)

        monkeypatch.setattr(skill_guarded_patch, "atomic_write_text", fail_rollback)

    result = _approved_guarded_dispatch(_operation(), monkeypatch)

    assert result["success"] is False
    assert result["result_kind"] == "failed"
    assert result["error_code"] == (
        "rollback_failed" if rollback_write_fails else "transaction_failed"
    )
    if not rollback_write_fails:
        assert target.read_bytes() == b"before\r\n"
        assert target.stat().st_mode & 0o777 == 0o640
    assert calls == {"bump": [], "sync": [], "clear": []}


def test_partial_guarded_intent_fails_closed_instead_of_using_legacy_batch(
    guarded_home, monkeypatch
):
    from tools import skill_manager_tool

    target = guarded_home / "skills" / "managed-skill" / "SKILL.md"
    calls = _track_guarded_side_effects(monkeypatch)
    result = json.loads(skill_manager_tool._skill_manage_from({
        "operations": [{
            "name": "managed-skill",
            "action": "patch",
            "old_string": "before",
            "new_string": "after",
            "expected_sha256": _sha(b"before\r\n"),
        }]
    }))

    assert result["success"] is False
    assert result["result_kind"] == "failed"
    assert target.read_bytes() == b"before\r\n"
    assert calls == {"bump": [], "sync": [], "clear": []}


def test_ordinary_unguarded_manager_patch_keeps_normal_sync_side_effect(
    guarded_home, monkeypatch
):
    from tools import skill_manager_tool

    target = guarded_home / "skills" / "managed-skill" / "references" / "ordinary.md"
    target.parent.mkdir()
    target.write_bytes(b"before\r\n")
    calls = _track_guarded_side_effects(monkeypatch)
    result = json.loads(skill_manager_tool.skill_manage(
        action="patch",
        name="managed-skill",
        file_path="references/ordinary.md",
        old_string="before",
        new_string="after",
    ))

    assert result["success"] is True
    assert target.read_bytes() == b"after\n"
    assert len(calls["bump"]) == 1
    assert len(calls["sync"]) == 1
