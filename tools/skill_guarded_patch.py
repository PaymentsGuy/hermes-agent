"""Validation, resolution, and exact execution for guarded skill patches."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from hermes_constants import get_hermes_home
from tools.skill_mutation_lock import skill_mutation_lock_held
from tools.skills_tool_plugin import _read_authorized_file_bytes
from utils import atomic_write_text

_FIELDS = frozenset(
    {
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
)
_STRING_FIELDS = _FIELDS - {"replace_all"}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SCOPE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_GUARDED_ONLY_FIELDS = _FIELDS - {
    "name", "action", "file_path", "old_string", "new_string", "replace_all"
}


class GuardedPatchValidationError(ValueError):
    """The closed guarded-patch operation contract was not satisfied."""


class GuardedPatchResolutionError(RuntimeError):
    """The guarded target could not be resolved to an executable exact state."""


def guarded_patch_failure(error_code: str, error_summary: str, **identity) -> dict:
    return {
        "success": False,
        "result_kind": "failed",
        "error_code": error_code,
        "error_summary": error_summary[:240],
        **identity,
    }


def has_guarded_patch_intent(operations: Any) -> bool:
    """Detect even a partial use of guarded-only operation fields."""
    candidates = operations if isinstance(operations, list) else [operations]
    return any(
        isinstance(operation, Mapping) and bool(set(operation) & _GUARDED_ONLY_FIELDS)
        for operation in candidates
    )


@dataclass(frozen=True)
class ValidatedGuardedPatch:
    name: str
    action: str
    file_path: str
    old_string: str
    new_string: str
    replace_all: bool
    expected_target_state: str
    expected_sha256: str
    expected_result_sha256: str
    authorization_scope_id: str
    authorization_scope_sha256: str
    sync_policy: str
    required_owner_class: str
    expected_profile_relative_skill_root: str


@dataclass(frozen=True)
class GuardedPatchTarget:
    """Immutable exact snapshot returned only for executable or idempotent states."""

    operation: ValidatedGuardedPatch
    resolution: str
    current_manifest: Mapping[str, Any]
    _target_path: Path = field(repr=False)
    _current_bytes: bytes = field(repr=False)
    _current_text: str = field(repr=False)
    _expected_result_bytes: bytes | None = field(repr=False)

    @property
    def target_path(self) -> Path:
        return self._target_path

    @property
    def current_bytes(self) -> bytes:
        return self._current_bytes

    @property
    def current_text(self) -> str:
        return self._current_text

    @property
    def expected_result_bytes(self) -> bytes | None:
        return self._expected_result_bytes


def _text(operation: Mapping[str, Any], field_name: str) -> str:
    value = operation[field_name]
    if not isinstance(value, str):
        raise GuardedPatchValidationError(f"{field_name} must be a string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise GuardedPatchValidationError(f"{field_name} must be strict UTF-8 encodable") from exc
    return value


def _validate_posix_parts(value: str, label: str) -> list[str]:
    if not value or value.startswith("/") or "\\" in value:
        raise GuardedPatchValidationError(f"{label} must be a normalized relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise GuardedPatchValidationError(f"{label} must be a normalized relative POSIX path")
    return parts


def validate_guarded_patch_operation(operation: Mapping[str, Any]) -> ValidatedGuardedPatch:
    """Validate the exact closed operation schema and detach it from caller state."""
    if not isinstance(operation, Mapping):
        raise GuardedPatchValidationError("guarded patch operation must be a mapping")
    if set(operation) != _FIELDS:
        raise GuardedPatchValidationError("guarded patch operation does not match the closed schema")

    values = {name: _text(operation, name) for name in _STRING_FIELDS}
    if operation["replace_all"] is not False:
        raise GuardedPatchValidationError("replace_all must be the literal false value")

    from tools.skill_manager_tool import _validate_name

    if error := _validate_name(values["name"]):
        raise GuardedPatchValidationError(error)
    constants = {
        "action": "patch",
        "expected_target_state": "present",
        "sync_policy": "suppress",
        "required_owner_class": "curator_managed",
    }
    for name, expected in constants.items():
        if values[name] != expected:
            raise GuardedPatchValidationError(f"{name} must equal {expected!r}")

    old_string, new_string = values["old_string"], values["new_string"]
    if not old_string or len(old_string) > 8000:
        raise GuardedPatchValidationError("old_string must contain 1 to 8000 Unicode characters")
    if len(new_string) > 12000:
        raise GuardedPatchValidationError("new_string must contain at most 12000 Unicode characters")

    for name in ("expected_sha256", "expected_result_sha256", "authorization_scope_sha256"):
        if _HASH_RE.fullmatch(values[name]) is None:
            raise GuardedPatchValidationError(f"{name} must be an exact lowercase SHA-256 hex digest")
    if _SCOPE_ID_RE.fullmatch(values["authorization_scope_id"]) is None:
        raise GuardedPatchValidationError("authorization_scope_id must use 1-128 safe printable ASCII characters")

    file_parts = _validate_posix_parts(values["file_path"], "file_path")
    if values["file_path"] != "SKILL.md" and not (
        len(file_parts) >= 2 and file_parts[0] == "references"
    ):
        raise GuardedPatchValidationError("file_path must be SKILL.md or strictly under references/")

    root_parts = _validate_posix_parts(
        values["expected_profile_relative_skill_root"],
        "expected_profile_relative_skill_root",
    )
    if len(root_parts) < 2 or root_parts[0] != "skills" or root_parts[-1] != values["name"]:
        raise GuardedPatchValidationError(
            "expected_profile_relative_skill_root must be skills/.../<name>"
        )

    return ValidatedGuardedPatch(
        **values,
        replace_all=False,
    )


def _find_plugin_skill(name: str) -> Path | None:
    from hermes_cli.plugins import get_plugin_manager

    return get_plugin_manager().find_plugin_skill(name)


def _reject_redirects(root: Path, descendant: Path) -> None:
    """Reject symlinks in root and each lexical component through descendant."""
    try:
        relative = descendant.relative_to(root)
    except ValueError as exc:
        raise GuardedPatchResolutionError("guarded target is outside the active profile skills root") from exc
    current = root
    for part in (None, *relative.parts):
        if part is not None:
            current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise GuardedPatchResolutionError("guarded target must exist as a regular file") from exc
        if stat.S_ISLNK(mode):
            raise GuardedPatchResolutionError("guarded target path contains a symlink")
        if current != descendant and not stat.S_ISDIR(mode):
            raise GuardedPatchResolutionError("guarded target path contains a non-directory component")


def _owner_skill_dir(operation: ValidatedGuardedPatch, skills_root: Path) -> Path:
    from tools import skill_usage

    name = operation.name
    try:
        for predicate, label in (
            (skill_usage.is_protected_builtin, "protected built-in"),
            (skill_usage.is_hub_installed, "hub-installed"),
            (skill_usage.is_bundled, "bundled"),
        ):
            if predicate(name):
                raise GuardedPatchResolutionError(f"guarded patch rejects {label} skills")
        if _find_plugin_skill(name) is not None:
            raise GuardedPatchResolutionError("guarded patch rejects plugin-provided skills")
        skill_dir = skill_usage._find_skill_dir(name)
    except GuardedPatchResolutionError:
        raise
    except Exception as exc:
        raise GuardedPatchResolutionError("guarded skill ownership lookup failed closed") from exc

    if skill_dir is None:
        declared = Path(get_hermes_home()) / operation.expected_profile_relative_skill_root
        if declared.is_symlink():
            raise GuardedPatchResolutionError("guarded skill root is a symlink")
        raise GuardedPatchResolutionError("skill does not exist locally in the active profile")

    skill_dir = Path(skill_dir)
    _reject_redirects(skills_root, skill_dir)
    try:
        skills_root_resolved = skills_root.resolve(strict=True)
        skill_dir_resolved = skill_dir.resolve(strict=True)
        relative_root = skill_dir_resolved.relative_to(skills_root_resolved)
    except ValueError as exc:
        raise GuardedPatchResolutionError("guarded skill root resolves outside active profile skills") from exc
    except OSError as exc:
        raise GuardedPatchResolutionError("guarded skill root could not be resolved") from exc

    actual_root = f"skills/{relative_root.as_posix()}"
    if actual_root != operation.expected_profile_relative_skill_root:
        raise GuardedPatchResolutionError("guarded skill root does not match the expected owner root")

    try:
        usage = skill_usage.load_usage()
        record = usage.get(name) if isinstance(usage, dict) else None
    except Exception as exc:
        raise GuardedPatchResolutionError("guarded skill ownership record is unavailable") from exc
    if not isinstance(record, dict) or not skill_usage._is_curator_managed_record(record):
        raise GuardedPatchResolutionError("guarded skill ownership is not curator-managed")
    if record.get("pinned") is not False:
        raise GuardedPatchResolutionError("guarded patch rejects pinned skills")
    return skill_dir


def resolve_guarded_patch_target(operation: ValidatedGuardedPatch) -> GuardedPatchTarget:
    """Resolve an exact read-only target state while the active profile lock is held."""
    if not isinstance(operation, ValidatedGuardedPatch):
        raise GuardedPatchResolutionError("guarded target resolution requires a validated operation")
    if not skill_mutation_lock_held():
        raise GuardedPatchResolutionError("active profile skill mutation lock is not held by this thread")

    try:
        home = Path(get_hermes_home()).expanduser().resolve(strict=True)
    except OSError as exc:
        raise GuardedPatchResolutionError("active profile identity is unavailable") from exc
    skills_root = home / "skills"
    skill_dir = _owner_skill_dir(operation, skills_root)
    target = skill_dir / operation.file_path
    _reject_redirects(skills_root, target)

    try:
        current_bytes = _read_authorized_file_bytes(target, skills_root)
    except (OSError, ValueError):
        raise GuardedPatchResolutionError("guarded target could not be read securely") from None
    current_sha256 = hashlib.sha256(current_bytes).hexdigest()
    try:
        current_text = current_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GuardedPatchResolutionError("guarded target is not strict UTF-8") from exc

    manifest = MappingProxyType(
        {
            "path": f"{operation.expected_profile_relative_skill_root}/{operation.file_path}",
            "state": "present",
            "byte_length": len(current_bytes),
            "sha256": current_sha256,
        }
    )
    if current_sha256 == operation.expected_result_sha256:
        return GuardedPatchTarget(
            operation,
            "already_applied",
            manifest,
            target,
            current_bytes,
            current_text,
            None,
        )
    if current_sha256 != operation.expected_sha256:
        raise GuardedPatchResolutionError("guarded target is stale")
    if current_text.count(operation.old_string) != 1:
        raise GuardedPatchResolutionError("old_string must occur exactly once in the current target")

    expected_result_bytes = current_text.replace(
        operation.old_string, operation.new_string, 1
    ).encode("utf-8", errors="strict")
    if hashlib.sha256(expected_result_bytes).hexdigest() != operation.expected_result_sha256:
        raise GuardedPatchResolutionError("computed patch result does not match expected_result_sha256")
    return GuardedPatchTarget(
        operation,
        "precondition_met",
        manifest,
        target,
        current_bytes,
        current_text,
        expected_result_bytes,
    )


def _result_identity(operation: ValidatedGuardedPatch, receipt_id: str) -> dict[str, Any]:
    return {
        "approval_receipt_id": receipt_id,
        "authorization_scope_id": operation.authorization_scope_id,
        "authorization_scope_sha256": operation.authorization_scope_sha256,
        "required_owner_class": operation.required_owner_class,
        "observed_owner_class": "curator_managed",
        "expected_profile_relative_skill_root": operation.expected_profile_relative_skill_root,
        "observed_profile_relative_skill_root": operation.expected_profile_relative_skill_root,
    }


def _success_result(
    operation: ValidatedGuardedPatch,
    receipt_id: str,
    result_kind: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "success": True,
        "result_kind": result_kind,
        **_result_identity(operation, receipt_id),
        "before_manifest": [dict(before)],
        "after_manifest": [dict(after)],
        "external_sync": "suppressed",
    }


def _rollback_original_target(
    target: Path,
    original_text: str,
    original_sha256: str,
    original_mode: int,
) -> bool:
    try:
        atomic_write_text(target, original_text, preserve_mode=True)
        skills_root = Path(get_hermes_home()).expanduser().resolve(strict=True) / "skills"
        restored = _read_authorized_file_bytes(target, skills_root)
        return (
            hashlib.sha256(restored).hexdigest() == original_sha256
            and stat.S_IMODE(target.stat().st_mode) == original_mode
        )
    except Exception:
        return False


def execute_guarded_patch(
    operation: ValidatedGuardedPatch,
    *,
    task_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Apply one exact patch under cooperative lock and transient dispatch authority."""
    if not skill_mutation_lock_held():
        return guarded_patch_failure(
            "approval_authority_denied", "Exact approved patch authority is unavailable."
        )
    from tools.intrinsic_approval import _current_exact_approved_skill_dispatch

    authority = _current_exact_approved_skill_dispatch()
    if authority is None:
        return guarded_patch_failure(
            "approval_authority_denied", "Exact approved patch authority is unavailable."
        )
    receipt, scope_hash = authority
    identity = _result_identity(operation, receipt.receipt_id)
    if scope_hash != operation.authorization_scope_sha256:
        return guarded_patch_failure(
            "approval_authority_denied",
            "Exact approved patch scope does not match the operation.",
            **identity,
        )

    try:
        initial = resolve_guarded_patch_target(operation)
        if initial.resolution == "already_applied":
            verified = resolve_guarded_patch_target(operation)
            if verified.resolution != "already_applied":
                raise GuardedPatchResolutionError("idempotent state did not remain exact")
            return _success_result(
                operation,
                receipt.receipt_id,
                "already_applied",
                initial.current_manifest,
                verified.current_manifest,
            )

    except GuardedPatchResolutionError:
        return guarded_patch_failure(
            "precondition_failed", "Exact guarded patch precondition was not satisfied.", **identity
        )

    target = initial.target_path
    skill_dir = target
    for _part in Path(operation.file_path).parts:
        skill_dir = skill_dir.parent
    original_mode = stat.S_IMODE(target.stat().st_mode)
    ledger_before = None
    with suppress(Exception):
        from tools import skill_ledger

        ledger_before = skill_ledger.capture_before(target, skill=operation.name)

    from tools import skill_manager_tool

    try:
        ready = resolve_guarded_patch_target(operation)
        if ready.resolution != "precondition_met" or ready.expected_result_bytes is None:
            raise GuardedPatchResolutionError("guarded patch precondition changed")
    except GuardedPatchResolutionError:
        return guarded_patch_failure(
            "precondition_failed", "Exact guarded patch precondition was not satisfied.", **identity
        )
    expected_result_text = ready.expected_result_bytes.decode("utf-8", errors="strict")
    try:
        write_error = skill_manager_tool._guarded_write(
            operation.name,
            skill_dir,
            target,
            "patch",
            operation.file_path,
            expected_result_text,
        )
    except Exception:
        return guarded_patch_failure(
            "transaction_failed", "Guarded patch transaction failed.", **identity
        )
    if write_error is not None:
        restored = _rollback_original_target(
            target,
            ready.current_text,
            operation.expected_sha256,
            original_mode,
        )
        return guarded_patch_failure(
            "transaction_failed" if restored else "rollback_failed",
            "Guarded patch transaction failed."
            if restored
            else "Guarded patch rollback could not be verified.",
            **identity,
        )

    try:
        committed = resolve_guarded_patch_target(operation)
        if (
            committed.resolution != "already_applied"
            or committed.current_manifest["sha256"] != operation.expected_result_sha256
        ):
            raise GuardedPatchResolutionError("guarded patch readback was not exact")
    except Exception:
        restored = _rollback_original_target(
            target,
            ready.current_text,
            operation.expected_sha256,
            original_mode,
        )
        return guarded_patch_failure(
            "transaction_failed" if restored else "rollback_failed",
            "Guarded patch transaction failed and the original target was restored."
            if restored
            else "Guarded patch rollback could not be verified.",
            **identity,
        )

    with suppress(Exception):
        from tools import skill_ledger

        skill_ledger.record_mutation(
            "patch",
            operation.name,
            before=ledger_before,
            after_root=target,
            evidence={
                "approval_receipt_id": receipt.receipt_id,
                "authorization_scope_id": operation.authorization_scope_id,
            },
        )
    with suppress(Exception):
        from tools.skill_usage import bump_patch

        bump_patch(
            operation.name, action="patch", task_id=task_id, session_id=session_id
        )
    with suppress(Exception):
        from agent.prompt_builder import clear_skills_system_prompt_cache

        clear_skills_system_prompt_cache(clear_snapshot=True)
    return _success_result(
        operation,
        receipt.receipt_id,
        "committed",
        ready.current_manifest,
        committed.current_manifest,
    )
