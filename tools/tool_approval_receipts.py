"""Domain owner for durable intrinsic tool approval receipts.

Receipts contain only bounded identities and canonical SHA-256 bindings. Raw tool
arguments and approval previews are canonicalized in memory and never reach state.db.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from hermes_constants import get_hermes_home, hermes_home_key, profile_name_for_home
from hermes_state_approval_receipts import IntrinsicApprovalReceiptCollisionError
from tools.registry import (
    ApprovalPreviewError,
    _canonical_json_mapping,
    approval_preview_hash,
)

if TYPE_CHECKING:
    from hermes_state import SessionDB


_RECEIPT_TTL_SECONDS = 15 * 60
_ID_MAX_BYTES = 1024
_RECEIPT_ID_MAX_BYTES = 128
_PROFILE_HOME_MAX_BYTES = 4096
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class IntrinsicApprovalReceiptValidationError(ValueError):
    """Receipt input failed closed before any database write."""


def _bounded_identity(value: Any, field_name: str, *, max_bytes: int = _ID_MAX_BYTES) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntrinsicApprovalReceiptValidationError(f"{field_name} must be a non-empty string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise IntrinsicApprovalReceiptValidationError(f"{field_name} must be valid UTF-8") from None
    if size > max_bytes:
        raise IntrinsicApprovalReceiptValidationError(f"{field_name} exceeds the bounded identity limit")
    if any(not ch.isprintable() for ch in value):
        raise IntrinsicApprovalReceiptValidationError(
            f"{field_name} contains an invalid non-printable character"
        )
    return value


def _sha256_identity(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise IntrinsicApprovalReceiptValidationError(
            f"{field_name} must be exact lowercase 64-hex SHA-256"
        )
    return value


def validate_intrinsic_approval_identity(value: Any, field_name: str) -> str:
    """Validate one receipt-bound identity before any preview or prompt runs."""
    return _bounded_identity(value, field_name)


def validate_intrinsic_approval_scope_sha256(value: Any) -> str:
    """Validate the exact scope digest supplied by an intrinsic preview."""
    return _sha256_identity(value, "approval_scope_sha256")


def current_intrinsic_approval_profile_binding() -> tuple[str, str]:
    """Return the bounded active profile identity and canonical home key."""
    try:
        active_home = get_hermes_home().expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise IntrinsicApprovalReceiptValidationError(
            "active profile identity is unavailable"
        ) from None
    profile_home = _bounded_identity(
        hermes_home_key(active_home),
        "profile_home",
        max_bytes=_PROFILE_HOME_MAX_BYTES,
    )
    profile_identity = profile_name_for_home(profile_home) or profile_home
    return _bounded_identity(profile_identity, "profile_identity"), profile_home


def canonical_original_args_sha256(original_args: Mapping) -> str:
    """Hash detached canonical finite JSON mapping bytes for original tool args."""
    try:
        _detached, encoded = _canonical_json_mapping(
            original_args, label="original_args",
        )
    except ApprovalPreviewError:
        raise IntrinsicApprovalReceiptValidationError(
            "original_args must be a finite JSON-serializable mapping"
        ) from None
    return hashlib.sha256(encoded).hexdigest()


def _canonical_preview_sha256(preview: Mapping) -> str:
    try:
        return approval_preview_hash(preview)
    except ApprovalPreviewError:
        raise IntrinsicApprovalReceiptValidationError(
            "preview must be a bounded finite JSON-serializable mapping"
        ) from None


def _profile_binding(session_db: "SessionDB") -> tuple[str, str]:
    try:
        active_home = get_hermes_home().expanduser().resolve(strict=True)
        supplied_path = session_db.db_path.expanduser()
        supplied_parent = supplied_path.parent.resolve(strict=True)
        supplied_file = supplied_path.resolve(strict=True)
        lexical_parent = supplied_path.parent.absolute()
    except (OSError, RuntimeError, ValueError):
        raise IntrinsicApprovalReceiptValidationError(
            "receipt database must be the active profile canonical state.db"
        ) from None
    expected_file = active_home / "state.db"
    if (
        supplied_parent != active_home
        or supplied_file != expected_file
        or lexical_parent != active_home
    ):
        raise IntrinsicApprovalReceiptValidationError(
            "receipt database must be the active profile canonical state.db"
        )
    return current_intrinsic_approval_profile_binding()


def _open_current_profile_db() -> "SessionDB":
    from hermes_state import SessionDB

    return SessionDB(db_path=get_hermes_home().expanduser().resolve() / "state.db")


def _immutable(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(record))


def _clock_timestamp(clock: Callable[[], float]) -> float:
    try:
        timestamp = clock()
    except Exception:
        raise IntrinsicApprovalReceiptValidationError(
            "clock must return a finite UTC epoch timestamp"
        ) from None
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(timestamp)
        or timestamp < 0
    ):
        raise IntrinsicApprovalReceiptValidationError(
            "clock must return a finite non-negative UTC epoch timestamp"
        )
    return float(timestamp)


def create_intrinsic_approval_receipt(
    *,
    session_id: str,
    turn_id: str,
    tool_call_id: str,
    tool_name: str,
    original_args: Mapping,
    preview: Mapping,
    approval_scope_sha256: str,
    decision: str,
    session_db: "SessionDB | None" = None,
    clock: Callable[[], float] = time.time,
) -> Mapping[str, Any]:
    """Create one unique profile-scoped, append-only 15-minute decision receipt."""
    bounded_receipt_id = _bounded_identity(
        uuid.uuid4().hex,
        "receipt_id",
        max_bytes=_RECEIPT_ID_MAX_BYTES,
    )
    identities = {
        "session_id": _bounded_identity(session_id, "session_id"),
        "turn_id": _bounded_identity(turn_id, "turn_id"),
        "tool_call_id": _bounded_identity(tool_call_id, "tool_call_id"),
        "tool_name": _bounded_identity(tool_name, "tool_name"),
    }
    scope_hash = _sha256_identity(approval_scope_sha256, "approval_scope_sha256")
    if decision not in {"approved", "denied"}:
        raise IntrinsicApprovalReceiptValidationError(
            "decision must be the exact string 'approved' or 'denied'"
        )
    decided_at = _clock_timestamp(clock)
    expires_at = decided_at + float(_RECEIPT_TTL_SECONDS)
    if (
        not math.isfinite(expires_at)
        or expires_at <= decided_at
        or expires_at - decided_at != float(_RECEIPT_TTL_SECONDS)
    ):
        raise IntrinsicApprovalReceiptValidationError(
            "clock timestamp must preserve an exact 900-second receipt lifetime"
        )

    args_hash = canonical_original_args_sha256(original_args)
    preview_hash = _canonical_preview_sha256(preview)

    owned_db = session_db is None
    db = session_db or _open_current_profile_db()
    try:
        profile_identity, profile_home = _profile_binding(db)
        record = {
            "receipt_id": bounded_receipt_id,
            "profile_identity": profile_identity,
            "profile_home": profile_home,
            **identities,
            "original_args_sha256": args_hash,
            "preview_sha256": preview_hash,
            "approval_scope_sha256": scope_hash,
            "decision": decision,
            "decided_at": decided_at,
            "expires_at": expires_at,
            "consumed_by": None,
            "consumed_at": None,
        }
        stored = db._insert_intrinsic_approval_receipt(record)
        return _immutable(stored)
    finally:
        if owned_db:
            db.close()


def get_intrinsic_approval_receipt(
    receipt_id: str, *, session_db: "SessionDB | None" = None,
) -> Mapping[str, Any] | None:
    """Read one detached immutable receipt from the active or explicit profile store."""
    bounded_receipt_id = _bounded_identity(
        receipt_id, "receipt_id", max_bytes=_RECEIPT_ID_MAX_BYTES,
    )
    owned_db = session_db is None
    db = session_db or _open_current_profile_db()
    try:
        profile_identity, profile_home = _profile_binding(db)
        record = db._get_intrinsic_approval_receipt(bounded_receipt_id)
        if record is None:
            return None
        if (
            record.get("profile_identity") != profile_identity
            or record.get("profile_home") != profile_home
        ):
            raise IntrinsicApprovalReceiptValidationError(
                "receipt stored profile provenance does not match the active profile"
            )
        return _immutable(record)
    finally:
        if owned_db:
            db.close()


def verify_and_consume_intrinsic_approval_receipt(
    *,
    receipt_id: str,
    session_id: str,
    tool_name: str,
    approval_scope_sha256: str,
    consume_id: str,
    session_db: "SessionDB | None" = None,
    clock: Callable[[], float] = time.time,
) -> Mapping[str, Any]:
    """Atomically verify exact active-profile bindings and consume one receipt."""
    bounded_receipt_id = _bounded_identity(
        receipt_id, "receipt_id", max_bytes=_RECEIPT_ID_MAX_BYTES,
    )
    bounded_session_id = _bounded_identity(session_id, "session_id")
    bounded_tool_name = _bounded_identity(tool_name, "tool_name")
    bounded_scope = _sha256_identity(approval_scope_sha256, "approval_scope_sha256")
    bounded_consume_id = _bounded_identity(consume_id, "consume_id")

    owned_db = session_db is None
    db = session_db or _open_current_profile_db()
    try:
        profile_identity, profile_home = _profile_binding(db)
        result = db._verify_and_consume_intrinsic_approval_receipt(
            receipt_id=bounded_receipt_id,
            profile_identity=profile_identity,
            profile_home=profile_home,
            session_id=bounded_session_id,
            tool_name=bounded_tool_name,
            approval_scope_sha256=bounded_scope,
            consume_id=bounded_consume_id,
            clock=lambda: _clock_timestamp(clock),
        )
        if result is None:
            raise IntrinsicApprovalReceiptValidationError(
                "approval receipt verification failed"
            )
        return _immutable(result)
    finally:
        if owned_db:
            db.close()
