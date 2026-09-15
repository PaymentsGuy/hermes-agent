"""Persistence owner for append-only intrinsic tool approval receipts.

Plain mixin for :class:`hermes_state.SessionDB`; domain validation and hashing live
in ``tools.tool_approval_receipts`` so state storage never receives raw payloads.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable
from typing import Any, TypeGuard


_RECEIPT_COLUMNS = (
    "receipt_id",
    "profile_identity",
    "profile_home",
    "session_id",
    "turn_id",
    "tool_call_id",
    "tool_name",
    "original_args_sha256",
    "preview_sha256",
    "approval_scope_sha256",
    "decision",
    "decided_at",
    "expires_at",
    "consumed_by",
    "consumed_at",
)
_RECEIPT_SELECT = ", ".join(_RECEIPT_COLUMNS)
_CONSUMED_RESULT_COLUMNS = (
    "receipt_id",
    "decision",
    "decided_at",
    "expires_at",
    "consumed_by",
    "consumed_at",
)


class IntrinsicApprovalReceiptCollisionError(ValueError):
    """A generated receipt id already exists, so creation failed closed."""


def _valid_timestamp(value: Any) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


class SessionApprovalReceiptsMixin:
    """SQLite operations for intrinsic approval receipts."""

    def _insert_intrinsic_approval_receipt(self, record: dict[str, Any]) -> dict[str, Any]:
        """Atomically insert one new receipt; every collision fails closed."""

        def _insert(conn):
            placeholders = ", ".join("?" for _ in _RECEIPT_COLUMNS)
            try:
                conn.execute(
                    f"INSERT INTO intrinsic_tool_approval_receipts ({_RECEIPT_SELECT}) "
                    f"VALUES ({placeholders})",
                    tuple(record[name] for name in _RECEIPT_COLUMNS),
                )
            except sqlite3.IntegrityError as exc:
                existing = conn.execute(
                    f"SELECT {_RECEIPT_SELECT} FROM intrinsic_tool_approval_receipts "
                    "WHERE receipt_id = ?",
                    (record["receipt_id"],),
                ).fetchone()
                if existing is not None:
                    raise IntrinsicApprovalReceiptCollisionError(
                        "generated receipt_id already exists"
                    ) from None
                raise IntrinsicApprovalReceiptCollisionError(
                    "approval receipt insert violated its immutable schema"
                ) from exc
            stored = conn.execute(
                f"SELECT {_RECEIPT_SELECT} FROM intrinsic_tool_approval_receipts "
                "WHERE receipt_id = ?",
                (record["receipt_id"],),
            ).fetchone()
            return dict(stored)

        return self._execute_write(_insert)

    def _get_intrinsic_approval_receipt(self, receipt_id: str) -> dict[str, Any] | None:
        row = self._read_one(
            f"SELECT {_RECEIPT_SELECT} FROM intrinsic_tool_approval_receipts "
            "WHERE receipt_id = ?",
            (receipt_id,),
        )
        return dict(row) if row is not None else None

    def _verify_and_consume_intrinsic_approval_receipt(
        self,
        *,
        receipt_id: str,
        profile_identity: str,
        profile_home: str,
        session_id: str,
        tool_name: str,
        approval_scope_sha256: str,
        consume_id: str,
        clock: Callable[[], float],
    ) -> dict[str, Any] | None:
        """Compare and consume one receipt under the existing writer transaction."""

        def _consume(conn):
            now = clock()
            row = conn.execute(
                f"SELECT {_RECEIPT_SELECT} FROM intrinsic_tool_approval_receipts "
                "WHERE receipt_id = ?",
                (receipt_id,),
            ).fetchone()
            if row is None:
                return None
            record = dict(row)
            if any(
                record.get(field) != expected
                for field, expected in (
                    ("profile_identity", profile_identity),
                    ("profile_home", profile_home),
                    ("session_id", session_id),
                    ("tool_name", tool_name),
                    ("approval_scope_sha256", approval_scope_sha256),
                )
            ) or record.get("decision") != "approved":
                return None

            decided_at = record.get("decided_at")
            expires_at = record.get("expires_at")
            if (
                not _valid_timestamp(decided_at)
                or not _valid_timestamp(expires_at)
                or expires_at - decided_at != 900
            ):
                return None

            consumed_by = record.get("consumed_by")
            consumed_at = record.get("consumed_at")
            if consumed_by is not None or consumed_at is not None:
                if (
                    consumed_by != consume_id
                    or not _valid_timestamp(consumed_at)
                    or not decided_at <= consumed_at < expires_at
                ):
                    return None
                return {name: record[name] for name in _CONSUMED_RESULT_COLUMNS}

            if now < decided_at or now >= expires_at:
                return None
            updated = conn.execute(
                "UPDATE intrinsic_tool_approval_receipts "
                "SET consumed_by = ?, consumed_at = ? "
                "WHERE receipt_id = ? AND profile_identity = ? AND profile_home = ? "
                "AND session_id = ? AND tool_name = ? AND approval_scope_sha256 = ? "
                "AND decision = 'approved' AND expires_at > ? "
                "AND consumed_by IS NULL AND consumed_at IS NULL",
                (
                    consume_id,
                    now,
                    receipt_id,
                    profile_identity,
                    profile_home,
                    session_id,
                    tool_name,
                    approval_scope_sha256,
                    now,
                ),
            )
            if updated.rowcount != 1:
                return None
            record["consumed_by"] = consume_id
            record["consumed_at"] = now
            return {name: record[name] for name in _CONSUMED_RESULT_COLUMNS}

        return self._execute_write(_consume)
