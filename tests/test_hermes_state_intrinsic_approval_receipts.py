"""SessionDB creates and additively restores the approval-receipt schema."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_state import SessionDB


_TABLE = "intrinsic_tool_approval_receipts"
_EXPECTED_COLUMNS = {
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
}


def _columns(conn) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info('{_TABLE}')")}


def test_fresh_session_db_has_receipt_table_and_lookup_indexes(tmp_path):
    db = SessionDB(db_path=tmp_path / "fresh" / "state.db")
    try:
        indexes = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
                (_TABLE,),
            ).fetchall()
        }
        assert _columns(db._conn) == _EXPECTED_COLUMNS
        assert {
            "idx_intrinsic_approval_receipts_session",
            "idx_intrinsic_approval_receipts_tool_call",
            "idx_intrinsic_approval_receipts_scope",
            "idx_intrinsic_approval_receipts_expiry",
        } <= indexes
    finally:
        db.close()


def test_existing_session_db_additively_restores_receipt_table(tmp_path):
    path = tmp_path / "existing" / "state.db"
    SessionDB(db_path=path).close()
    conn = sqlite3.connect(path)
    try:
        conn.execute(f"DROP TABLE {_TABLE}")
        conn.commit()
    finally:
        conn.close()

    reopened = SessionDB(db_path=path)
    try:
        assert _columns(reopened._conn) == _EXPECTED_COLUMNS
        assert reopened._conn.execute(
            f"SELECT COUNT(*) FROM {_TABLE}"
        ).fetchone()[0] == 0
    finally:
        reopened.close()


def test_receipt_decision_and_bindings_are_immutable_in_schema(tmp_path):
    db = SessionDB(db_path=tmp_path / "immutable" / "state.db")
    try:
        values = (
            "receipt", "default", str(tmp_path), "session", "turn", "call", "tool",
            "1" * 64, "2" * 64, "3" * 64, "approved", 100.0, 1000.0,
        )
        db._conn.execute(
            f"INSERT INTO {_TABLE} ("
            "receipt_id, profile_identity, profile_home, session_id, turn_id, "
            "tool_call_id, tool_name, original_args_sha256, preview_sha256, "
            "approval_scope_sha256, decision, decided_at, expires_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        db._conn.commit()

        for column, value in (("decision", "denied"), ("tool_name", "other-tool")):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                db._conn.execute(
                    f"UPDATE {_TABLE} SET {column} = ? WHERE receipt_id = 'receipt'",
                    (value,),
                )
        row = db._conn.execute(
            f"SELECT decision, tool_name FROM {_TABLE} WHERE receipt_id = 'receipt'"
        ).fetchone()
        assert tuple(row) == ("approved", "tool")
    finally:
        db.close()


def test_receipts_cannot_be_deleted_but_consumption_columns_remain_updateable(tmp_path):
    db = SessionDB(db_path=tmp_path / "append-only" / "state.db")
    try:
        values = (
            "receipt", "default", str(tmp_path), "session", "turn", "call", "tool",
            "1" * 64, "2" * 64, "3" * 64, "approved", 100.0, 1000.0,
        )
        db._conn.execute(
            f"INSERT INTO {_TABLE} ("
            "receipt_id, profile_identity, profile_home, session_id, turn_id, "
            "tool_call_id, tool_name, original_args_sha256, preview_sha256, "
            "approval_scope_sha256, decision, decided_at, expires_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        db._conn.execute(
            f"UPDATE {_TABLE} SET consumed_by = ?, consumed_at = ? WHERE receipt_id = ?",
            ("consumer", 200.0, "receipt"),
        )
        with pytest.raises(sqlite3.IntegrityError, match="cannot be deleted"):
            db._conn.execute(f"DELETE FROM {_TABLE} WHERE receipt_id = 'receipt'")
        assert tuple(db._conn.execute(
            f"SELECT consumed_by, consumed_at FROM {_TABLE} WHERE receipt_id = 'receipt'"
        ).fetchone()) == ("consumer", 200.0)
    finally:
        db.close()


def test_existing_database_restores_delete_guard_idempotently(tmp_path):
    path = tmp_path / "migration" / "state.db"
    SessionDB(db_path=path).close()
    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP TRIGGER IF EXISTS intrinsic_tool_approval_receipts_no_delete")
        conn.commit()
    finally:
        conn.close()

    for _ in range(2):
        SessionDB(db_path=path).close()
    conn = sqlite3.connect(path)
    try:
        triggers = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name=?",
            ("intrinsic_tool_approval_receipts_no_delete",),
        ).fetchone()[0]
        assert triggers == 1
    finally:
        conn.close()
