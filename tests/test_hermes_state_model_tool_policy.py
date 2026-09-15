"""Existing SessionDB schemas receive model-tool policy columns additively."""

import sqlite3

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL


def test_existing_sessions_table_adds_policy_columns_without_binding_legacy_rows(tmp_path):
    path = tmp_path / "state.db"
    legacy_schema = SCHEMA_SQL.replace(
        "    model_tool_policy_version INTEGER,\n    model_tool_policy TEXT,\n", ""
    )
    conn = sqlite3.connect(path)
    conn.executescript(legacy_schema)
    conn.execute("INSERT INTO schema_version (version) VALUES (30)")
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
        ("legacy", "tui", 1.0),
    )
    conn.commit()
    conn.close()

    db = SessionDB(path)
    try:
        columns = {
            row[1] for row in db._conn.execute("PRAGMA table_info('sessions')").fetchall()
        }
        row = db.get_session("legacy")
    finally:
        db.close()

    assert {"model_tool_policy_version", "model_tool_policy"} <= columns
    assert row["model_tool_policy_version"] is None
    assert row["model_tool_policy"] is None
