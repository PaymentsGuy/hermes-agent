"""Durable intrinsic tool approval receipts."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType

import pytest

from hermes_constants import (
    hermes_home_key,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_state import SessionDB
from tools.tool_approval_receipts import (
    IntrinsicApprovalReceiptCollisionError,
    IntrinsicApprovalReceiptValidationError,
    canonical_original_args_sha256,
    create_intrinsic_approval_receipt,
    get_intrinsic_approval_receipt,
)


def _create(db: SessionDB, **overrides):
    values = {
        "session_db": db,
        "session_id": "session-1",
        "turn_id": "turn-1",
        "tool_call_id": "call-1",
        "tool_name": "plugin_write",
        "original_args": {"z": "café", "a": {"b": 2, "a": 1}},
        "preview": {"summary": "Write café", "target": {"z": 2, "a": 1}},
        "approval_scope_sha256": "a" * 64,
        "decision": "approved",
        "clock": lambda: 1_700_000_000.25,
    }
    values.update(overrides)
    return create_intrinsic_approval_receipt(**values)


@pytest.fixture
def db(tmp_path):
    profile_home = tmp_path / "profile"
    token = set_hermes_home_override(profile_home)
    instance = SessionDB(db_path=profile_home / "state.db")
    try:
        yield instance
    finally:
        instance.close()
        reset_hermes_home_override(token)


def test_create_binds_exact_canonical_hashes_and_fifteen_minute_expiry(db):
    receipt = _create(db)
    expected_args = '{"a":{"a":1,"b":2},"z":"café"}'.encode()
    expected_preview = '{"summary":"Write café","target":{"a":1,"z":2}}'.encode()

    assert isinstance(receipt, MappingProxyType)
    assert receipt["original_args_sha256"] == hashlib.sha256(expected_args).hexdigest()
    assert receipt["preview_sha256"] == hashlib.sha256(expected_preview).hexdigest()
    assert receipt["approval_scope_sha256"] == "a" * 64
    assert receipt["decided_at"] == 1_700_000_000.25
    assert receipt["expires_at"] - receipt["decided_at"] == 900
    assert receipt["profile_home"] == hermes_home_key(db.db_path.parent)
    assert receipt["profile_identity"]
    assert receipt["receipt_id"]


def test_canonical_original_args_hash_is_detached_finite_json_mapping():
    first = {"z": [1, {"é": True}], "a": 2}
    second = {"a": 2, "z": [1, {"é": True}]}
    expected = hashlib.sha256(
        json.dumps(first, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()

    assert canonical_original_args_sha256(first) == expected
    assert canonical_original_args_sha256(second) == expected
    for malformed in (None, [], {"bad": float("nan")}, {"bad": object()}):
        with pytest.raises(IntrinsicApprovalReceiptValidationError):
            canonical_original_args_sha256(malformed)


def test_receipt_stores_identities_and_hashes_but_no_raw_payload_or_canary(db):
    canary = "RAW_SECRET_CANARY_1f68ab"
    receipt = _create(
        db,
        original_args={"secret": canary},
        preview={"summary": canary},
    )

    columns = {
        row[1]
        for row in db._conn.execute(
            "PRAGMA table_info('intrinsic_tool_approval_receipts')"
        ).fetchall()
    }
    dump = "\n".join(db._conn.iterdump())
    assert "original_args" not in columns
    assert "preview" not in columns
    assert canary not in dump
    assert receipt["original_args_sha256"] == hashlib.sha256(
        ('{"secret":"' + canary + '"}').encode()
    ).hexdigest()


@pytest.mark.parametrize("decision", ["approved", "denied"])
def test_approved_and_denied_decisions_are_storable_and_lookup_is_detached(db, decision):
    created = _create(db, decision=decision)
    loaded = get_intrinsic_approval_receipt(created["receipt_id"], session_db=db)

    assert loaded == created
    assert loaded is not created
    assert loaded["decision"] == decision
    with pytest.raises(TypeError):
        loaded["decision"] = "denied"
    assert get_intrinsic_approval_receipt("missing-receipt", session_db=db) is None


def test_lookup_without_explicit_db_uses_and_validates_current_profile(db):
    created = _create(db)

    assert get_intrinsic_approval_receipt(created["receipt_id"]) == created


def test_lookup_rejects_previous_profile_db_before_receipt_query(tmp_path):
    first_home = tmp_path / "profiles" / "first"
    first_token = set_hermes_home_override(first_home)
    first_db = SessionDB(db_path=first_home / "state.db")
    try:
        created = _create(first_db)
        statements = []
        first_db._conn.set_trace_callback(statements.append)
        second_home = tmp_path / "profiles" / "second"
        second_home.mkdir(parents=True)
        second_token = set_hermes_home_override(second_home)
        try:
            with pytest.raises(
                IntrinsicApprovalReceiptValidationError,
                match="active profile canonical state.db",
            ):
                get_intrinsic_approval_receipt(created["receipt_id"], session_db=first_db)
        finally:
            reset_hermes_home_override(second_token)
            first_db._conn.set_trace_callback(None)

        assert not any(
            "intrinsic_tool_approval_receipts" in statement.lower()
            for statement in statements
        )
    finally:
        first_db.close()
        reset_hermes_home_override(first_token)


@pytest.mark.parametrize("field", ["profile_identity", "profile_home"])
def test_lookup_rejects_tampered_stored_profile_provenance(db, field):
    created = _create(db)
    db._conn.execute("DROP TRIGGER intrinsic_tool_approval_receipts_immutable")
    db._conn.execute(
        f"UPDATE intrinsic_tool_approval_receipts SET {field} = ? WHERE receipt_id = ?",
        ("tampered-profile", created["receipt_id"]),
    )
    db._conn.commit()

    with pytest.raises(
        IntrinsicApprovalReceiptValidationError,
        match="stored profile provenance",
    ):
        get_intrinsic_approval_receipt(created["receipt_id"], session_db=db)


def test_receipt_id_is_not_a_creation_argument_and_rejection_writes_nothing(db):
    with pytest.raises(TypeError, match="receipt_id"):
        _create(db, receipt_id="caller-chosen")
    assert db._conn.execute(
        "SELECT COUNT(*) FROM intrinsic_tool_approval_receipts"
    ).fetchone()[0] == 0


def test_generated_uuid_collision_fails_closed(db, monkeypatch):
    fixed_uuid = type("FixedUUID", (), {"hex": "1" * 32})()
    monkeypatch.setattr("tools.tool_approval_receipts.uuid.uuid4", lambda: fixed_uuid)

    first = _create(db)
    with pytest.raises(IntrinsicApprovalReceiptCollisionError):
        _create(db)

    assert first["receipt_id"] == "1" * 32
    assert db._conn.execute(
        "SELECT COUNT(*) FROM intrinsic_tool_approval_receipts"
    ).fetchone()[0] == 1


def test_named_profile_contexts_bind_only_their_canonical_state_db(tmp_path):
    original_env = dict(os.environ)
    homes = [tmp_path / "profiles" / name for name in ("first", "second")]
    receipts = []
    for number, home in enumerate(homes):
        token = set_hermes_home_override(home)
        db = SessionDB(db_path=home / "state.db")
        try:
            receipt = _create(db, session_id=f"session-{number}")
            assert receipt["profile_identity"] == home.name
            assert receipt["profile_home"] == hermes_home_key(home)
            assert get_intrinsic_approval_receipt(receipt["receipt_id"], session_db=db) == receipt
            receipts.append(receipt)
        finally:
            db.close()
            reset_hermes_home_override(token)

    assert receipts[0]["profile_home"] != receipts[1]["profile_home"]
    for index, home in enumerate(homes):
        token = set_hermes_home_override(home)
        try:
            assert get_intrinsic_approval_receipt(receipts[index]["receipt_id"]) == receipts[index]
            assert get_intrinsic_approval_receipt(receipts[1 - index]["receipt_id"]) is None
        finally:
            reset_hermes_home_override(token)
    assert os.environ == original_env


@pytest.mark.parametrize("kind", ["foreign", "alternate", "directory_symlink", "file_symlink"])
def test_noncanonical_profile_database_is_rejected_without_receipt_rows(tmp_path, kind):
    active_home = tmp_path / "active"
    active_home.mkdir()
    token = set_hermes_home_override(active_home)
    foreign_path = tmp_path / "foreign" / "state.db"
    if kind == "foreign":
        path = foreign_path
    elif kind == "alternate":
        path = active_home / "alternate.db"
    elif kind == "directory_symlink":
        alias = tmp_path / "active-alias"
        alias.symlink_to(active_home, target_is_directory=True)
        path = alias / "state.db"
    else:
        foreign = SessionDB(db_path=foreign_path)
        foreign.close()
        (active_home / "state.db").symlink_to(foreign_path)
        path = active_home / "state.db"

    db = SessionDB(db_path=path)
    try:
        with pytest.raises(IntrinsicApprovalReceiptValidationError, match="active profile"):
            _create(db)
        with pytest.raises(IntrinsicApprovalReceiptValidationError, match="active profile"):
            get_intrinsic_approval_receipt("missing-receipt", session_db=db)
        assert db._conn.execute(
            "SELECT COUNT(*) FROM intrinsic_tool_approval_receipts"
        ).fetchone()[0] == 0
    finally:
        db.close()
        reset_hermes_home_override(token)


def test_concurrent_unique_inserts_share_wal_without_loss(tmp_path):
    home = tmp_path / "profile"
    token = set_hermes_home_override(home)
    path = home / "state.db"
    databases = [SessionDB(db_path=path) for _ in range(3)]
    try:
        def insert(number: int):
            thread_token = set_hermes_home_override(home)
            try:
                return _create(
                    databases[number % len(databases)],
                    tool_call_id=f"call-{number}",
                    approval_scope_sha256=f"{number:064x}",
                )
            finally:
                reset_hermes_home_override(thread_token)

        with ThreadPoolExecutor(max_workers=6) as pool:
            receipts = list(pool.map(insert, range(18)))

        assert len({item["receipt_id"] for item in receipts}) == 18
        stored = databases[0]._conn.execute(
            "SELECT COUNT(*) FROM intrinsic_tool_approval_receipts"
        ).fetchone()[0]
        assert stored == 18
    finally:
        for instance in databases:
            instance.close()
        reset_hermes_home_override(token)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"session_id": ""}, "session_id"),
        ({"turn_id": " "}, "turn_id"),
        ({"tool_call_id": "x" * 1025}, "tool_call_id"),
        ({"tool_name": 7}, "tool_name"),
        ({"approval_scope_sha256": "A" * 64}, "approval_scope_sha256"),
        ({"approval_scope_sha256": "a" * 63}, "approval_scope_sha256"),
        ({"decision": "allow"}, "decision"),
        ({"clock": lambda: float("nan")}, "clock"),
    ],
)
def test_malformed_fields_are_rejected_before_write(db, override, message):
    before = db._conn.execute(
        "SELECT COUNT(*) FROM intrinsic_tool_approval_receipts"
    ).fetchone()[0]
    with pytest.raises(IntrinsicApprovalReceiptValidationError, match=message):
        _create(db, **override)
    after = db._conn.execute(
        "SELECT COUNT(*) FROM intrinsic_tool_approval_receipts"
    ).fetchone()[0]
    assert after == before


@pytest.mark.parametrize("field", ["session_id", "turn_id", "tool_call_id", "tool_name"])
@pytest.mark.parametrize("control", ["\n", "\t", "\x1b", "\x7f", "\x85"])
def test_all_identity_control_characters_are_rejected_before_write(db, field, control):
    with pytest.raises(IntrinsicApprovalReceiptValidationError, match=field):
        _create(db, **{field: f"safe{control}unsafe"})
    assert db._conn.execute(
        "SELECT COUNT(*) FROM intrinsic_tool_approval_receipts"
    ).fetchone()[0] == 0


def test_profile_identity_control_character_is_rejected_before_write(tmp_path):
    home = tmp_path / "profiles" / "bad\nprofile"
    token = set_hermes_home_override(home)
    db = SessionDB(db_path=home / "state.db")
    try:
        with pytest.raises(IntrinsicApprovalReceiptValidationError, match="profile"):
            _create(db)
        assert db._conn.execute(
            "SELECT COUNT(*) FROM intrinsic_tool_approval_receipts"
        ).fetchone()[0] == 0
    finally:
        db.close()
        reset_hermes_home_override(token)


@pytest.mark.parametrize("control", ["\n", "\t", "\x1b", "\x7f", "\x85"])
def test_receipt_lookup_rejects_control_characters(db, control):
    with pytest.raises(IntrinsicApprovalReceiptValidationError, match="receipt_id"):
        get_intrinsic_approval_receipt(f"receipt{control}id", session_db=db)


def test_huge_finite_timestamp_that_loses_ttl_precision_is_rejected_before_write(db):
    with pytest.raises(IntrinsicApprovalReceiptValidationError, match="900"):
        _create(db, clock=lambda: 1e308)
    assert db._conn.execute(
        "SELECT COUNT(*) FROM intrinsic_tool_approval_receipts"
    ).fetchone()[0] == 0
