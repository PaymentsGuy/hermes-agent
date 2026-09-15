"""Atomic verification and consumption of intrinsic approval receipts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import MappingProxyType

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from tools.tool_approval_receipts import (
    IntrinsicApprovalReceiptValidationError,
    create_intrinsic_approval_receipt,
    verify_and_consume_intrinsic_approval_receipt,
)

_SCOPE = "a" * 64
_FIELDS = {
    "receipt_id",
    "decision",
    "decided_at",
    "expires_at",
    "consumed_by",
    "consumed_at",
}


@pytest.fixture
def receipt_db(tmp_path):
    home = tmp_path / "profile"
    token = set_hermes_home_override(home)
    db = SessionDB(db_path=home / "state.db")
    try:
        yield home, db
    finally:
        db.close()
        reset_hermes_home_override(token)


def _create(db: SessionDB, *, decision: str = "approved", clock=lambda: 100.0):
    return create_intrinsic_approval_receipt(
        session_db=db,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-1",
        tool_name="apply_action_os_learning",
        original_args={"finding": "bounded"},
        preview={"summary": "Apply one finding"},
        approval_scope_sha256=_SCOPE,
        decision=decision,
        clock=clock,
    )


def _consume(db: SessionDB, receipt_id: str, *, clock=lambda: 200.0, **overrides):
    values = {
        "session_db": db,
        "receipt_id": receipt_id,
        "session_id": "session-1",
        "tool_name": "apply_action_os_learning",
        "approval_scope_sha256": _SCOPE,
        "consume_id": "action-os-command-1",
        "clock": clock,
    }
    values.update(overrides)
    return verify_and_consume_intrinsic_approval_receipt(**values)


def _stored_consumption(db: SessionDB, receipt_id: str):
    row = db._conn.execute(
        "SELECT consumed_by, consumed_at FROM intrinsic_tool_approval_receipts WHERE receipt_id = ?",
        (receipt_id,),
    ).fetchone()
    return tuple(row)


def _stored_temporal_state(db: SessionDB, receipt_id: str):
    row = db._conn.execute(
        "SELECT decided_at, expires_at, consumed_by, consumed_at "
        "FROM intrinsic_tool_approval_receipts WHERE receipt_id = ?",
        (receipt_id,),
    ).fetchone()
    return tuple(row)


def _tamper(db: SessionDB, receipt_id: str, **values):
    db._conn.execute("DROP TRIGGER intrinsic_tool_approval_receipts_immutable")
    db._conn.execute("PRAGMA ignore_check_constraints = ON")
    assignments = ", ".join(f"{field} = ?" for field in values)
    db._conn.execute(
        f"UPDATE intrinsic_tool_approval_receipts SET {assignments} WHERE receipt_id = ?",
        (*values.values(), receipt_id),
    )
    db._conn.commit()


def test_fresh_consume_returns_only_detached_bounded_result(receipt_db):
    _home, db = receipt_db
    receipt = _create(db)

    result = _consume(db, receipt["receipt_id"])

    assert isinstance(result, MappingProxyType)
    assert set(result) == _FIELDS
    assert dict(result) == {
        "receipt_id": receipt["receipt_id"],
        "decision": "approved",
        "decided_at": 100.0,
        "expires_at": 1000.0,
        "consumed_by": "action-os-command-1",
        "consumed_at": 200.0,
    }
    with pytest.raises(TypeError):
        result["decision"] = "denied"
    assert _stored_consumption(db, receipt["receipt_id"]) == ("action-os-command-1", 200.0)


def test_exact_replay_before_and_after_expiry_returns_original_result(receipt_db):
    _home, db = receipt_db
    receipt = _create(db)
    first = _consume(db, receipt["receipt_id"], clock=lambda: 200.0)

    before_expiry = _consume(db, receipt["receipt_id"], clock=lambda: 999.0)
    after_expiry = _consume(db, receipt["receipt_id"], clock=lambda: 1000.0)

    assert before_expiry == first
    assert after_expiry == first
    assert before_expiry is not first
    assert after_expiry["consumed_at"] == 200.0


@pytest.mark.parametrize(
    "kind",
    ["missing", "denied", "expired", "session", "tool", "scope", "consume"],
)
def test_invalid_receipt_or_binding_fails_without_consuming(receipt_db, kind):
    _home, db = receipt_db
    receipt = _create(db, decision="denied" if kind == "denied" else "approved")
    receipt_id = receipt["receipt_id"]
    kwargs = {}
    if kind == "missing":
        receipt_id = "missing-receipt"
    elif kind == "expired":
        kwargs["clock"] = lambda: 1000.0
    elif kind == "session":
        kwargs["session_id"] = "other-session"
    elif kind == "tool":
        kwargs["tool_name"] = "other-tool"
    elif kind == "scope":
        kwargs["approval_scope_sha256"] = "b" * 64
    elif kind == "consume":
        _consume(db, receipt_id)
        kwargs["consume_id"] = "other-command"

    with pytest.raises(
        IntrinsicApprovalReceiptValidationError,
        match="approval receipt verification failed",
    ):
        _consume(db, receipt_id, **kwargs)

    if kind == "consume":
        assert _stored_consumption(db, receipt["receipt_id"]) == (
            "action-os-command-1",
            200.0,
        )
    else:
        assert _stored_consumption(db, receipt["receipt_id"]) == (None, None)


def test_replay_still_validates_requested_bindings(receipt_db):
    _home, db = receipt_db
    receipt = _create(db)
    _consume(db, receipt["receipt_id"])

    for override in (
        {"session_id": "other-session"},
        {"tool_name": "other-tool"},
        {"approval_scope_sha256": "b" * 64},
    ):
        with pytest.raises(IntrinsicApprovalReceiptValidationError):
            _consume(db, receipt["receipt_id"], clock=lambda: 5000.0, **override)
    assert _stored_consumption(db, receipt["receipt_id"]) == (
        "action-os-command-1",
        200.0,
    )


def test_expiry_clock_is_sampled_once_after_write_lock_acquisition(receipt_db, monkeypatch):
    home, creator = receipt_db
    receipt = _create(creator)
    blocker = SessionDB(db_path=home / "state.db")
    consumer = SessionDB(db_path=home / "state.db")
    lock_acquired = Event()
    release_lock = Event()
    execute_started = Event()
    clock_calls = []
    original_execute_write = consumer._execute_write

    def hold_write_lock(conn):
        lock_acquired.set()
        assert release_lock.wait(timeout=5)

    def observed_execute_write(fn, patience_s=None):
        execute_started.set()
        return original_execute_write(fn, patience_s)

    def crossing_expiry_clock():
        clock_calls.append(release_lock.is_set())
        return 1000.0 if release_lock.is_set() else 999.0

    def consume_while_locked():
        token = set_hermes_home_override(home)
        try:
            return _consume(
                consumer,
                receipt["receipt_id"],
                clock=crossing_expiry_clock,
            )
        finally:
            reset_hermes_home_override(token)

    monkeypatch.setattr(consumer, "_execute_write", observed_execute_write)
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        holding = pool.submit(blocker._execute_write, hold_write_lock)
        assert lock_acquired.wait(timeout=5)
        consuming = pool.submit(consume_while_locked)
        assert execute_started.wait(timeout=5)
        assert clock_calls == []
        release_lock.set()
        holding.result(timeout=5)
        with pytest.raises(
            IntrinsicApprovalReceiptValidationError,
            match="approval receipt verification failed",
        ):
            consuming.result(timeout=5)
    finally:
        release_lock.set()
        pool.shutdown(wait=True)
        consumer.close()
        blocker.close()

    assert clock_calls == [True]
    assert _stored_consumption(creator, receipt["receipt_id"]) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("decided_at", True, id="decided-bool-coerced-by-sqlite"),
        pytest.param("decided_at", "not-a-number", id="decided-nonnumeric"),
        pytest.param("decided_at", float("inf"), id="decided-positive-infinity"),
        pytest.param("decided_at", float("-inf"), id="decided-negative-infinity"),
        pytest.param("decided_at", -1.0, id="decided-negative"),
        pytest.param("expires_at", True, id="expires-bool-coerced-by-sqlite"),
        pytest.param("expires_at", "not-a-number", id="expires-nonnumeric"),
        pytest.param("expires_at", float("inf"), id="expires-positive-infinity"),
        pytest.param("expires_at", float("-inf"), id="expires-negative-infinity"),
        pytest.param("expires_at", -1.0, id="expires-negative"),
    ],
)
def test_malformed_stored_decision_timestamps_fail_without_mutation(receipt_db, field, value):
    _home, db = receipt_db
    receipt = _create(db)
    _tamper(db, receipt["receipt_id"], **{field: value})
    before = _stored_temporal_state(db, receipt["receipt_id"])

    with pytest.raises(IntrinsicApprovalReceiptValidationError):
        _consume(db, receipt["receipt_id"])

    assert _stored_temporal_state(db, receipt["receipt_id"]) == before


@pytest.mark.parametrize("expires_at", [999.0, 1001.0])
def test_stored_non_900_second_ttl_fails_without_mutation(receipt_db, expires_at):
    _home, db = receipt_db
    receipt = _create(db)
    _tamper(db, receipt["receipt_id"], expires_at=expires_at)
    before = _stored_temporal_state(db, receipt["receipt_id"])

    with pytest.raises(IntrinsicApprovalReceiptValidationError):
        _consume(db, receipt["receipt_id"])

    assert _stored_temporal_state(db, receipt["receipt_id"]) == before


@pytest.mark.parametrize(
    "values",
    [
        pytest.param({"consumed_by": "action-os-command-1"}, id="missing-consumed-at"),
        pytest.param({"consumed_at": 200.0}, id="missing-consumed-by"),
        pytest.param(
            {"consumed_by": "action-os-command-1", "consumed_at": True},
            id="consumed-bool-coerced-by-sqlite",
        ),
        pytest.param(
            {"consumed_by": "action-os-command-1", "consumed_at": "not-a-number"},
            id="consumed-nonnumeric",
        ),
        pytest.param(
            {"consumed_by": "action-os-command-1", "consumed_at": float("inf")},
            id="consumed-positive-infinity",
        ),
        pytest.param(
            {"consumed_by": "action-os-command-1", "consumed_at": float("-inf")},
            id="consumed-negative-infinity",
        ),
        pytest.param(
            {"consumed_by": "action-os-command-1", "consumed_at": -1.0},
            id="consumed-negative",
        ),
        pytest.param(
            {"consumed_by": "action-os-command-1", "consumed_at": 99.0},
            id="consumed-before-decision",
        ),
        pytest.param(
            {"consumed_by": "action-os-command-1", "consumed_at": 1000.0},
            id="consumed-at-expiry",
        ),
        pytest.param(
            {"consumed_by": "action-os-command-1", "consumed_at": 1001.0},
            id="consumed-after-expiry",
        ),
    ],
)
def test_malformed_stored_consumption_fails_exact_replay_without_mutation(receipt_db, values):
    _home, db = receipt_db
    receipt = _create(db)
    _tamper(db, receipt["receipt_id"], **values)
    before = _stored_temporal_state(db, receipt["receipt_id"])

    with pytest.raises(IntrinsicApprovalReceiptValidationError):
        _consume(db, receipt["receipt_id"], clock=lambda: 5000.0)

    assert _stored_temporal_state(db, receipt["receipt_id"]) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("receipt_id", ""),
        ("receipt_id", "bad\nreceipt"),
        ("session_id", " "),
        ("tool_name", "bad\u200btool"),
        ("consume_id", "x" * 1025),
        ("approval_scope_sha256", "A" * 64),
        ("approval_scope_sha256", "a" * 63),
    ],
)
def test_malformed_request_fails_before_mutation(receipt_db, field, value):
    _home, db = receipt_db
    receipt = _create(db)
    kwargs = {field: value}
    receipt_id = kwargs.pop("receipt_id", receipt["receipt_id"])

    with pytest.raises(IntrinsicApprovalReceiptValidationError, match=field):
        _consume(db, receipt_id, **kwargs)
    assert _stored_consumption(db, receipt["receipt_id"]) == (None, None)


@pytest.mark.parametrize("field", ["profile_identity", "profile_home"])
def test_tampered_stored_provenance_fails_without_consuming(receipt_db, field):
    _home, db = receipt_db
    receipt = _create(db)
    db._conn.execute("DROP TRIGGER intrinsic_tool_approval_receipts_immutable")
    db._conn.execute(
        f"UPDATE intrinsic_tool_approval_receipts SET {field} = ? WHERE receipt_id = ?",
        ("tampered", receipt["receipt_id"]),
    )
    db._conn.commit()

    with pytest.raises(IntrinsicApprovalReceiptValidationError):
        _consume(db, receipt["receipt_id"])
    assert _stored_consumption(db, receipt["receipt_id"]) == (None, None)


def test_wrong_active_profile_and_noncanonical_db_fail_without_query_or_mutation(tmp_path):
    first_home = tmp_path / "profiles" / "first"
    token = set_hermes_home_override(first_home)
    db = SessionDB(db_path=first_home / "state.db")
    try:
        receipt = _create(db)
        statements = []
        db._conn.set_trace_callback(statements.append)
        second_home = tmp_path / "profiles" / "second"
        second_home.mkdir(parents=True)
        second_token = set_hermes_home_override(second_home)
        try:
            with pytest.raises(IntrinsicApprovalReceiptValidationError, match="active profile"):
                _consume(db, receipt["receipt_id"])
        finally:
            reset_hermes_home_override(second_token)
            db._conn.set_trace_callback(None)
        assert not any("intrinsic_tool_approval_receipts" in sql.lower() for sql in statements)
        assert _stored_consumption(db, receipt["receipt_id"]) == (None, None)
    finally:
        db.close()
        reset_hermes_home_override(token)


def test_alternate_and_symlink_database_paths_are_rejected(tmp_path):
    active = tmp_path / "active"
    active.mkdir()
    token = set_hermes_home_override(active)
    canonical = SessionDB(db_path=active / "state.db")
    receipt = _create(canonical)
    alternate = SessionDB(db_path=active / "other.db")
    alias = tmp_path / "active-alias"
    alias.symlink_to(active, target_is_directory=True)
    symlinked = SessionDB(db_path=alias / "state.db")
    try:
        for db in (alternate, symlinked):
            with pytest.raises(IntrinsicApprovalReceiptValidationError, match="canonical state.db"):
                _consume(db, receipt["receipt_id"])
        assert _stored_consumption(canonical, receipt["receipt_id"]) == (None, None)
    finally:
        symlinked.close()
        alternate.close()
        canonical.close()
        reset_hermes_home_override(token)


@pytest.mark.parametrize("same_consume", [True, False])
def test_concurrent_consumers_are_atomic(receipt_db, same_consume):
    home, creator = receipt_db
    receipt = _create(creator)
    databases = [SessionDB(db_path=home / "state.db") for _ in range(2)]

    def consume(index: int):
        token = set_hermes_home_override(home)
        try:
            consume_id = "shared-command" if same_consume else f"command-{index}"
            try:
                return _consume(
                    databases[index],
                    receipt["receipt_id"],
                    consume_id=consume_id,
                    clock=lambda: 250.0 + index,
                )
            except IntrinsicApprovalReceiptValidationError:
                return None
        finally:
            reset_hermes_home_override(token)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(consume, range(2)))
        successes = [result for result in results if result is not None]
        if same_consume:
            assert len(successes) == 2
            assert successes[0] == successes[1]
            assert successes[0]["consumed_at"] in {250.0, 251.0}
        else:
            assert len(successes) == 1
            assert successes[0]["consumed_by"] in {"command-0", "command-1"}
    finally:
        for db in databases:
            db.close()
