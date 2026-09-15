"""Regression tests for stale writes after a compression session split."""

from __future__ import annotations

import json
import time

import pytest

from agent.model_tool_policy import MODEL_TOOL_POLICY_VERSION
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _compression_parent(db: SessionDB, session_id: str = "parent") -> None:
    db.create_session(session_id, source="webui")
    db.append_message(session_id, "user", "before split")
    db.end_session(session_id, "compression")


def _model_tool_policy(*allowed_tools: str) -> dict:
    return {
        "policy_id": "compression-policy",
        "policy_sha256": "a" * 64,
        "allowed_tools": list(allowed_tools),
        "approval_required_tools": [],
    }


def test_find_live_compression_child_returns_unique_direct_child(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("child", source="webui", parent_session_id="parent")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "child"
    assert child["parent_session_id"] == "parent"
    assert child["ended_at"] is None


def test_find_live_compression_child_fails_closed_when_ambiguous(db: SessionDB) -> None:
    _compression_parent(db)
    db.create_session("child-a", source="webui", parent_session_id="parent")
    db.create_session("child-b", source="webui", parent_session_id="parent")

    assert db.find_live_compression_child("parent") is None


def test_reopen_orphaned_compression_session_reopens_parent_without_child(
    db: SessionDB,
) -> None:
    _compression_parent(db, "orphan")

    assert db.reopen_orphaned_compression_session("orphan") is True
    assert db.get_session("orphan")["ended_at"] is None
    assert db.get_session("orphan")["end_reason"] is None

    db.append_message("orphan", "user", "recovered turn")
    assert [m["content"] for m in db.get_messages("orphan")] == [
        "before split",
        "recovered turn",
    ]


def test_reopen_orphaned_compression_session_fails_closed_with_child(
    db: SessionDB,
) -> None:
    _compression_parent(db, "parent-with-child")
    db.create_session("child", source="webui", parent_session_id="parent-with-child")

    assert db.reopen_orphaned_compression_session("parent-with-child") is False
    parent = db.get_session("parent-with-child")
    assert parent["end_reason"] == "compression"
    assert parent["ended_at"] is not None


def test_reopen_orphaned_compression_session_ignores_non_continuation_children(
    db: SessionDB,
) -> None:
    _compression_parent(db, "parent-with-non-continuation-children")
    db.create_session(
        "branch",
        source="webui",
        parent_session_id="parent-with-non-continuation-children",
        model_config={"_branched_from": "parent-with-non-continuation-children"},
    )
    db.create_session(
        "delegate",
        source="tool",
        parent_session_id="parent-with-non-continuation-children",
        model_config={"_delegate_from": "parent-with-non-continuation-children"},
    )

    assert db.reopen_orphaned_compression_session(
        "parent-with-non-continuation-children"
    ) is True


def test_reopen_fails_closed_when_continuation_inherits_foreign_markers(
    db: SessionDB,
) -> None:
    """A REAL continuation can carry ``_delegate_from``/``_branched_from``
    pointing at some OTHER session: ``publish_compression_child`` callers
    pass the rotated agent's ``_session_init_model_config`` verbatim, so a
    delegate subagent's continuation inherits ``_delegate_from=<the
    delegate's own parent>``. Marker-presence matching misclassified it as
    a delegate child — reopen returned True with a live continuation
    present, forking the lineage. Markers only disqualify a child when
    they point at the queried parent."""
    _compression_parent(db, "delegate-session")
    db.create_session(
        "delegate-continuation",
        source="subagent",
        parent_session_id="delegate-session",
        model_config={"_delegate_from": "some-original-parent"},
    )

    assert db.reopen_orphaned_compression_session("delegate-session") is False
    parent = db.get_session("delegate-session")
    assert parent["end_reason"] == "compression"


def test_find_live_child_returns_continuation_with_foreign_markers(
    db: SessionDB,
) -> None:
    """Adoption-side twin of the reopen test above: the continuation that
    inherited a foreign ``_delegate_from`` must still be adoptable."""
    _compression_parent(db, "delegate-session-2")
    db.create_session(
        "inherited-continuation",
        source="subagent",
        parent_session_id="delegate-session-2",
        model_config={"_delegate_from": "some-original-parent"},
    )

    child = db.find_live_compression_child("delegate-session-2")
    assert child is not None
    assert child["id"] == "inherited-continuation"


def test_compression_lineage_includes_continuation_with_foreign_markers(
    db: SessionDB,
) -> None:
    """Lineage walk uses the same parent-bound marker rule as orphan recovery."""
    _compression_parent(db, "delegate-session-3")
    db.create_session(
        "inherited-tip",
        source="subagent",
        parent_session_id="delegate-session-3",
        model_config={"_delegate_from": "some-original-parent"},
    )

    assert db.get_compression_lineage("inherited-tip") == [
        "delegate-session-3",
        "inherited-tip",
    ]
    assert db.get_compression_lineage("delegate-session-3") == [
        "delegate-session-3",
        "inherited-tip",
    ]


def test_reopen_orphaned_compression_session_fails_closed_with_active_lease(
    db: SessionDB,
) -> None:
    _compression_parent(db, "leased-parent")
    assert db.try_acquire_compression_lock("leased-parent", "compressor")

    assert db.reopen_orphaned_compression_session("leased-parent") is False
    assert db.get_session("leased-parent")["end_reason"] == "compression"


def test_reopen_orphaned_compression_session_reclaims_expired_lease(
    db: SessionDB,
) -> None:
    _compression_parent(db, "expired-lease-parent")
    now = time.time()
    db._conn.execute(
        "INSERT INTO compression_locks "
        "(session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        ("expired-lease-parent", "old-compressor", now - 60, now - 30),
    )
    db._conn.commit()

    assert db.reopen_orphaned_compression_session("expired-lease-parent") is True
    assert db.refresh_compression_lock(
        "expired-lease-parent", "old-compressor"
    ) is False
    assert db.get_compression_lock_holder("expired-lease-parent") is None


def test_reopen_orphaned_compression_session_loses_to_expired_lease_refresh(
    db: SessionDB,
) -> None:
    _compression_parent(db, "refreshed-lease-parent")
    now = time.time()
    db._conn.execute(
        "INSERT INTO compression_locks "
        "(session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        ("refreshed-lease-parent", "live-compressor", now - 60, now - 30),
    )
    db._conn.commit()

    assert db.refresh_compression_lock(
        "refreshed-lease-parent", "live-compressor"
    ) is True
    assert db.reopen_orphaned_compression_session("refreshed-lease-parent") is False
    assert db.get_session("refreshed-lease-parent")["end_reason"] == "compression"


def test_find_live_compression_child_ignores_non_continuation_children(
    db: SessionDB,
) -> None:
    _compression_parent(db)
    db.create_session("canonical", source="webui", parent_session_id="parent")
    db.create_session(
        "branch",
        source="webui",
        parent_session_id="parent",
        model_config={"_branched_from": "parent"},
    )
    db.create_session(
        "delegate",
        source="webui",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )
    db.create_session("tool-child", source="tool", parent_session_id="parent")

    child = db.find_live_compression_child("parent")

    assert child is not None
    assert child["id"] == "canonical"


def test_publish_compression_child_is_atomic_on_handoff_failure(
    db: SessionDB, monkeypatch
) -> None:
    db.create_session("atomic-parent", source="webui")
    db.append_message("atomic-parent", "user", "original")
    assert db.try_acquire_compression_lock("atomic-parent", "winner", ttl_seconds=60)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("handoff insert failed")

    monkeypatch.setattr(db, "_insert_message_rows", _boom)
    with pytest.raises(RuntimeError, match="handoff insert failed"):
        db.publish_compression_child(
            parent_session_id="atomic-parent",
            child_session_id="atomic-child",
            source="webui",
            messages=[{"role": "user", "content": "summary"}],
            compression_lock_holder="winner",
        )

    parent = db.get_session("atomic-parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert db.get_session("atomic-child") is None


def test_publish_compression_child_exposes_complete_child(db: SessionDB) -> None:
    db.create_session("atomic-parent", source="webui")
    db.append_message("atomic-parent", "user", "original")
    assert db.try_acquire_compression_lock("atomic-parent", "winner", ttl_seconds=60)

    db.publish_compression_child(
        parent_session_id="atomic-parent",
        child_session_id="atomic-child",
        source="webui",
        system_prompt="compressed system",
        messages=[{"role": "user", "content": "summary"}],
        compression_lock_holder="winner",
    )

    assert db.get_session("atomic-parent")["end_reason"] == "compression"
    child = db.find_live_compression_child("atomic-parent")
    assert child is not None
    assert child["id"] == "atomic-child"
    assert child["system_prompt"] == "compressed system"
    assert [m["content"] for m in db.get_messages("atomic-child")] == ["summary"]


def test_publish_compression_child_inherits_exact_model_tool_policy(db: SessionDB) -> None:
    policy = _model_tool_policy("read_file", "search_files")
    db.create_session(
        "policy-parent", source="webui", model_tool_policy=policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    assert db.try_acquire_compression_lock("policy-parent", "winner", ttl_seconds=60)
    parent_before = db.get_session("policy-parent")

    db.publish_compression_child(
        parent_session_id="policy-parent",
        child_session_id="policy-child",
        source="webui",
        messages=[{"role": "user", "content": "summary"}],
        compression_lock_holder="winner",
    )

    child = db.get_session("policy-child")
    assert child["model_tool_policy_version"] == parent_before["model_tool_policy_version"]
    assert child["model_tool_policy"] == parent_before["model_tool_policy"]
    assert json.loads(child["model_tool_policy"]) == policy


def test_publish_compression_child_preserves_legacy_policy_absence(db: SessionDB) -> None:
    db.create_session("legacy-parent", source="webui")
    assert db.try_acquire_compression_lock("legacy-parent", "winner", ttl_seconds=60)

    db.publish_compression_child(
        parent_session_id="legacy-parent",
        child_session_id="legacy-child",
        source="webui",
        messages=[{"role": "user", "content": "summary"}],
        compression_lock_holder="winner",
    )

    child = db.get_session("legacy-child")
    assert child["model_tool_policy_version"] is None
    assert child["model_tool_policy"] is None


@pytest.mark.parametrize(
    ("version", "payload"),
    [
        (MODEL_TOOL_POLICY_VERSION, "{"),
        (None, json.dumps(_model_tool_policy("read_file"))),
        (99, json.dumps(_model_tool_policy("read_file"))),
    ],
)
def test_publish_compression_child_rejects_corrupt_parent_without_state_change(
    db: SessionDB, version, payload
) -> None:
    db.create_session("corrupt-parent", source="webui")
    db.append_message("corrupt-parent", "user", "original")
    db._conn.execute(
        "UPDATE sessions SET model_tool_policy_version = ?, model_tool_policy = ? WHERE id = ?",
        (version, payload, "corrupt-parent"),
    )
    db._conn.commit()
    parent_before = db.get_session("corrupt-parent")
    messages_before = db.get_messages("corrupt-parent")

    with pytest.raises(ValueError, match="policy"):
        db.publish_compression_child(
            parent_session_id="corrupt-parent",
            child_session_id="corrupt-child",
            source="webui",
            messages=[{"role": "user", "content": "summary"}],
            require_compression_lease=False,
        )

    assert db.get_session("corrupt-parent") == parent_before
    assert db.get_messages("corrupt-parent") == messages_before
    assert db.get_session("corrupt-child") is None


@pytest.mark.parametrize(
    ("child_version", "child_payload", "error"),
    [
        (None, None, "does not match"),
        (MODEL_TOOL_POLICY_VERSION, "{", "policy"),
        (MODEL_TOOL_POLICY_VERSION, None, "policy"),
    ],
)
def test_publish_compression_child_rejects_existing_child_policy_mismatch_or_corruption(
    db: SessionDB, child_version, child_payload, error
) -> None:
    policy = _model_tool_policy("read_file")
    db.create_session(
        "replay-parent", source="webui", model_tool_policy=policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    db.append_message("replay-parent", "user", "original")
    db.create_session("replay-child", source="webui", parent_session_id="replay-parent")
    db._conn.execute(
        "UPDATE sessions SET model_tool_policy_version = ?, model_tool_policy = ? WHERE id = ?",
        (child_version, child_payload, "replay-child"),
    )
    db._conn.commit()
    parent_before = db.get_session("replay-parent")
    child_before = db.get_session("replay-child")

    with pytest.raises((RuntimeError, ValueError), match=error):
        db.publish_compression_child(
            parent_session_id="replay-parent",
            child_session_id="replay-child",
            source="webui",
            messages=[{"role": "user", "content": "summary"}],
            require_compression_lease=False,
        )

    assert db.get_session("replay-parent") == parent_before
    assert db.get_session("replay-child") == child_before
    assert db.get_messages("replay-parent")[0]["content"] == "original"
    assert db.get_messages("replay-child") == []


def test_publish_compression_child_replay_requires_exact_policy_carrier(db: SessionDB) -> None:
    policy = _model_tool_policy("read_file")
    db.create_session(
        "replay-parent", source="webui", model_tool_policy=policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    db.create_session(
        "replay-child", source="webui", parent_session_id="replay-parent",
        model_tool_policy=policy, model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    parent_before = db.get_session("replay-parent")
    child_before = db.get_session("replay-child")

    with pytest.raises(RuntimeError, match="already exists"):
        db.publish_compression_child(
            parent_session_id="replay-parent",
            child_session_id="replay-child",
            source="webui",
            messages=[{"role": "user", "content": "summary"}],
            require_compression_lease=False,
        )

    assert db.get_session("replay-parent") == parent_before
    assert db.get_session("replay-child") == child_before
    assert db.get_messages("replay-child") == []


@pytest.mark.parametrize(
    ("child_version", "child_payload"),
    [
        (None, None),
        (MODEL_TOOL_POLICY_VERSION, json.dumps(_model_tool_policy("search_files"))),
        (MODEL_TOOL_POLICY_VERSION, "{"),
    ],
)
def test_cold_compression_resolution_rejects_nonidentical_policy_carrier(
    db: SessionDB, child_version, child_payload
) -> None:
    policy = _model_tool_policy("read_file")
    db.create_session(
        "cold-parent", source="webui", model_tool_policy=policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    db.append_message("cold-parent", "user", "before compression")
    db.end_session("cold-parent", "compression")
    db.create_session("cold-child", source="webui", parent_session_id="cold-parent")
    db.append_message("cold-child", "user", "continued")
    db._conn.execute(
        "UPDATE sessions SET model_tool_policy_version = ?, model_tool_policy = ? WHERE id = ?",
        (child_version, child_payload, "cold-child"),
    )
    db._conn.commit()

    with pytest.raises(ValueError, match="policy"):
        db.resolve_resume_session_id("cold-parent")


@pytest.mark.parametrize("policy", [None, _model_tool_policy("read_file")])
def test_cold_compression_resolution_accepts_equal_or_legacy_carriers(
    db: SessionDB, policy
) -> None:
    kwargs = ({
        "model_tool_policy": policy,
        "model_tool_policy_version": MODEL_TOOL_POLICY_VERSION,
    } if policy is not None else {})
    db.create_session("cold-parent", source="webui", **kwargs)
    db.append_message("cold-parent", "user", "before compression")
    db.end_session("cold-parent", "compression")
    db.create_session("cold-child", source="webui", parent_session_id="cold-parent", **kwargs)
    db.append_message("cold-child", "user", "continued")

    assert db.resolve_resume_session_id("cold-parent") == "cold-child"


def test_publish_compression_child_rejects_lost_or_expired_lease(db: SessionDB) -> None:
    db.create_session("lease-parent", source="webui")
    db.append_message("lease-parent", "user", "new durable turn")
    assert db.try_acquire_compression_lock("lease-parent", "new-winner", ttl_seconds=60)

    with pytest.raises(RuntimeError, match="lease lost"):
        db.publish_compression_child(
            parent_session_id="lease-parent",
            child_session_id="stale-child",
            source="webui",
            messages=[{"role": "user", "content": "stale summary"}],
            compression_lock_holder="old-loser",
        )

    parent = db.get_session("lease-parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert db.get_session("stale-child") is None
    assert [m["content"] for m in db.get_messages("lease-parent")] == [
        "new durable turn"
    ]


def test_compression_lease_blocks_non_owner_but_allows_owner_flush(
    db: SessionDB,
) -> None:
    """Contract flipped by the watermark commit (#75316): a live lease no
    longer fences ordinary appends — both the owner's flush and a concurrent
    turn land immediately, and the commit-side watermark decides what
    survives compaction (see test_compression_watermark_commit.py)."""
    db.create_session("leased", source="webui")
    assert db.try_acquire_compression_lock("leased", "winner", ttl_seconds=60)

    db.append_message("leased", "user", "late concurrent turn")
    db.append_message(
        "leased",
        "assistant",
        "winner flush",
        compression_lock_holder="winner",
    )
    assert [m["content"] for m in db.get_messages("leased")] == [
        "late concurrent turn",
        "winner flush",
    ]
