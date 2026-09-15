"""Live compression adoption preserves model-tool policy authority."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.conversation_compression import recover_rotated_compression_session
from agent.model_tool_policy import MODEL_TOOL_POLICY_VERSION
from hermes_state import SessionDB


def _policy(*allowed_tools: str) -> dict:
    return {
        "policy_id": "live-compression-policy",
        "policy_sha256": "e" * 64,
        "allowed_tools": list(allowed_tools),
        "approval_required_tools": [],
    }


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _agent(db: SessionDB, session_id: str):
    compressor = SimpleNamespace(bind_session_state=lambda **_kwargs: None)
    return SimpleNamespace(
        _session_db=db,
        session_id=session_id,
        context_compressor=compressor,
        _memory_manager=None,
        platform="cli",
        _gateway_session_key=None,
        _cached_system_prompt="parent prompt",
        _session_db_created=True,
        _last_flushed_db_idx=1,
        _flushed_db_message_session_id=session_id,
        _flushed_db_message_ids=set(),
    )


def test_live_adoption_rejects_mismatched_policy_before_rebind(db: SessionDB) -> None:
    parent_policy = _policy("read_file")
    child_policy = _policy("search_files")
    db.create_session(
        "parent", source="cli", model_tool_policy=parent_policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    db.append_message("parent", "user", "before compression")
    db.end_session("parent", "compression")
    db.create_session(
        "child", source="cli", parent_session_id="parent",
        model_tool_policy=child_policy, model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    db.append_message("child", "user", "continued")
    agent = _agent(db, "parent")

    assert recover_rotated_compression_session(agent) is None

    assert agent.session_id == "parent"
    assert agent._flushed_db_message_session_id == "parent"
    assert db.get_session("parent")["end_reason"] == "compression"
    assert json.loads(db.get_session("child")["model_tool_policy"]) == child_policy


@pytest.mark.parametrize("policy", [None, _policy("read_file")])
def test_live_adoption_accepts_equal_or_legacy_policy(db: SessionDB, policy) -> None:
    kwargs = ({
        "model_tool_policy": policy,
        "model_tool_policy_version": MODEL_TOOL_POLICY_VERSION,
    } if policy is not None else {})
    db.create_session("parent", source="cli", **kwargs)
    db.append_message("parent", "user", "before compression")
    db.end_session("parent", "compression")
    db.create_session("child", source="cli", parent_session_id="parent", **kwargs)
    db.append_message("child", "user", "continued")
    agent = _agent(db, "parent")

    recovered = recover_rotated_compression_session(agent)

    assert recovered is not None
    assert agent.session_id == "child"
    assert agent._flushed_db_message_session_id == "child"
