"""Model-tool policy persistence across session portability boundaries."""

from __future__ import annotations

import json

import pytest

from agent.model_tool_policy import MODEL_TOOL_POLICY_VERSION
from hermes_state import SessionDB


def _policy(*allowed_tools: str, policy_id: str = "portable-policy") -> dict:
    return {
        "policy_id": policy_id,
        "policy_sha256": "d" * 64,
        "allowed_tools": list(allowed_tools),
        "approval_required_tools": [],
    }


@pytest.fixture
def stores(tmp_path):
    source = SessionDB(db_path=tmp_path / "source.db")
    target = SessionDB(db_path=tmp_path / "target.db")
    try:
        yield source, target
    finally:
        source.close()
        target.close()


def _create_bound(db: SessionDB, session_id: str, policy: dict, **kwargs) -> None:
    db.create_session(
        session_id,
        source="desktop",
        model_tool_policy=policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
        **kwargs,
    )


def test_export_to_new_id_import_preserves_canonical_policy_carrier(stores) -> None:
    source, target = stores
    policy = _policy("read_file", "search_files")
    _create_bound(source, "bound-source", policy)
    source.append_message("bound-source", "user", "portable")
    exported = source.export_session("bound-source")
    exported["id"] = "bound-copy"

    result = target.import_sessions([exported])

    assert result["ok"] is True
    row = target.get_session("bound-copy")
    assert row["model_tool_policy_version"] == MODEL_TOOL_POLICY_VERSION
    assert row["model_tool_policy"] == json.dumps(
        policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def test_import_corrupt_policy_is_atomic_across_payload(stores) -> None:
    _source, target = stores
    result = target.import_sessions([
        {"id": "would-be-valid", "messages": [{"role": "user", "content": "first"}]},
        {
            "id": "corrupt",
            "model_tool_policy_version": MODEL_TOOL_POLICY_VERSION,
            "model_tool_policy": "{",
            "messages": [{"role": "user", "content": "second"}],
        },
    ])

    assert result["ok"] is False
    assert "policy" in result["errors"][0]["error"]
    assert target.get_session("would-be-valid") is None
    assert target.get_session("corrupt") is None
    assert target.message_count() == 0


@pytest.mark.parametrize("unsupported", ["delegate_task", "execute_code"])
def test_import_rejects_nested_execution_policy_atomically(stores, unsupported) -> None:
    _source, target = stores
    result = target.import_sessions([
        {"id": "would-be-valid", "messages": [{"role": "user", "content": "first"}]},
        {
            "id": "unsupported",
            "model_tool_policy_version": MODEL_TOOL_POLICY_VERSION,
            "model_tool_policy": _policy("read_file", unsupported),
            "messages": [{"role": "user", "content": "second"}],
        },
    ])

    assert result["ok"] is False
    assert "unsupported V1 nested execution authority" in result["errors"][0]["error"]
    assert target.get_session("would-be-valid") is None
    assert target.get_session("unsupported") is None
    assert target.message_count() == 0


def test_import_conflict_cannot_drop_existing_policy(stores) -> None:
    _source, target = stores
    policy = _policy("read_file")
    _create_bound(target, "existing", policy)

    result = target.import_sessions([{"id": "existing", "messages": []}])

    assert result["ok"] is False
    row = target.get_session("existing")
    assert row["model_tool_policy_version"] == MODEL_TOOL_POLICY_VERSION
    assert json.loads(row["model_tool_policy"]) == policy


def test_import_conflict_cannot_broaden_existing_legacy_session(stores) -> None:
    _source, target = stores
    target.create_session("existing-legacy", source="desktop")
    policy = _policy("read_file")

    result = target.import_sessions([{
        "id": "existing-legacy",
        "model_tool_policy_version": MODEL_TOOL_POLICY_VERSION,
        "model_tool_policy": policy,
        "messages": [],
    }])

    assert result["ok"] is False
    row = target.get_session("existing-legacy")
    assert row["model_tool_policy_version"] is None
    assert row["model_tool_policy"] is None


def test_import_rejects_mismatched_compression_lineage_atomically(stores) -> None:
    _source, target = stores
    policy = _policy("read_file")
    result = target.import_sessions([
        {
            "id": "parent", "source": "desktop", "end_reason": "compression",
            "model_tool_policy_version": MODEL_TOOL_POLICY_VERSION,
            "model_tool_policy": policy, "messages": [],
        },
        {
            "id": "child", "source": "desktop", "parent_session_id": "parent",
            "messages": [{"role": "user", "content": "continued"}],
        },
    ])

    assert result["ok"] is False
    assert target.get_session("parent") is None
    assert target.get_session("child") is None
    assert target.message_count() == 0


def test_import_rejects_branch_that_drops_inherited_policy(stores) -> None:
    _source, target = stores
    policy = _policy("read_file")
    result = target.import_sessions([
        {
            "id": "branch-parent", "source": "desktop",
            "model_tool_policy_version": MODEL_TOOL_POLICY_VERSION,
            "model_tool_policy": policy, "messages": [],
        },
        {
            "id": "branch-child", "source": "desktop", "parent_session_id": "branch-parent",
            "model_config": {"_branched_from": "branch-parent"}, "messages": [],
        },
    ])

    assert result["ok"] is False
    assert target.get_session("branch-parent") is None
    assert target.get_session("branch-child") is None


def test_adopt_compression_lineage_preserves_exact_policy_carrier(stores) -> None:
    donor, target = stores
    policy = _policy("read_file")
    _create_bound(donor, "parent", policy)
    donor.end_session("parent", "compression")
    _create_bound(donor, "child", policy, parent_session_id="parent")
    donor.append_message("child", "user", "continued")

    result = target.adopt_session_lineage_from(donor, "parent", retire_donor=False)

    assert result["adopted"] is True
    for session_id in ("parent", "child"):
        source_row = donor.get_session(session_id)
        target_row = target.get_session(session_id)
        assert target_row["model_tool_policy_version"] == source_row["model_tool_policy_version"]
        assert target_row["model_tool_policy"] == source_row["model_tool_policy"]


def test_adopt_legacy_compression_lineage_preserves_absence(stores) -> None:
    donor, target = stores
    donor.create_session("legacy-parent", source="desktop")
    donor.end_session("legacy-parent", "compression")
    donor.create_session("legacy-child", source="desktop", parent_session_id="legacy-parent")
    donor.append_message("legacy-child", "user", "continued")

    result = target.adopt_session_lineage_from(donor, "legacy-parent", retire_donor=False)

    assert result["adopted"] is True
    for session_id in ("legacy-parent", "legacy-child"):
        row = target.get_session(session_id)
        assert row["model_tool_policy_version"] is None
        assert row["model_tool_policy"] is None


def test_adopt_rejects_mismatched_compression_lineage_before_copy_or_retire(stores) -> None:
    donor, target = stores
    policy = _policy("read_file")
    _create_bound(donor, "bad-parent", policy)
    donor.end_session("bad-parent", "compression")
    donor.create_session("bad-child", source="desktop", parent_session_id="bad-parent")
    donor.append_message("bad-child", "user", "continued")

    with pytest.raises(ValueError, match="policy"):
        target.adopt_session_lineage_from(donor, "bad-parent")

    assert target.get_session("bad-parent") is None
    assert target.get_session("bad-child") is None
    assert donor.get_session("bad-parent")["archived"] == 0
    assert donor.get_session("bad-child")["archived"] == 0
