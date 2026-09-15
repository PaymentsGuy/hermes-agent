"""H3a: persisted session-scoped model-tool policy security boundary."""

import json

import pytest

import model_tools
from agent.model_tool_policy import (
    MODEL_TOOL_POLICY_VERSION,
    apply_model_tool_policy,
    decode_stored_model_tool_policy,
    model_tool_policy_identity,
    normalize_model_tool_policy,
)
from hermes_state import SessionDB


SHA = "a" * 64


def policy(*allowed, approval_required=()):
    return {
        "policy_id": "external-policy-1",
        "policy_sha256": SHA,
        "allowed_tools": list(allowed),
        "approval_required_tools": list(approval_required),
    }


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {**policy("read_file"), "extra": True},
        {k: v for k, v in policy("read_file").items() if k != "policy_id"},
        {**policy("read_file"), "policy_id": ""},
        {**policy("read_file"), "policy_sha256": "A" * 64},
        {**policy("read_file"), "policy_sha256": "a" * 63},
        {**policy("read_file"), "allowed_tools": "read_file"},
        {**policy("read_file"), "allowed_tools": ["read_file", "read_file"]},
        {**policy("read_file"), "allowed_tools": [""]},
        policy("read_file", approval_required=("skill_manage",)),
    ],
)
def test_policy_shape_is_closed_and_fails_invalid_values(bad):
    with pytest.raises(ValueError):
        normalize_model_tool_policy(bad)


def test_policy_external_identity_is_preserved_and_names_are_exact():
    raw = policy("read_file", "proposal.tool", approval_required=("proposal.tool",))
    assert normalize_model_tool_policy(raw) == raw
    assert model_tool_policy_identity(raw) == {"policy_id": "external-policy-1", "policy_sha256": SHA}


@pytest.mark.parametrize(
    ("unsupported", "expected_names"),
    [
        (("delegate_task",), ("delegate_task",)),
        (("execute_code",), ("execute_code",)),
        (("delegate_task", "execute_code"), ("delegate_task", "execute_code")),
    ],
)
def test_v1_policy_rejects_nested_execution_authority(unsupported, expected_names):
    with pytest.raises(ValueError) as exc_info:
        normalize_model_tool_policy(policy("read_file", *unsupported))

    message = str(exc_info.value)
    assert "unsupported V1 nested execution authority" in message
    assert all(name in message for name in expected_names)


def test_v1_policy_keeps_connector_bridge_authority_supported():
    raw = policy("tool_call")
    assert normalize_model_tool_policy(raw) == raw


def test_policy_filters_exact_schema_without_copying_or_reordering():
    retained_a = {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
    dropped = {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}
    retained_b = {"type": "function", "function": {"name": "search_files", "parameters": {"type": "object"}}}
    result = apply_model_tool_policy(policy("search_files", "read_file"), [retained_a, dropped, retained_b])
    assert result == [retained_a, retained_b]
    assert result[0] is retained_a and result[1] is retained_b


def test_policy_rejects_unavailable_declared_tool_before_prompt():
    with pytest.raises(ValueError, match="unavailable"):
        apply_model_tool_policy(policy("read_file", "missing_tool"), [
            {"type": "function", "function": {"name": "read_file"}}
        ])


def test_persisted_policy_round_trips_and_legacy_remains_unbound(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        bound = policy("read_file")
        db.create_session(
            "bound", "api_server", model_tool_policy=bound,
            model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
        )
        db.create_session("legacy", "api_server")
        row = db.get_session("bound")
        assert decode_stored_model_tool_policy(
            row["model_tool_policy_version"], row["model_tool_policy"]
        ) == bound
        legacy = db.get_session("legacy")
        assert decode_stored_model_tool_policy(
            legacy["model_tool_policy_version"], legacy["model_tool_policy"]
        ) is None
    finally:
        db.close()


@pytest.mark.parametrize(
    ("version", "stored"),
    [(MODEL_TOOL_POLICY_VERSION, None), (None, "{}"), (99, json.dumps(policy("read_file"))),
     (MODEL_TOOL_POLICY_VERSION, "{"), (MODEL_TOOL_POLICY_VERSION, "{}")],
)
def test_declared_missing_or_corrupt_stored_policy_fails_closed(version, stored):
    with pytest.raises(ValueError):
        decode_stored_model_tool_policy(version, stored)


@pytest.mark.parametrize("unsupported", ["delegate_task", "execute_code"])
def test_stored_v1_policy_with_nested_execution_authority_is_corrupt(unsupported):
    with pytest.raises(ValueError, match="unsupported V1 nested execution authority"):
        decode_stored_model_tool_policy(
            MODEL_TOOL_POLICY_VERSION, json.dumps(policy("read_file", unsupported)),
        )


def test_model_dispatch_denies_before_request_middleware_hooks_and_handler(monkeypatch):
    called = []
    monkeypatch.setattr(model_tools, "_apply_request_middleware", lambda *a, **k: called.append("middleware"))
    monkeypatch.setattr(model_tools, "_pre_dispatch_guards", lambda *a, **k: called.append("hook"))
    monkeypatch.setattr(model_tools, "_execute_tool", lambda *a, **k: called.append("handler"))

    result = model_tools.handle_function_call(
        "skill_manage", {}, session_id="s1", call_origin="model",
        model_tool_policy=policy("read_file"),
    )

    assert "not allowed" in result
    assert called == []


def test_model_policy_survives_connector_batch_bridge_reentry(monkeypatch):
    import model_tools_connectors

    called = []
    monkeypatch.setattr(model_tools, "_apply_request_middleware", lambda *a, **k: called.append("middleware"))
    monkeypatch.setattr(model_tools, "_pre_dispatch_guards", lambda *a, **k: called.append("hook"))
    monkeypatch.setattr(model_tools, "_execute_tool", lambda *a, **k: called.append("handler"))

    result = json.loads(model_tools_connectors.dispatch_connector_batch(
        [{"name": "connectors__fixture__write", "arguments": {}}],
        model_tools._CallIds(None, "session", None, None, None),
        user_task=None, enabled_tools=None, middleware_trace=[],
        enabled_toolsets=None, disabled_toolsets=None,
        call_origin="model", model_tool_policy=policy("tool_call"),
    ))

    assert "not allowed" in result["results"][0]["error"]["message"]
    assert called == []


def test_trusted_internal_dispatch_does_not_inherit_model_authority(monkeypatch):
    monkeypatch.setattr(model_tools, "_execute_tool", lambda *a, **k: '{"ok": true}')
    monkeypatch.setattr(model_tools, "_pre_dispatch_guards", lambda args_name, args, *a, **k: (args, None))
    assert json.loads(model_tools.handle_function_call("internal_fixture", {}, call_origin="internal")) == {"ok": True}


def test_two_sessions_keep_independent_policies():
    schemas = [
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "search_files"}},
    ]
    assert [d["function"]["name"] for d in apply_model_tool_policy(policy("read_file"), schemas)] == ["read_file"]
    assert [d["function"]["name"] for d in apply_model_tool_policy(policy("search_files"), schemas)] == ["search_files"]
