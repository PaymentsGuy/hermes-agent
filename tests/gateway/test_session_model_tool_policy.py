"""API session behavior for persisted model-tool policy."""

import json
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from agent.memory_provider import MemoryProvider
from agent.model_tool_policy import MODEL_TOOL_POLICY_VERSION, decode_stored_model_tool_policy
from gateway.config import PlatformConfig
from gateway.platforms import api_server
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


SHA = "d" * 64


class _DynamicMemoryProvider(MemoryProvider):
    @property
    def name(self):
        return "api-memory"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return [{
            "name": "api_memory_lookup",
            "description": "Look up API profile memory.",
            "parameters": {"type": "object", "properties": {}},
        }]


def _policy(*allowed):
    return {
        "policy_id": "persisted-api-policy",
        "policy_sha256": SHA,
        "allowed_tools": list(allowed),
        "approval_required_tools": [],
    }


def _json_response(payload, *, status=200, headers=None):
    return SimpleNamespace(payload=payload, status=status, headers=headers)


async def _fork(instance, parent_id, body, monkeypatch):
    monkeypatch.setattr(api_server, "web", SimpleNamespace(json_response=_json_response))
    request = SimpleNamespace(
        match_info={"session_id": parent_id},
        json=AsyncMock(return_value=body),
    )
    return await APIServerAdapter._handle_fork_session.__wrapped__(instance, request)


@pytest.mark.asyncio
async def test_api_run_agent_rebuild_passes_persisted_policy_into_agent(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    instance = APIServerAdapter(PlatformConfig(enabled=True))
    instance._session_db = db
    policy = _policy("read_file")
    db.create_session(
        "bound-api-chat", "desktop", model_tool_policy=policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    row = db.get_session("bound-api-chat")
    restored = decode_stored_model_tool_policy(
        row["model_tool_policy_version"], row["model_tool_policy"]
    )
    captured = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id):
            self.session_id = session_id

        def run_conversation(self, **_kwargs):
            return {"final_response": "ok"}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return FakeAgent(kwargs["session_id"])

    monkeypatch.setattr(instance, "_create_agent", fake_create_agent)
    try:
        result, _usage = await instance._run_agent(
            user_message="hello",
            conversation_history=[],
            session_id="bound-api-chat",
            model_tool_policy=restored,
        )
    finally:
        db.close()

    assert result["final_response"] == "ok"
    assert captured["model_tool_policy"] == policy


def test_api_policy_validator_accepts_configured_dynamic_memory_tool(monkeypatch):
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"memory": {"provider": "api-memory"}, "platform_toolsets": {"api_server": ["memory"]}},
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"memory": {"provider": "api-memory"}},
    )
    monkeypatch.setattr(
        "plugins.memory.load_memory_provider", lambda _name, **_kwargs: _DynamicMemoryProvider(),
    )
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kwargs: [])

    policy = _policy("api_memory_lookup")
    policy["approval_required_tools"] = ["api_memory_lookup"]
    assert APIServerAdapter._validate_api_model_tool_policy(policy) == policy

    with pytest.raises(ValueError, match="unavailable"):
        APIServerAdapter._validate_api_model_tool_policy(_policy("invented_tool"))


@pytest.mark.asyncio
async def test_api_fork_inherits_exact_parent_policy_and_transcript(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    instance = APIServerAdapter(PlatformConfig(enabled=True))
    instance._session_db = db
    policy = _policy("read_file")
    db.create_session(
        "policy-parent", "api_server", model_tool_policy=policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    db.append_message("policy-parent", "user", "hello")

    try:
        response = await _fork(
            instance, "policy-parent", {"id": "policy-child"}, monkeypatch,
        )
        child = db.get_session("policy-child")
        child_messages = db.get_messages("policy-child")
    finally:
        db.close()

    assert response.status == 201
    assert child["model_tool_policy_version"] == MODEL_TOOL_POLICY_VERSION
    assert json.loads(child["model_tool_policy"]) == policy
    assert response.payload["session"]["model_tool_policy"] == {
        "policy_id": policy["policy_id"],
        "policy_sha256": policy["policy_sha256"],
    }
    assert [(message["role"], message["content"]) for message in child_messages] == [
        ("user", "hello"),
    ]


@pytest.mark.asyncio
async def test_api_fork_leaves_legacy_parent_policy_unbound(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    instance = APIServerAdapter(PlatformConfig(enabled=True))
    instance._session_db = db
    db.create_session("legacy-parent", "api_server")
    db.append_message("legacy-parent", "user", "legacy hello")

    try:
        response = await _fork(
            instance, "legacy-parent", {"id": "legacy-child"}, monkeypatch,
        )
        child = db.get_session("legacy-child")
        child_messages = db.get_messages("legacy-child")
    finally:
        db.close()

    assert response.status == 201
    assert child["model_tool_policy_version"] is None
    assert child["model_tool_policy"] is None
    assert "model_tool_policy" not in response.payload["session"]
    assert [(message["role"], message["content"]) for message in child_messages] == [
        ("user", "legacy hello"),
    ]


@pytest.mark.parametrize(
    ("version", "stored"),
    [
        (MODEL_TOOL_POLICY_VERSION, None),
        (None, json.dumps(_policy("read_file"))),
        (MODEL_TOOL_POLICY_VERSION, "{"),
        (MODEL_TOOL_POLICY_VERSION + 1, json.dumps(_policy("read_file"))),
    ],
)
@pytest.mark.asyncio
async def test_api_fork_rejects_corrupt_parent_before_mutation(
    tmp_path, monkeypatch, version, stored,
):
    db = SessionDB(tmp_path / "state.db")
    instance = APIServerAdapter(PlatformConfig(enabled=True))
    instance._session_db = db
    db.create_session("corrupt-parent", "api_server")
    db.append_message("corrupt-parent", "user", "must remain")
    db._execute_write(lambda conn: conn.execute(
        """UPDATE sessions
           SET model_tool_policy_version = ?, model_tool_policy = ?
           WHERE id = ?""",
        (version, stored, "corrupt-parent"),
    ))

    try:
        response = await _fork(
            instance, "corrupt-parent", {"id": "must-not-exist"}, monkeypatch,
        )
        parent = db.get_session("corrupt-parent")
        parent_messages = db.get_messages("corrupt-parent")
        child = db.get_session("must-not-exist")
    finally:
        db.close()

    assert response.status == 409
    assert response.payload["error"]["code"] == "invalid_model_tool_policy"
    assert parent["ended_at"] is None
    assert parent["end_reason"] is None
    assert child is None
    assert [(message["role"], message["content"]) for message in parent_messages] == [
        ("user", "must remain"),
    ]
