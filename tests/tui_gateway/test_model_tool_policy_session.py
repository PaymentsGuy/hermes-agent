"""TUI/Desktop session policy creation and resume integration."""

import json
import threading
from types import SimpleNamespace

import pytest

from agent.memory_provider import MemoryProvider
from agent.model_tool_policy import MODEL_TOOL_POLICY_VERSION
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from tui_gateway import server


SHA = "c" * 64


def _policy(*allowed, policy_id="desktop-policy"):
    return {
        "policy_id": policy_id,
        "policy_sha256": SHA,
        "allowed_tools": list(allowed),
        "approval_required_tools": [],
    }


def _definitions(*names):
    return [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        for name in names
    ]


class _ProfileMemoryProvider(MemoryProvider):
    def __init__(self, tool_name, *, available=True):
        self._tool_name = tool_name
        self._available = available

    @property
    def name(self):
        return "profile-memory"

    def is_available(self):
        return self._available

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return [{
            "name": self._tool_name,
            "description": "Profile-scoped memory lookup.",
            "parameters": {"type": "object", "properties": {}},
        }]


@pytest.fixture
def gateway_db(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_completion_cwd", lambda *_a, **_k: str(tmp_path))
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda _platform: ["file"])
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **_kwargs: _definitions("read_file", "search_files"),
    )
    server._sessions.clear()
    try:
        yield db
    finally:
        server._sessions.clear()
        db.close()


def test_session_create_validates_identity_and_persists_policy_before_first_prompt(gateway_db):
    policy = _policy("read_file")
    response = server.handle_request({
        "id": "create",
        "method": "session.create",
        "params": {"source": "desktop", "model_tool_policy": policy},
    })

    result = response["result"]
    assert result["info"]["model_tool_policy"] == {
        "policy_id": policy["policy_id"],
        "policy_sha256": policy["policy_sha256"],
    }
    row = gateway_db.get_session(result["stored_session_id"])
    assert row["model_tool_policy_version"] == MODEL_TOOL_POLICY_VERSION
    assert json.loads(row["model_tool_policy"]) == policy
    assert server._sessions[result["session_id"]]["model_tool_policy"] == policy


def test_session_create_rejects_unavailable_tool_without_live_or_persisted_session(gateway_db):
    response = server.handle_request({
        "id": "invalid",
        "method": "session.create",
        "params": {"model_tool_policy": _policy("missing_tool")},
    })

    assert response["error"]["code"] == 4027
    assert "unavailable" in response["error"]["message"]
    assert server._sessions == {}
    assert gateway_db.list_sessions_rich(limit=10) == []


@pytest.mark.parametrize("unsupported", ["delegate_task", "execute_code"])
def test_session_create_rejects_nested_execution_before_live_or_persisted_session(
    gateway_db, unsupported,
):
    response = server.handle_request({
        "id": f"invalid-{unsupported}",
        "method": "session.create",
        "params": {"model_tool_policy": _policy("read_file", unsupported)},
    })

    assert response["error"]["code"] == 4027
    assert "unsupported V1 nested execution authority" in response["error"]["message"]
    assert unsupported in response["error"]["message"]
    assert server._sessions == {}
    assert gateway_db.list_sessions_rich(limit=10) == []


@pytest.mark.parametrize("unsupported", ["delegate_task", "execute_code"])
def test_resume_rejects_stored_nested_execution_policy_before_runtime_registration(
    gateway_db, unsupported,
):
    gateway_db.create_session("unsupported-policy", "desktop")
    gateway_db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET model_tool_policy_version = ?, model_tool_policy = ? WHERE id = ?",
        (MODEL_TOOL_POLICY_VERSION, json.dumps(_policy("read_file", unsupported)), "unsupported-policy"),
    ))

    response = server.handle_request({
        "id": f"resume-{unsupported}",
        "method": "session.resume",
        "params": {"session_id": "unsupported-policy"},
    })

    assert response["error"]["code"] == 4027
    assert "unsupported V1 nested execution authority" in response["error"]["message"]
    assert server._sessions == {}


def test_tui_policy_validation_uses_profile_scoped_dynamic_memory_surface(
    gateway_db, monkeypatch, tmp_path,
):
    homes = {name: tmp_path / name for name in ("alpha", "beta", "disabled", "unconfigured")}
    for home in homes.values():
        home.mkdir()
    providers = {
        str(homes["alpha"]): _ProfileMemoryProvider("alpha_recall"),
        str(homes["beta"]): _ProfileMemoryProvider("beta_recall"),
        str(homes["disabled"]): _ProfileMemoryProvider("disabled_recall", available=False),
    }

    monkeypatch.setattr(server, "_profile_home", lambda name: homes.get(name))
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda _platform: ["memory"])
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: ({"memory": {}} if get_hermes_home() == homes["unconfigured"]
                 else {"memory": {"provider": "profile-memory"}}),
    )
    monkeypatch.setattr(
        "plugins.memory.load_memory_provider",
        lambda _name, **_kwargs: providers[str(get_hermes_home())],
    )
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kwargs: [])

    alpha = server.handle_request({
        "id": "alpha", "method": "session.create",
        "params": {"profile": "alpha", "model_tool_policy": _policy("alpha_recall")},
    })
    beta = server.handle_request({
        "id": "beta", "method": "session.create",
        "params": {"profile": "beta", "model_tool_policy": _policy("beta_recall")},
    })
    leaked = server.handle_request({
        "id": "leaked", "method": "session.create",
        "params": {"profile": "alpha", "model_tool_policy": _policy("beta_recall")},
    })
    disabled = server.handle_request({
        "id": "disabled", "method": "session.create",
        "params": {"profile": "disabled", "model_tool_policy": _policy("disabled_recall")},
    })
    unconfigured = server.handle_request({
        "id": "unconfigured", "method": "session.create",
        "params": {"profile": "unconfigured", "model_tool_policy": _policy("alpha_recall")},
    })
    unknown = server.handle_request({
        "id": "unknown", "method": "session.create",
        "params": {"profile": "alpha", "model_tool_policy": _policy("invented_tool")},
    })

    assert alpha is not None and "result" in alpha
    assert beta is not None and "result" in beta
    assert leaked is not None
    assert disabled is not None
    assert unconfigured is not None
    assert unknown is not None
    assert leaked["error"]["code"] == 4027
    assert disabled["error"]["code"] == 4027
    assert unconfigured["error"]["code"] == 4027
    assert unknown["error"]["code"] == 4027


def test_persisted_policy_restores_into_cold_record_and_deferred_agent_build(gateway_db):
    policy = _policy("search_files")
    gateway_db.create_session(
        "resume-policy", "desktop", model_tool_policy=policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )

    response = server.handle_request({
        "id": "resume",
        "method": "session.resume",
        "params": {"session_id": "resume-policy"},
    })

    result = response["result"]
    record = server._sessions[result["session_id"]]
    assert result["model_tool_policy"] == {
        "policy_id": policy["policy_id"], "policy_sha256": policy["policy_sha256"]
    }
    assert record["model_tool_policy"] == policy
    assert server._deferred_build_agent_kwargs(record, gateway_db)["model_tool_policy"] == policy


@pytest.mark.parametrize(
    ("version", "stored"),
    [(MODEL_TOOL_POLICY_VERSION, None), (None, json.dumps(_policy("read_file"))),
     (MODEL_TOOL_POLICY_VERSION, "{")],
)
def test_resume_rejects_declared_corrupt_policy_before_registering_runtime(
    gateway_db, version, stored
):
    gateway_db.create_session("corrupt-policy", "desktop")
    gateway_db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET model_tool_policy_version = ?, model_tool_policy = ? WHERE id = ?",
        (version, stored, "corrupt-policy"),
    ))

    response = server.handle_request({
        "id": "resume-corrupt",
        "method": "session.resume",
        "params": {"session_id": "corrupt-policy"},
    })

    assert response["error"]["code"] == 4027
    assert "stored model-tool policy invalid" in response["error"]["message"]
    assert server._sessions == {}


def test_legacy_resume_remains_unbound(gateway_db):
    gateway_db.create_session("legacy-session", "desktop")

    response = server.handle_request({
        "id": "resume-legacy",
        "method": "session.resume",
        "params": {"session_id": "legacy-session"},
    })

    result = response["result"]
    record = server._sessions[result["session_id"]]
    assert "model_tool_policy" not in result
    assert record["model_tool_policy"] is None
    assert server._deferred_build_agent_kwargs(record, gateway_db)["model_tool_policy"] is None


def test_resume_rejects_compression_tip_with_mismatched_policy(gateway_db):
    policy = _policy("read_file")
    gateway_db.create_session(
        "policy-parent", "desktop", model_tool_policy=policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    gateway_db.append_message("policy-parent", "user", "before compression")
    gateway_db.end_session("policy-parent", "compression")
    gateway_db.create_session(
        "policyless-child", "desktop", parent_session_id="policy-parent",
    )
    gateway_db.append_message("policyless-child", "user", "continued")

    response = server.handle_request({
        "id": "resume-invalid-continuation",
        "method": "session.resume",
        "params": {"session_id": "policy-parent"},
    })

    assert response["error"]["code"] == 4027
    assert "model-tool policy invalid" in response["error"]["message"]
    assert server._sessions == {}


def test_concurrent_live_sessions_keep_independent_policy_objects(gateway_db):
    first = server.handle_request({
        "id": "one", "method": "session.create",
        "params": {"model_tool_policy": _policy("read_file", policy_id="one")},
    })["result"]
    second = server.handle_request({
        "id": "two", "method": "session.create",
        "params": {"model_tool_policy": _policy("search_files", policy_id="two")},
    })["result"]

    assert server._sessions[first["session_id"]]["model_tool_policy"]["allowed_tools"] == ["read_file"]
    assert server._sessions[second["session_id"]]["model_tool_policy"]["allowed_tools"] == ["search_files"]


@pytest.mark.parametrize("rebuild", ["new", "bot"])
def test_policy_bound_agent_rebuilds_keep_full_policy_and_marker(
    gateway_db, monkeypatch, rebuild
):
    policy = _policy("read_file")
    gateway_db.create_session(
        "bound-rebuild", "desktop", model_tool_policy=policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    old = SimpleNamespace(
        _session_db=gateway_db, _owns_session_db=False, _session_title_hint="Bot Chat",
    )
    session = {
        "agent": old, "session_key": "bound-rebuild", "source": "desktop",
        "model_tool_policy": policy,
        "model_tool_policy_version": MODEL_TOOL_POLICY_VERSION,
        "history": [], "history_lock": threading.Lock(), "history_version": 0,
        "cwd": str(gateway_db.db_path.parent), "bot_caps_seen": "before",
    }
    captured = []

    def make_agent(*_args, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(_session_db=kwargs.get("session_db"), _owns_session_db=False)

    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("test-model", ""))
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_emit", lambda *_args: None)
    if rebuild == "new":
        server._reset_session_agent("bound-runtime", session)
    else:
        monkeypatch.setattr("tools.bot_mode_probe.capability_fingerprint", lambda _home: "after")
        server._sync_bot_capabilities("bound-runtime", session)

    assert captured[0]["model_tool_policy"] == policy
    assert session["model_tool_policy_version"] == MODEL_TOOL_POLICY_VERSION
    assert session["agent"] is not old


def test_corrupt_declared_policy_aborts_rebuild_without_replacing_agent(gateway_db, monkeypatch):
    old = SimpleNamespace(_session_db=gateway_db, _owns_session_db=False)
    session = {
        "agent": old, "session_key": "corrupt-rebuild", "source": "desktop",
        "model_tool_policy": None,
        "model_tool_policy_version": MODEL_TOOL_POLICY_VERSION,
    }
    built = []
    monkeypatch.setattr(server, "_make_agent", lambda *_a, **_k: built.append(True))

    with pytest.raises(ValueError, match="marker or payload"):
        server._rebuild_session_agent("corrupt-runtime", session)

    assert session["agent"] is old
    assert built == []


def test_legacy_rebuild_remains_unbound(gateway_db, monkeypatch):
    old = SimpleNamespace(_session_db=gateway_db, _owns_session_db=False)
    session = {"agent": old, "session_key": "legacy-rebuild", "source": "desktop"}
    captured = {}

    def make_agent(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(_session_db=gateway_db, _owns_session_db=False)

    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("test-model", ""))
    server._rebuild_session_agent("legacy-runtime", session)

    assert captured["model_tool_policy"] is None
    assert session.get("model_tool_policy_version") is None


def test_seeded_tui_branch_inherits_parent_policy_in_storage_and_build(gateway_db):
    policy = _policy("read_file")
    gateway_db.create_session(
        "bound-parent", "desktop", model_tool_policy=policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    gateway_db.append_message("bound-parent", "user", "parent message")

    response = server.handle_request({
        "id": "branch", "method": "session.create",
        "params": {
            "source": "desktop", "parent_session_id": "bound-parent",
            "messages": [{"role": "user", "content": "parent message"}],
        },
    })

    assert "result" in response, response
    result = response["result"]
    child = gateway_db.get_session(result["stored_session_id"])
    record = server._sessions[result["session_id"]]
    assert child["model_tool_policy_version"] == MODEL_TOOL_POLICY_VERSION
    assert json.loads(child["model_tool_policy"]) == policy
    assert record["model_tool_policy"] == policy
    assert record["model_tool_policy_version"] == MODEL_TOOL_POLICY_VERSION
    assert server._deferred_build_agent_kwargs(record, gateway_db)["model_tool_policy"] == policy


def test_seeded_tui_branch_rejects_corrupt_parent_policy(gateway_db):
    gateway_db.create_session("corrupt-parent", "desktop")
    gateway_db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET model_tool_policy_version = ? WHERE id = ?",
        (MODEL_TOOL_POLICY_VERSION, "corrupt-parent"),
    ))

    response = server.handle_request({
        "id": "branch", "method": "session.create",
        "params": {
            "source": "desktop", "parent_session_id": "corrupt-parent",
            "messages": [{"role": "user", "content": "parent message"}],
        },
    })

    assert response["error"]["code"] == 4027
    assert server._sessions == {}
