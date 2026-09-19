"""Per-session capability manifests stay frozen across every agent-construction path."""

import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server


def _create(monkeypatch, params: dict) -> tuple[dict, dict]:
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    response = server._methods["session.create"]("create", params)
    assert "result" in response, response
    result = response["result"]
    return result, server._sessions[result["session_id"]]


def _clear_session(result: dict | None) -> None:
    if result:
        server._sessions.pop(result["session_id"], None)


def test_explicit_toolsets_are_frozen_and_survive_deferred_build(monkeypatch):
    result = None
    try:
        result, record = _create(monkeypatch, {"enabled_toolsets": ["file", "web"]})

        assert record["enabled_toolsets"] == ("file", "web")
        assert result["info"]["enabled_toolsets"] == ["file", "web"]
        kwargs = server._deferred_build_agent_kwargs(record, None)
        assert kwargs["enabled_toolsets"] == ["file", "web"]
    finally:
        _clear_session(result)


def test_explicit_empty_toolsets_stays_empty(monkeypatch):
    result = None
    try:
        result, record = _create(monkeypatch, {"enabled_toolsets": []})

        assert record["enabled_toolsets"] == ()
        assert server._deferred_build_agent_kwargs(record, None)["enabled_toolsets"] == []
        assert result["info"]["enabled_toolsets"] == []
    finally:
        _clear_session(result)


def test_omitted_toolsets_preserve_platform_defaults(monkeypatch):
    result = None
    try:
        result, record = _create(monkeypatch, {})

        assert "enabled_toolsets" not in record
        assert "enabled_toolsets" not in server._deferred_build_agent_kwargs(record, None)
        assert "enabled_toolsets" not in result["info"]
    finally:
        _clear_session(result)


@pytest.mark.parametrize("name", ["", "   ", "not-a-real-toolset"])
def test_invalid_explicit_toolset_fails_before_session_creation(monkeypatch, name):
    scheduled = []
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: scheduled.append(sid))
    before = set(server._sessions)

    response = server._methods["session.create"]("create", {"enabled_toolsets": [name]})

    assert response["error"]["code"] == 4000
    assert "enabled_toolsets" in response["error"]["message"]
    assert set(server._sessions) == before
    assert scheduled == []


@pytest.mark.parametrize("selection", [["file"], []])
def test_explicit_toolsets_reach_aiagent_without_platform_expansion(monkeypatch, selection):
    captured = {}
    monkeypatch.setattr("run_agent.AIAgent", lambda **kwargs: captured.update(kwargs) or SimpleNamespace(model="m"))
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_startup_system_prompt", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *_args: ("m", {}))
    monkeypatch.setattr(server, "_load_provider_routing", lambda: {})
    monkeypatch.setattr(server, "_resolve_agent_platform", lambda *_args: "desktop")
    monkeypatch.setattr(server, "_agent_cbs", lambda _sid: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)

    server._make_agent("sid", "key", enabled_toolsets=selection)

    assert captured["enabled_toolsets"] == selection


def test_preloaded_skills_do_not_change_inherited_toolsets(monkeypatch):
    captured = {}
    monkeypatch.setattr("run_agent.AIAgent", lambda **kwargs: captured.update(kwargs) or SimpleNamespace(model="m"))
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_startup_system_prompt", lambda *_args, **_kwargs: "SKILL BODY")
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *_args: ("m", {}))
    monkeypatch.setattr(server, "_load_provider_routing", lambda: {})
    monkeypatch.setattr(server, "_resolve_agent_platform", lambda *_args: "desktop")
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_args: ["hermes-cli"])
    monkeypatch.setattr(server, "_agent_cbs", lambda _sid: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)

    server._make_agent("sid", "key", preload_skills=["installed-skill"])

    assert captured["enabled_toolsets"] == ["hermes-cli"]
    assert captured.get("disabled_toolsets") is None


def test_exact_preload_skills_are_appended_to_startup_prompt(monkeypatch):
    calls = []

    def fake_build(names, task_id=None):
        calls.append((names, task_id))
        return "EXACT SKILL BODY", ["installed-skill"], []

    monkeypatch.setattr("agent.skill_commands.build_preloaded_skills_prompt", fake_build)
    monkeypatch.setattr(
        "hermes_cli.config.resolve_ephemeral_system_prompt_from_config",
        lambda _cfg: "BASE PROMPT",
    )

    prompt = server._startup_system_prompt({}, "session-key", ("installed-skill",))

    assert prompt == "BASE PROMPT\n\nEXACT SKILL BODY"
    assert calls == [(["installed-skill"], "session-key")]


def test_unknown_explicit_preload_skill_fails_agent_build(monkeypatch):
    monkeypatch.setattr(
        "agent.skill_commands.build_preloaded_skills_prompt",
        lambda names, task_id=None: ("", [], list(names)),
    )
    monkeypatch.setattr(
        "hermes_cli.config.resolve_ephemeral_system_prompt_from_config",
        lambda _cfg: "BASE",
    )

    with pytest.raises(ValueError, match=r"Unknown skill\(s\): missing-skill"):
        server._startup_system_prompt({}, "session-key", ("missing-skill",))


def test_sibling_sessions_keep_independent_capability_manifests(monkeypatch):
    restricted = ordinary = None
    try:
        restricted, restricted_record = _create(
            monkeypatch,
            {"enabled_toolsets": ["file"], "preload_skills": ["installed-skill"]},
        )
        ordinary, ordinary_record = _create(monkeypatch, {})

        assert restricted_record["enabled_toolsets"] == ("file",)
        assert restricted_record["preload_skills"] == ("installed-skill",)
        assert "enabled_toolsets" not in ordinary_record
        assert "preload_skills" not in ordinary_record
        assert "enabled_toolsets" not in server._deferred_build_agent_kwargs(ordinary_record, None)
        assert "preload_skills" not in server._deferred_build_agent_kwargs(ordinary_record, None)
    finally:
        _clear_session(restricted)
        _clear_session(ordinary)


def test_null_capability_fields_are_wire_compatible_omissions(monkeypatch):
    result = None
    try:
        result, record = _create(monkeypatch, {"enabled_toolsets": None, "preload_skills": None})
        assert "enabled_toolsets" not in record
        assert "preload_skills" not in record
        assert "enabled_toolsets" not in result["info"]
        assert "preload_skills" not in result["info"]
    finally:
        _clear_session(result)


@pytest.mark.parametrize("kwargs, expected", [({"enabled_toolsets": []}, True), ({"preload_skills": []}, False)])
def test_only_explicit_toolset_manifest_restricts_dynamic_tools(monkeypatch, kwargs, expected):
    agent = SimpleNamespace(model="m")
    monkeypatch.setattr("run_agent.AIAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_startup_system_prompt", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *_args: ("m", {}))
    monkeypatch.setattr(server, "_load_provider_routing", lambda: {})
    monkeypatch.setattr(server, "_resolve_agent_platform", lambda *_args: "desktop")
    monkeypatch.setattr(server, "_agent_cbs", lambda _sid: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)

    server._make_agent("sid", "key", **kwargs)

    assert getattr(agent, "_session_toolset_manifest_explicit") is expected


def test_compute_host_turn_frame_preserves_present_capability_fields(tmp_path):
    record = {
        "session_key": "stored",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "cols": 80,
        "cwd": str(tmp_path),
        "source": "desktop",
        "enabled_toolsets": (),
        "preload_skills": ("grounded-citations",),
    }

    frame = server._compute_host_turn_frame("rid", "runtime", record, "question")

    assert frame["enabled_toolsets"] == []
    assert frame["preload_skills"] == ["grounded-citations"]
