"""Model-tool policy continuity for TUI derivative agents."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tui_gateway import server


SHA = "d" * 64


def _policy(*allowed, policy_id="derivative-policy"):
    return {
        "policy_id": policy_id,
        "policy_sha256": SHA,
        "allowed_tools": list(allowed),
        "approval_required_tools": list(allowed[:1]),
    }


def _definition(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _parent(policy):
    return SimpleNamespace(
        model_tool_policy=policy,
        base_url="https://example.invalid/v1",
        api_key="test-key-1234567890",
        provider="openai",
        api_mode="chat_completions",
        model="test-model",
        enabled_toolsets=["file", "terminal"],
        disabled_toolsets=[],
        request_overrides={},
        reasoning_config=None,
        service_tier=None,
        fallback_model=[],
    )


def _helper_patches(definitions):
    return (
        patch.object(server, "_load_cfg", return_value={}),
        patch.object(server, "_get_db", return_value=None),
        patch("model_tools.get_tool_definitions", return_value=definitions),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    )


@pytest.mark.parametrize(
    "helper",
    [server._background_agent_kwargs, server._ephemeral_preview_agent_kwargs],
)
def test_side_agent_kwargs_carry_full_normalized_policy_and_construct_no_wider_schema(helper):
    policy = _policy("read_file")
    parent = _parent(policy)
    definitions = [_definition("terminal"), _definition("read_file"), _definition("search_files")]

    patches = _helper_patches(definitions)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        kwargs = helper(parent, "side-task")
        from run_agent import AIAgent

        kwargs["skip_context_files"] = True
        kwargs["skip_memory"] = True
        child = AIAgent(**kwargs)

    assert kwargs["model_tool_policy"] == policy
    assert kwargs["model_tool_policy"] is not policy
    assert child.model_tool_policy == policy
    assert [tool["function"]["name"] for tool in child.tools] == ["read_file"]
    assert child.tools[0] is definitions[1]


@pytest.mark.parametrize(
    ("method_name", "params", "policy"),
    [
        ("prompt.background", {"text": "work", "session_id": "ui"}, {"policy_id": "broken"}),
        (
            "preview.restart",
            {"url": "http://localhost:5173", "cwd": "", "session_id": "ui"},
            _policy("desktop_preview"),
        ),
    ],
)
def test_side_agent_invalid_or_unsatisfied_parent_policy_fails_before_thread_or_provider(
    method_name, params, policy,
):
    parent = _parent(policy)
    session = {"agent": parent, "session_key": "session", "profile_home": None, "history": []}
    threads = []
    providers = MagicMock()

    class RecordingThread:
        def __init__(self, *args, **kwargs):
            threads.append((args, kwargs))

        def start(self):
            raise AssertionError("thread must not start")

    with (
        patch.object(server, "_sess", return_value=(session, None)),
        patch.object(server, "_load_cfg", return_value={}),
        patch.object(server, "_get_db", return_value=None),
        patch.object(server, "_preview_restart_history", return_value=[]),
        patch.object(server.threading, "Thread", RecordingThread),
        patch("model_tools.get_tool_definitions", return_value=[_definition("read_file")]),
        patch("agent.process_bootstrap.OpenAI", providers),
        pytest.raises(ValueError),
    ):
        server._methods[method_name]("rid", params)

    assert threads == []
    providers.assert_not_called()


def test_legacy_side_agents_remain_unbound_and_independent_policy_copies_do_not_leak():
    first_policy = _policy("read_file", policy_id="first")
    second_policy = _policy("search_files", policy_id="second")
    first = _parent(first_policy)
    second = _parent(second_policy)
    legacy = _parent(None)
    definitions = [_definition("read_file"), _definition("search_files")]

    patches = _helper_patches(definitions)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        first_kwargs = server._background_agent_kwargs(first, "first-task")
        second_kwargs = server._background_agent_kwargs(second, "second-task")
        legacy_kwargs = server._background_agent_kwargs(legacy, "legacy-task")

    first_kwargs["model_tool_policy"]["allowed_tools"].append("search_files")
    assert first.model_tool_policy == first_policy
    assert second_kwargs["model_tool_policy"] == second_policy
    assert legacy_kwargs["model_tool_policy"] is None
