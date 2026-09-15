"""Real AIAgent schema and model-dispatch policy integration."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
from agent.memory_provider import MemoryProvider
from run_agent import AIAgent


SHA = "b" * 64


def _policy(*allowed):
    return {
        "policy_id": "session-policy",
        "policy_sha256": SHA,
        "allowed_tools": list(allowed),
        "approval_required_tools": [],
    }


def _definitions(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _agent(policy, definitions):
    with (
        patch("model_tools.get_tool_definitions", return_value=definitions),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://example.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            model_tool_policy=policy,
        )


def _tool_call(name, call_id):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def test_agent_construction_filters_exact_schema_without_copying_or_reordering():
    definitions = _definitions("read_file", "terminal", "search_files")
    agent = _agent(_policy("search_files", "read_file"), definitions)

    assert agent.tools == [definitions[0], definitions[2]]
    assert agent.tools[0] is definitions[0]
    assert agent.tools[1] is definitions[2]
    assert agent.valid_tool_names == {"read_file", "search_files"}


def test_agent_final_policy_validation_keeps_configured_dynamic_memory_tool():
    class DynamicProvider(MemoryProvider):
        @property
        def name(self):
            return "dynamic-memory"

        def is_available(self):
            return True

        def initialize(self, session_id, **kwargs):
            pass

        def get_tool_schemas(self):
            return [{
                "name": "dynamic_recall",
                "description": "Recall dynamic memory.",
                "parameters": {"type": "object", "properties": {}},
            }]

    cfg = {"memory": {"provider": "dynamic-memory"}, "agent": {}}
    with (
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=DynamicProvider()),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://example.invalid/v1",
            enabled_toolsets=["memory"],
            quiet_mode=True,
            skip_context_files=True,
            model_tool_policy=_policy("dynamic_recall"),
        )

    assert [tool["function"]["name"] for tool in getattr(agent, "tools", [])] == ["dynamic_recall"]
    assert getattr(agent, "valid_tool_names", set()) == {"dynamic_recall"}


def test_dynamic_schema_drift_fails_agent_build_before_first_provider_request():
    class DriftingProvider(MemoryProvider):
        def __init__(self):
            self.initialized = False

        @property
        def name(self):
            return "drifting-memory"

        def is_available(self):
            return True

        def initialize(self, session_id, **kwargs):
            self.initialized = True

        def get_tool_schemas(self):
            if self.initialized:
                return []
            return [{
                "name": "drifting_recall",
                "description": "A pre-init recognized dynamic tool.",
                "parameters": {"type": "object", "properties": {}},
            }]

    cfg = {"memory": {"provider": "drifting-memory"}, "agent": {}}
    policy = _policy("drifting_recall")
    client_factory = MagicMock()
    with (
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", side_effect=lambda *_a, **_k: DriftingProvider()),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", client_factory),
    ):
        from agent.model_tool_policy import validate_model_tool_policy_for_eventual_surface

        assert validate_model_tool_policy_for_eventual_surface(
            policy, enabled_toolsets=["memory"],
        ) == policy
        with pytest.raises(ValueError, match="unavailable.*drifting_recall"):
            AIAgent(
                api_key="test-key-1234567890",
                base_url="https://example.invalid/v1",
                enabled_toolsets=["memory"],
                quiet_mode=True,
                skip_context_files=True,
                model_tool_policy=policy,
            )

    assert client_factory.return_value.chat.completions.create.call_count == 0


@pytest.mark.parametrize("unsupported", ["delegate_task", "execute_code"])
def test_nested_execution_policy_fails_before_client_child_or_code_kernel_start(unsupported):
    client_factory = MagicMock()
    with (
        patch("agent.process_bootstrap.OpenAI", client_factory),
        patch("tools.delegate_tool._run_single_child") as delegate_entry,
        patch("tools.code_kernel.execute_in_session_kernel") as kernel_entry,
        pytest.raises(ValueError, match="unsupported V1 nested execution authority"),
    ):
        AIAgent(
            api_key="test-key-1234567890",
            base_url="https://example.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            model_tool_policy=_policy("read_file", unsupported),
        )

    client_factory.assert_not_called()
    delegate_entry.assert_not_called()
    kernel_entry.assert_not_called()


@pytest.mark.parametrize("executor", ["sequential", "concurrent"])
def test_forbidden_model_call_reaches_no_middleware_hook_or_handler(executor, monkeypatch):
    agent = _agent(_policy("read_file"), _definitions("read_file", "memory", "terminal"))
    calls = []
    inline_handler = MagicMock(side_effect=lambda *_a, **_k: calls.append("inline"))

    monkeypatch.setattr(
        "hermes_cli.middleware.apply_tool_request_middleware",
        lambda *_a, **_k: calls.append("request_middleware"),
    )
    monkeypatch.setattr(
        "hermes_cli.middleware.run_tool_execution_middleware",
        lambda *_a, **_k: calls.append("execution_middleware"),
    )
    monkeypatch.setattr(
        "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
        lambda *_a, **_k: calls.append("plugin_hook"),
    )
    monkeypatch.setattr(
        "agent.relay_tools.execute", lambda *_a, **_k: calls.append("relay")
    )
    monkeypatch.setattr(
        "model_tools.handle_function_call",
        lambda *_a, **_k: calls.append("registry"),
    )

    assistant = SimpleNamespace(tool_calls=[_tool_call("memory", f"{executor}-blocked")])
    messages = []
    with patch.dict(INLINE_TOOL_EXECUTORS, {"memory": inline_handler}):
        if executor == "sequential":
            agent._execute_tool_calls_sequential(assistant, messages, "policy-task")
        else:
            agent._execute_tool_calls_concurrent(assistant, messages, "policy-task")

    assert calls == []
    assert inline_handler.call_count == 0
    assert len(messages) == 1
    assert "not allowed" in json.loads(messages[0]["content"])["error"]
