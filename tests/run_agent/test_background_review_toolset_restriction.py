"""Tests that the background review agent restricts tools at runtime, not at schema time.

Regression coverage for issue #15204 (the background skill-review agent must
not perform non-skill side effects like terminal, send_message, delegate_task)
combined with issue #25322 / PR #17276 (the review fork must hit the parent's
Anthropic/OpenRouter prefix cache).

Reconciling the two: the fork now inherits the parent's full ``tools`` schema
so the cache-key matches, and enforces the memory+skills restriction at
runtime via a thread-local whitelist on the existing
``get_pre_tool_call_block_message`` gate. Safety is preserved mechanically
(any non-whitelisted dispatch is blocked) without the schema-level narrowing
that caused the prefix-cache miss.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _make_agent_stub(agent_cls):
    """Create a minimal AIAgent-like object with just enough state for _spawn_background_review."""
    agent = object.__new__(agent_cls)
    agent.model = "test-model"
    agent.platform = "test"
    agent.provider = "openai"
    agent.session_id = "sess-123"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 5
    agent._skill_nudge_interval = 5
    agent.background_review_callback = None
    agent.status_callback = None
    agent._cached_system_prompt = None
    import datetime as _dt
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    # Non-None so the test catches a missing-kwarg regression.
    agent.enabled_toolsets = ["memory", "skills", "terminal"]
    agent.disabled_toolsets = ["spotify", "feishu_doc"]
    return agent


class _SyncThread:
    """Drop-in replacement for threading.Thread that runs the target inline."""

    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def test_background_review_matches_parent_toolset_config():
    """Fork must receive parent's toolset config so ``tools[]`` cache key matches."""
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)
    captured = {}

    def _capture_init(self, *args, **kwargs):
        captured["enabled_toolsets"] = kwargs.get("enabled_toolsets", "UNSET")
        captured["disabled_toolsets"] = kwargs.get("disabled_toolsets", "UNSET")
        raise RuntimeError("stop after capturing init args")

    with patch.object(run_agent.AIAgent, "__init__", _capture_init), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "enabled_toolsets" in captured, "AIAgent.__init__ was not called"
    assert captured["enabled_toolsets"] == agent.enabled_toolsets, (
        f"enabled_toolsets mismatch: {captured['enabled_toolsets']!r} "
        f"vs expected {agent.enabled_toolsets!r}"
    )
    assert captured["disabled_toolsets"] == agent.disabled_toolsets, (
        f"disabled_toolsets mismatch: {captured['disabled_toolsets']!r} "
        f"vs expected {agent.disabled_toolsets!r}"
    )


def test_background_review_installs_thread_local_whitelist():
    """The review fork must install a memory/skills-only thread-local whitelist.

    The schema-level toolset narrowing was lifted (for prefix-cache parity),
    so #15204's safety contract now relies on the runtime whitelist gate to
    deny terminal/send_message/delegate_task at dispatch time. Verify the
    whitelist is set with exactly the memory+skills tool names.
    """
    import run_agent
    from hermes_cli import plugins as _plugins

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)
        captured["deny_msg_fmt"] = deny_msg_fmt
        # Stop here — we just want to see what gets installed.
        raise RuntimeError("stop after capturing whitelist")

    agent = _make_agent_stub(run_agent.AIAgent)

    def _no_init(self, *args, **kwargs):
        # Don't crash AIAgent.__init__; let execution flow reach
        # set_thread_tool_whitelist.
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "whitelist" in captured, "set_thread_tool_whitelist was not called"
    whitelist = captured["whitelist"]
    # memory + skills tools must be allowed
    assert "memory" in whitelist
    assert "skill_manage" in whitelist
    assert "skill_view" in whitelist
    assert "skills_list" in whitelist
    # read-only file tools are allowed too (#61521): the model reaches for
    # read_file to inspect a skill before patching; denying it caused a
    # per-review denial storm that starved the self-improvement loop.
    assert "read_file" in whitelist
    assert "search_files" in whitelist
    # write/dangerous tools must NOT be in the whitelist
    assert "write_file" not in whitelist
    assert "patch" not in whitelist
    assert "terminal" not in whitelist
    assert "send_message" not in whitelist
    assert "delegate_task" not in whitelist
    assert "web_search" not in whitelist
    assert "execute_code" not in whitelist
    # The deny message must name the correct substitutes so a single denial
    # redirects the model instead of a 142-denial storm (#61521).
    deny = captured.get("deny_msg_fmt") or ""
    assert "skill_manage" in deny
    assert "skill_view" in deny


def test_read_file_registers_background_review_read_mark(tmp_path):
    """read_file inside a review fork must satisfy the read-before-write guard.

    The whitelist now allows read_file; without this mark, the model would
    read SKILL.md via read_file and still get "content has not been loaded
    in this review turn" on the follow-up skill_manage patch (#61521).
    """
    from tools.file_tools import read_file_tool
    from tools.skill_manager_guards import (
        _background_review_has_read,
        _reset_background_review_read_marks,
    )
    from tools.skill_provenance import (
        BACKGROUND_REVIEW,
        reset_current_write_origin,
        set_current_write_origin,
    )

    target = tmp_path / "SKILL.md"
    target.write_text("---\nname: t\n---\nbody\n")

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        _reset_background_review_read_marks()
        assert not _background_review_has_read(target)
        out = read_file_tool(str(target), task_id="bg-review-test")
        assert "body" in out
        assert _background_review_has_read(target), (
            "full read_file inside a review fork must register with the "
            "read-before-write guard"
        )

        # A partial read must NOT satisfy the guard.
        _reset_background_review_read_marks()
        read_file_tool(str(target), offset=2, task_id="bg-review-test2")
        assert not _background_review_has_read(target)
    finally:
        reset_current_write_origin(token)


def test_read_file_outside_review_does_not_mark(tmp_path):
    """Foreground reads must not populate the review-fork read set."""
    from tools.file_tools import read_file_tool
    from tools.skill_manager_guards import (
        _background_review_has_read,
        _reset_background_review_read_marks,
    )

    target = tmp_path / "SKILL.md"
    target.write_text("content\n")
    _reset_background_review_read_marks()
    read_file_tool(str(target), task_id="fg-test")
    assert not _background_review_has_read(target)


def test_background_review_whitelist_includes_configured_extra_tools(
    tmp_path, monkeypatch
):
    """A profile may opt a specific proposal tool into background review.

    The review fork inherits the parent's full tool schema for cache parity,
    but runtime dispatch remains denied unless the tool is also present in the
    thread-local whitelist.  This config hook lets profiles grant a narrowly
    scoped, human-gated proposal tool without enabling unrelated side effects.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "auxiliary:\n"
        "  background_review:\n"
        "    extra_tools:\n"
        "      - propose_shared_memory\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import run_agent
    from hermes_cli import config as config_module
    from hermes_cli import plugins as _plugins

    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()

    captured = {}

    def _capture_whitelist(whitelist, deny_msg_fmt=None):
        captured["whitelist"] = set(whitelist)

    def _capture_run_conversation(self, *, user_message, **kwargs):
        captured["review_prompt"] = user_message
        return {"final_response": "Nothing to save."}

    agent = _make_agent_stub(run_agent.AIAgent)

    def _no_init(self, *args, **kwargs):
        return None

    with patch.object(run_agent.AIAgent, "__init__", _no_init), \
         patch.object(
             run_agent.AIAgent,
             "run_conversation",
             _capture_run_conversation,
         ), \
         patch.object(run_agent.AIAgent, "shutdown_memory_provider", lambda self: None), \
         patch.object(run_agent.AIAgent, "close", lambda self: None), \
         patch.object(_plugins, "set_thread_tool_whitelist", _capture_whitelist), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "propose_shared_memory" in captured["whitelist"]
    assert "terminal" not in captured["whitelist"]
    assert "propose_shared_memory" in captured["review_prompt"]


def test_propose_mode_uses_only_proposal_and_read_tools_for_automatic_and_refine():
    """Proposal mode keeps persistence behind one configured parent tool on every review path."""
    import agent.background_review as bg
    from hermes_cli import plugins as plugin_runtime
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    defaults = DEFAULT_CONFIG["auxiliary"]["background_review"]
    assert defaults["mode"] == "apply"
    assert defaults["proposal_tool"] == ""
    assert bg._background_review_mode({}) == "apply"
    assert bg._background_review_mode({"mode": "apply"}) == "apply"
    assert bg._background_review_mode({"mode": "propose"}) == "propose"

    task_cfg = {
        "mode": "propose",
        "proposal_tool": "propose_shared_memory",
        "extra_tools": ["propose_shared_memory", "memory", "skill_manage"],
    }
    parent = SimpleNamespace(
        tools=[
            {"type": "function", "function": {"name": name, "parameters": {}}}
            for name in ("propose_shared_memory", "memory", "skill_manage", "terminal")
        ]
    )
    calls = []

    def fake_build(agent, task_cfg=None, *, max_iterations, write_origin="background_review"):
        fork = SimpleNamespace(
            _memory_enabled=True,
            _user_profile_enabled=True,
            _session_messages=[],
            run_conversation=lambda **kwargs: calls.append(kwargs),
        )
        return fork, {}, False

    whitelists = []
    noop = lambda *args, **kwargs: None
    with patch.object(bg, "build_cache_parity_fork", fake_build), \
            patch.object(bg, "_track_review_fork", noop), \
            patch.object(bg, "_snapshot_review_usage", lambda agent: {}), \
            patch.object(bg, "_record_review_usage_to_parent", noop), \
            patch.object(bg, "finish_background_review_run", noop), \
            patch.object(bg, "_release_fork_clients", noop), \
            patch.object(
                plugin_runtime,
                "set_thread_tool_whitelist",
                lambda whitelist, deny_msg_fmt=None: whitelists.append(set(whitelist)),
            ), \
            patch.object(plugin_runtime, "clear_thread_tool_whitelist", noop):
        for explicit in (False, True):
            bg._run_review_fork(
                parent, [], "review both", task_cfg, None, bg._ReviewForkState(),
                review_memory=True, explicit=explicit,
            )

    expected = {
        "propose_shared_memory", "skill_view", "skills_list", "read_file", "search_files",
    }
    assert whitelists == [expected, expected]
    assert len(calls) == 2
    for call in calls:
        prompt = call["user_message"]
        assert "propose_shared_memory" in prompt
        assert "sole persistence action" in prompt.lower()
        assert "direct memory or skill mutation" in prompt.lower()


def test_brace_bearing_proposal_tool_cannot_break_denial_formatting():
    """A registered proposal name containing braces must not make a denied call fail open."""
    import agent.background_review as bg
    from hermes_cli.plugins import get_pre_tool_call_block_message

    proposal_tool = "propose_{shared}_memory"
    task_cfg = {
        "mode": "propose",
        "proposal_tool": proposal_tool,
        "extra_tools": [proposal_tool],
    }
    parent = SimpleNamespace(
        tools=[
            {"type": "function", "function": {"name": proposal_tool, "parameters": {}}}
        ]
    )
    denials = []

    def fake_build(agent, task_cfg=None, *, max_iterations, write_origin="background_review"):
        fork = SimpleNamespace(
            _memory_enabled=True,
            _user_profile_enabled=True,
            _session_messages=[],
            run_conversation=lambda **kwargs: denials.append(
                get_pre_tool_call_block_message("terminal", {})
            ),
        )
        return fork, {}, False

    noop = lambda *args, **kwargs: None
    with patch.object(bg, "build_cache_parity_fork", fake_build), \
            patch.object(bg, "_track_review_fork", noop), \
            patch.object(bg, "_snapshot_review_usage", lambda agent: {}), \
            patch.object(bg, "_record_review_usage_to_parent", noop), \
            patch.object(bg, "finish_background_review_run", noop), \
            patch.object(bg, "_release_fork_clients", noop):
        bg._run_review_fork(
            parent, [], "review both", task_cfg, None, bg._ReviewForkState(),
            review_memory=True, explicit=False,
        )

    assert len(denials) == 1
    assert denials[0] is not None
    assert proposal_tool in denials[0]
    assert "terminal" in denials[0]


@pytest.mark.parametrize(
    ("task_cfg", "parent_tools"),
    [
        ({"mode": "propose", "extra_tools": ["propose_shared_memory"]}, ["propose_shared_memory"]),
        (
            {"mode": "propose", "proposal_tool": " ", "extra_tools": ["propose_shared_memory"]},
            ["propose_shared_memory"],
        ),
        (
            {"mode": "propose", "proposal_tool": "propose_shared_memory", "extra_tools": []},
            ["propose_shared_memory"],
        ),
        (
            {
                "mode": "propose",
                "proposal_tool": "propose_shared_memory",
                "extra_tools": ["propose_shared_memory"],
            },
            ["memory", "skill_manage"],
        ),
        pytest.param(
            {"mode": "propose", "proposal_tool": True, "extra_tools": ["True"]},
            ["True"],
            id="boolean-proposal-tool",
        ),
        pytest.param(
            {"mode": "propose", "proposal_tool": 7, "extra_tools": ["7"]},
            ["7"],
            id="numeric-proposal-tool",
        ),
        pytest.param(
            {
                "mode": "propose",
                "proposal_tool": {"name": "propose_shared_memory"},
                "extra_tools": ["{'name': 'propose_shared_memory'}"],
            },
            ["{'name': 'propose_shared_memory'}"],
            id="mapping-proposal-tool",
        ),
        pytest.param(
            {
                "mode": "propose",
                "proposal_tool": ["propose_shared_memory"],
                "extra_tools": ["['propose_shared_memory']"],
            },
            ["['propose_shared_memory']"],
            id="list-proposal-tool",
        ),
        pytest.param(
            {"mode": "propose", "proposal_tool": "memory", "extra_tools": ["memory"]},
            ["memory"],
            id="memory-cannot-be-proposal-tool",
        ),
        pytest.param(
            {
                "mode": "propose",
                "proposal_tool": "skill_manage",
                "extra_tools": ["skill_manage"],
            },
            ["skill_manage"],
            id="skill-manage-cannot-be-proposal-tool",
        ),
    ],
)
def test_invalid_propose_mode_fails_before_fork_or_provider_call(
    task_cfg, parent_tools, caplog
):
    """A proposal review without an available, explicitly admitted tool fails visibly and closed."""
    import agent.background_review as bg

    failures = []
    parent = SimpleNamespace(
        provider="openai",
        client=None,
        tools=[
            {"type": "function", "function": {"name": name, "parameters": {}}}
            for name in parent_tools
        ],
        _emit_auxiliary_failure=lambda task, error: failures.append((task, str(error))),
    )

    with patch.object(
        bg, "build_cache_parity_fork", side_effect=AssertionError("review fork must not be built")
    ), patch.object(bg, "_set_thread_approval_callback"), caplog.at_level(
        logging.WARNING, logger="agent.background_review"
    ):
        bg._run_review_in_thread(parent, [], "review both", task_cfg=task_cfg)

    assert failures and failures[0][0] == "background review"
    assert "proposal" in failures[0][1].lower()
    assert any("proposal" in record.message.lower() for record in caplog.records)


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(None, id="null"),
        pytest.param(False, id="false"),
        pytest.param(0, id="zero"),
        pytest.param("", id="empty"),
        pytest.param("unknown", id="unknown"),
    ],
)
def test_explicit_invalid_mode_fails_before_fork_or_provider_call(mode, caplog):
    """Only omission from a valid task mapping may select the legacy apply mode."""
    import agent.background_review as bg

    failures = []
    parent = SimpleNamespace(
        provider="openai",
        client=None,
        tools=[],
        _emit_auxiliary_failure=lambda task, error: failures.append((task, str(error))),
    )

    with patch.object(bg, "build_cache_parity_fork") as build_fork, patch.object(
        bg, "_set_thread_approval_callback"
    ), caplog.at_level(logging.WARNING, logger="agent.background_review"):
        bg._run_review_in_thread(parent, [], "review both", task_cfg={"mode": mode})

    build_fork.assert_not_called()
    assert failures and failures[0][0] == "background review"
    assert "mode" in failures[0][1].lower()
    assert any("mode" in record.message.lower() for record in caplog.records)


@pytest.mark.parametrize(
    "config_result",
    [
        pytest.param(
            {"auxiliary": {"background_review": []}},
            id="malformed-background-review-block",
        ),
        pytest.param(OSError("config unavailable"), id="config-read-failure"),
    ],
)
def test_invalid_or_unavailable_loaded_config_fails_before_fork_or_provider_call(
    config_result, caplog
):
    """A review without an explicit valid task mapping must not infer apply from bad config."""
    import agent.background_review as bg

    failures = []
    parent = SimpleNamespace(
        provider="openai",
        client=None,
        tools=[],
        _emit_auxiliary_failure=lambda task, error: failures.append((task, str(error))),
    )
    with patch("hermes_cli.config.load_config_readonly") as load_config, patch.object(
        bg, "build_cache_parity_fork"
    ) as build_fork, patch.object(bg, "_set_thread_approval_callback"), caplog.at_level(
        logging.WARNING, logger="agent.background_review"
    ):
        if isinstance(config_result, BaseException):
            load_config.side_effect = config_result
        else:
            load_config.return_value = config_result
        enabled, task_cfg = bg.load_background_review_settings()
        assert enabled is True
        bg._run_review_in_thread(parent, [], "review both", task_cfg=task_cfg)

    build_fork.assert_not_called()
    assert failures and failures[0][0] == "background review"
    assert "config" in failures[0][1].lower() or "background_review" in failures[0][1]
    assert caplog.records


@pytest.mark.parametrize(
    "malformed_yaml",
    [
        pytest.param("null", id="null"),
        pytest.param("[]", id="list"),
    ],
)
def test_malformed_task_from_real_config_load_blocks_review_before_fork(
    tmp_path, monkeypatch, malformed_yaml, caplog
):
    """The merged loader must retain explicit malformed task provenance for review preflight."""
    import agent.background_review as bg

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"auxiliary:\n  background_review: {malformed_yaml}\n",
        encoding="utf-8",
    )
    failures = []
    parent = SimpleNamespace(
        provider="openai",
        client=None,
        tools=[],
        _emit_auxiliary_failure=lambda task, error: failures.append((task, str(error))),
    )

    with patch.object(bg, "build_cache_parity_fork") as build_fork, patch.object(
        bg, "_set_thread_approval_callback"
    ), caplog.at_level(logging.WARNING, logger="agent.background_review"):
        enabled, task_cfg = bg.load_background_review_settings()
        bg._run_review_in_thread(parent, [], "review both", task_cfg=task_cfg)

    assert enabled is True
    build_fork.assert_not_called()
    assert failures and failures[0][0] == "background review"
    assert "background review" in failures[0][1].lower()
    assert caplog.records


@pytest.mark.parametrize("failure_kind", ["parse", "read"])
def test_cold_config_failure_fallback_blocks_review_before_fork(
    tmp_path, monkeypatch, caplog, failure_kind
):
    """A real cold parse/read failure must not turn the default apply mode into authority."""
    import agent.background_review as bg
    from hermes_cli import config as config_module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "auxiliary:\n  background_review: [unterminated\n"
        if failure_kind == "parse"
        else "auxiliary:\n  background_review:\n    mode: propose\n",
        encoding="utf-8",
    )
    path_key = str(config_path)
    config_module._LOAD_CONFIG_CACHE.pop(path_key, None)
    config_module._LAST_EXPANDED_CONFIG_BY_PATH.pop(path_key, None)

    # Exercise the actual cold loader first; load_background_review_settings then consumes its
    # cached effective result, matching the production flow that previously exposed mode=apply.
    if failure_kind == "read":
        real_open = open

        def deny_config_read(file, mode="r", *args, **kwargs):
            if str(file) == str(config_path) and "r" in mode:
                raise PermissionError("config read denied")
            return real_open(file, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=deny_config_read):
            loaded = config_module.load_config_readonly()
    else:
        loaded = config_module.load_config_readonly()
    assert isinstance(loaded, dict)

    failures = []
    parent = SimpleNamespace(
        provider="openai",
        client=None,
        tools=[],
        _emit_auxiliary_failure=lambda task, error: failures.append((task, str(error))),
    )
    with patch.object(
        bg,
        "build_cache_parity_fork",
        side_effect=AssertionError("cold config fallback must block before fork"),
    ) as build_fork, patch.object(bg, "_set_thread_approval_callback"), caplog.at_level(
        logging.WARNING, logger="agent.background_review"
    ):
        enabled, task_cfg = bg.load_background_review_settings()
        bg._run_review_in_thread(parent, [], "review both", task_cfg=task_cfg)

    assert enabled is True
    build_fork.assert_not_called()
    assert failures and failures[0][0] == "background review"
    assert "config" in failures[0][1].lower()


def test_malformed_config_keeps_prior_proposal_mode_last_known_good(
    tmp_path, monkeypatch
):
    """A later parse failure keeps a valid in-process proposal configuration."""
    import agent.background_review as bg
    from hermes_cli import config as config_module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "auxiliary:\n"
        "  background_review:\n"
        "    mode: propose\n"
        "    proposal_tool: propose_shared_memory\n"
        "    extra_tools: [propose_shared_memory]\n",
        encoding="utf-8",
    )
    path_key = str(config_path)
    config_module._LOAD_CONFIG_CACHE.pop(path_key, None)
    config_module._LAST_EXPANDED_CONFIG_BY_PATH.pop(path_key, None)

    good = config_module.load_config_readonly()["auxiliary"]["background_review"]
    assert good["mode"] == "propose"

    config_path.write_text(
        "auxiliary:\n  background_review: [unterminated and changed\n",
        encoding="utf-8",
    )
    after = config_module.load_config_readonly()["auxiliary"]["background_review"]
    enabled, task_cfg = bg.load_background_review_settings()

    assert enabled is True
    assert after == good
    assert task_cfg == good
    assert bg._background_review_mode(task_cfg) == "propose"
