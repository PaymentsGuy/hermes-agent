"""Deferred background review on the managed local runtime.

Behavior contracts for agent/review_idle_queue.py and the decision
wrapper in run_agent.AIAgent._spawn_background_review:

- defer: auto + review runtime == managed local  -> queued, not spawned
- defer: never, or non-managed runtime, or /refine -> immediate spawn
- queue coalesces per session (newest snapshot wins, age preserved)
- dispatch requires sustained process-quiet AND server idle
- aged-out items dispatch regardless of idleness (delay, never lose)
- preempted deferred reviews requeue with a bounded attempt cap
"""

import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.review_idle_queue import (
    ReviewIdleQueue,
    _IDLE_SETTLE_S,
    defer_max_age_s,
    defer_mode,
)


# ── config parsing ───────────────────────────────────────────────


def test_defer_mode_values():
    assert defer_mode(None) == "auto"
    assert defer_mode({}) == "auto"
    assert defer_mode({"defer": "auto"}) == "auto"
    assert defer_mode({"defer": "never"}) == "never"
    assert defer_mode({"defer": "NEVER"}) == "never"
    # Unknown values fall back to auto (the safe, documented default).
    assert defer_mode({"defer": "sometimes"}) == "auto"
    assert defer_mode({"defer": 3}) == "auto"


def test_defer_max_age_parsing():
    assert defer_max_age_s(None) == 30 * 60
    assert defer_max_age_s({"defer_max_age_s": 120}) == 120.0
    assert defer_max_age_s({"defer_max_age_s": "600"}) == 600.0
    # Nonsense and non-positive fall back to the default.
    assert defer_max_age_s({"defer_max_age_s": "soon"}) == 30 * 60
    assert defer_max_age_s({"defer_max_age_s": 0}) == 30 * 60
    assert defer_max_age_s({"defer_max_age_s": -5}) == 30 * 60


# ── queue harness ────────────────────────────────────────────────


class _FakeAgent:
    def __init__(self):
        self.spawned = []
        self.session_id = "sess-x"
        self.tools = []
        self.failures = []

    def _spawn_background_review_now(self, **kwargs):
        self.spawned.append(kwargs)

    def _emit_auxiliary_failure(self, task, error):
        self.failures.append((task, str(error)))


def _make_queue(now=None, server_idle=True):
    q = ReviewIdleQueue()
    clock = {"t": 0.0}
    if now is None:
        q._now = lambda: clock["t"]
    else:
        q._now = now
    q._server_idle = lambda: server_idle
    # Never start the real dispatcher thread in unit tests.
    q._ensure_thread = lambda: None
    return q, clock


def test_enqueue_coalesces_per_session_newest_wins_oldest_age():
    q, clock = _make_queue()
    agent = _FakeAgent()

    clock["t"] = 100.0
    q.enqueue(agent, "s1", {"messages_snapshot": ["old"], "task_cfg": {}})
    clock["t"] = 200.0
    q.enqueue(agent, "s1", {"messages_snapshot": ["new"], "task_cfg": {}})
    q.enqueue(agent, "s2", {"messages_snapshot": ["other"], "task_cfg": {}})

    assert q.pending_count() == 2
    with q._lock:
        item = q._pending["s1"]
    # Newest snapshot won, but the age clock kept the ORIGINAL enqueue
    # time so a busy session cannot push its own age-out forever.
    assert item.kwargs["messages_snapshot"] == ["new"]
    assert item.enqueued_at == 100.0


def test_dispatch_waits_for_sustained_quiet():
    q, clock = _make_queue()
    agent = _FakeAgent()
    q.enqueue(agent, "s1", {"task_cfg": {}})

    # A live turn: nothing dispatches.
    q.note_turn_started()
    assert q._pop_dispatchable() is None

    # Turn finished, but the settle window hasn't elapsed.
    q.note_turn_finished()
    assert q._pop_dispatchable() is None

    # Quiet long enough -> dispatchable.
    clock["t"] += _IDLE_SETTLE_S + 1
    item = q._pop_dispatchable()
    assert item is not None and item.session_key == "s1"
    assert q.pending_count() == 0


def test_dispatch_blocked_by_busy_server():
    q, clock = _make_queue(server_idle=False)
    agent = _FakeAgent()
    q.enqueue(agent, "s1", {"task_cfg": {}})
    q.note_turn_started()
    q.note_turn_finished()
    clock["t"] += _IDLE_SETTLE_S + 1
    # Process is quiet but the managed server has a processing slot
    # (another profile's session, a live prefill): hold.
    assert q._pop_dispatchable() is None
    assert q.pending_count() == 1


def test_aged_out_item_dispatches_despite_busy_server():
    q, clock = _make_queue(server_idle=False)
    agent = _FakeAgent()
    q.enqueue(agent, "s1", {"task_cfg": {"defer_max_age_s": 60}})
    q.note_turn_started()  # never goes quiet
    clock["t"] += 61
    item = q._pop_dispatchable()
    assert item is not None
    assert item.session_key == "s1"


def test_new_turn_resets_the_quiet_clock():
    q, clock = _make_queue()
    agent = _FakeAgent()
    q.enqueue(agent, "s1", {"task_cfg": {}})
    q.note_turn_started()
    q.note_turn_finished()
    clock["t"] += _IDLE_SETTLE_S - 2
    # A new prompt arrives just before the settle window closes.
    q.note_turn_started()
    clock["t"] += 30
    assert q._pop_dispatchable() is None  # still live
    q.note_turn_finished()
    assert q._pop_dispatchable() is None  # settle restarts
    clock["t"] += _IDLE_SETTLE_S + 1
    assert q._pop_dispatchable() is not None


def test_nested_turns_require_all_to_finish():
    q, clock = _make_queue()
    agent = _FakeAgent()
    q.enqueue(agent, "s1", {"task_cfg": {}})
    q.note_turn_started()
    q.note_turn_started()
    q.note_turn_finished()
    clock["t"] += _IDLE_SETTLE_S + 1
    assert q._pop_dispatchable() is None  # one turn still live
    q.note_turn_finished()
    clock["t"] += _IDLE_SETTLE_S + 1
    assert q._pop_dispatchable() is not None


# ── the decision wrapper ─────────────────────────────────────────


def _wrapper_agent(monkeypatch, defer="auto", managed=True):
    """A minimal object wearing the real _spawn_background_review."""
    import run_agent
    from agent import review_idle_queue as riq

    agent = _FakeAgent()
    agent._delegate_depth = 0
    calls = {"enqueued": [], "spawned": []}

    monkeypatch.setattr(
        "agent.background_review.load_background_review_settings",
        lambda: (True, {"defer": defer}),
    )
    monkeypatch.setattr(
        riq, "review_targets_managed_local", lambda a, cfg: managed
    )
    monkeypatch.setattr(
        riq.QUEUE, "enqueue",
        lambda a, key, kw: calls["enqueued"].append((key, kw)),
    )
    agent._spawn_background_review_now = (
        lambda **kw: calls["spawned"].append(kw)
    )
    bound = types.MethodType(run_agent.AIAgent._spawn_background_review, agent)
    return bound, calls


def test_wrapper_defers_managed_local_auto(monkeypatch):
    spawn, calls = _wrapper_agent(monkeypatch, defer="auto", managed=True)
    spawn([{"role": "user", "content": "hi"}], review_memory=True)
    assert len(calls["enqueued"]) == 1
    assert calls["spawned"] == []
    key, kwargs = calls["enqueued"][0]
    assert key == "sess-x"
    assert kwargs["review_memory"] is True


def test_wrapper_spawns_immediately_for_non_managed(monkeypatch):
    spawn, calls = _wrapper_agent(monkeypatch, defer="auto", managed=False)
    spawn([{"role": "user", "content": "hi"}], review_skills=True)
    assert calls["enqueued"] == []
    assert len(calls["spawned"]) == 1


def test_wrapper_defer_never_is_old_behavior(monkeypatch):
    spawn, calls = _wrapper_agent(monkeypatch, defer="never", managed=True)
    spawn([{"role": "user", "content": "hi"}], review_memory=True)
    assert calls["enqueued"] == []
    assert len(calls["spawned"]) == 1


def test_wrapper_refine_bypasses_queue(monkeypatch):
    spawn, calls = _wrapper_agent(monkeypatch, defer="auto", managed=True)
    spawn([{"role": "user", "content": "hi"}], review_memory=True,
          focus="save the deploy workflow")
    assert calls["enqueued"] == []
    assert len(calls["spawned"]) == 1
    assert calls["spawned"][0]["focus"] == "save the deploy workflow"


def test_wrapper_bare_refine_bypasses_queue(monkeypatch):
    """/refine with no focus text is still explicit: never deferred."""
    spawn, calls = _wrapper_agent(monkeypatch, defer="auto", managed=True)
    spawn([{"role": "user", "content": "hi"}], review_memory=True,
          focus=None, explicit=True)
    assert calls["enqueued"] == []
    assert len(calls["spawned"]) == 1


def test_wrapper_cloud_fast_path_skips_runtime_resolution(monkeypatch):
    """No managed server on the machine -> the classifier answers from the
    TTL-cached netloc probe alone, without resolving the review runtime.
    Guards the cloud-only turn tail from growing new work."""
    from agent import review_idle_queue as riq

    resolved = {"count": 0}

    def _explode(agent, cfg):
        resolved["count"] += 1
        raise AssertionError("runtime resolution must not run")

    monkeypatch.setattr(
        "agent.auxiliary_client._managed_local_netloc", lambda: "")
    monkeypatch.setattr(
        "agent.background_review._resolve_review_runtime", _explode)
    assert riq.review_targets_managed_local(object(), {}) is False
    assert resolved["count"] == 0


def _write_review_profile(home, review_yaml, marker):
    from hermes_cli import config as config_module

    home.mkdir(parents=True, exist_ok=True)
    config_path = home / "config.yaml"
    config_path.write_text(
        f"auxiliary:\n  background_review:\n{review_yaml}", encoding="utf-8"
    )
    (home / "review-profile-marker").write_text(marker, encoding="utf-8")
    path_key = str(config_path)
    config_module._LOAD_CONFIG_CACHE.pop(path_key, None)
    config_module._LAST_EXPANDED_CONFIG_BY_PATH.pop(path_key, None)


class _ProfileAgent(_FakeAgent):
    def __init__(self, proposal_tool):
        super().__init__()
        self.tools = [
            {
                "type": "function",
                "function": {"name": proposal_tool, "parameters": {}},
            }
        ]

    def _spawn_background_review_now(self, **kwargs):
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        self.spawned.append(
            {
                "home": home,
                "marker": (home / "review-profile-marker").read_text(encoding="utf-8"),
                "kwargs": kwargs,
            }
        )


def _enqueue_in_profile(q, agent, profile_home, session_key, messages):
    import agent.background_review as bg
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(profile_home)
    try:
        _enabled, task_cfg = bg.load_background_review_settings()
        q.enqueue(
            agent,
            session_key,
            {"messages_snapshot": messages, "task_cfg": task_cfg},
        )
    finally:
        reset_hermes_home_override(token)


def _take_pending(q, session_key):
    with q._lock:
        return q._pending.pop(session_key)


def test_dispatch_uses_named_profile_disabled_policy_after_scope_exit(
    tmp_path, monkeypatch
):
    default_home = tmp_path / "default"
    named_home = default_home / "profiles" / "disabled"
    _write_review_profile(default_home, "    mode: apply\n", "default")
    _write_review_profile(named_home, "    enabled: false\n    mode: propose\n", "disabled")
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    q, _clock = _make_queue()
    agent = _ProfileAgent("proposal_disabled")
    _enqueue_in_profile(q, agent, named_home, "named-disabled", ["named"])

    q._dispatch(_take_pending(q, "named-disabled"))

    assert agent.spawned == []


def test_dispatch_uses_named_profile_proposal_policy_and_home_after_scope_exit(
    tmp_path, monkeypatch
):
    default_home = tmp_path / "default"
    named_home = default_home / "profiles" / "propose"
    _write_review_profile(default_home, "    mode: apply\n", "default")
    _write_review_profile(
        named_home,
        "    mode: propose\n"
        "    proposal_tool: proposal_named\n"
        "    extra_tools: [proposal_named]\n",
        "named",
    )
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    q, _clock = _make_queue()
    agent = _ProfileAgent("proposal_named")
    _enqueue_in_profile(q, agent, named_home, "named-propose", ["named"])

    q._dispatch(_take_pending(q, "named-propose"))

    assert len(agent.spawned) == 1
    spawned = agent.spawned[0]
    assert spawned["home"] == named_home
    assert spawned["marker"] == "named"
    assert spawned["kwargs"]["task_cfg"]["mode"] == "propose"
    assert spawned["kwargs"]["task_cfg"]["proposal_tool"] == "proposal_named"


def test_spawned_review_worker_inherits_queued_named_profile_context(
    tmp_path, monkeypatch
):
    import agent.background_review as bg
    import run_agent

    default_home = tmp_path / "default"
    named_home = default_home / "profiles" / "worker"
    _write_review_profile(default_home, "    mode: apply\n", "default")
    _write_review_profile(
        named_home,
        "    mode: propose\n"
        "    proposal_tool: proposal_worker\n"
        "    extra_tools: [proposal_worker]\n",
        "worker",
    )
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    captured = {}
    completed = threading.Event()
    agent = _ProfileAgent("proposal_worker")
    setattr(agent, "_maybe_requeue_preempted_review", lambda *_args, **_kwargs: None)
    agent._spawn_background_review_now = types.MethodType(
        run_agent.AIAgent._spawn_background_review_now, agent
    )

    def worker_target():
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        captured["home"] = home
        captured["marker"] = (home / "review-profile-marker").read_text(
            encoding="utf-8"
        )
        completed.set()

    monkeypatch.setattr(bg, "prepare_background_review_run", lambda _agent: object())
    monkeypatch.setattr(
        bg,
        "spawn_background_review_thread",
        lambda *_args, **_kwargs: (worker_target, "prompt"),
    )

    q, _clock = _make_queue()
    _enqueue_in_profile(q, agent, named_home, "worker", ["named"])
    q._dispatch(_take_pending(q, "worker"))

    assert completed.wait(timeout=2)
    assert captured == {"home": named_home, "marker": "worker"}


def test_concurrent_named_profile_dispatches_do_not_cross_policy_or_home(
    tmp_path, monkeypatch
):
    default_home = tmp_path / "default"
    homes = [default_home / "profiles" / name for name in ("alpha", "beta")]
    _write_review_profile(default_home, "    mode: apply\n", "default")
    for name, home in zip(("alpha", "beta"), homes):
        _write_review_profile(
            home,
            "    mode: propose\n"
            f"    proposal_tool: proposal_{name}\n"
            f"    extra_tools: [proposal_{name}]\n",
            name,
        )
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    q, _clock = _make_queue()
    agents = [_ProfileAgent("proposal_alpha"), _ProfileAgent("proposal_beta")]
    for name, home, agent in zip(("alpha", "beta"), homes, agents):
        _enqueue_in_profile(q, agent, home, name, [name])
    items = [_take_pending(q, name) for name in ("alpha", "beta")]

    threads = [threading.Thread(target=q._dispatch, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    for name, home, agent in zip(("alpha", "beta"), homes, agents):
        assert len(agent.spawned) == 1
        spawned = agent.spawned[0]
        assert spawned["home"] == home
        assert spawned["marker"] == name
        assert spawned["kwargs"]["task_cfg"]["proposal_tool"] == f"proposal_{name}"


def test_coalescing_replaces_kwargs_and_profile_context_together(tmp_path, monkeypatch):
    default_home = tmp_path / "default"
    old_home = default_home / "profiles" / "old"
    new_home = default_home / "profiles" / "new"
    _write_review_profile(default_home, "    mode: apply\n", "default")
    _write_review_profile(
        old_home,
        "    mode: propose\n"
        "    proposal_tool: proposal_old\n"
        "    extra_tools: [proposal_old]\n",
        "old",
    )
    _write_review_profile(
        new_home,
        "    mode: propose\n"
        "    proposal_tool: proposal_new\n"
        "    extra_tools: [proposal_new]\n",
        "new",
    )
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    q, clock = _make_queue()
    agent = _ProfileAgent("proposal_new")
    clock["t"] = 10
    _enqueue_in_profile(q, agent, old_home, "same-session", ["old"])
    clock["t"] = 20
    _enqueue_in_profile(q, agent, new_home, "same-session", ["new"])
    item = _take_pending(q, "same-session")

    q._dispatch(item)

    assert item.enqueued_at == 10
    assert len(agent.spawned) == 1
    spawned = agent.spawned[0]
    assert spawned["home"] == new_home
    assert spawned["marker"] == "new"
    assert spawned["kwargs"]["messages_snapshot"] == ["new"]
    assert spawned["kwargs"]["task_cfg"]["proposal_tool"] == "proposal_new"


def test_context_capture_failure_drops_existing_coalesced_item(monkeypatch):
    q, _clock = _make_queue()
    agent = _FakeAgent()
    q.enqueue(agent, "same-session", {"messages_snapshot": ["old"], "task_cfg": {}})

    def fail_capture(_target):
        raise RuntimeError("context capture failed")

    monkeypatch.setattr("tools.thread_context.propagate_context_to_thread", fail_capture)
    q.enqueue(agent, "same-session", {"messages_snapshot": ["new"], "task_cfg": {}})

    assert q.pending_count() == 0
    assert agent.spawned == []
    assert agent.failures == [("background review", "context capture failed")]


def test_context_entry_failure_blocks_policy_reload_and_spawn():
    q, _clock = _make_queue()
    agent = _FakeAgent()
    q.enqueue(agent, "s1", {"messages_snapshot": [], "task_cfg": {}})
    item = _take_pending(q, "s1")

    def fail_entry(*_args, **_kwargs):
        raise RuntimeError("context entry failed")

    item.run_in_context = fail_entry
    q._dispatch(item)

    assert agent.spawned == []
    assert agent.failures == [("background review", "context entry failed")]


def test_dispatch_reloads_apply_as_propose_and_uses_current_proposal_policy(
    tmp_path, monkeypatch
):
    """Queued apply authority must not survive an operator switch to propose."""
    import agent.background_review as bg
    from hermes_cli import config as config_module
    from hermes_cli import plugins as plugin_runtime

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "auxiliary:\n  background_review:\n    mode: apply\n",
        encoding="utf-8",
    )
    path_key = str(config_path)
    config_module._LOAD_CONFIG_CACHE.pop(path_key, None)
    config_module._LAST_EXPANDED_CONFIG_BY_PATH.pop(path_key, None)
    _enabled, queued_cfg = bg.load_background_review_settings()

    captured = {}
    fork = SimpleNamespace(
        _memory_enabled=True,
        _user_profile_enabled=True,
        _session_messages=[],
        run_conversation=lambda **kwargs: captured.setdefault("prompt", kwargs["user_message"]),
    )

    class PolicyAgent(_FakeAgent):
        def __init__(self):
            super().__init__()
            self.tools = [
                {"type": "function", "function": {"name": name, "parameters": {}}}
                for name in ("memory", "skill_manage", "propose_current")
            ]

        def _spawn_background_review_now(self, **kwargs):
            captured["task_cfg"] = kwargs["task_cfg"]
            bg._run_review_fork(
                self,
                kwargs["messages_snapshot"],
                "review both",
                kwargs["task_cfg"],
                None,
                bg._ReviewForkState(),
                review_memory=True,
            )

    q, _clock = _make_queue()
    agent = PolicyAgent()
    q.enqueue(agent, "s1", {"messages_snapshot": [], "task_cfg": queued_cfg})
    with q._lock:
        item = q._pending.pop("s1")

    config_path.write_text(
        "auxiliary:\n"
        "  background_review:\n"
        "    mode: propose\n"
        "    proposal_tool: propose_current\n"
        "    extra_tools: [propose_current]\n",
        encoding="utf-8",
    )

    with patch.object(
        bg, "build_cache_parity_fork", return_value=(fork, {}, False)
    ), patch.object(bg, "_track_review_fork"), patch.object(
        bg, "finish_background_review_run"
    ), patch.object(bg, "_record_review_usage_to_parent"), patch.object(
        bg, "_release_fork_clients"
    ), patch.object(
        plugin_runtime,
        "set_thread_tool_whitelist",
        lambda whitelist, deny_msg_fmt=None: captured.setdefault("whitelist", set(whitelist)),
    ), patch.object(plugin_runtime, "clear_thread_tool_whitelist"):
        q._dispatch(item)

    assert captured["task_cfg"]["mode"] == "propose"
    assert captured["whitelist"] == {
        "propose_current", "skill_view", "skills_list", "read_file", "search_files",
    }
    assert "propose_current is the sole persistence action" in captured["prompt"]
    assert "memory" not in captured["whitelist"]
    assert "skill_manage" not in captured["whitelist"]


_INVALID_CURRENT_CONFIG = object()


@pytest.mark.parametrize(
    ("current", "expected_mode", "expected_tool", "should_spawn"),
    [
        (
            (True, {"mode": "propose", "proposal_tool": "proposal_new", "extra_tools": ["proposal_new"]}),
            "propose", "proposal_new", True,
        ),
        (
            (
                True,
                {
                    "mode": "apply",
                    "provider": "new-provider",
                    "model": "new-model",
                    "base_url": "https://new.example/v1",
                    "reasoning_effort": "low",
                    "max_input_tokens": 12345,
                    "extra_tools": ["proposal_new"],
                    "defer_max_age_s": 77,
                },
            ),
            "apply", None, True,
        ),
        ((False, {"mode": "apply"}), None, None, False),
        ((True, _INVALID_CURRENT_CONFIG), None, None, False),
        ((True, {"mode": "propose", "extra_tools": ["proposal_new"]}), None, None, False),
        (
            (True, {"mode": "propose", "proposal_tool": "proposal_new", "extra_tools": []}),
            None, None, False,
        ),
        (
            (True, {"mode": "propose", "proposal_tool": "proposal_absent", "extra_tools": ["proposal_absent"]}),
            None, None, False,
        ),
    ],
    ids=(
        "changed-proposal-tool",
        "propose-to-apply",
        "disabled",
        "unavailable-config",
        "missing-proposal-tool",
        "proposal-tool-not-admitted",
        "proposal-tool-unavailable",
    ),
)
def test_dispatch_replaces_stale_policy_or_fails_closed(
    monkeypatch, current, expected_mode, expected_tool, should_spawn
):
    import agent.background_review as bg

    agent = _FakeAgent()
    agent.tools = [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in ("proposal_old", "proposal_new")
    ]

    q, _clock = _make_queue()
    queued = {
        "mode": "propose",
        "proposal_tool": "proposal_old",
        "extra_tools": ["proposal_old"],
        "provider": "old-provider",
        "model": "old-model",
    }
    q.enqueue(agent, "s1", {"messages_snapshot": [], "task_cfg": queued})
    with q._lock:
        item = q._pending.pop("s1")

    if current[1] is _INVALID_CURRENT_CONFIG:
        current = (True, bg._InvalidBackgroundReviewConfig())
    monkeypatch.setattr(bg, "load_background_review_settings", lambda: current)

    q._dispatch(item)

    assert bool(agent.spawned) is should_spawn
    if should_spawn:
        dispatched = agent.spawned[0]["task_cfg"]
        assert dispatched == current[1]
        assert bg._background_review_mode(dispatched) == expected_mode
        assert (bg._proposal_tool(agent, dispatched) if expected_tool else None) == expected_tool
        assert dispatched.get("provider") != "old-provider"
    else:
        assert not agent.spawned
        if current[0]:
            assert agent.failures


def test_dispatch_uses_valid_last_known_good_policy_after_malformed_edit(
    tmp_path, monkeypatch
):
    import agent.background_review as bg
    from hermes_cli import config as config_module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "auxiliary:\n"
        "  background_review:\n"
        "    mode: propose\n"
        "    proposal_tool: proposal_lkg\n"
        "    extra_tools: [proposal_lkg]\n",
        encoding="utf-8",
    )
    path_key = str(config_path)
    config_module._LOAD_CONFIG_CACHE.pop(path_key, None)
    config_module._LAST_EXPANDED_CONFIG_BY_PATH.pop(path_key, None)
    _enabled, queued_cfg = bg.load_background_review_settings()

    agent = _FakeAgent()
    agent.tools = [
        {"type": "function", "function": {"name": "proposal_lkg", "parameters": {}}}
    ]
    q, _clock = _make_queue()
    q.enqueue(agent, "s1", {"messages_snapshot": [], "task_cfg": queued_cfg})
    with q._lock:
        item = q._pending.pop("s1")

    config_path.write_text(
        "auxiliary:\n  background_review: [unterminated and changed\n",
        encoding="utf-8",
    )
    q._dispatch(item)

    assert len(agent.spawned) == 1
    assert agent.spawned[0]["task_cfg"]["proposal_tool"] == "proposal_lkg"


# ── requeue on preemption ────────────────────────────────────────


class _Run:
    def __init__(self, cancelled):
        self.cancel_requested = threading.Event()
        if cancelled:
            self.cancel_requested.set()


def _requeue_agent(monkeypatch, managed=True):
    import run_agent
    from agent import review_idle_queue as riq

    agent = _FakeAgent()
    calls = {"enqueued": []}
    monkeypatch.setattr(
        riq, "review_targets_managed_local", lambda a, cfg: managed
    )
    monkeypatch.setattr(
        riq.QUEUE, "enqueue",
        lambda a, key, kw: calls["enqueued"].append(kw),
    )
    agent._REVIEW_REQUEUE_MAX_ATTEMPTS = (
        run_agent.AIAgent._REVIEW_REQUEUE_MAX_ATTEMPTS
    )
    bound = types.MethodType(
        run_agent.AIAgent._maybe_requeue_preempted_review, agent
    )
    return bound, calls


def test_preempted_review_requeues(monkeypatch):
    requeue, calls = _requeue_agent(monkeypatch)
    requeue(_Run(cancelled=True),
            {"task_cfg": {"defer": "auto"}, "focus": None,
             "_requeue_attempts": 1})
    assert len(calls["enqueued"]) == 1
    # The attempt counter rides along so the cap survives the round trip.
    assert calls["enqueued"][0]["_requeue_attempts"] == 1


def test_completed_review_does_not_requeue(monkeypatch):
    requeue, calls = _requeue_agent(monkeypatch)
    requeue(_Run(cancelled=False),
            {"task_cfg": {"defer": "auto"}, "focus": None,
             "_requeue_attempts": 1})
    assert calls["enqueued"] == []


def test_requeue_attempt_cap(monkeypatch):
    requeue, calls = _requeue_agent(monkeypatch)
    requeue(_Run(cancelled=True),
            {"task_cfg": {"defer": "auto"}, "focus": None,
             "_requeue_attempts": 4})
    assert calls["enqueued"] == []


def test_requeue_skips_non_managed(monkeypatch):
    requeue, calls = _requeue_agent(monkeypatch, managed=False)
    requeue(_Run(cancelled=True),
            {"task_cfg": {"defer": "auto"}, "focus": None,
             "_requeue_attempts": 1})
    assert calls["enqueued"] == []
