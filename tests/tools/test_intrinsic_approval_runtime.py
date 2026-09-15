"""Runtime enforcement for plugin tools with intrinsic always-human approval."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

import model_tools
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from tools.intrinsic_approval import get_intrinsic_approval_receipt_context
from tools.registry import ToolRegistry


def _schema(name: str) -> dict:
    return {
        "name": name,
        "description": "Intrinsic approval runtime fixture",
        "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
    }


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import tools.registry as registry_module
    from tools import approval, approval_context, terminal_tool

    home = tmp_path / "profile"
    home.mkdir()
    home_token = set_hermes_home_override(home)
    registry = ToolRegistry()
    monkeypatch.setattr(registry_module, "registry", registry)
    monkeypatch.setattr(model_tools, "registry", registry)
    manager = PluginManager(scope_key=registry.current_scope_key())
    context = PluginContext(PluginManifest(name="runtime", key="runtime"), manager)
    interactive_token = approval_context.set_hermes_interactive_context(True)
    monkeypatch.setattr(approval_context, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_context, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_context, "_is_unattended_platform_approval_context", lambda: False)
    monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: False)
    terminal_tool.set_approval_callback(lambda *args, **kwargs: "once")
    try:
        yield home, registry, context
    finally:
        terminal_tool.set_approval_callback(None)
        approval.clear_session("session-1")
        approval.unregister_gateway_notify("session-1")
        approval_context.reset_hermes_interactive_context(interactive_token)
        reset_hermes_home_override(home_token)


def _register(context, *, name="intrinsic_write", handler=None, preview=None):
    return context.register_tool(
        name=name,
        toolset="runtime-fixture",
        schema=_schema(name),
        handler=handler or (lambda args, **kwargs: json.dumps({"ran": True})),
        human_approval="always",
        approval_preview=preview or (
            lambda args, preview_context: {
                "summary": "Write fixture",
                "authorization_scope_sha256": "a" * 64,
            }
        ),
    )


def _call(name="intrinsic_write", args=None, **kwargs):
    return model_tools.handle_function_call(
        name,
        {"value": "original"} if args is None else args,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-1",
        call_origin="model",
        **kwargs,
    )


def _receipts(home):
    db = SessionDB(db_path=home / "state.db")
    try:
        rows = db._conn.execute(
            "SELECT receipt_id, profile_identity, session_id, turn_id, tool_call_id, "
            "tool_name, original_args_sha256, preview_sha256, approval_scope_sha256, "
            "decision, decided_at, expires_at FROM intrinsic_tool_approval_receipts "
            "ORDER BY rowid"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        db.close()


@pytest.mark.parametrize("concurrent", [False, True], ids=["sequential", "concurrent"])
def test_intrinsic_bridge_scope_block_precedes_approval_and_dispatch(
    runtime, monkeypatch, concurrent
):
    home, _registry, context = runtime
    from agent import tool_executor
    from tools import terminal_tool

    called = []
    _register(
        context,
        handler=lambda args, **kwargs: called.append("handler") or "{}",
        preview=lambda args, preview_context: called.append("preview") or {
            "authorization_scope_sha256": "a" * 64
        },
    )
    terminal_tool.set_approval_callback(
        lambda *args, **kwargs: called.append("prompt") or "once"
    )

    def request_middleware(name, args, ids, trace):
        called.append("request_middleware")
        return dict(args), dict(args), trace

    def pre_hook(name, args, skip, ids, trace):
        called.append("pre_hook")
        return args, None

    def execution_middleware(name, args, dispatch, **kwargs):
        called.append("execution_middleware")
        return dispatch(dict(args))

    def transform(name, args, result, duration, ids):
        called.append("transform_hook")
        return result

    monkeypatch.setattr(model_tools, "_apply_request_middleware", request_middleware)
    monkeypatch.setattr(model_tools, "_pre_dispatch_guards", pre_hook)
    monkeypatch.setattr(
        "hermes_cli.middleware.run_tool_execution_middleware", execution_middleware
    )
    monkeypatch.setattr(
        model_tools,
        "_emit_post_tool_call_hook",
        lambda *args, **kwargs: called.append("post_hook"),
    )
    monkeypatch.setattr(model_tools, "_apply_transform_tool_result_hook", transform)
    scope_block = (
        "'intrinsic_write' is not available in this session. "
        "Use tool_search to find tools you can call."
    )
    begin_calls = []

    outcome = tool_executor._run_agent_tool_execution_middleware(
        type("Agent", (), {"model_tool_policy": None})(),
        function_name="intrinsic_write",
        function_args={"value": "original"},
        effective_task_id="task-1",
        tool_call_id="call-1",
        execute=lambda args: _call(args=args),
        scope_block=scope_block,
        middleware_trace=[],
        begin_execution=(lambda callback=None: begin_calls.append(callback))
        if concurrent
        else None,
    )

    assert outcome.result == json.dumps({"error": scope_block}, ensure_ascii=False)
    assert outcome.args == {"value": "original"}
    assert outcome.middleware_trace == []
    assert outcome.blocked is True
    assert outcome.dispatched is False
    assert begin_calls == ([None] if concurrent else [])
    assert called == []
    assert _receipts(home) == []


def test_policy_denial_precedes_intrinsic_preview_prompt_and_hooks(runtime, monkeypatch):
    _home, _registry, context = runtime
    called = []
    _register(
        context,
        preview=lambda args, preview_context: called.append("preview") or {
            "authorization_scope_sha256": "a" * 64
        },
    )
    monkeypatch.setattr(
        "agent.model_tool_policy.model_tool_policy_denial",
        lambda *args, **kwargs: "policy denied",
    )
    monkeypatch.setattr(model_tools, "_apply_request_middleware", lambda *a, **k: called.append("middleware"))
    monkeypatch.setattr(model_tools, "_pre_dispatch_guards", lambda *a, **k: called.append("hook"))

    result = _call(model_tool_policy={"version": 1})

    assert "policy denied" in result
    assert called == []


def test_intrinsic_gate_precedes_middleware_and_context_exists_only_in_handler(runtime, monkeypatch):
    home, _registry, context = runtime
    seen = []
    handler_context = {}

    def handler(args, **kwargs):
        current = get_intrinsic_approval_receipt_context()
        assert current is not None
        seen.append(("handler", dict(args), current))
        handler_context.update(asdict(current))
        return json.dumps({"ok": True})

    _register(
        context,
        handler=handler,
        preview=lambda args, preview_context: seen.append(("preview", dict(args), get_intrinsic_approval_receipt_context())) or {
            "summary": "Exact preview",
            "authorization_scope_sha256": "b" * 64,
        },
    )

    def request_middleware(name, args, ids, trace):
        seen.append(("request", dict(args), get_intrinsic_approval_receipt_context()))
        return dict(args), dict(args), trace

    def prehook(name, args, skip, ids, trace):
        seen.append(("prehook", dict(args), get_intrinsic_approval_receipt_context()))
        return args, None

    def execution_middleware(name, args, dispatch, **kwargs):
        seen.append(("execution", dict(args), get_intrinsic_approval_receipt_context()))
        return dispatch(dict(args))

    def posthook(**kwargs):
        seen.append(("post", None, get_intrinsic_approval_receipt_context()))

    def transform(name, args, result, duration, ids):
        seen.append(("transform", None, get_intrinsic_approval_receipt_context()))
        return result

    monkeypatch.setattr(model_tools, "_apply_request_middleware", request_middleware)
    monkeypatch.setattr(model_tools, "_pre_dispatch_guards", prehook)
    monkeypatch.setattr("hermes_cli.middleware.run_tool_execution_middleware", execution_middleware)
    monkeypatch.setattr(model_tools, "_emit_post_tool_call_hook", posthook)
    monkeypatch.setattr(model_tools, "_apply_transform_tool_result_hook", transform)

    result = _call()

    assert json.loads(result) == {"ok": True}
    assert [item[0] for item in seen] == ["preview", "request", "prehook", "execution", "handler", "post", "transform"]
    assert all(item[2] is None for item in seen if item[0] != "handler")
    assert seen[4][1] == {"value": "original"}
    receipt = _receipts(home)[0]
    assert handler_context["receipt_id"] == receipt["receipt_id"]
    assert handler_context["decision"] == "approved"
    assert handler_context["tool_call_id"] == "call-1"
    assert get_intrinsic_approval_receipt_context() is None
    assert receipt["expires_at"] - receipt["decided_at"] == 900
    assert receipt["receipt_id"] not in result


@pytest.mark.parametrize("choice", ["session", "always", "approve", "deny", "timeout", "cancel"])
def test_only_exact_once_approves_and_every_prompt_outcome_is_receipted(runtime, choice):
    home, _registry, context = runtime
    from tools import terminal_tool

    calls = []
    _register(context, handler=lambda args, **kwargs: calls.append(dict(args)) or "{}")
    terminal_tool.set_approval_callback(lambda *args, **kwargs: choice)

    result = _call()

    assert "approval" in result.lower()
    assert calls == []
    assert _receipts(home)[0]["decision"] == "denied"


def test_receipt_hashes_exact_original_args_preview_and_scope_without_raw_payload(runtime):
    home, _registry, context = runtime
    canary = "RAW_SECRET_CANARY_5e18"
    args = {"secret": canary, "nested": {"z": 2, "a": 1}}
    preview = {
        "summary": f"Show {canary}",
        "authorization_scope_sha256": "c" * 64,
    }
    _register(context, preview=lambda supplied, preview_context: preview)

    assert json.loads(_call(args=args)) == {"ran": True}

    receipt = _receipts(home)[0]
    canonical_args = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    canonical_preview = json.dumps(preview, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert receipt["original_args_sha256"] == hashlib.sha256(canonical_args).hexdigest()
    assert receipt["preview_sha256"] == hashlib.sha256(canonical_preview).hexdigest()
    assert receipt["approval_scope_sha256"] == "c" * 64
    assert canary not in json.dumps(receipt)
    assert canary not in (home / "state.db").read_bytes().decode("utf-8", errors="ignore")


@pytest.mark.parametrize("scope", [None, "A" * 64, "a" * 63, 7])
def test_missing_or_malformed_preview_scope_blocks_before_prompt_and_receipt(runtime, scope):
    home, _registry, context = runtime
    from tools import terminal_tool

    prompted = []
    handler_calls = []
    _register(
        context,
        handler=lambda args, **kwargs: handler_calls.append(True) or "{}",
        preview=lambda args, preview_context: {"authorization_scope_sha256": scope},
    )
    terminal_tool.set_approval_callback(lambda *args, **kwargs: prompted.append(True) or "once")

    result = _call()

    assert "approval" in result.lower()
    assert prompted == []
    assert handler_calls == []
    assert _receipts(home) == []


def test_no_human_surface_blocks_before_preview_even_if_modes_grants_or_yolo_would_allow(runtime, monkeypatch):
    home, _registry, context = runtime
    from tools import approval, approval_context, terminal_tool

    called = []
    _register(
        context,
        preview=lambda args, preview_context: called.append("preview") or {
            "authorization_scope_sha256": "a" * 64
        },
    )
    terminal_tool.set_approval_callback(None)
    approval.enable_session_yolo("session-1")
    approval._permanent_approved.add("anything")
    monkeypatch.setattr(approval_context, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: False)

    result = _call()

    assert "approval" in result.lower()
    assert called == []
    assert _receipts(home) == []


def test_gateway_native_once_uses_only_once_and_deny_choices(runtime, monkeypatch):
    home, _registry, context = runtime
    from tools import approval, approval_context

    notices = []
    monkeypatch.setattr(approval_context, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: True)

    def notify(data):
        notices.append(data)
        approval.resolve_gateway_approval("session-1", "once", request_id=data["request_id"])

    approval.register_gateway_notify("session-1", notify)
    _register(context)

    result = _call()

    assert notices
    assert json.loads(result) == {"ran": True}
    assert notices[0]["choices"] == ["once", "deny"]
    assert notices[0]["allow_session"] is False
    assert notices[0]["allow_permanent"] is False
    assert _receipts(home)[0]["decision"] == "approved"


def test_receipt_persistence_failure_blocks_handler(runtime, monkeypatch):
    _home, _registry, context = runtime
    calls = []
    _register(context, handler=lambda args, **kwargs: calls.append(True) or "{}")
    monkeypatch.setattr(
        "tools.intrinsic_approval.create_intrinsic_approval_receipt",
        lambda **kwargs: (_ for _ in ()).throw(OSError("SECRET_PERSISTENCE_DETAIL")),
    )

    result = _call()

    assert "approval" in result.lower()
    assert "SECRET_PERSISTENCE_DETAIL" not in result
    assert calls == []


def test_middleware_argument_mutation_blocks_original_approved_handler(runtime, monkeypatch):
    home, _registry, context = runtime
    calls = []
    _register(context, handler=lambda args, **kwargs: calls.append(dict(args)) or "{}")
    monkeypatch.setattr(
        model_tools,
        "_apply_request_middleware",
        lambda name, args, ids, trace: ({"value": "changed"}, dict(args), trace),
    )

    result = _call()

    assert "approval" in result.lower()
    assert calls == []
    assert _receipts(home)[0]["decision"] == "approved"


def test_direct_registry_dispatch_has_no_receipt_context(runtime):
    _home, registry, context = runtime
    observed = []
    _register(
        context,
        handler=lambda args, **kwargs: observed.append(get_intrinsic_approval_receipt_context()) or "{}",
    )

    assert registry.dispatch("intrinsic_write", {}, scope=context._manager.scope_key) == "{}"
    assert observed == [None]


def test_entry_replacement_during_prompt_cannot_execute_replacement(runtime):
    home, _registry, context = runtime
    from tools import terminal_tool

    old_calls = []
    new_calls = []
    _register(context, handler=lambda args, **kwargs: old_calls.append(True) or "{}")

    def replace_then_approve(*args, **kwargs):
        _register(
            context,
            handler=lambda handler_args, **handler_kwargs: new_calls.append(True) or "{}",
            preview=lambda handler_args, preview_context: {
                "authorization_scope_sha256": "d" * 64
            },
        )
        return "once"

    terminal_tool.set_approval_callback(replace_then_approve)

    result = _call()

    assert "approval" in result.lower()
    assert old_calls == new_calls == []
    assert _receipts(home)[0]["decision"] == "approved"


def test_profile_switch_during_callback_blocks_without_receipt(runtime, tmp_path):
    first_home, _registry, context = runtime
    from tools import terminal_tool

    second_home = tmp_path / "other-profile"
    second_home.mkdir()
    switched = []
    calls = []
    _register(context, handler=lambda args, **kwargs: calls.append(True) or "{}")

    def switch_profile(*args, **kwargs):
        switched.append(set_hermes_home_override(second_home))
        return "once"

    terminal_tool.set_approval_callback(switch_profile)
    try:
        result = _call()
    finally:
        if switched:
            reset_hermes_home_override(switched.pop())

    assert "approval" in result.lower()
    assert calls == []
    assert _receipts(first_home) == []
    assert _receipts(second_home) == []


def test_handler_context_resets_after_error(runtime):
    _home, _registry, context = runtime
    seen = []

    def handler(args, **kwargs):
        seen.append(get_intrinsic_approval_receipt_context())
        raise RuntimeError("handler failed")

    _register(context, handler=handler)

    assert "handler failed" in _call()
    assert seen[0] is not None
    assert get_intrinsic_approval_receipt_context() is None


def test_concurrent_handler_contexts_are_isolated(runtime):
    _home, _registry, context = runtime
    from tools import approval_context, terminal_tool

    seen = {}

    def handler(args, **kwargs):
        current = get_intrinsic_approval_receipt_context()
        assert current is not None
        seen[current.tool_call_id] = (
            current.receipt_id,
            args["value"],
            get_intrinsic_approval_receipt_context() is current,
        )
        return "{}"

    _register(context, handler=handler)

    def invoke(number):
        home_token = set_hermes_home_override(_home)
        interactive = approval_context.set_hermes_interactive_context(True)
        terminal_tool.set_approval_callback(lambda *args, **kwargs: "once")
        try:
            return model_tools.handle_function_call(
                "intrinsic_write",
                {"value": str(number)},
                session_id=f"session-{number}",
                turn_id=f"turn-{number}",
                tool_call_id=f"call-{number}",
                call_origin="model",
            )
        finally:
            terminal_tool.set_approval_callback(None)
            approval_context.reset_hermes_interactive_context(interactive)
            reset_hermes_home_override(home_token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(invoke, (1, 2)))

    assert results == ["{}", "{}"]
    assert set(seen) == {"call-1", "call-2"}
    assert seen["call-1"][1:] == ("1", True)
    assert seen["call-2"][1:] == ("2", True)
    assert seen["call-1"][0] != seen["call-2"][0]
    assert get_intrinsic_approval_receipt_context() is None


def _approved_skill_arguments(scope_hash="e" * 64):
    return {
        "operations": [{
            "action": "patch",
            "name": "fixture-skill",
            "old_string": "before",
            "new_string": "after",
            "replace_all": False,
            "authorization_scope_sha256": scope_hash,
        }]
    }


def _approved_dispatch_preview(arguments=None, *, scope_hash="e" * 64):
    return {
        "summary": "Apply one approved skill patch",
        "authorization_scope_sha256": scope_hash,
        "approved_dispatch": {
            "tool_name": "skill_manage",
            "arguments": _approved_skill_arguments(scope_hash) if arguments is None else arguments,
        },
    }


def _register_skill_manage(registry, calls):
    registry.register(
        name="skill_manage",
        toolset="skills",
        schema=_schema("skill_manage"),
        handler=lambda args, **kwargs: calls.append((args, kwargs)) or json.dumps({"applied": True}),
    )


def test_preview_without_approved_dispatch_receipts_but_cannot_dispatch(runtime):
    home, _registry, context = runtime
    downstream = []
    _register_skill_manage(_registry, downstream)

    def handler(args, **kwargs):
        receipt = get_intrinsic_approval_receipt_context()
        assert receipt is not None
        return context.dispatch_approved_tool("skill_manage", _approved_skill_arguments("a" * 64))

    _register(context, handler=handler)

    result = json.loads(_call())

    assert "error" in result
    assert downstream == []
    assert _receipts(home)[0]["decision"] == "approved"


@pytest.mark.parametrize(
    "approved_dispatch",
    [
        None,
        [],
        {"tool_name": "skill_manage"},
        {"arguments": _approved_skill_arguments()},
        {"tool_name": "skill_manage", "arguments": _approved_skill_arguments(), "extra": True},
        {"tool_name": "write_file", "arguments": _approved_skill_arguments()},
        {"tool_name": "skill_manage", "arguments": []},
        {"tool_name": "skill_manage", "arguments": {"operations": []}},
        {"tool_name": "skill_manage", "arguments": {"operations": [{}, {}]}},
        {"tool_name": "skill_manage", "arguments": {"operations": ["patch"]}},
        {"tool_name": "skill_manage", "arguments": {"operations": [{
            "action": "create", "replace_all": False,
            "authorization_scope_sha256": "e" * 64,
        }]}},
        {"tool_name": "skill_manage", "arguments": {"operations": [{
            "action": "patch", "authorization_scope_sha256": "e" * 64,
        }]}},
        {"tool_name": "skill_manage", "arguments": {"operations": [{
            "action": "patch", "replace_all": 0,
            "authorization_scope_sha256": "e" * 64,
        }]}},
        {"tool_name": "skill_manage", "arguments": _approved_skill_arguments("f" * 64)},
        {"tool_name": "skill_manage", "arguments": {
            "operations": [{
                "action": "patch", "replace_all": False,
                "authorization_scope_sha256": "e" * 64,
                "value": float("nan"),
            }]
        }},
    ],
    ids=[
        "nonmapping-none", "nonmapping-list", "missing-arguments", "missing-tool",
        "extra-envelope-key", "wrong-tool", "arguments-nonmapping", "operations-empty",
        "operations-many", "operation-nonmapping", "wrong-action", "missing-replace-all",
        "replace-all-not-literal-false", "scope-mismatch", "nonfinite-arguments",
    ],
)
def test_malformed_approved_dispatch_blocks_before_prompt_receipt_and_handler(
    runtime, approved_dispatch,
):
    home, _registry, context = runtime
    from tools import terminal_tool

    called = []
    terminal_tool.set_approval_callback(lambda *args, **kwargs: called.append("prompt") or "once")
    _register(
        context,
        handler=lambda args, **kwargs: called.append("handler") or "{}",
        preview=lambda args, preview_context: {
            "authorization_scope_sha256": "e" * 64,
            "approved_dispatch": approved_dispatch,
        },
    )

    result = _call()

    assert "approval" in result.lower()
    assert called == []
    assert _receipts(home) == []


def test_exact_approved_dispatch_uses_detached_canonical_arguments_once(runtime):
    _home, registry, context = runtime
    from tools import terminal_tool

    downstream = []
    displayed = []
    _register_skill_manage(registry, downstream)
    approved = _approved_skill_arguments()
    preview = _approved_dispatch_preview(approved)
    terminal_tool.set_approval_callback(
        lambda command, description, **kwargs: displayed.append(json.loads(command)) or "once"
    )

    def handler(args, **kwargs):
        # Key order is immaterial under canonical JSON.
        reordered = {"operations": [dict(reversed(list(approved["operations"][0].items())))]}
        return context.dispatch_approved_tool("skill_manage", reordered)

    _register(context, handler=handler, preview=lambda args, preview_context: preview)

    assert json.loads(_call()) == {"applied": True}
    assert len(downstream) == 1
    assert downstream[0][0] == approved
    assert downstream[0][0] is not approved
    assert downstream[0][0]["operations"][0] is not approved["operations"][0]
    assert downstream[0][1] == {}
    assert displayed == [preview]


def test_approved_dispatch_capability_is_absent_from_preview_middleware_and_hooks(
    runtime, monkeypatch,
):
    _home, registry, context = runtime
    downstream = []
    attempts = []
    approved = _approved_skill_arguments()
    _register_skill_manage(registry, downstream)

    def attempt(stage):
        result = json.loads(context.dispatch_approved_tool("skill_manage", approved))
        attempts.append((stage, "error" in result))

    def preview(args, preview_context):
        attempt("preview")
        return _approved_dispatch_preview(approved)

    def request(name, args, ids, trace):
        attempt("request")
        return dict(args), dict(args), trace

    def prehook(name, args, skip, ids, trace):
        attempt("prehook")
        return args, None

    def execution(name, args, dispatch, **kwargs):
        attempt("execution")
        return dispatch(dict(args))

    def posthook(**kwargs):
        attempt("post")

    def transform(name, args, result, duration, ids):
        attempt("transform")
        return result

    _register(context, handler=lambda args, **kwargs: "{}", preview=preview)
    monkeypatch.setattr(model_tools, "_apply_request_middleware", request)
    monkeypatch.setattr(model_tools, "_pre_dispatch_guards", prehook)
    monkeypatch.setattr("hermes_cli.middleware.run_tool_execution_middleware", execution)
    monkeypatch.setattr(model_tools, "_emit_post_tool_call_hook", posthook)
    monkeypatch.setattr(model_tools, "_apply_transform_tool_result_hook", transform)

    assert json.loads(_call()) == {}
    assert attempts == [
        ("preview", True), ("request", True), ("prehook", True),
        ("execution", True), ("post", True), ("transform", True),
    ]
    assert downstream == []


@pytest.mark.parametrize("mismatch", ["tool", "nested"], ids=["wrong-tool", "nested-args"])
def test_first_approved_dispatch_mismatch_burns_capability(runtime, mismatch):
    _home, registry, context = runtime
    downstream = []
    _register_skill_manage(registry, downstream)
    approved = _approved_skill_arguments()

    def handler(args, **kwargs):
        if mismatch == "tool":
            first = context.dispatch_approved_tool("write_file", approved)
        else:
            changed = json.loads(json.dumps(approved))
            changed["operations"][0]["new_string"] = "different"
            first = context.dispatch_approved_tool("skill_manage", changed)
        second = context.dispatch_approved_tool("skill_manage", approved)
        return json.dumps({"first": json.loads(first), "second": json.loads(second)})

    _register(
        context, handler=handler,
        preview=lambda args, preview_context: _approved_dispatch_preview(approved),
    )

    result = json.loads(_call())

    assert "error" in result["first"]
    assert "error" in result["second"]
    assert downstream == []


def test_downstream_dispatch_error_burns_capability(runtime, monkeypatch):
    _home, registry, context = runtime
    approved = _approved_skill_arguments()
    dispatch_calls = []

    def failing_dispatch(name, args, **kwargs):
        dispatch_calls.append((name, args, kwargs))
        raise RuntimeError("private downstream detail")

    monkeypatch.setattr(registry, "dispatch", failing_dispatch)

    def handler(args, **kwargs):
        first = context.dispatch_approved_tool("skill_manage", approved)
        second = context.dispatch_approved_tool("skill_manage", approved)
        return json.dumps({"first": json.loads(first), "second": json.loads(second)})

    _register(
        context, handler=handler,
        preview=lambda args, preview_context: _approved_dispatch_preview(approved),
    )

    result = json.loads(_call())

    assert "error" in result["first"]
    assert "private downstream detail" not in result["first"]["error"]
    assert "error" in result["second"]
    assert len(dispatch_calls) == 1


def test_different_scope_context_cannot_borrow_and_burns_capability(runtime):
    _home, registry, context = runtime
    downstream = []
    _register_skill_manage(registry, downstream)
    approved = _approved_skill_arguments()
    other_manager = PluginManager(scope_key=context._manager.scope_key + "-other")
    other_context = PluginContext(
        PluginManifest(name="other", key="other"), other_manager,
    )

    def handler(args, **kwargs):
        first = other_context.dispatch_approved_tool("skill_manage", approved)
        second = context.dispatch_approved_tool("skill_manage", approved)
        return json.dumps({"first": json.loads(first), "second": json.loads(second)})

    _register(
        context, handler=handler,
        preview=lambda args, preview_context: _approved_dispatch_preview(approved),
    )

    result = json.loads(_call())

    assert "error" in result["first"]
    assert "error" in result["second"]
    assert downstream == []


def test_second_and_concurrent_approved_dispatch_make_one_registry_call(runtime, monkeypatch):
    import contextvars
    import threading

    _home, registry, context = runtime
    approved = _approved_skill_arguments()
    calls = []
    results = []
    call_lock = threading.Lock()
    threads = []
    barrier = threading.Barrier(3)

    original_dispatch = registry.dispatch

    def counted_dispatch(name, args, **kwargs):
        with call_lock:
            calls.append((name, args, kwargs))
        return json.dumps({"applied": True})

    monkeypatch.setattr(registry, "dispatch", counted_dispatch)

    def handler(args, **kwargs):
        contexts = [contextvars.copy_context(), contextvars.copy_context()]

        def dispatch(copied):
            barrier.wait()
            results.append(json.loads(copied.run(
                context.dispatch_approved_tool, "skill_manage", approved
            )))

        for copied in contexts:
            thread = threading.Thread(target=dispatch, args=(copied,))
            thread.start()
            threads.append(thread)
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)
        return "{}"

    _register(
        context, handler=handler,
        preview=lambda args, preview_context: _approved_dispatch_preview(approved),
    )

    assert json.loads(_call()) == {}
    for thread in threads:
        assert not thread.is_alive()
    assert len(calls) == 1
    assert sum(item == {"applied": True} for item in results) == 1
    assert sum("error" in item for item in results) == 1
    monkeypatch.setattr(registry, "dispatch", original_dispatch)


def test_copied_handler_context_cannot_dispatch_after_handler_returns(runtime):
    import contextvars
    import threading

    _home, registry, context = runtime
    approved = _approved_skill_arguments()
    downstream = []
    results = []
    release = threading.Event()
    threads = []
    _register_skill_manage(registry, downstream)

    def handler(args, **kwargs):
        copied = contextvars.copy_context()
        thread = threading.Thread(
            target=lambda: (
                release.wait(),
                results.append(json.loads(copied.run(
                    context.dispatch_approved_tool, "skill_manage", approved
                ))),
            )
        )
        thread.start()
        threads.append(thread)
        return "{}"

    _register(
        context, handler=handler,
        preview=lambda args, preview_context: _approved_dispatch_preview(approved),
    )

    assert json.loads(_call()) == {}
    release.set()
    threads[0].join(timeout=2)

    assert not threads[0].is_alive()
    assert "error" in results[0]
    assert downstream == []


def test_approved_dispatch_absent_outside_handler_and_in_plain_sibling_thread(runtime):
    _home, registry, context = runtime
    approved = _approved_skill_arguments()
    downstream = []
    _register_skill_manage(registry, downstream)
    observed = []

    assert "error" in json.loads(context.dispatch_approved_tool("skill_manage", approved))

    def handler(args, **kwargs):
        import threading

        thread = threading.Thread(
            target=lambda: observed.append(
                json.loads(context.dispatch_approved_tool("skill_manage", approved))
            )
        )
        thread.start()
        thread.join()
        raise RuntimeError("reset probe")

    _register(
        context, handler=handler,
        preview=lambda args, preview_context: _approved_dispatch_preview(approved),
    )

    assert "reset probe" in _call()
    assert "error" in observed[0]
    assert "error" in json.loads(context.dispatch_approved_tool("skill_manage", approved))
    assert downstream == []


def test_unloaded_context_cannot_use_approved_dispatch(runtime):
    _home, registry, context = runtime
    downstream = []
    _register_skill_manage(registry, downstream)
    approved = _approved_skill_arguments()

    def handler(args, **kwargs):
        context._manager._invalidate_plugin_contexts([context.plugin_id])
        return context.dispatch_approved_tool("skill_manage", approved)

    _register(
        context, handler=handler,
        preview=lambda args, preview_context: _approved_dispatch_preview(approved),
    )

    assert "error" in json.loads(_call())
    assert downstream == []


def test_approved_handler_direct_dispatch_cannot_spoof_cli_parent_policy(runtime):
    _home, registry, context = runtime
    downstream = []
    _register_skill_manage(registry, downstream)
    approved = _approved_skill_arguments()
    context._manager._cli_ref = type("CLI", (), {
        "agent": type("Agent", (), {"model_tool_policy": {
            "policy_id": "fixture",
            "policy_sha256": "1" * 64,
            "allowed_tools": ["read_file"],
            "approval_required_tools": [],
        }})()
    })()
    fake_parent = type("FakeParent", (), {})()

    def handler(args, **kwargs):
        direct = context.dispatch_tool("skill_manage", approved, parent_agent=fake_parent)
        exact = context.dispatch_approved_tool("skill_manage", approved)
        return json.dumps({"direct": json.loads(direct), "exact": json.loads(exact)})

    _register(
        context, handler=handler,
        preview=lambda args, preview_context: _approved_dispatch_preview(approved),
    )

    result = json.loads(_call())

    assert result["direct"] == {"error": "Tool dispatch denied"}
    assert result["exact"] == {"applied": True}
    assert downstream == [(approved, {})]


def test_approved_handler_direct_dispatch_denied_without_cli_reference(runtime):
    _home, registry, context = runtime
    downstream = []
    _register_skill_manage(registry, downstream)
    approved = _approved_skill_arguments()
    context._manager._cli_ref = None

    def handler(args, **kwargs):
        direct = context.dispatch_tool("skill_manage", approved)
        exact = context.dispatch_approved_tool("skill_manage", approved)
        return json.dumps({"direct": json.loads(direct), "exact": json.loads(exact)})

    _register(
        context, handler=handler,
        preview=lambda args, preview_context: _approved_dispatch_preview(approved),
    )

    result = json.loads(_call())

    assert result["direct"] == {"error": "Tool dispatch denied"}
    assert result["exact"] == {"applied": True}
    assert downstream == [(approved, {})]


def test_fresh_context_cannot_direct_dispatch_but_exact_approved_dispatch_succeeds(runtime):
    import contextvars

    _home, registry, context = runtime
    downstream = []
    approved = _approved_skill_arguments()
    _register_skill_manage(registry, downstream)
    context._manager._cli_ref = None

    def handler(args, **kwargs):
        direct = contextvars.Context().run(
            context.dispatch_tool, "skill_manage", approved
        )
        exact = context.dispatch_approved_tool("skill_manage", approved)
        return json.dumps({"direct": json.loads(direct), "exact": json.loads(exact)})

    _register(
        context, handler=handler,
        preview=lambda args, preview_context: _approved_dispatch_preview(approved),
    )

    result = json.loads(_call())

    assert result["direct"] == {"error": "Tool dispatch denied"}
    assert result["exact"] == {"applied": True}
    assert downstream == [(approved, {})]


def test_child_thread_cannot_direct_dispatch_without_deadlocking_handler(runtime):
    import threading

    _home, registry, context = runtime
    downstream = []
    approved = _approved_skill_arguments()
    direct_results = []
    threads = []
    entered = threading.Event()
    joined_in_handler = []
    _register_skill_manage(registry, downstream)
    context._manager._cli_ref = None

    def handler(args, **kwargs):
        def direct_dispatch():
            entered.set()
            direct_results.append(json.loads(
                context.dispatch_tool("skill_manage", approved)
            ))

        thread = threading.Thread(target=direct_dispatch)
        threads.append(thread)
        thread.start()
        assert entered.wait(timeout=1)
        thread.join(timeout=1)
        joined_in_handler.append(not thread.is_alive())
        exact = context.dispatch_approved_tool("skill_manage", approved)
        return json.dumps({"exact": json.loads(exact)})

    _register(
        context, handler=handler,
        preview=lambda args, preview_context: _approved_dispatch_preview(approved),
    )

    result = json.loads(_call())
    threads[0].join(timeout=1)

    assert joined_in_handler == [True]
    assert not threads[0].is_alive()
    assert direct_results == [{"error": "Tool dispatch denied"}]
    assert result["exact"] == {"applied": True}
    assert downstream == [(approved, {})]


def test_intrinsic_handler_marker_counts_nesting_and_isolates_scopes():
    registry = ToolRegistry()
    scope = "profile-a"

    assert not registry._intrinsic_handler_active_in_scope(scope)
    with registry._intrinsic_handler_scope(scope):
        assert registry._intrinsic_handler_active_in_scope(scope)
        assert not registry._intrinsic_handler_active_in_scope("profile-b")
        with registry._intrinsic_handler_scope(scope):
            assert registry._intrinsic_handler_active_in_scope(scope)
        assert registry._intrinsic_handler_active_in_scope(scope)
    assert not registry._intrinsic_handler_active_in_scope(scope)


@pytest.mark.parametrize("raises", [False, True], ids=["success", "error"])
def test_intrinsic_handler_marker_clears_and_legacy_dispatch_resumes(runtime, raises):
    _home, registry, context = runtime
    scope = context._manager.scope_key
    downstream = []
    active_in_handler = []
    _register_skill_manage(registry, downstream)
    context._manager._cli_ref = None

    def handler(args, **kwargs):
        active_in_handler.append(registry._intrinsic_handler_active_in_scope(scope))
        if raises:
            raise RuntimeError("handler failed")
        return "{}"

    _register(context, handler=handler)

    result = _call()
    direct = context.dispatch_tool("skill_manage", {"ordinary": True})

    assert ("handler failed" in result) is raises
    assert active_in_handler == [True]
    assert not registry._intrinsic_handler_active_in_scope(scope)
    assert json.loads(direct) == {"applied": True}
    assert downstream == [({"ordinary": True}, {})]


def test_dispatch_capability_does_not_leak_public_receipt_result_or_database(runtime):
    home, registry, context = runtime
    canary = "EXPECTED_APPROVED_ARGUMENT_CANARY_7c41"
    approved = _approved_skill_arguments()
    approved["operations"][0]["new_string"] = canary
    downstream = []
    _register_skill_manage(registry, downstream)

    def handler(args, **kwargs):
        receipt = get_intrinsic_approval_receipt_context()
        assert receipt is not None
        assert set(asdict(receipt)) == set(receipt.__dataclass_fields__)
        assert all("dispatch" not in key and "expected" not in key for key in asdict(receipt))
        return context.dispatch_approved_tool("skill_manage", approved)

    _register(
        context, handler=handler,
        preview=lambda args, preview_context: _approved_dispatch_preview(approved),
    )

    result = _call()
    approved_bytes = json.dumps(
        approved, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    approved_hash = hashlib.sha256(approved_bytes).hexdigest()
    receipt_json = json.dumps(_receipts(home))
    database_text = (home / "state.db").read_bytes().decode("utf-8", errors="ignore")

    assert json.loads(result) == {"applied": True}
    assert canary not in result
    assert canary not in receipt_json
    assert canary not in database_text
    assert approved_hash not in receipt_json
    assert approved_hash not in database_text
