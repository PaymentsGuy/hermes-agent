"""Intrinsic always-human metadata for plugin-provided tools."""

import hashlib
import json
import traceback
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools.registry import (
    APPROVAL_PREVIEW_MAX_BYTES,
    ApprovalPreviewError,
    ToolRegistry,
    approval_preview_hash,
    canonical_approval_preview_bytes,
)


def _schema(name: str) -> dict:
    return {
        "name": name,
        "description": "Approval test tool",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
    }


@pytest.fixture
def plugin_context(monkeypatch):
    import tools.registry as registry_module

    registry = ToolRegistry()
    monkeypatch.setattr(registry_module, "registry", registry)
    manager = PluginManager(scope_key=registry.current_scope_key())
    context = PluginContext(
        PluginManifest(name="approval-test", key="approval-test"), manager
    )
    return registry, context


def _register(
    context: PluginContext,
    *,
    name: str = "approval_test_tool",
    handler=None,
    preview=None,
    human_approval="always",
):
    return context.register_tool(
        name=name,
        toolset="approval-test",
        schema=_schema(name),
        handler=handler or (lambda args, **kwargs: json.dumps({"ran": True})),
        human_approval=human_approval,
        approval_preview=preview,
    )


def test_plugin_always_metadata_is_intrinsic_and_preview_is_read_only(plugin_context, monkeypatch):
    registry, context = plugin_context
    calls = {"handler": 0, "hook": 0}
    original_args = {"z": 1, "nested": {"before": True}}
    received = {}

    def handler(args, **kwargs):
        calls["handler"] += 1
        return "{}"

    def preview(args, preview_context):
        received["args"] = args
        received["context"] = preview_context
        args["nested"]["before"] = False
        return {"summary": "café", "target": args["z"]}

    def unexpected_hook(*args, **kwargs):
        calls["hook"] += 1
        raise AssertionError("preview resolution must not invoke plugin hooks")

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", unexpected_hook)
    assert _register(context, handler=handler, preview=preview) is not None

    entry = registry.get_entry("approval_test_tool", scope=context._manager.scope_key)
    assert entry is not None
    assert entry.human_approval == "always"
    assert entry.approval_preview is preview
    with pytest.raises(AttributeError):
        entry.human_approval = None
    with pytest.raises(AttributeError):
        entry.approval_preview = lambda args, context: {}
    with pytest.raises(AttributeError):
        entry._approval = None

    definitions = registry.get_definitions({"approval_test_tool"})
    assert definitions[0]["function"] == _schema("approval_test_tool")
    encoded_definitions = json.dumps(definitions)
    assert "human_approval" not in encoded_definitions
    assert "approval_preview" not in encoded_definitions

    resolved = registry.resolve_approval_preview(
        "approval_test_tool",
        original_args,
        scope=context._manager.scope_key,
        session_id="session-1",
        profile_name="approval-test",
        tool_call_id="call-1",
    )

    assert resolved == {"summary": "café", "target": 1}
    assert original_args == {"z": 1, "nested": {"before": True}}
    assert received["context"].tool_name == "approval_test_tool"
    assert received["context"].session_id == "session-1"
    assert received["context"].profile_name == "approval-test"
    assert received["context"].tool_call_id == "call-1"
    with pytest.raises(AttributeError):
        received["context"].session_id = "changed"
    assert calls == {"handler": 0, "hook": 0}


@pytest.mark.parametrize("human_approval", [True, False, "Always", "sometimes", 1, {}, []])
def test_plugin_rejects_invalid_human_approval(plugin_context, human_approval):
    registry, context = plugin_context

    with pytest.raises((TypeError, ValueError), match="human_approval"):
        _register(
            context,
            name=f"invalid_{type(human_approval).__name__}",
            human_approval=human_approval,
            preview=lambda args, preview_context: {},
        )

    assert registry.get_entry(
        f"invalid_{type(human_approval).__name__}", scope=context._manager.scope_key
    ) is None


@pytest.mark.parametrize(
    ("human_approval", "preview"),
    [("always", None), ("always", "not callable"), (None, lambda args, context: {})],
)
def test_plugin_rejects_missing_noncallable_or_dangling_preview(
    plugin_context, human_approval, preview
):
    registry, context = plugin_context

    with pytest.raises((TypeError, ValueError), match="approval_preview"):
        _register(
            context,
            name="invalid_preview",
            human_approval=human_approval,
            preview=preview,
        )

    assert registry.get_entry("invalid_preview", scope=context._manager.scope_key) is None


def test_plugin_rejects_async_preview_resolver(plugin_context):
    registry, context = plugin_context

    async def preview(args, preview_context):
        return {}

    with pytest.raises(TypeError, match="synchronous"):
        _register(context, name="async_preview", preview=preview)

    assert registry.get_entry("async_preview", scope=context._manager.scope_key) is None


@pytest.mark.parametrize("callable_object", [False, True])
def test_plugin_rejects_async_generator_preview_resolver(plugin_context, callable_object):
    registry, context = plugin_context

    async def preview(args, preview_context):
        yield {}

    if callable_object:
        class AsyncGeneratorPreview:
            async def __call__(self, args, preview_context):
                yield {}

        resolver = AsyncGeneratorPreview()
    else:
        resolver = preview

    with pytest.raises(TypeError, match="synchronous"):
        _register(context, name="async_generator_preview", preview=resolver)

    assert registry.get_entry(
        "async_generator_preview", scope=context._manager.scope_key
    ) is None


def test_core_registration_cannot_claim_plugin_approval_metadata():
    registry = ToolRegistry()

    with pytest.raises(PermissionError, match="plugin-provided"):
        registry.register(
            name="core_claim",
            toolset="core",
            schema=_schema("core_claim"),
            handler=lambda args, **kwargs: "{}",
            human_approval="always",
            approval_preview=lambda args, preview_context: {},
        )

    assert registry.get_entry("core_claim") is None


def test_imported_registry_internals_cannot_forge_plugin_approval_metadata():
    import tools.registry as registry_module

    registry = ToolRegistry()
    old_sentinel = getattr(registry_module, "_PLUGIN_APPROVAL_METADATA_TOKEN", object())

    with pytest.raises((PermissionError, TypeError), match="plugin|unexpected keyword"):
        registry.register(
            name="forged_claim",
            toolset="core",
            schema=_schema("forged_claim"),
            handler=lambda args, **kwargs: "{}",
            human_approval="always",
            approval_preview=lambda args, preview_context: {},
            _plugin_approval_metadata_token=old_sentinel,
        )

    assert registry.get_entry("forged_claim") is None

    forged_metadata = registry_module._ToolApprovalMetadata(
        "always", lambda args, preview_context: {}
    )
    with pytest.raises(PermissionError, match="plugin registration provenance"):
        registry._register(
            name="forged_private_claim",
            toolset="core",
            schema=_schema("forged_private_claim"),
            handler=lambda args, **kwargs: "{}",
            approval=forged_metadata,
        )
    assert registry.get_entry("forged_private_claim") is None


def _capture_plugin_registration_capability(monkeypatch, context, *, name):
    registry = context._tool_registration_registry
    captured = {}
    original = registry._register_plugin_tool

    def capture(capability, **kwargs):
        captured.update(capability=capability, kwargs=kwargs)
        raise RuntimeError("capture registration capability")

    monkeypatch.setattr(registry, "_register_plugin_tool", capture)
    with pytest.raises(RuntimeError, match="capture registration capability"):
        _register(context, name=name, preview=lambda args, preview_context: {})
    monkeypatch.setattr(registry, "_register_plugin_tool", original)
    return captured["capability"], captured["kwargs"], original


def test_plugin_registration_capability_is_exact_and_single_attempt(plugin_context, monkeypatch):
    registry, context = plugin_context
    other_manager = PluginManager(scope_key=context._manager.scope_key)
    other_context = PluginContext(
        PluginManifest(name="other-plugin", key="other-plugin"), other_manager
    )

    mutations = [
        {"context": other_context},
        {"manager": other_manager},
        {"plugin_key": "other-plugin"},
        {"scope": f"{context._manager.scope_key}-other"},
        {"scope": None},
        {"manager_generation": object()},
        {"name": "different_tool", "schema": _schema("different_tool")},
    ]
    for mutation in mutations:
        capability, kwargs, register_plugin_tool = _capture_plugin_registration_capability(
            monkeypatch, context, name="capability_bound_tool"
        )
        attempt = {**kwargs, **mutation}
        with pytest.raises(PermissionError, match="plugin registration provenance"):
            register_plugin_tool(capability, **attempt)
        with pytest.raises(PermissionError, match="plugin registration provenance"):
            register_plugin_tool(capability, **kwargs)

    assert registry.snapshot_registration(
        "capability_bound_tool", scope=context._manager.scope_key
    ) is None

    capability, kwargs, register_plugin_tool = _capture_plugin_registration_capability(
        monkeypatch, context, name="capability_bound_tool"
    )
    registered = register_plugin_tool(capability, **kwargs)
    assert registered is registry.snapshot_registration(
        "capability_bound_tool", scope=context._manager.scope_key
    )
    with pytest.raises(PermissionError, match="plugin registration provenance"):
        register_plugin_tool(capability, **kwargs)


def test_failed_plugin_registration_attempts_are_purged_and_retryable(
    plugin_context, monkeypatch
):
    registry, context = plugin_context
    baseline = len(registry._plugin_tool_registration_capabilities)

    for _ in range(2_000):
        with pytest.raises(TypeError, match="approval_preview"):
            _register(context, name="failed_attempt", preview="not callable")
    assert len(registry._plugin_tool_registration_capabilities) == baseline
    assert registry.snapshot_registration(
        "failed_attempt", scope=context._manager.scope_key
    ) is None

    cases = []

    capability, kwargs, register_plugin_tool = _capture_plugin_registration_capability(
        monkeypatch, context, name="invalid_metadata_attempt"
    )
    cases.append((capability, {**kwargs, "approval_preview": "not callable"}, kwargs, TypeError))

    capability, kwargs, register_plugin_tool = _capture_plugin_registration_capability(
        monkeypatch, context, name="collision_attempt"
    )
    registry.register(
        name="collision_attempt", toolset="other-toolset",
        schema=_schema("collision_attempt"), handler=lambda args, **kw: "{}",
    )
    cases.append((capability, kwargs, kwargs, None))

    _register(
        context, name="downgrade_attempt",
        preview=lambda args, preview_context: {"summary": "original"},
    )
    original = registry.snapshot_registration(
        "downgrade_attempt", scope=context._manager.scope_key
    )
    capability, kwargs, register_plugin_tool = _capture_plugin_registration_capability(
        monkeypatch, context, name="downgrade_attempt"
    )
    cases.append((
        capability, {**kwargs, "human_approval": None, "approval_preview": None}, kwargs,
        ValueError,
    ))

    class ExplodingSchema(dict):
        def get(self, key, default=None):
            raise RuntimeError("registration failed")

    capability, kwargs, register_plugin_tool = _capture_plugin_registration_capability(
        monkeypatch, context, name="exception_attempt"
    )
    cases.append((capability, {**kwargs, "schema": ExplodingSchema()}, kwargs, RuntimeError))

    generation = registry._generation
    for capability, attempt, replay, error_type in cases:
        if error_type is None:
            assert register_plugin_tool(capability, **attempt) is None
        else:
            with pytest.raises(error_type):
                register_plugin_tool(capability, **attempt)
        with pytest.raises(PermissionError, match="plugin registration provenance"):
            register_plugin_tool(capability, **replay)

    assert registry.snapshot_registration(
        "downgrade_attempt", scope=context._manager.scope_key
    ) is original
    assert registry.snapshot_registration(
        "exception_attempt", scope=context._manager.scope_key
    ) is None
    assert registry._generation == generation
    assert len(registry._plugin_tool_registration_capabilities) == baseline

    assert _register(
        context, name="failed_attempt",
        preview=lambda args, preview_context: {"summary": "valid"},
    ) is not None


def test_unload_revokes_pending_plugin_registration_capability(plugin_context, monkeypatch):
    registry, context = plugin_context
    baseline = len(registry._plugin_tool_registration_capabilities)
    assert context.register_tool(
        name="generation_anchor",
        toolset="approval-test",
        schema=_schema("generation_anchor"),
        handler=lambda args, **kwargs: "{}",
    ) is not None
    capability, kwargs, register_plugin_tool = _capture_plugin_registration_capability(
        monkeypatch, context, name="stale_generation_tool"
    )
    assert len(registry._plugin_tool_registration_capabilities) == baseline + 1

    assert context._manager.unload(context.plugin_id)
    assert len(registry._plugin_tool_registration_capabilities) == baseline
    with pytest.raises(PermissionError, match="plugin registration provenance"):
        register_plugin_tool(capability, **kwargs)
    assert registry.snapshot_registration(
        "stale_generation_tool", scope=context._manager.scope_key
    ) is None


@pytest.mark.parametrize("unload_mode", ["targeted", "all", "force_reload"])
def test_unload_waits_for_approval_tool_ledger_tracking(
    plugin_context, monkeypatch, unload_mode
):
    registry, context = plugin_context
    manager = context._manager
    name = f"registration_first_{unload_mode}"
    registry_written = Event()
    finish_tracking = Event()
    original_track = manager._track_scoped_registration

    def paused_track(*args, **kwargs):
        registry_written.set()
        assert finish_tracking.wait(timeout=2)
        return original_track(*args, **kwargs)

    monkeypatch.setattr(manager, "_track_scoped_registration", paused_track)
    if unload_mode == "force_reload":
        manager._discovered = True
        monkeypatch.setattr(manager, "_discover_and_load_inner", lambda: None)
        monkeypatch.setattr(manager, "_evict_stale_persistent_registrations", lambda: None)
        monkeypatch.setattr(manager, "_refresh_secret_sources_after_discovery", lambda: None)
        monkeypatch.setattr(manager, "_re_register_config_hooks_after_force", lambda: None)

    def unload():
        if unload_mode == "targeted":
            return manager.unload(context.plugin_id)
        if unload_mode == "all":
            return manager.unload()
        manager.discover_and_load(force=True)
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        registration = pool.submit(
            _register,
            context,
            name=name,
            preview=lambda args, preview_context: {"summary": "atomic"},
        )
        assert registry_written.wait(timeout=1)
        unloading = pool.submit(unload)
        try:
            unloading.result(timeout=0.05)
        except TimeoutError:
            pass
        finally:
            finish_tracking.set()
        assert registration.result(timeout=1) is not None
        assert unloading.result(timeout=1) is True

    assert registry.snapshot_registration(name, scope=manager.scope_key) is None
    assert context.plugin_id not in manager._ownership_ledger
    assert context.plugin_id not in manager._plugin_context_generations
    assert context not in registry._plugin_contexts
    assert not registry._plugin_tool_registration_capabilities


def test_approval_tool_registration_waits_for_unload_invalidation(
    plugin_context, monkeypatch
):
    registry, context = plugin_context
    manager = context._manager
    name = "unload_first_approval_tool"
    invalidation_reached = Event()
    finish_invalidation = Event()
    original_invalidate = manager._invalidate_plugin_contexts

    def paused_invalidate(plugin_keys):
        invalidation_reached.set()
        assert finish_invalidation.wait(timeout=2)
        original_invalidate(plugin_keys)

    monkeypatch.setattr(manager, "_invalidate_plugin_contexts", paused_invalidate)

    with ThreadPoolExecutor(max_workers=2) as pool:
        unloading = pool.submit(manager.unload, context.plugin_id)
        assert invalidation_reached.wait(timeout=1)
        registration = pool.submit(
            _register,
            context,
            name=name,
            preview=lambda args, preview_context: {"summary": "must fail closed"},
        )
        try:
            registration.result(timeout=0.05)
        except TimeoutError:
            pass
        finally:
            finish_invalidation.set()
        assert unloading.result(timeout=1) is True
        with pytest.raises(PermissionError, match="plugin registration provenance"):
            registration.result(timeout=1)

    assert registry.snapshot_registration(name, scope=manager.scope_key) is None
    assert context.plugin_id not in manager._ownership_ledger
    assert context.plugin_id not in manager._plugin_context_generations
    assert context not in registry._plugin_contexts
    assert not registry._plugin_tool_registration_capabilities


def test_failed_registration_does_not_poison_next_context_registration(plugin_context):
    registry, context = plugin_context

    with pytest.raises(TypeError, match="approval_preview"):
        _register(context, name="retry_after_failure", preview="not callable")

    assert _register(
        context,
        name="retry_after_failure",
        preview=lambda args, preview_context: {"summary": "valid"},
    ) is not None
    assert registry.get_entry(
        "retry_after_failure", scope=context._manager.scope_key
    ).human_approval == "always"


def test_preview_has_stable_canonical_utf8_bytes_and_sha256(plugin_context):
    registry, context = plugin_context
    _register(
        context,
        preview=lambda args, preview_context: {"z": "é", "a": {"b": 2, "a": 1}},
    )

    resolved = registry.resolve_approval_preview(
        "approval_test_tool", {}, scope=context._manager.scope_key
    )
    expected = '{"a":{"a":1,"b":2},"z":"é"}'.encode("utf-8")

    assert canonical_approval_preview_bytes(resolved) == expected
    assert approval_preview_hash(resolved) == hashlib.sha256(expected).hexdigest()


def test_preview_accepts_exact_byte_limit_and_rejects_one_byte_over(plugin_context):
    registry, context = plugin_context
    empty_size = len(canonical_approval_preview_bytes({"text": ""}))
    at_limit = "x" * (APPROVAL_PREVIEW_MAX_BYTES - empty_size)
    current = {"value": at_limit}

    def preview(args, preview_context):
        return {"text": current["value"]}

    _register(context, preview=preview)
    assert len(
        canonical_approval_preview_bytes(
            registry.resolve_approval_preview(
                "approval_test_tool", {}, scope=context._manager.scope_key
            )
        )
    ) == APPROVAL_PREVIEW_MAX_BYTES

    current["value"] += "x"
    with pytest.raises(ApprovalPreviewError, match="24 KiB"):
        registry.resolve_approval_preview(
            "approval_test_tool", {}, scope=context._manager.scope_key
        )


@pytest.mark.parametrize(
    "bad_result",
    [None, "text", ["list"], {"value": float("nan")}, {"value": object()}],
)
def test_preview_rejects_non_mapping_nonfinite_and_nonserializable_results(
    plugin_context, bad_result
):
    registry, context = plugin_context
    _register(context, preview=lambda args, preview_context: bad_result)

    with pytest.raises(ApprovalPreviewError):
        registry.resolve_approval_preview(
            "approval_test_tool", {}, scope=context._manager.scope_key
        )


def test_preview_wraps_resolver_exception_without_dispatch(plugin_context):
    registry, context = plugin_context
    handler_calls = 0

    def handler(args, **kwargs):
        nonlocal handler_calls
        handler_calls += 1
        return "{}"

    def preview(args, preview_context):
        raise RuntimeError("resolver failed with SECRET_VALUE")

    _register(context, handler=handler, preview=preview)

    with pytest.raises(ApprovalPreviewError, match="resolver failed") as excinfo:
        registry.resolve_approval_preview(
            "approval_test_tool", {}, scope=context._manager.scope_key
        )
    assert "SECRET_VALUE" not in str(excinfo.value)
    assert handler_calls == 0


@pytest.mark.parametrize("boundary", ["arguments", "result"])
@pytest.mark.parametrize("nested", [False, True])
def test_preview_canonicalization_sanitizes_all_mapping_exceptions(
    plugin_context, caplog, boundary, nested
):
    registry, context = plugin_context
    canary = "SECRET_CANARY"

    class HostileTopLevelMapping(Mapping):
        def __getitem__(self, key):
            raise KeyError(key)

        def __iter__(self):
            raise RuntimeError(canary)

        def __len__(self):
            return 1

    class HostileNestedMapping(dict):
        def items(self):
            raise RuntimeError(canary)

    hostile = {"nested": HostileNestedMapping(value=1)} if nested else HostileTopLevelMapping()
    result = hostile if boundary == "result" else {"summary": "safe"}
    _register(context, preview=lambda args, preview_context: result)
    arguments = hostile if boundary == "arguments" else {}

    with pytest.raises(ApprovalPreviewError) as excinfo:
        registry.resolve_approval_preview(
            "approval_test_tool", arguments, scope=context._manager.scope_key
        )

    rendered = "".join(
        traceback.format_exception(type(excinfo.value), excinfo.value, excinfo.value.__traceback__)
    )
    assert canary not in str(excinfo.value)
    assert canary not in repr(excinfo.value)
    assert canary not in rendered
    assert canary not in caplog.text


def test_model_arguments_cannot_change_intrinsic_metadata(plugin_context):
    registry, context = plugin_context
    received = {}
    preview = lambda args, preview_context: {"summary": "fixed"}

    def handler(args, **kwargs):
        received.update(args)
        return "{}"

    _register(context, handler=handler, preview=preview)
    registry.dispatch(
        "approval_test_tool",
        {"human_approval": None, "approval_preview": "replace"},
        scope=context._manager.scope_key,
    )

    entry = registry.get_entry("approval_test_tool", scope=context._manager.scope_key)
    assert received == {"human_approval": None, "approval_preview": "replace"}
    assert entry.human_approval == "always"
    assert entry.approval_preview is preview


def test_preview_context_identity_is_bounded_before_resolver_runs(plugin_context):
    registry, context = plugin_context
    resolver_calls = 0

    def preview(args, preview_context):
        nonlocal resolver_calls
        resolver_calls += 1
        return {}

    _register(context, preview=preview)

    with pytest.raises(ApprovalPreviewError, match="bounded identity"):
        registry.resolve_approval_preview(
            "approval_test_tool",
            {},
            scope=context._manager.scope_key,
            session_id="x" * 1025,
        )
    assert resolver_calls == 0


def test_same_toolset_reregistration_cannot_downgrade_always(plugin_context):
    registry, context = plugin_context
    preview = lambda args, preview_context: {"summary": "first"}
    _register(context, preview=preview)
    original = registry.get_entry("approval_test_tool", scope=context._manager.scope_key)

    with pytest.raises(ValueError, match="cannot downgrade"):
        context.register_tool(
            name="approval_test_tool",
            toolset="approval-test",
            schema=_schema("approval_test_tool"),
            handler=lambda args, **kwargs: "{}",
        )

    current = registry.get_entry("approval_test_tool", scope=context._manager.scope_key)
    assert current is original
    assert current.human_approval == "always"
    assert current.approval_preview is preview


def test_legacy_plugin_tool_has_no_approval_resolver(plugin_context):
    registry, context = plugin_context
    handle = context.register_tool(
        name="legacy_plugin_tool",
        toolset="approval-test",
        schema=_schema("legacy_plugin_tool"),
        handler=lambda args, **kwargs: "{}",
    )

    assert handle is not None
    entry = registry.get_entry("legacy_plugin_tool", scope=context._manager.scope_key)
    assert entry is not None
    assert entry.human_approval is None
    assert entry.approval_preview is None
    with pytest.raises(ApprovalPreviewError, match="does not require"):
        registry.resolve_approval_preview(
            "legacy_plugin_tool", {}, scope=context._manager.scope_key
        )
