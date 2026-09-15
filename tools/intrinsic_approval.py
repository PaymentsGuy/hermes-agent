"""Fail-closed runtime gate for plugin tools marked ``human_approval='always'``."""

from __future__ import annotations

import contextvars
import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from tools.registry import ToolEntry, ToolRegistry, _canonical_json_mapping, tool_error
from tools.tool_approval_receipts import (
    canonical_original_args_sha256,
    create_intrinsic_approval_receipt,
    current_intrinsic_approval_profile_binding,
    validate_intrinsic_approval_identity,
    validate_intrinsic_approval_scope_sha256,
)


class IntrinsicApprovalDenied(RuntimeError):
    """An intrinsic tool call did not obtain exact live approve-once authority."""


@dataclass(frozen=True, slots=True)
class IntrinsicApprovalReceiptContext:
    receipt_id: str
    profile_identity: str
    profile_home: str
    session_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str
    original_args_sha256: str
    preview_sha256: str
    approval_scope_sha256: str
    decision: str
    decided_at: float
    expires_at: float


@dataclass(slots=True)
class _ApprovedDispatchCapability:
    registry: ToolRegistry
    scope: str
    tool_name: str
    arguments_sha256: str
    arguments_json: bytes
    approval_scope_sha256: str
    bound: bool = False
    active: bool = False
    attempted: bool = False
    attempt_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(frozen=True, slots=True)
class IntrinsicApprovalGrant:
    entry: ToolEntry
    scope: str
    profile_identity: str
    profile_home: str
    original_args_sha256: str
    original_args_json: bytes
    receipt_context: IntrinsicApprovalReceiptContext
    dispatch_capability: _ApprovedDispatchCapability | None

    def original_args(self) -> dict:
        return json.loads(self.original_args_json)


_handler_receipt: contextvars.ContextVar[IntrinsicApprovalReceiptContext | None] = (
    contextvars.ContextVar("intrinsic_approval_handler_receipt", default=None)
)
_handler_dispatch_capability: contextvars.ContextVar[_ApprovedDispatchCapability | None] = (
    contextvars.ContextVar("intrinsic_approval_handler_dispatch_capability", default=None)
)
_exact_dispatch_capability: contextvars.ContextVar[_ApprovedDispatchCapability | None] = (
    contextvars.ContextVar("intrinsic_approval_exact_dispatch_capability", default=None)
)


def get_intrinsic_approval_receipt_context() -> IntrinsicApprovalReceiptContext | None:
    """Return the exact receipt only while its approved registry handler is running."""
    return _handler_receipt.get()


def _current_exact_approved_skill_dispatch(
) -> tuple[IntrinsicApprovalReceiptContext, str] | None:
    """Return authority only during the exact approved ``skill_manage`` dispatch."""
    receipt = _handler_receipt.get()
    capability = _handler_dispatch_capability.get()
    if (
        receipt is None
        or capability is None
        or _exact_dispatch_capability.get() is not capability
        or not capability.active
        or not capability.attempted
        or capability.tool_name != "skill_manage"
        or receipt.decision != "approved"
        or receipt.approval_scope_sha256 != capability.approval_scope_sha256
    ):
        return None
    return receipt, capability.approval_scope_sha256


@contextmanager
def bind_intrinsic_approval_receipt_context(
    receipt: IntrinsicApprovalReceiptContext,
    dispatch_capability: _ApprovedDispatchCapability | None = None,
) -> Iterator[None]:
    bound_capability = None
    if dispatch_capability is not None:
        with dispatch_capability.attempt_lock:
            if not dispatch_capability.bound:
                dispatch_capability.bound = True
                dispatch_capability.active = True
                bound_capability = dispatch_capability
    receipt_token = _handler_receipt.set(receipt)
    dispatch_token = _handler_dispatch_capability.set(bound_capability)
    try:
        yield
    finally:
        if bound_capability is not None:
            with bound_capability.attempt_lock:
                bound_capability.active = False
        _handler_dispatch_capability.reset(dispatch_token)
        _handler_receipt.reset(receipt_token)


def _approved_dispatch_capability(
    registry: ToolRegistry,
    scope: str,
    preview: Mapping,
    approval_scope_sha256: str,
) -> _ApprovedDispatchCapability | None:
    if "approved_dispatch" not in preview:
        return None
    envelope = preview["approved_dispatch"]
    if not isinstance(envelope, Mapping) or set(envelope) != {"tool_name", "arguments"}:
        raise IntrinsicApprovalDenied("approved dispatch envelope is malformed")
    if envelope["tool_name"] != "skill_manage":
        raise IntrinsicApprovalDenied("approved dispatch tool is unavailable")
    arguments, arguments_json = _canonical_json_mapping(
        envelope["arguments"], label="approved_dispatch arguments"
    )
    operations = arguments.get("operations")
    if not isinstance(operations, list) or len(operations) != 1:
        raise IntrinsicApprovalDenied("approved dispatch operations are malformed")
    operation = operations[0]
    if (
        not isinstance(operation, Mapping)
        or operation.get("action") != "patch"
        or operation.get("replace_all") is not False
        or operation.get("authorization_scope_sha256") != approval_scope_sha256
    ):
        raise IntrinsicApprovalDenied("approved dispatch operation is malformed")
    return _ApprovedDispatchCapability(
        registry=registry,
        scope=scope,
        tool_name="skill_manage",
        arguments_sha256=hashlib.sha256(arguments_json).hexdigest(),
        arguments_json=arguments_json,
        approval_scope_sha256=approval_scope_sha256,
    )


def dispatch_intrinsic_approved_tool(context, tool_name: str, args: Mapping) -> str | dict:
    """Consume the exact private handler capability and dispatch its frozen envelope once."""
    denied = tool_error("Approved tool dispatch denied")
    capability = _handler_dispatch_capability.get()
    if capability is None:
        return denied
    with capability.attempt_lock:
        if not capability.active or capability.attempted:
            return denied
        capability.attempted = True
    try:
        registry = capability.registry
        if (
            getattr(context, "_tool_registration_registry", None) is not registry
            or tool_name != capability.tool_name
            or not registry._plugin_context_active_in_scope(context, capability.scope)
        ):
            return denied
        canonical_args, encoded = _canonical_json_mapping(
            args, label="approved tool arguments"
        )
        if (
            hashlib.sha256(encoded).hexdigest() != capability.arguments_sha256
            or encoded != capability.arguments_json
        ):
            return denied
        dispatch_token = _exact_dispatch_capability.set(capability)
        try:
            return registry.dispatch(tool_name, canonical_args, scope=capability.scope)
        finally:
            _exact_dispatch_capability.reset(dispatch_token)
    except Exception:
        return denied


def _surface_available(default_session_key: str) -> tuple[str, Any, str]:
    from tools import approval, approval_context

    session_key = approval_context.get_current_session_key(default_session_key)
    from agent.delegation_context import is_delegated_child_process_context
    if (
        is_delegated_child_process_context()
        or approval_context._is_cron_approval_context()
        or approval_context._is_single_query_approval_context()
        or approval_context._is_unattended_platform_approval_context()
    ):
        raise IntrinsicApprovalDenied("intrinsic approval requires a live human surface")
    if approval_context._is_gateway_approval_context():
        callback = approval._gateway_notify_cb(session_key)
        if callback is None:
            raise IntrinsicApprovalDenied("intrinsic approval requires a live human surface")
        return "gateway", callback, session_key
    callback = approval_context._resolve_cli_approval_callback()
    if not approval_context._is_interactive_cli() or callback is None:
        raise IntrinsicApprovalDenied("intrinsic approval requires a live human surface")
    return "cli", callback, session_key


def _ask_once(surface: str, callback, session_key: str, preview: Mapping, tool_name: str) -> str:
    from agent.redact import redact_sensitive_text
    from tools.approval_gateway_wait import _await_gateway_decision
    from tools.approval_prompt import prompt_dangerous_approval
    from tools.registry import canonical_approval_preview_bytes

    display = redact_sensitive_text(
        canonical_approval_preview_bytes(preview).decode("utf-8"), force=True
    )
    description = f"Allow plugin tool {tool_name} once?"
    if surface == "gateway":
        result = _await_gateway_decision(
            session_key,
            callback,
            {
                "command": display,
                "description": description,
                "pattern_key": "intrinsic-plugin-approval",
                "pattern_keys": ["intrinsic-plugin-approval"],
                "allow_session": False,
                "allow_permanent": False,
                "choices": ["once", "deny"],
            },
            surface="intrinsic",
        )
        return result.get("choice") if result.get("resolved") else "timeout"
    return prompt_dangerous_approval(
        display,
        description,
        allow_permanent=False,
        allow_session=False,
        approval_callback=callback,
    )


def require_intrinsic_approval(
    registry: ToolRegistry,
    name: str,
    args: Mapping,
    *,
    session_id: str,
    turn_id: str,
    tool_call_id: str,
) -> IntrinsicApprovalGrant | None:
    """Approve and durably receipt one exact intrinsic entry, or fail closed."""
    scope = registry.current_scope_key()
    entry = registry.get_entry(name, scope=scope)
    if entry is None or entry.human_approval != "always":
        return None

    session_id = validate_intrinsic_approval_identity(session_id, "session_id")
    turn_id = validate_intrinsic_approval_identity(turn_id, "turn_id")
    tool_call_id = validate_intrinsic_approval_identity(tool_call_id, "tool_call_id")
    validate_intrinsic_approval_identity(name, "tool_name")
    profile_identity, profile_home = current_intrinsic_approval_profile_binding()
    surface, callback, session_key = _surface_available(session_id)
    canonical_args, args_json = _canonical_json_mapping(args, label="original_args")
    preview = registry.resolve_approval_preview(
        name,
        canonical_args,
        session_id=session_id,
        profile_name=profile_identity,
        tool_call_id=tool_call_id,
        scope=scope,
    )
    scope_hash = validate_intrinsic_approval_scope_sha256(
        preview.get("authorization_scope_sha256")
    )
    dispatch_capability = _approved_dispatch_capability(
        registry, scope, preview, scope_hash
    )
    choice = _ask_once(surface, callback, session_key, preview, name)

    if current_intrinsic_approval_profile_binding() != (profile_identity, profile_home):
        raise IntrinsicApprovalDenied("active profile changed during intrinsic approval")
    decision = "approved" if choice == "once" else "denied"
    receipt = create_intrinsic_approval_receipt(
        session_id=session_id,
        turn_id=turn_id,
        tool_call_id=tool_call_id,
        tool_name=name,
        original_args=canonical_args,
        preview=preview,
        approval_scope_sha256=scope_hash,
        decision=decision,
    )
    if decision != "approved":
        raise IntrinsicApprovalDenied("intrinsic approval was denied")
    context = IntrinsicApprovalReceiptContext(**{
        key: receipt[key]
        for key in IntrinsicApprovalReceiptContext.__dataclass_fields__
    })
    return IntrinsicApprovalGrant(
        entry=entry,
        scope=scope,
        profile_identity=profile_identity,
        profile_home=profile_home,
        original_args_sha256=canonical_original_args_sha256(canonical_args),
        original_args_json=args_json,
        receipt_context=context,
        dispatch_capability=dispatch_capability,
    )


def approved_args_unchanged(grant: IntrinsicApprovalGrant, args: Mapping) -> bool:
    try:
        return canonical_original_args_sha256(args) == grant.original_args_sha256
    except Exception:
        return False


def approved_profile_unchanged(grant: IntrinsicApprovalGrant) -> bool:
    try:
        return current_intrinsic_approval_profile_binding() == (
            grant.profile_identity,
            grant.profile_home,
        )
    except Exception:
        return False
