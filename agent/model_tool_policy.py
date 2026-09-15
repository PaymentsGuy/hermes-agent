"""Validation, persistence decoding, and schema filtering for model-tool policy.

``policy_sha256`` is an opaque external policy identity.  Hermes validates its
wire shape and preserves it; it does not derive it from the other fields.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, Mapping


MODEL_TOOL_POLICY_VERSION = 1
_POLICY_FIELDS = frozenset({
    "policy_id", "policy_sha256", "allowed_tools", "approval_required_tools",
})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_UNSUPPORTED_V1_NESTED_EXECUTION_TOOLS = frozenset({"delegate_task", "execute_code"})
logger = logging.getLogger(__name__)


class ModelToolPolicyContinuityError(ValueError):
    """A same-session continuation changed or corrupted its policy carrier."""


def enforce_model_tool_policy_runtime(policy: Mapping[str, Any] | None, api_mode: str | None) -> None:
    """Reject transports that cannot prove the policy covers every model-owned tool."""
    if policy is not None and api_mode == "codex_app_server":
        raise RuntimeError(
            "`model_tool_policy` is unsupported with `codex_app_server`: Codex-owned built-in "
            "tools do not expose a proven exact tool policy allowlist; refusing to start the "
            "provider transport. Select a runtime that dispatches tools through Hermes policy."
        )


def _tool_names(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"model_tool_policy.{field} must be an array")
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValueError(f"model_tool_policy.{field} entries must be non-empty exact tool names")
    if len(value) != len(set(value)):
        raise ValueError(f"model_tool_policy.{field} entries must be unique")
    return list(value)


def normalize_model_tool_policy(
    value: Any, *, available_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a validated copy of the closed policy mapping."""
    if not isinstance(value, Mapping):
        raise ValueError("model_tool_policy must be an object")
    fields = set(value)
    if fields != _POLICY_FIELDS:
        missing = sorted(_POLICY_FIELDS - fields)
        unknown = sorted(fields - _POLICY_FIELDS)
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if unknown:
            detail.append("unknown: " + ", ".join(unknown))
        raise ValueError("model_tool_policy fields are closed (" + "; ".join(detail) + ")")
    policy_id = value["policy_id"]
    policy_sha256 = value["policy_sha256"]
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError("model_tool_policy.policy_id must be a non-empty opaque string")
    if not isinstance(policy_sha256, str) or _SHA256_RE.fullmatch(policy_sha256) is None:
        raise ValueError("model_tool_policy.policy_sha256 must be 64 lowercase hexadecimal characters")
    allowed = _tool_names(value["allowed_tools"], "allowed_tools")
    unsupported = sorted(set(allowed) & _UNSUPPORTED_V1_NESTED_EXECUTION_TOOLS)
    if unsupported:
        raise ValueError(
            "model_tool_policy.allowed_tools contains unsupported V1 nested execution authority: "
            + ", ".join(unsupported)
        )
    approval = _tool_names(value["approval_required_tools"], "approval_required_tools")
    if not set(approval).issubset(allowed):
        raise ValueError("model_tool_policy.approval_required_tools must be a subset of allowed_tools")
    if available_names is not None:
        available = set(available_names)
        unavailable = [name for name in allowed if name not in available]
        if unavailable:
            raise ValueError("model_tool_policy declares unavailable tool(s): " + ", ".join(unavailable))
    return {
        "policy_id": policy_id,
        "policy_sha256": policy_sha256,
        "allowed_tools": allowed,
        "approval_required_tools": approval,
    }


def inherit_model_tool_policy(parent: Any) -> dict[str, Any] | None:
    """Return an exact normalized policy copy for a derivative model agent."""
    policy = getattr(parent, "model_tool_policy", None)
    return None if policy is None else normalize_model_tool_policy(policy)


def validate_inherited_model_tool_policy(
    parent: Any, *, enabled_toolsets: list[str] | None,
    disabled_toolsets: list[str] | None = None,
) -> dict[str, Any] | None:
    """Fail before derivative startup when its requested surface cannot honor the parent."""
    policy = inherit_model_tool_policy(parent)
    if policy is None:
        return None
    definitions = resolve_eventual_model_tool_surface(
        enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
    )
    apply_model_tool_policy(policy, definitions)
    return policy


def encode_model_tool_policy(value: Mapping[str, Any]) -> str:
    return json.dumps(normalize_model_tool_policy(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def decode_model_tool_policy_carrier(version: Any, payload: Any) -> dict[str, Any] | None:
    """Decode one durable/live/internal policy carrier, preserving legacy absence."""
    if version is None and payload is None:
        return None
    if version != MODEL_TOOL_POLICY_VERSION or payload is None:
        raise ValueError("model-tool policy marker or payload is missing/corrupt")
    if isinstance(payload, str):
        if not payload:
            raise ValueError("model-tool policy marker or payload is missing/corrupt")
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("model-tool policy payload is unreadable") from exc
    return normalize_model_tool_policy(payload)


def decode_stored_model_tool_policy(version: Any, stored: Any) -> dict[str, Any] | None:
    """Decode persisted policy; distinguish a legacy absence from corruption."""
    if version is None and stored is None:
        return None
    if not isinstance(stored, str):
        raise ValueError("stored model-tool policy marker or payload is missing/corrupt")
    try:
        return decode_model_tool_policy_carrier(version, stored)
    except ValueError as exc:
        if "unreadable" not in str(exc):
            raise
        raise ValueError("stored model-tool policy is unreadable") from exc


def require_matching_model_tool_policy_carriers(
    left_version: Any,
    left_payload: Any,
    right_version: Any,
    right_payload: Any,
    *,
    context: str,
) -> dict[str, Any] | None:
    """Decode two persisted carriers and require exact normalized equality."""
    try:
        left = decode_stored_model_tool_policy(left_version, left_payload)
        right = decode_stored_model_tool_policy(right_version, right_payload)
    except ValueError as exc:
        raise ModelToolPolicyContinuityError(f"{context} model-tool policy is corrupt: {exc}") from exc
    if left != right:
        raise ModelToolPolicyContinuityError(
            f"{context} model-tool policy does not match exactly"
        )
    return left


def apply_model_tool_policy(policy: Mapping[str, Any], tool_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate availability and retain exact schema objects in their original order."""
    names = [
        item.get("function", {}).get("name")
        for item in tool_definitions
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    ]
    normalized = normalize_model_tool_policy(policy, available_names=(name for name in names if isinstance(name, str)))
    allowed = set(normalized["allowed_tools"])
    return [item for item in tool_definitions if item.get("function", {}).get("name") in allowed]


def _append_dynamic_schemas(
    definitions: list[dict[str, Any]], schemas: Iterable[Any],
) -> None:
    """Append locally discovered provider schemas using the agent's normalization/dedup rules."""
    from agent.memory_manager import normalize_tool_schema

    names = {
        item.get("function", {}).get("name")
        for item in definitions
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    for raw_schema in schemas:
        schema = normalize_tool_schema(raw_schema)
        if schema is not None and schema["name"] not in names:
            definitions.append({"type": "function", "function": schema})
            names.add(schema["name"])


def resolve_eventual_model_tool_surface(
    *, enabled_toolsets: list[str] | None, disabled_toolsets: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve the pre-prompt model-visible candidate surface without provider I/O.

    Registry/toolset definitions are followed by schemas from the configured, locally
    available memory provider and context engine under the caller's current profile and
    working-directory scopes. Provider initialization is deliberately not run: it may open
    clients or start workers. Dynamic names exposed by the provider's pre-init schema contract
    are therefore recognized candidates; :func:`apply_model_tool_policy` remains the final,
    fail-closed check after real agent initialization and dynamic injection.
    """
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
    except Exception:
        logger.warning("Plugin discovery failed during model-tool policy validation", exc_info=True)

    import model_tools

    definitions = list(model_tools.get_tool_definitions(
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        quiet_mode=True,
        skip_tool_search_assembly=True,
    ) or [])

    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:
        config = {}

    try:
        from agent.memory_manager import MemoryManager, memory_provider_tools_enabled
        from plugins.memory import load_memory_provider
        from tools.memory_tool import get_builtin_memory_config

        memory_config = get_builtin_memory_config(config)
        provider_name = str(memory_config.get("provider") or "").strip()
        memory_present = any(
            item.get("function", {}).get("name") == "memory"
            for item in definitions if isinstance(item, dict)
        )
        if provider_name and memory_provider_tools_enabled(
            enabled_toolsets, disabled_toolsets, memory_tool_present=memory_present,
        ):
            provider = load_memory_provider(provider_name, register_skills=False)
            if provider is not None and provider.is_available():
                manager = MemoryManager()
                manager.add_provider(provider)
                _append_dynamic_schemas(definitions, manager.get_all_tool_schemas())
    except Exception:
        # Agent init treats memory-provider load/availability/schema failures as unavailable.
        logger.debug("Memory-provider schema discovery skipped during policy validation", exc_info=True)

    try:
        from agent.agent_init import _select_context_engine

        engine = _select_context_engine(config)
        if engine is not None and (enabled_toolsets is None or "context_engine" in enabled_toolsets):
            _append_dynamic_schemas(definitions, engine.get_tool_schemas())
    except Exception:
        # A provider that cannot expose local schemas cannot authorize an unknown name here.
        logger.debug("Context-engine schema discovery skipped during policy validation", exc_info=True)

    return definitions


def validate_model_tool_policy_for_eventual_surface(
    value: Any, *, enabled_toolsets: list[str] | None,
    disabled_toolsets: list[str] | None = None,
) -> dict[str, Any]:
    """Normalize and validate a policy against one canonical eventual session surface."""
    policy = normalize_model_tool_policy(value)
    definitions = resolve_eventual_model_tool_surface(
        enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
    )
    apply_model_tool_policy(policy, definitions)
    return policy


def model_tool_policy_identity(policy: Mapping[str, Any] | None) -> dict[str, str] | None:
    if policy is None:
        return None
    normalized = normalize_model_tool_policy(policy)
    return {key: normalized[key] for key in ("policy_id", "policy_sha256")}


def model_tool_policy_denial(
    function_name: str, *, call_origin: str, policy: Mapping[str, Any] | None,
) -> str | None:
    """Return a denial for a model-origin call outside its exact session grant."""
    if call_origin != "model" or policy is None:
        return None
    normalized = normalize_model_tool_policy(policy)
    if function_name not in normalized["allowed_tools"]:
        return f"Tool '{function_name}' is not allowed by this session's model-tool policy."
    return None
