"""Disposable real-plugin composition fixture for H1-H4 skill learning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

import model_tools
from hermes_cli.plugins import get_plugin_manager
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


PROPOSAL_TOOL = "propose_action_os_learning"
EXECUTOR_TOOL = "apply_action_os_learning"
PLUGIN_TOOLSET = "learning-fixture"
BEFORE = b"---\nname: managed-skill\ndescription: Use when testing learning. Fixture.\n---\n\nBefore fixture body.\n"
AFTER = BEFORE.replace(b"Before fixture body.", b"After fixture body.", 1)
SCOPE_HASH = "a" * 64


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _guarded_arguments() -> dict:
    return {
        "operations": [
            {
                "name": "managed-skill",
                "action": "patch",
                "file_path": "SKILL.md",
                "old_string": "Before fixture body.",
                "new_string": "After fixture body.",
                "replace_all": False,
                "expected_target_state": "present",
                "expected_sha256": _sha(BEFORE),
                "expected_result_sha256": _sha(AFTER),
                "authorization_scope_id": "learning-fixture:patch",
                "authorization_scope_sha256": SCOPE_HASH,
                "sync_policy": "suppress",
                "required_owner_class": "curator_managed",
                "expected_profile_relative_skill_root": "skills/managed-skill",
            }
        ]
    }


def _write_disposable_plugin(home: Path, approved_arguments: dict) -> None:
    plugin = home / "plugins" / PLUGIN_TOOLSET
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump({"name": PLUGIN_TOOLSET, "version": "1.0.0"}),
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        f'''import json
from dataclasses import asdict

from tools.intrinsic_approval import get_intrinsic_approval_receipt_context

APPROVED_ARGUMENTS = {approved_arguments!r}
PREVIEW_CALLS = 0
SEEN_RECEIPTS = []


def _schema(name):
    return {{
        "name": name,
        "description": "Disposable learning integration fixture.",
        "parameters": {{"type": "object", "properties": {{}}}},
    }}


def _propose(args, **kwargs):
    return json.dumps({{"accepted": True, "mode": "proposal"}})


def _preview(args, context):
    global PREVIEW_CALLS
    PREVIEW_CALLS += 1
    return {{
        "summary": "Apply one approved learning patch",
        "authorization_scope_sha256": "{SCOPE_HASH}",
        "approved_dispatch": {{
            "tool_name": "skill_manage",
            "arguments": APPROVED_ARGUMENTS,
        }},
    }}


def _apply(args, **kwargs):
    receipt = get_intrinsic_approval_receipt_context()
    SEEN_RECEIPTS.append(asdict(receipt) if receipt is not None else None)
    return CTX.dispatch_approved_tool("skill_manage", APPROVED_ARGUMENTS)


def register(ctx):
    global CTX
    CTX = ctx
    ctx.register_tool(
        name="{PROPOSAL_TOOL}", toolset="{PLUGIN_TOOLSET}",
        schema=_schema("{PROPOSAL_TOOL}"), handler=_propose,
    )
    ctx.register_tool(
        name="{EXECUTOR_TOOL}", toolset="{PLUGIN_TOOLSET}",
        schema=_schema("{EXECUTOR_TOOL}"), handler=_apply,
        human_approval="always", approval_preview=_preview,
    )
''',
        encoding="utf-8",
    )


def test_real_plugin_manager_composes_proposal_approval_and_guarded_patch(
    tmp_path, monkeypatch
):
    from agent import background_review, prompt_builder
    from hermes_cli import plugins as plugins_module
    from tools import approval, approval_context, skill_manager_tool, skill_usage, terminal_tool
    from tools.intrinsic_approval import get_intrinsic_approval_receipt_context
    from tools.registry import registry

    home = tmp_path / "home"
    target = home / "skills" / "managed-skill" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_bytes(BEFORE)
    (home / "skills" / ".usage.json").write_text(
        json.dumps({"managed-skill": {"created_by": "agent", "pinned": False}}),
        encoding="utf-8",
    )
    approved_arguments = _guarded_arguments()
    _write_disposable_plugin(home, approved_arguments)
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {"enabled": [PLUGIN_TOOLSET]},
                "tools": {"tool_search": {"enabled": "off"}},
                "auxiliary": {
                    "background_review": {
                        "mode": "propose",
                        "proposal_tool": PROPOSAL_TOOL,
                        "extra_tools": [PROPOSAL_TOOL, "skill_manage"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setattr(plugins_module, "get_bundled_plugins_dir", lambda: empty_bundled)

    writes = []
    bumps = []
    cache_clears = []
    sync_calls = []
    original_guarded_write = skill_manager_tool._guarded_write

    def counted_guarded_write(*args, **kwargs):
        writes.append((args, kwargs))
        return original_guarded_write(*args, **kwargs)

    monkeypatch.setattr(skill_manager_tool, "_guarded_write", counted_guarded_write)
    monkeypatch.setattr(
        skill_usage, "bump_patch", lambda *args, **kwargs: bumps.append((args, kwargs))
    )
    monkeypatch.setattr(
        prompt_builder,
        "clear_skills_system_prompt_cache",
        lambda *args, **kwargs: cache_clears.append((args, kwargs)),
    )
    monkeypatch.setattr(
        skill_manager_tool,
        "_maybe_debounced_sync_push",
        lambda *args, **kwargs: sync_calls.append((args, kwargs)),
    )

    prompts = []
    home_token = set_hermes_home_override(home)
    interactive_token = approval_context.set_hermes_interactive_context(True)
    monkeypatch.setattr(approval_context, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_context, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(
        approval_context, "_is_unattended_platform_approval_context", lambda: False
    )
    monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: False)
    terminal_tool.set_approval_callback(
        lambda command, description, **kwargs: prompts.append(json.loads(command)) or "once"
    )
    manager = get_plugin_manager()
    try:
        manager.discover_and_load()
        loaded = manager._plugins[PLUGIN_TOOLSET]
        assert loaded.enabled is True
        assert loaded.tools_registered == [PROPOSAL_TOOL, EXECUTOR_TOOL]
        entries = [
            registry.get_entry(name, scope=manager.scope_key)
            for name in (PROPOSAL_TOOL, EXECUTOR_TOOL)
        ]
        assert all(entry is not None for entry in entries)
        assert {entry.toolset for entry in entries if entry is not None} == {
            PLUGIN_TOOLSET
        }

        definitions = model_tools.get_tool_definitions(
            enabled_toolsets=[PLUGIN_TOOLSET], quiet_mode=True
        )
        parent = SimpleNamespace(tools=definitions)
        enabled, task_cfg = background_review.load_background_review_settings()
        assert enabled is True
        assert background_review._proposal_tool(parent, task_cfg) == PROPOSAL_TOOL
        whitelist, configured = background_review._review_tool_whitelist(parent, task_cfg)
        assert whitelist == {
            PROPOSAL_TOOL,
            "skill_view",
            "skills_list",
            "read_file",
            "search_files",
        }
        assert configured == {PROPOSAL_TOOL}
        assert {"skill_manage", EXECUTOR_TOOL}.isdisjoint(whitelist)

        result = json.loads(
            model_tools.handle_function_call(
                EXECUTOR_TOOL,
                {},
                session_id="session-1",
                turn_id="turn-1",
                tool_call_id="call-1",
                call_origin="model",
            )
        )

        assert "approval_receipt_id" in result, result
        receipt_id = result.pop("approval_receipt_id")
        assert isinstance(receipt_id, str) and receipt_id
        assert result == {
            "success": True,
            "result_kind": "committed",
            "authorization_scope_id": "learning-fixture:patch",
            "authorization_scope_sha256": SCOPE_HASH,
            "required_owner_class": "curator_managed",
            "observed_owner_class": "curator_managed",
            "expected_profile_relative_skill_root": "skills/managed-skill",
            "observed_profile_relative_skill_root": "skills/managed-skill",
            "before_manifest": [
                {
                    "path": "skills/managed-skill/SKILL.md",
                    "state": "present",
                    "byte_length": len(BEFORE),
                    "sha256": _sha(BEFORE),
                }
            ],
            "after_manifest": [
                {
                    "path": "skills/managed-skill/SKILL.md",
                    "state": "present",
                    "byte_length": len(AFTER),
                    "sha256": _sha(AFTER),
                }
            ],
            "external_sync": "suppressed",
        }
        module = loaded.module
        assert module is not None
        assert module.PREVIEW_CALLS == 1
        assert prompts == [
            {
                "summary": "Apply one approved learning patch",
                "authorization_scope_sha256": SCOPE_HASH,
                "approved_dispatch": {
                    "tool_name": "skill_manage",
                    "arguments": approved_arguments,
                },
            }
        ]
        assert len(module.SEEN_RECEIPTS) == 1
        assert module.SEEN_RECEIPTS[0]["receipt_id"] == receipt_id
        assert module.SEEN_RECEIPTS[0]["decision"] == "approved"
        assert module.SEEN_RECEIPTS[0]["approval_scope_sha256"] == SCOPE_HASH
        assert get_intrinsic_approval_receipt_context() is None
        assert target.read_bytes() == AFTER
        assert _sha(target.read_bytes()) == _sha(AFTER)
        assert len(writes) == len(bumps) == len(cache_clears) == 1
        assert sync_calls == []
    finally:
        manager.unload()
        terminal_tool.set_approval_callback(None)
        approval.clear_session("session-1")
        approval.unregister_gateway_notify("session-1")
        approval_context.reset_hermes_interactive_context(interactive_token)
        reset_hermes_home_override(home_token)
