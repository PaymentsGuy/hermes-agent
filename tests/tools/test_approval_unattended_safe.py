from types import SimpleNamespace

from tools import approval
from tools.approval_context import reset_current_session_key, set_current_session_key


def with_session(key: str):
    return set_current_session_key(key)


def cleanup(key: str, token) -> None:
    approval.clear_session(key)
    reset_current_session_key(token)


def test_unattended_safe_mode_allows_unflagged_commands_without_prompt():
    key = "safe-session-command"
    token = with_session(key)
    try:
        approval.enable_session_unattended_safe_mode(key)
        result = approval.check_all_command_guards("python -m pytest -q", "local")
        assert result["approved"] is True
    finally:
        cleanup(key, token)


def test_unattended_safe_mode_blocks_dangerous_command_even_when_yolo_is_enabled():
    key = "safe-session-dangerous"
    token = with_session(key)
    try:
        approval.enable_session_unattended_safe_mode(key)
        approval.enable_session_yolo(key)
        result = approval.check_all_command_guards("rm -rf ./important-data", "local")
        assert result["approved"] is False
        assert result["outcome"] == "blocked"
        assert "unattended" in result["message"].lower()
    finally:
        cleanup(key, token)


def test_unattended_safe_mode_blocks_dangerous_sandbox_command_before_fast_path():
    key = "safe-session-sandbox-command"
    token = with_session(key)
    try:
        approval.enable_session_unattended_safe_mode(key)
        result = approval.check_all_command_guards("rm -rf ./important-data", "docker")
        assert result["approved"] is False
        assert result["outcome"] == "blocked"
        assert "unattended" in result["message"].lower()
    finally:
        cleanup(key, token)


def test_unattended_safe_mode_blocks_arbitrary_tool_approval_without_human_prompt():
    key = "safe-session-tool"
    token = with_session(key)
    try:
        approval.enable_session_unattended_safe_mode(key)
        result = approval.request_tool_approval(
            "paid_or_destructive_operation",
            "Paid or destructive operation requires explicit human confirmation",
        )
        assert result["approved"] is False
        assert result["outcome"] == "blocked"
        assert result["user_consent"] is False
    finally:
        cleanup(key, token)


def test_unattended_safe_mode_blocks_execute_code_instead_of_prompting():
    key = "safe-session-execute-code"
    token = with_session(key)
    try:
        approval.enable_session_unattended_safe_mode(key)
        result = approval.check_execute_code_guard("print('hello')", "local")
        assert result["approved"] is False
        assert result["outcome"] == "blocked"
        assert "unattended" in result["message"].lower()
    finally:
        cleanup(key, token)


def test_unattended_safe_mode_blocks_execute_code_before_sandbox_fast_path():
    key = "safe-session-sandbox-code"
    token = with_session(key)
    try:
        approval.enable_session_unattended_safe_mode(key)
        result = approval.check_execute_code_guard("print('hello')", "vercel_sandbox")
        assert result["approved"] is False
        assert result["outcome"] == "blocked"
        assert "unattended" in result["message"].lower()
    finally:
        cleanup(key, token)


def test_clear_session_revokes_unattended_safe_mode():
    key = "safe-session-clear"
    approval.enable_session_unattended_safe_mode(key)
    assert approval.is_session_unattended_safe_mode(key) is True
    approval.clear_session(key)
    assert approval.is_session_unattended_safe_mode(key) is False


def test_compression_continuation_transfers_unattended_safe_mode(monkeypatch):
    from tui_gateway import session_compression

    old_key = "safe-session-old"
    new_key = "safe-session-new"
    approval.enable_session_unattended_safe_mode(old_key)
    monkeypatch.setattr(session_compression, "_transfer_active_session_slot", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(session_compression, "_restart_slash_worker", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(session_compression, "_emit_approval_request", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(session_compression, "logger", SimpleNamespace(warning=lambda *_a, **_k: None), raising=False)
    session = {"agent": SimpleNamespace(session_id=new_key), "session_key": old_key}
    try:
        session_compression._sync_session_key_after_compress("runtime", session)
        assert session["session_key"] == new_key
        assert approval.is_session_unattended_safe_mode(old_key) is False
        assert approval.is_session_unattended_safe_mode(new_key) is True
    finally:
        approval.clear_session(old_key)
        approval.clear_session(new_key)


def test_failed_compression_lease_transfer_drops_authority_instead_of_moving_it(monkeypatch):
    from tui_gateway import session_compression

    old_key = "safe-session-failed-old"
    new_key = "safe-session-failed-new"
    approval.enable_session_unattended_safe_mode(old_key)
    monkeypatch.setattr(session_compression, "_transfer_active_session_slot", lambda *_a, **_k: False, raising=False)
    monkeypatch.setattr(session_compression, "_restart_slash_worker", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(session_compression, "_emit_approval_request", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(session_compression, "logger", SimpleNamespace(warning=lambda *_a, **_k: None), raising=False)
    session = {"agent": SimpleNamespace(session_id=new_key), "session_key": old_key}
    try:
        session_compression._sync_session_key_after_compress("runtime", session)
        assert approval.is_session_unattended_safe_mode(old_key) is False
        assert approval.is_session_unattended_safe_mode(new_key) is False
    finally:
        approval.clear_session(old_key)
        approval.clear_session(new_key)


def test_unattended_safe_mode_blocks_protected_instruction_prompt():
    from tools import file_tools_write_guards as guards

    key = "safe-session-protected-write"
    token = with_session(key)
    try:
        approval.enable_session_unattended_safe_mode(key)
        result = guards._request_protected_instruction_approval(["AGENTS.md"])
        assert result is not None
        assert "unattended-safe" in result
    finally:
        cleanup(key, token)


def test_unattended_safe_mode_blocks_memory_write_instead_of_staging(monkeypatch):
    from tools import write_approval

    key = "safe-session-memory-write"
    token = with_session(key)
    try:
        approval.enable_session_unattended_safe_mode(key)
        monkeypatch.setattr(write_approval, "write_approval_enabled", lambda _subsystem: True)
        result = write_approval.evaluate_gate("memory")
        assert result.blocked is True
        assert result.allow is False
        assert result.stage is False
        assert "unattended-safe" in result.message
    finally:
        cleanup(key, token)


def test_action_os_unattended_activation_claim_is_durable_and_consume_once(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-claim", source="desktop")
    fingerprint = "launch_test:" + "a" * 64

    assert db.claim_action_os_unattended_activation("session-claim", fingerprint) is True
    assert db.claim_action_os_unattended_activation("session-claim", fingerprint) is False
    assert db.get_session_model_config_value(
        "session-claim", "action_os_unattended_activation_consumed"
    ) == fingerprint
