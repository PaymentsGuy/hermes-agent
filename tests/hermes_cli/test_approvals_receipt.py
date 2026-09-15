"""CLI contract for atomic approval-receipt verification and consumption."""

from __future__ import annotations

import argparse
import json
import time

import pytest

from hermes_cli.approvals_suggest import approvals_command
from hermes_cli.subcommands.approvals import build_approvals_parser
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from tools.tool_approval_receipts import create_intrinsic_approval_receipt

_SCOPE = "a" * 64
_RESULT_FIELDS = {
    "receipt_id",
    "decision",
    "decided_at",
    "expires_at",
    "consumed_by",
    "consumed_at",
}


def _parser():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_approvals_parser(subparsers, cmd_approvals=approvals_command)
    return parser


def _argv(receipt_id: str, **overrides):
    values = {
        "id": receipt_id,
        "session": "session-1",
        "tool": "apply_action_os_learning",
        "scope_sha256": _SCOPE,
        "consume_id": "action-os-command-1",
    }
    values.update(overrides)
    return [
        "approvals",
        "verify-receipt",
        "--id",
        values["id"],
        "--session",
        values["session"],
        "--tool",
        values["tool"],
        "--scope-sha256",
        values["scope_sha256"],
        "--consume-id",
        values["consume_id"],
        "--json",
    ]


def _create(db: SessionDB, *, decision="approved", clock=time.time):
    return create_intrinsic_approval_receipt(
        session_db=db,
        session_id="session-1",
        turn_id="turn-1",
        tool_call_id="call-1",
        tool_name="apply_action_os_learning",
        original_args={"finding": "bounded"},
        preview={"summary": "Apply one finding"},
        approval_scope_sha256=_SCOPE,
        decision=decision,
        clock=clock,
    )


@pytest.fixture
def active_db(tmp_path):
    home = tmp_path / "profiles" / "named"
    token = set_hermes_home_override(home)
    db = SessionDB(db_path=home / "state.db")
    try:
        yield home, db
    finally:
        db.close()
        reset_hermes_home_override(token)


def test_parser_exposes_verify_receipt_help_and_exact_arguments(capsys):
    parser = _parser()
    args = parser.parse_args(_argv("receipt-1"))

    assert args.approvals_command == "verify-receipt"
    assert args.receipt_id == "receipt-1"
    assert args.session_id == "session-1"
    assert args.tool_name == "apply_action_os_learning"
    assert args.approval_scope_sha256 == _SCOPE
    assert args.consume_id == "action-os-command-1"
    assert args.json is True

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["approvals", "verify-receipt", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--id", "--session", "--tool", "--scope-sha256", "--consume-id", "--json"):
        assert flag in help_text


@pytest.mark.parametrize(
    "flag",
    ["--id", "--session", "--tool", "--scope-sha256", "--consume-id"],
)
def test_parser_requires_each_receipt_binding(flag):
    argv = _argv("receipt-1")
    index = argv.index(flag)
    del argv[index : index + 2]

    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(argv)
    assert exc.value.code == 2


def test_success_json_has_exact_allowlisted_fields_and_exit_zero(active_db, capsys):
    _home, db = active_db
    receipt = _create(db)
    args = _parser().parse_args(_argv(receipt["receipt_id"]))

    assert approvals_command(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert set(output) == _RESULT_FIELDS
    assert output["receipt_id"] == receipt["receipt_id"]
    assert output["decision"] == "approved"
    assert output["consumed_by"] == "action-os-command-1"


@pytest.mark.parametrize("kind", ["missing", "denied", "expired", "session", "tool", "scope"])
def test_failures_are_nonzero_and_bounded_json(active_db, capsys, kind):
    _home, db = active_db
    receipt = _create(
        db,
        decision="denied" if kind == "denied" else "approved",
        clock=(lambda: 0.0) if kind == "expired" else time.time,
    )
    receipt_id = "missing-receipt" if kind == "missing" else receipt["receipt_id"]
    overrides = {
        "session": "SECRET_SESSION_CANARY" if kind == "session" else "session-1",
        "tool": "SECRET_TOOL_CANARY" if kind == "tool" else "apply_action_os_learning",
        "scope_sha256": "b" * 64 if kind == "scope" else _SCOPE,
    }
    args = _parser().parse_args(_argv(receipt_id, **overrides))

    assert approvals_command(args) != 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {"error": "approval receipt verification failed"}
    assert "CANARY" not in captured.out
    row = db._conn.execute(
        "SELECT consumed_by, consumed_at FROM intrinsic_tool_approval_receipts WHERE receipt_id = ?",
        (receipt["receipt_id"],),
    ).fetchone()
    assert tuple(row) == (None, None)


def test_malformed_identifier_is_nonzero_without_echoing_value(active_db, capsys):
    _home, db = active_db
    receipt = _create(db)
    args = _parser().parse_args(_argv(receipt["receipt_id"], session="CANARY\nSESSION"))

    assert approvals_command(args) != 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"error": "approval receipt verification failed"}
    assert "CANARY" not in captured.out


def test_command_uses_only_the_active_named_profile(tmp_path, capsys):
    homes = [tmp_path / "profiles" / name for name in ("first", "second")]
    token = set_hermes_home_override(homes[0])
    first_db = SessionDB(db_path=homes[0] / "state.db")
    try:
        receipt = _create(first_db)
    finally:
        first_db.close()
        reset_hermes_home_override(token)

    second_token = set_hermes_home_override(homes[1])
    second_db = SessionDB(db_path=homes[1] / "state.db")
    try:
        args = _parser().parse_args(_argv(receipt["receipt_id"]))
        assert approvals_command(args) != 0
        assert json.loads(capsys.readouterr().out) == {
            "error": "approval receipt verification failed"
        }
    finally:
        second_db.close()
        reset_hermes_home_override(second_token)

    first_token = set_hermes_home_override(homes[0])
    try:
        args = _parser().parse_args(_argv(receipt["receipt_id"]))
        assert approvals_command(args) == 0
        assert json.loads(capsys.readouterr().out)["receipt_id"] == receipt["receipt_id"]
    finally:
        reset_hermes_home_override(first_token)
