import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.model_tool_policy import MODEL_TOOL_POLICY_VERSION
from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.compute_host import ComputeHost


POLICY_SHA = "f" * 64


def _policy(*allowed):
    return {
        "policy_id": "compute-policy", "policy_sha256": POLICY_SHA,
        "allowed_tools": list(allowed), "approval_required_tools": [],
    }


def _stdout_queue(proc: subprocess.Popen) -> queue.Queue[dict]:
    out: queue.Queue[dict] = queue.Queue()
    assert proc.stdout is not None

    def drain() -> None:
        for line in proc.stdout or []:
            out.put(json.loads(line))

    threading.Thread(target=drain, daemon=True).start()
    return out


def _read_json_line(out: queue.Queue[dict], timeout: float = 2.0) -> dict:
    try:
        return out.get(timeout=timeout)
    except queue.Empty as exc:
        raise AssertionError("timed out waiting for compute host JSON") from exc


def test_compute_host_line_json_hello_and_shutdown():
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tui_gateway.compute_host"],
        cwd=str(repo),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    out = _stdout_queue(proc)
    try:
        hello = _read_json_line(out)
        assert hello["type"] == "hello"
        assert hello["host_pid"] == proc.pid

        proc.stdin.write(json.dumps({"type": "bogus", "request_id": "b"}) + "\n")
        proc.stdin.flush()
        error = _read_json_line(out)
        assert error["type"] == "error"
        assert error["message"] == "unknown frame type: bogus"

        proc.stdin.write(json.dumps({"type": "shutdown", "request_id": "stop"}) + "\n")
        proc.stdin.flush()
        assert _read_json_line(out)["type"] == "shutdown.ack"
        proc.wait(timeout=2)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_compute_host_frame_and_host_agent_keep_full_policy_and_marker(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    policy = _policy("read_file")
    db.create_session(
        "compute-key", "desktop", model_tool_policy=policy,
        model_tool_policy_version=MODEL_TOOL_POLICY_VERSION,
    )
    parent = {
        "agent": SimpleNamespace(_session_db=db), "session_key": "compute-key",
        "source": "desktop", "model_tool_policy": policy,
        "model_tool_policy_version": MODEL_TOOL_POLICY_VERSION,
        "history": [], "history_lock": threading.Lock(), "history_version": 0,
        "cwd": str(tmp_path), "cols": 80,
    }
    frame = server._compute_host_turn_frame("rid", "compute-sid", parent, "hello")
    assert frame["model_tool_policy"] == policy
    assert frame["model_tool_policy_version"] == MODEL_TOOL_POLICY_VERSION

    captured = {}
    fake_agent = SimpleNamespace(_session_db=None, _owns_session_db=False)
    monkeypatch.setattr(server, "_create_model_tool_policy", lambda params, **_kwargs: params["model_tool_policy"])
    monkeypatch.setattr(server, "_make_agent", lambda *_args, **kwargs: captured.update(kwargs) or fake_agent)

    def init_session(sid, key, agent, history, **kwargs):
        captured["init"] = kwargs
        server._sessions[sid] = {
            "agent": agent, "session_key": key, "history": history,
            "history_lock": threading.Lock(), "source": "desktop",
        }

    monkeypatch.setattr(server, "_init_session", init_session)
    sink = open(os.devnull, "w")
    host = ComputeHost(stdout=sink, heartbeat_secs=0)
    try:
        child = host._build_server_session(server, frame, "compute-sid")
    finally:
        host.close()
        sink.close()
        server._sessions.pop("compute-sid", None)
        db.close()

    assert captured["model_tool_policy"] == policy
    assert captured["init"]["model_tool_policy"] == policy
    assert captured["init"]["model_tool_policy_version"] == MODEL_TOOL_POLICY_VERSION
    assert child["model_tool_policy"] == policy
    assert child["model_tool_policy_version"] == MODEL_TOOL_POLICY_VERSION


def test_compute_host_rejects_policy_payload_without_marker_before_agent_build(monkeypatch):
    built = []
    monkeypatch.setattr(server, "_make_agent", lambda *_a, **_k: built.append(True))
    frame = {
        "type": "turn.start", "sid": "corrupt-compute", "session_key": "key",
        "source": "desktop", "model_tool_policy": _policy("read_file"),
        "history": [], "cwd": "", "cols": 80,
    }
    sink = open(os.devnull, "w")
    host = ComputeHost(stdout=sink, heartbeat_secs=0)
    try:
        with pytest.raises(ValueError, match="marker or payload"):
            host._build_server_session(server, frame, "corrupt-compute")
    finally:
        host.close()
        sink.close()
        server._sessions.pop("corrupt-compute", None)

    assert built == []
