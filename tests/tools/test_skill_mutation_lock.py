"""Profile-scoped serialization for skill package/classification mutations."""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest


SKILL = """---
name: {name}
description: Use when testing mutation locking. Serialize writes.
---

# Test

Keep this package consistent.
"""


def _hold_profile_lock(home: str, entered, release, attempted=None) -> None:
    os.environ["HERMES_HOME"] = home
    from tools.skill_mutation_lock import skill_mutation_lock

    if attempted is not None:
        attempted.set()
    with skill_mutation_lock():
        entered.set()
        release.wait(10)


def _acquire_in_forked_child(home: str, attempted, entered) -> None:
    os.environ["HERMES_HOME"] = home
    from tools.skill_mutation_lock import skill_mutation_lock

    attempted.set()
    with skill_mutation_lock():
        entered.set()


def _mutate_while_instrumented(home: str, name: str, entered, release, attempted=None) -> None:
    os.environ["HERMES_HOME"] = home
    os.environ["HERMES_YOLO_MODE"] = "1"
    from tools import skill_manager_tool as smt

    original = smt._skill_manage_unlocked

    def instrumented(*args, **kwargs):
        entered.set()
        release.wait(10)
        return original(*args, **kwargs)

    smt._skill_manage_unlocked = instrumented
    if attempted is not None:
        attempted.set()
    result = json.loads(
        smt.skill_manage(action="create", name=name, content=SKILL.format(name=name))
    )
    if not result.get("success"):
        raise RuntimeError(result)


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    return home


def test_same_thread_reentry_uses_one_process_lock_and_rejects_profile_switch(
    profile_home, monkeypatch
):
    from tools import skill_mutation_lock as lock_mod

    calls = 0
    real_acquire = lock_mod._acquire_file_lock

    def counted(fd):
        nonlocal calls
        calls += 1
        return real_acquire(fd)

    monkeypatch.setattr(lock_mod, "_acquire_file_lock", counted)
    with lock_mod.skill_mutation_lock():
        with lock_mod.skill_mutation_lock():
            assert calls == 1
        other = profile_home.parent / "other"
        monkeypatch.setenv("HERMES_HOME", str(other))
        with pytest.raises(RuntimeError, match="profile changed"):
            with lock_mod.skill_mutation_lock():
                pass
    assert calls == 1


def test_lock_releases_after_success_and_body_exception(profile_home):
    from tools.skill_mutation_lock import skill_mutation_lock

    with skill_mutation_lock():
        pass
    with pytest.raises(ValueError, match="body failed"):
        with skill_mutation_lock():
            raise ValueError("body failed")

    entered = threading.Event()

    def acquire_again():
        with skill_mutation_lock():
            entered.set()

    thread = threading.Thread(target=acquire_again)
    thread.start()
    thread.join(timeout=3)
    assert entered.is_set()
    assert not thread.is_alive()


def test_two_threads_serialize(profile_home):
    from tools.skill_mutation_lock import skill_mutation_lock

    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()

    def first():
        with skill_mutation_lock():
            first_entered.set()
            release_first.wait(5)

    def second():
        first_entered.wait(5)
        second_attempted.set()
        with skill_mutation_lock():
            second_entered.set()

    one = threading.Thread(target=first)
    two = threading.Thread(target=second)
    one.start()
    assert first_entered.wait(3)
    two.start()
    assert second_attempted.wait(3)
    assert not second_entered.is_set()
    release_first.set()
    one.join(timeout=3)
    two.join(timeout=3)
    assert second_entered.is_set()


def test_two_processes_in_same_profile_serialize(profile_home):
    ctx = mp.get_context("spawn")
    first_entered, release_first = ctx.Event(), ctx.Event()
    second_entered, release_second = ctx.Event(), ctx.Event()
    second_attempted = ctx.Event()
    one = ctx.Process(
        target=_mutate_while_instrumented,
        args=(str(profile_home), "process-one", first_entered, release_first),
    )
    two = ctx.Process(
        target=_mutate_while_instrumented,
        args=(str(profile_home), "process-two", second_entered, release_second, second_attempted),
    )
    one.start()
    assert first_entered.wait(5)
    two.start()
    assert second_attempted.wait(5)
    assert not second_entered.is_set()
    release_first.set()
    assert second_entered.wait(5)
    release_second.set()
    one.join(timeout=5)
    two.join(timeout=5)
    assert one.exitcode == two.exitcode == 0
    assert (profile_home / "skills" / "process-one" / "SKILL.md").exists()
    assert (profile_home / "skills" / "process-two" / "SKILL.md").exists()


def test_forked_child_does_not_inherit_parent_reentry_state(profile_home):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("multiprocessing fork start method is unavailable")

    from tools.skill_mutation_lock import skill_mutation_lock

    ctx = mp.get_context("fork")
    attempted, entered = ctx.Event(), ctx.Event()
    child = ctx.Process(
        target=_acquire_in_forked_child,
        args=(str(profile_home), attempted, entered),
    )

    with skill_mutation_lock():
        child.start()
        assert attempted.wait(5)
        assert not entered.wait(0.5), "forked child bypassed the parent's file lock"

    assert entered.wait(5)
    child.join(timeout=5)
    assert child.exitcode == 0


def test_different_profiles_do_not_share_process_lock(profile_home):
    ctx = mp.get_context("spawn")
    other = profile_home.parent / "other-profile"
    (other / "skills").mkdir(parents=True)
    first_entered, release_first = ctx.Event(), ctx.Event()
    second_entered, release_second = ctx.Event(), ctx.Event()
    one = ctx.Process(target=_hold_profile_lock, args=(str(profile_home), first_entered, release_first))
    two = ctx.Process(target=_hold_profile_lock, args=(str(other), second_entered, release_second))
    one.start()
    assert first_entered.wait(5)
    two.start()
    assert second_entered.wait(5)
    release_second.set()
    release_first.set()
    one.join(timeout=5)
    two.join(timeout=5)
    assert one.exitcode == two.exitcode == 0


def test_no_lock_backend_fails_before_skill_manage_dispatch(profile_home, monkeypatch):
    from tools import skill_manager_tool as smt
    from tools import skill_mutation_lock as lock_mod

    called = False

    def writer(*args, **kwargs):
        nonlocal called
        called = True
        return "unreachable"

    monkeypatch.setattr(lock_mod, "fcntl", None)
    monkeypatch.setattr(lock_mod, "msvcrt", None)
    monkeypatch.setattr(smt, "_skill_manage_unlocked", writer)
    with pytest.raises(RuntimeError, match="no supported file-lock backend"):
        smt.skill_manage(action="create", name="blocked", content=SKILL.format(name="blocked"))
    assert not called


def test_windows_backend_locks_one_initialized_byte(profile_home, monkeypatch):
    from tools import skill_mutation_lock as lock_mod

    class FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        def __init__(self):
            self.calls = []

        def locking(self, fileno, mode, count):
            self.calls.append((fileno, mode, count))

    backend = FakeMsvcrt()
    monkeypatch.setattr(lock_mod, "fcntl", None)
    monkeypatch.setattr(lock_mod, "msvcrt", backend)

    with lock_mod.skill_mutation_lock():
        pass

    assert [mode for _, mode, _ in backend.calls] == [backend.LK_LOCK, backend.LK_UNLCK]
    assert all(count == 1 for _, _, count in backend.calls)
    assert (profile_home / "skills" / ".mutation.lock").read_bytes() == b" "


def test_flat_actions_and_batch_enter_live_skill_manage_lock(profile_home, monkeypatch):
    from tools import skill_manager_tool as smt

    entered = []

    @contextmanager
    def observed_lock():
        entered.append(True)
        yield

    monkeypatch.setattr(smt, "skill_mutation_lock", observed_lock)
    for action in ("create", "edit", "patch", "delete", "write_file", "remove_file"):
        smt.skill_manage(action=action, name="missing")
    smt.skill_manage(action="", name="", operations=[])
    assert len(entered) == 7


def test_batch_nested_calls_take_process_file_lock_once(profile_home, monkeypatch):
    from tools import skill_manager_tool as smt
    from tools import skill_mutation_lock as lock_mod

    calls = 0
    real_acquire = lock_mod._acquire_file_lock

    def counted(fd):
        nonlocal calls
        calls += 1
        return real_acquire(fd)

    monkeypatch.setattr(lock_mod, "_acquire_file_lock", counted)
    result = json.loads(
        smt.skill_manage(
            action="",
            name="",
            operations=[
                {"name": "batch-lock", "action": "create", "content": SKILL.format(name="batch-lock")},
                {
                    "name": "batch-lock",
                    "action": "write_file",
                    "file_path": "references/note.md",
                    "file_content": "locked",
                },
            ],
        )
    )
    assert result["success"] is True, result
    assert calls == 1


def test_curator_classification_transitions_enter_shared_lock(profile_home, monkeypatch):
    from tools import skill_usage

    importlib.reload(skill_usage)
    skills = profile_home / "skills"
    for name in ("managed", "adopted"):
        directory = skills / name
        directory.mkdir()
        (directory / "SKILL.md").write_text(SKILL.format(name=name), encoding="utf-8")
    skill_usage.record_created("managed", agent_created=True)

    entries = []

    @contextmanager
    def observed_lock():
        entries.append(True)
        yield

    monkeypatch.setattr(skill_usage, "skill_mutation_lock", observed_lock)
    transitions = [
        lambda: skill_usage.adopt_skill("adopted"),
        lambda: skill_usage.mark_agent_created("managed"),
        lambda: skill_usage.set_pinned("managed", True),
        lambda: skill_usage.set_pinned("managed", False),
        lambda: skill_usage.set_state("managed", skill_usage.STATE_STALE),
        lambda: skill_usage.archive_skill("managed"),
        lambda: skill_usage.restore_skill("managed"),
        lambda: skill_usage.record_created("managed", agent_created=True),
        lambda: skill_usage.record_installed("installed"),
    ]
    for transition in transitions:
        before = len(entries)
        transition()
        assert len(entries) > before


def test_owner_read_can_hold_lock_against_concurrent_pin_transition(profile_home):
    from tools import skill_usage
    from tools.skill_mutation_lock import skill_mutation_lock

    importlib.reload(skill_usage)
    directory = profile_home / "skills" / "guarded"
    directory.mkdir()
    (directory / "SKILL.md").write_text(SKILL.format(name="guarded"), encoding="utf-8")
    skill_usage.record_created("guarded", agent_created=True)

    attempted = threading.Event()
    completed = threading.Event()

    def pin():
        attempted.set()
        skill_usage.set_pinned("guarded", True)
        completed.set()

    with skill_mutation_lock():
        assert skill_usage.get_record("guarded")["created_by"] == "agent"
        thread = threading.Thread(target=pin)
        thread.start()
        assert attempted.wait(3)
        assert not completed.is_set()
    thread.join(timeout=3)
    assert completed.is_set()
    assert skill_usage.get_record("guarded")["pinned"] is True


def test_telemetry_bump_does_not_take_mutation_lock(profile_home, monkeypatch):
    from tools import skill_usage

    importlib.reload(skill_usage)

    @contextmanager
    def forbidden_lock():
        raise AssertionError("telemetry must not acquire the mutation lock")
        yield

    monkeypatch.setattr(skill_usage, "skill_mutation_lock", forbidden_lock)
    skill_usage.bump_view("telemetry-only")
    assert skill_usage.get_record("telemetry-only")["view_count"] == 1
