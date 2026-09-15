"""Profile-wide serialization for skill package mutations."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import BinaryIO, Dict, Iterator

from hermes_constants import get_hermes_home

try:  # Unix
    import fcntl
except ImportError:  # pragma: no cover - platform-specific
    fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - platform-specific
    msvcrt = None  # type: ignore[assignment]


_profile_locks: Dict[Path, threading.RLock] = {}
_profile_locks_guard = threading.Lock()
_thread_state = threading.local()


def _reset_after_fork_in_child() -> None:
    """Discard inherited thread locks and reentry state in a forked child."""
    global _profile_locks, _profile_locks_guard, _thread_state
    _profile_locks = {}
    _profile_locks_guard = threading.Lock()
    _thread_state = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork_in_child)


def _canonical_profile_home() -> Path:
    try:
        return Path(get_hermes_home()).expanduser().resolve()
    except Exception as exc:
        raise RuntimeError("skill mutation lock profile identity is unavailable") from exc


def _profile_lock(home: Path) -> threading.RLock:
    with _profile_locks_guard:
        return _profile_locks.setdefault(home, threading.RLock())


def skill_mutation_lock_held() -> bool:
    """Whether this thread holds the lock for the still-active canonical profile."""
    if getattr(_thread_state, "depth", 0) <= 0:
        return False
    try:
        return _canonical_profile_home() == _thread_state.home
    except (AttributeError, RuntimeError):
        return False


def _acquire_file_lock(fd: BinaryIO) -> None:
    if fcntl is not None:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        return
    if msvcrt is not None:
        fd.seek(0)
        getattr(msvcrt, "locking")(fd.fileno(), getattr(msvcrt, "LK_LOCK"), 1)
        return
    raise RuntimeError("no supported file-lock backend for skill mutations")


def _release_file_lock(fd: BinaryIO) -> None:
    if fcntl is not None:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:
        fd.seek(0)
        getattr(msvcrt, "locking")(fd.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)


@contextmanager
def skill_mutation_lock() -> Iterator[None]:
    """Lock all skill package/classification writes for the active profile."""
    home = _canonical_profile_home()
    depth = getattr(_thread_state, "depth", 0)
    if depth:
        if home != _thread_state.home:
            raise RuntimeError("active profile changed during nested skill mutation lock")
        _thread_state.depth = depth + 1
        try:
            yield
        finally:
            _thread_state.depth -= 1
        return

    if fcntl is None and msvcrt is None:
        raise RuntimeError("no supported file-lock backend for skill mutations")

    with _profile_lock(home):
        if _canonical_profile_home() != home:
            raise RuntimeError("active profile changed while acquiring skill mutation lock")
        lock_path = home / "skills" / ".mutation.lock"
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = open(lock_path, "a+b")
        except Exception as exc:
            raise RuntimeError(f"cannot open skill mutation lock for profile {home}") from exc
        try:
            if msvcrt is not None and fcntl is None and lock_path.stat().st_size == 0:
                fd.write(b" ")
                fd.flush()
            try:
                _acquire_file_lock(fd)
            except Exception as exc:
                raise RuntimeError(f"cannot acquire skill mutation lock for profile {home}") from exc
            _thread_state.home = home
            _thread_state.depth = 1
            try:
                yield
            finally:
                _thread_state.depth = 0
                del _thread_state.home
                with suppress(OSError, IOError):
                    _release_file_lock(fd)
        finally:
            fd.close()
