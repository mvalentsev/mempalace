"""Regression tests for the bash 3.2 compatibility fix (#1440).

The legacy hooks/*.sh scripts run on the user's system. On stock macOS
that is GNU bash 3.2.57 (Apple GPLv3 freeze, 2006). Using bash 4.0-only
builtins like ``mapfile`` silently breaks parsing: every JSON field
falls back to its default, the hook logs ``Session unknown: 0 exchanges``,
and zero drawers are saved.

These tests cover:
1. Source-level shape (mapfile/readarray absent, sed-based extraction).
2. Behavioral parse contract (session_id reaches the log, not 'unknown').
3. The fail-loud guard: fires only when the parser sentinel is missing,
   never on a legitimately empty/unicode/literal-'unknown' session_id.
4. The guard's disk discipline (bounded dump, overwrite-not-append).
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAVE_HOOK = REPO_ROOT / "hooks" / "mempal_save_hook.sh"
PRECOMPACT_HOOK = REPO_ROOT / "hooks" / "mempal_precompact_hook.sh"

pytestmark = pytest.mark.skipif(os.name == "nt", reason="bash hook scripts are POSIX-only")


def _hook_src_no_comments(hook: Path) -> str:
    return "\n".join(
        line for line in hook.read_text().splitlines() if not line.lstrip().startswith("#")
    )


def _run_hook(hook: Path, stdin: str, home: Path) -> tuple[int, str]:
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }
    p = subprocess.run(
        ["bash", str(hook)],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    return p.returncode, p.stdout


class TestNoBash4OnlyBuiltins:
    """Source-level regression: mapfile/readarray unavailable on macOS bash 3.2."""

    @pytest.mark.parametrize("hook", [SAVE_HOOK, PRECOMPACT_HOOK])
    def test_no_mapfile(self, hook):
        code = _hook_src_no_comments(hook)
        assert "mapfile" not in code, (
            f"{hook.name} uses mapfile, unavailable on macOS /bin/bash 3.2 (#1440)"
        )
        assert "readarray" not in code, (
            f"{hook.name} uses readarray, unavailable on macOS /bin/bash 3.2 (#1440)"
        )

    @pytest.mark.parametrize("hook", [SAVE_HOOK, PRECOMPACT_HOOK])
    def test_sed_extraction_present(self, hook):
        src = hook.read_text()
        # Each hook reads at least two values via ``sed -n 'Np'`` (sentinel + session_id).
        assert src.count("sed -n '") >= 2, (
            f"{hook.name} must use sed -n 'Np' for POSIX-portable line extraction"
        )

    @pytest.mark.parametrize("hook", [SAVE_HOOK, PRECOMPACT_HOOK])
    def test_bash_syntax_clean(self, hook):
        p = subprocess.run(
            ["bash", "-n", str(hook)],
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, f"{hook.name} syntax error: {p.stderr}"


class TestSessionIdExtraction:
    """Hook must parse session_id from valid JSON, not fall back to 'unknown'."""

    def test_save_hook_extracts_session_id(self, tmp_path):
        rc, out = _run_hook(
            SAVE_HOOK,
            json.dumps(
                {"session_id": "abc12345", "stop_hook_active": False, "transcript_path": ""}
            ),
            tmp_path,
        )
        assert rc == 0
        # Stdout must be valid JSON; Claude Code parses it. A regression that
        # leaks debug output here would silently break the harness contract.
        assert json.loads(out) == {}, f"hook stdout must be valid JSON, got: {out!r}"
        log = (tmp_path / ".mempalace" / "hook_state" / "hook.log").read_text()
        assert "Session abc12345:" in log, f"got fallback 'unknown'; log was: {log!r}"
        # Negative cross-check: the sentinel-distinguishes-success-from-failure
        # contract has no value if the guard fires on the happy path too.
        assert "WARN: input parse failed" not in log
        state_dir = tmp_path / ".mempalace" / "hook_state"
        assert not (state_dir / "last_input.log").exists()
        assert not (state_dir / "last_python_err.log").exists(), (
            "successful parse must leave no last_python_err.log behind"
        )

    def test_precompact_hook_extracts_session_id(self, tmp_path):
        rc, out = _run_hook(
            PRECOMPACT_HOOK,
            json.dumps({"session_id": "abc12345", "transcript_path": ""}),
            tmp_path,
        )
        assert rc == 0
        assert json.loads(out) == {}, f"hook stdout must be valid JSON, got: {out!r}"
        log = (tmp_path / ".mempalace" / "hook_state" / "hook.log").read_text()
        assert "PRE-COMPACT triggered for session abc12345" in log
        assert "WARN: input parse failed" not in log
        state_dir = tmp_path / ".mempalace" / "hook_state"
        assert not (state_dir / "last_input.log").exists()
        assert not (state_dir / "last_python_err.log").exists()


class TestFailLoudGuard:
    """Non-parseable stdin must dump the input and warn in hook.log so
    future silent failures are loud, and the dump must stay bounded and
    user-private. The guard must NOT fire on legitimately empty inputs
    or on sanitizer-stripped session_ids (#1440)."""

    @pytest.mark.parametrize("hook", [SAVE_HOOK, PRECOMPACT_HOOK])
    def test_malformed_input_logs_warning_and_dumps_input(self, hook, tmp_path):
        rc, _ = _run_hook(hook, "not-json garbage", tmp_path)
        assert rc == 0
        state_dir = tmp_path / ".mempalace" / "hook_state"
        log = (state_dir / "hook.log").read_text()
        last_input = (state_dir / "last_input.log").read_text()
        assert "WARN: input parse failed (sentinel missing)" in log
        assert "not-json garbage" in last_input

    @pytest.mark.parametrize("hook", [SAVE_HOOK, PRECOMPACT_HOOK])
    def test_empty_stdin_does_not_dump_or_warn(self, hook, tmp_path):
        """Empty stdin is a legitimate state (e.g. a hook re-fire on Stop
        with no message body). The guard's ``[ -n "$INPUT" ]`` short-circuit
        must hold so nothing is written to last_input.log."""
        rc, _ = _run_hook(hook, "", tmp_path)
        assert rc == 0
        state_dir = tmp_path / ".mempalace" / "hook_state"
        assert not (state_dir / "last_input.log").exists()
        log_path = state_dir / "hook.log"
        if log_path.exists():
            assert "WARN: input parse failed" not in log_path.read_text()

    def test_unicode_session_id_does_not_trip_guard(self, tmp_path):
        """A session_id with non-ASCII characters (Cyrillic, CJK, emoji)
        is stripped by the sanitizer to '', defaults to 'unknown'. The
        sentinel still printed, so the guard must skip and NOT spam disk."""
        rc, _ = _run_hook(
            SAVE_HOOK,
            json.dumps({"session_id": "сессия", "stop_hook_active": False, "transcript_path": ""}),
            tmp_path,
        )
        assert rc == 0
        state_dir = tmp_path / ".mempalace" / "hook_state"
        assert not (state_dir / "last_input.log").exists(), (
            "unicode-only session_id sanitized to empty must NOT trip the guard"
        )

    def test_literal_unknown_session_id_does_not_trip_guard(self, tmp_path):
        """A user who literally passes session_id='unknown' is parsing
        cleanly; the sentinel-based guard must distinguish that from a
        crash and skip the dump."""
        rc, _ = _run_hook(
            SAVE_HOOK,
            json.dumps({"session_id": "unknown", "stop_hook_active": False, "transcript_path": ""}),
            tmp_path,
        )
        assert rc == 0
        state_dir = tmp_path / ".mempalace" / "hook_state"
        assert not (state_dir / "last_input.log").exists(), (
            "literal session_id='unknown' must NOT trip the guard"
        )

    def test_dump_is_bounded_and_overwritten(self, tmp_path):
        """The dump caps at exactly 4096 bytes and overwrites on each
        failure so a repeating misconfiguration cannot grow the file
        unbounded."""
        # 4097 bytes: one over the cap, proves the cutoff fires at exactly
        # 4096 (a regression that silently shrinks the cap to e.g. 1024
        # would slip past a looser ``<= 4096`` check).
        big_payload = "x" * 4097
        rc, _ = _run_hook(SAVE_HOOK, big_payload, tmp_path)
        assert rc == 0
        last_input = tmp_path / ".mempalace" / "hook_state" / "last_input.log"
        assert last_input.stat().st_size == 4096, (
            f"cap must be exactly 4096 bytes; got {last_input.stat().st_size}"
        )
        # Second failure with a smaller payload overwrites the first; the
        # file shrinks instead of accumulating.
        rc, _ = _run_hook(SAVE_HOOK, "tiny", tmp_path)
        assert rc == 0
        assert last_input.read_text() == "tiny", "dump must overwrite on each failure, not append"

    def test_dump_cap_holds_under_utf8_locale(self, tmp_path):
        """Under a UTF-8 locale, a multibyte payload of 2000 CJK chars =
        6000 bytes would slip a character-counted substring (`${var:0:N}`)
        past the 4096-byte cap. The hook uses ``head -c`` precisely so
        the bound stays byte-based regardless of locale."""
        # ``C.UTF-8`` is available on every mainstream Linux distribution
        # (Debian/Ubuntu, Fedora, RHEL 8+, Alpine via musl) and macOS
        # bash falls back to the byte-based C locale gracefully;
        # ``en_US.UTF-8`` would silently degrade to no-op on minimal CI
        # images (Alpine, distroless) where that locale is not generated.
        env = {
            "HOME": str(tmp_path),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        # 2000 copies of U+4E2D (3 bytes each in UTF-8) = 6000 bytes.
        big_payload = "中" * 2000
        p = subprocess.run(
            ["bash", str(SAVE_HOOK)],
            input=big_payload,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert p.returncode == 0
        last_input = tmp_path / ".mempalace" / "hook_state" / "last_input.log"
        size = last_input.stat().st_size
        assert size == 4096, (
            f"UTF-8 payload must still cap at 4096 bytes (got {size}); "
            "regression to ${var:0:N} would let multibyte input bypass the bound"
        )

    def test_dump_is_not_world_readable(self, tmp_path):
        """The dump mirrors the raw Stop payload (transcript_path reveals
        the user's home + project layout). Permissions must be 600 so
        other users on a shared box cannot read it."""
        rc, _ = _run_hook(SAVE_HOOK, "not-json garbage", tmp_path)
        assert rc == 0
        last_input = tmp_path / ".mempalace" / "hook_state" / "last_input.log"
        mode = stat.S_IMODE(last_input.stat().st_mode)
        assert mode == 0o600, f"last_input.log mode should be 0600, got {oct(mode)}"

    @pytest.mark.parametrize("hook", [SAVE_HOOK, PRECOMPACT_HOOK])
    def test_python_stderr_captured_on_parse_failure(self, hook, tmp_path):
        """When the inline Python parser crashes (malformed JSON, missing
        interpreter, future regression), its stderr must land in
        last_python_err.log so a debugger can distinguish 'bad user
        input' from 'broken interpreter or broken inline script'."""
        rc, _ = _run_hook(hook, "not-json garbage", tmp_path)
        assert rc == 0
        err_log = tmp_path / ".mempalace" / "hook_state" / "last_python_err.log"
        assert err_log.exists(), "Python stderr must be captured on parse failure"
        contents = err_log.read_text()
        # Python's json.load raises JSONDecodeError with a recognizable
        # traceback. Don't pin the exact message (it varies by Python
        # version) but assert at least one canonical marker is present.
        assert "Traceback" in contents or "json" in contents.lower(), (
            f"expected Python traceback or json error, got: {contents!r}"
        )

    @pytest.mark.parametrize("hook", [SAVE_HOOK, PRECOMPACT_HOOK])
    def test_python_stderr_log_is_not_world_readable_on_failure(self, hook, tmp_path):
        """The stderr capture mirrors the privacy expectation of
        last_input.log: on a populated failure write it must be 0600."""
        rc, _ = _run_hook(hook, "not-json garbage", tmp_path)
        assert rc == 0
        err_log = tmp_path / ".mempalace" / "hook_state" / "last_python_err.log"
        mode = stat.S_IMODE(err_log.stat().st_mode)
        assert mode == 0o600, f"last_python_err.log mode should be 0600 on failure, got {oct(mode)}"
