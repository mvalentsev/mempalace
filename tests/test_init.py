"""__init__-level guards that must take effect before transitive imports."""

import os
import subprocess
import sys

import pytest


_LEAK_PREFIX = "/__mempalace_leak_test_sentinel__"


@pytest.mark.parametrize(
    "pythonpath",
    [
        f"{_LEAK_PREFIX}/single",
        f"{_LEAK_PREFIX}/a{os.pathsep}{_LEAK_PREFIX}/b",
        f"{_LEAK_PREFIX}/with-trailing{os.sep}",
        f"{os.pathsep}{_LEAK_PREFIX}/leading-sep",
        "",
        None,
    ],
    ids=["single", "multi", "trailing-sep", "leading-pathsep", "empty", "unset"],
)
def test_init_strips_leaked_pythonpath(pythonpath):
    """Package init must clear PYTHONPATH (env) AND remove all of its
    leaked entries from sys.path, normalizing case and trailing
    separators so Windows and POSIX behave alike."""
    env = os.environ.copy()
    if pythonpath is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = pythonpath
    code = (
        "import mempalace, os, sys; "
        f"leaked = {pythonpath!r}; "
        "print('ENV:', repr(os.environ.get('PYTHONPATH'))); "
        "tokens = leaked.split(os.pathsep) if leaked else []; "
        "entries = [t.strip() for t in tokens if t.strip()]; "
        "norm = lambda p: os.path.normcase(os.path.normpath(p)); "
        "leaked_norm = {norm(e) for e in entries}; "
        "leaked_in_path = any(norm(p) in leaked_norm for p in sys.path); "
        "print('SYSPATH_LEAK:', leaked_in_path)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    diag = (
        f"input={pythonpath!r}; rc={result.returncode}; "
        f"stdout={result.stdout!r}; stderr={result.stderr!r}"
    )
    assert result.returncode == 0, f"subprocess failed: {diag}"
    out = result.stdout
    assert "ENV: None" in out, f"PYTHONPATH not cleared: {diag}"
    assert "SYSPATH_LEAK: False" in out, f"sys.path retains leaked entry: {diag}"
