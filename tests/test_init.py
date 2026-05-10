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
        ".",
        "",
        None,
    ],
    ids=["single", "multi", "trailing-sep", "leading-pathsep", "dot", "empty", "unset"],
)
def test_init_strips_leaked_pythonpath(pythonpath):
    """Package init must clear PYTHONPATH (env) AND remove all of its
    sentinel-prefixed entries from sys.path. Asserts on the sentinel
    substring directly so the test does not couple to the production
    normalization logic. The dot/empty/unset cases additionally
    exercise the early-return / collision paths without crashing."""
    env = os.environ.copy()
    if pythonpath is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = pythonpath
    code = (
        "import mempalace, os, sys; "
        f"prefix = {_LEAK_PREFIX!r}; "
        "print('ENV:', repr(os.environ.get('PYTHONPATH'))); "
        "print('SENTINEL_IN_PATH:', any(prefix in (p or '') for p in sys.path))"
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
    assert "SENTINEL_IN_PATH: False" in out, f"sentinel-prefix leak: {diag}"


def test_init_preserves_cwd_marker_when_pythonpath_collides():
    """PYTHONPATH='.' normalizes to the same value as the empty-string
    CWD marker on sys.path. The strip must remove '.' from sys.path
    without collapsing the implicit current-directory entry."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    code = (
        "import mempalace, sys; "
        "print('CWD_IN_PATH:', '' in sys.path); "
        "print('DOT_IN_PATH:', '.' in sys.path)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    diag = f"rc={result.returncode}; stdout={result.stdout!r}; " f"stderr={result.stderr!r}"
    assert result.returncode == 0, f"subprocess failed: {diag}"
    assert "CWD_IN_PATH: True" in result.stdout, f"cwd marker dropped: {diag}"
    assert "DOT_IN_PATH: False" in result.stdout, f"dot leak survived: {diag}"
