"""Tests for mempalace.llm_client.

HTTP and subprocess are mocked throughout — these tests do not require a
running Ollama, network access, or the ``claude`` CLI binary. Live-provider
smoke tests live outside the unit-test suite (see
``test_claude_code_real_invocation`` for the gated integration probe).
"""

import json
import os
import subprocess
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from mempalace.llm_client import (
    AnthropicProvider,
    ClaudeCodeProvider,
    LLMError,
    OllamaProvider,
    OpenAICompatProvider,
    _http_post_json,
    get_provider,
)


# ── factory ─────────────────────────────────────────────────────────────


def test_get_provider_ollama():
    p = get_provider("ollama", "gemma4:e4b")
    assert isinstance(p, OllamaProvider)
    assert p.model == "gemma4:e4b"
    assert p.endpoint == OllamaProvider.DEFAULT_ENDPOINT


def test_get_provider_openai_compat():
    p = get_provider("openai-compat", "foo", endpoint="http://localhost:1234")
    assert isinstance(p, OpenAICompatProvider)


def test_get_provider_anthropic():
    p = get_provider("anthropic", "claude-haiku", api_key="sk-xxx")
    assert isinstance(p, AnthropicProvider)
    assert p.api_key == "sk-xxx"


def test_get_provider_claude_code():
    p = get_provider("claude-code", "claude-haiku-4-5")
    assert isinstance(p, ClaudeCodeProvider)
    assert p.model == "claude-haiku-4-5"


def test_get_provider_unknown_raises():
    with pytest.raises(LLMError, match="Unknown provider"):
        get_provider("nonsense", "x")


# ── _http_post_json ─────────────────────────────────────────────────────


def test_http_post_json_success():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"ok": true}'
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    with patch("mempalace.llm_client.urlopen", return_value=mock_resp):
        result = _http_post_json("http://x/y", {"a": 1}, {}, timeout=5)
    assert result == {"ok": True}


def test_http_post_json_http_error_wraps_as_llm_error():
    from urllib.error import HTTPError
    import io

    err = HTTPError("http://x", 404, "Not Found", {}, io.BytesIO(b"model missing"))
    with patch("mempalace.llm_client.urlopen", side_effect=err):
        with pytest.raises(LLMError, match="HTTP 404"):
            _http_post_json("http://x", {}, {}, timeout=5)


def test_http_post_json_url_error_wraps_as_llm_error():
    from urllib.error import URLError

    with patch("mempalace.llm_client.urlopen", side_effect=URLError("conn refused")):
        with pytest.raises(LLMError, match="Cannot reach"):
            _http_post_json("http://x", {}, {}, timeout=5)


def test_http_post_json_malformed_response():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"not json"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    with patch("mempalace.llm_client.urlopen", return_value=mock_resp):
        with pytest.raises(LLMError, match="Malformed"):
            _http_post_json("http://x", {}, {}, timeout=5)


# ── OllamaProvider ──────────────────────────────────────────────────────


def _mock_ollama_chat_response(content: str):
    mock = MagicMock()
    mock.read.return_value = json.dumps({"message": {"content": content}}).encode()
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = False
    return mock


def test_ollama_check_available_finds_model():
    tags = {"models": [{"name": "gemma4:e4b"}, {"name": "other:latest"}]}
    mock = MagicMock()
    mock.read.return_value = json.dumps(tags).encode()
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = False
    with patch("mempalace.llm_client.urlopen", return_value=mock):
        p = OllamaProvider(model="gemma4:e4b")
        ok, msg = p.check_available()
    assert ok
    assert msg == "ok"


def test_ollama_check_available_accepts_latest_suffix():
    tags = {"models": [{"name": "mymodel:latest"}]}
    mock = MagicMock()
    mock.read.return_value = json.dumps(tags).encode()
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = False
    with patch("mempalace.llm_client.urlopen", return_value=mock):
        p = OllamaProvider(model="mymodel")
        ok, _ = p.check_available()
    assert ok


def test_ollama_check_available_missing_model():
    tags = {"models": [{"name": "other:latest"}]}
    mock = MagicMock()
    mock.read.return_value = json.dumps(tags).encode()
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = False
    with patch("mempalace.llm_client.urlopen", return_value=mock):
        p = OllamaProvider(model="absent")
        ok, msg = p.check_available()
    assert not ok
    assert "ollama pull absent" in msg


def test_ollama_check_available_unreachable():
    from urllib.error import URLError

    with patch("mempalace.llm_client.urlopen", side_effect=URLError("refused")):
        p = OllamaProvider(model="gemma4:e4b")
        ok, msg = p.check_available()
    assert not ok
    assert "Cannot reach Ollama" in msg


def test_ollama_classify_sends_json_format():
    captured = {}

    def fake_urlopen(req, *, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _mock_ollama_chat_response('{"classifications": []}')

    with patch("mempalace.llm_client.urlopen", side_effect=fake_urlopen):
        p = OllamaProvider(model="gemma4:e4b")
        resp = p.classify("sys", "user", json_mode=True)

    assert captured["body"]["format"] == "json"
    assert captured["body"]["model"] == "gemma4:e4b"
    assert captured["url"].endswith("/api/chat")
    assert resp.provider == "ollama"
    assert resp.text == '{"classifications": []}'


def test_ollama_classify_empty_content_raises():
    with patch("mempalace.llm_client.urlopen", return_value=_mock_ollama_chat_response("")):
        p = OllamaProvider(model="x")
        with pytest.raises(LLMError, match="Empty response"):
            p.classify("s", "u")


# ── OpenAICompatProvider ────────────────────────────────────────────────


def _mock_openai_response(content: str):
    mock = MagicMock()
    payload = {"choices": [{"message": {"content": content}}]}
    mock.read.return_value = json.dumps(payload).encode()
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = False
    return mock


def test_openai_compat_resolves_url_with_v1_suffix():
    captured = {}

    def fake_urlopen(req, *, timeout):
        captured["url"] = req.full_url
        return _mock_openai_response('{"ok": true}')

    with patch("mempalace.llm_client.urlopen", side_effect=fake_urlopen):
        p = OpenAICompatProvider(model="x", endpoint="http://h:1234")
        p.classify("s", "u")
    assert captured["url"] == "http://h:1234/v1/chat/completions"


def test_openai_compat_resolves_url_with_existing_v1():
    captured = {}

    def fake_urlopen(req, *, timeout):
        captured["url"] = req.full_url
        return _mock_openai_response('{"ok": true}')

    with patch("mempalace.llm_client.urlopen", side_effect=fake_urlopen):
        p = OpenAICompatProvider(model="x", endpoint="http://h:1234/v1")
        p.classify("s", "u")
    assert captured["url"] == "http://h:1234/v1/chat/completions"


def test_openai_compat_requires_endpoint():
    p = OpenAICompatProvider(model="x")
    with pytest.raises(LLMError, match="requires --llm-endpoint"):
        p.classify("s", "u")


def test_openai_compat_sends_authorization_when_key_present():
    captured = {}

    def fake_urlopen(req, *, timeout):
        captured["auth"] = req.get_header("Authorization")
        return _mock_openai_response('{"ok": true}')

    with patch("mempalace.llm_client.urlopen", side_effect=fake_urlopen):
        p = OpenAICompatProvider(model="x", endpoint="http://h", api_key="sk-aaa")
        p.classify("s", "u")
    assert captured["auth"] == "Bearer sk-aaa"


def test_openai_compat_uses_env_var_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    p = OpenAICompatProvider(model="x", endpoint="http://h")
    assert p.api_key == "sk-from-env"


def test_openai_compat_sends_response_format_json():
    captured = {}

    def fake_urlopen(req, *, timeout):
        captured["body"] = json.loads(req.data.decode())
        return _mock_openai_response('{"ok": true}')

    with patch("mempalace.llm_client.urlopen", side_effect=fake_urlopen):
        p = OpenAICompatProvider(model="x", endpoint="http://h")
        p.classify("s", "u", json_mode=True)
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_openai_compat_unexpected_shape_raises():
    mock = MagicMock()
    mock.read.return_value = b'{"nothing": "here"}'
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = False
    with patch("mempalace.llm_client.urlopen", return_value=mock):
        p = OpenAICompatProvider(model="x", endpoint="http://h")
        with pytest.raises(LLMError, match="Unexpected response shape"):
            p.classify("s", "u")


# ── AnthropicProvider ───────────────────────────────────────────────────


def _mock_anthropic_response(text: str):
    mock = MagicMock()
    payload = {"content": [{"type": "text", "text": text}]}
    mock.read.return_value = json.dumps(payload).encode()
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = False
    return mock


def test_anthropic_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = AnthropicProvider(model="claude-haiku")
    ok, msg = p.check_available()
    assert not ok
    assert "ANTHROPIC_API_KEY" in msg


def test_anthropic_reads_env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    p = AnthropicProvider(model="claude-haiku")
    assert p.api_key == "sk-ant-env"
    ok, _ = p.check_available()
    assert ok


def test_anthropic_classify_sends_version_and_key():
    captured = {}

    def fake_urlopen(req, *, timeout):
        captured["api_key"] = req.get_header("X-api-key")
        captured["version"] = req.get_header("Anthropic-version")
        return _mock_anthropic_response('{"ok": true}')

    with patch("mempalace.llm_client.urlopen", side_effect=fake_urlopen):
        p = AnthropicProvider(model="claude-haiku", api_key="sk-ant-abc")
        resp = p.classify("s", "u")
    assert captured["api_key"] == "sk-ant-abc"
    assert captured["version"] == AnthropicProvider.API_VERSION
    assert resp.text == '{"ok": true}'


def test_anthropic_joins_multiple_text_blocks():
    mock = MagicMock()
    payload = {
        "content": [
            {"type": "text", "text": "part one. "},
            {"type": "text", "text": "part two."},
        ]
    }
    mock.read.return_value = json.dumps(payload).encode()
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = False
    with patch("mempalace.llm_client.urlopen", return_value=mock):
        p = AnthropicProvider(model="claude-haiku", api_key="sk-ant")
        resp = p.classify("s", "u")
    assert resp.text == "part one. part two."


def test_anthropic_no_key_raises_on_classify(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = AnthropicProvider(model="claude-haiku")
    with pytest.raises(LLMError, match="requires ANTHROPIC_API_KEY"):
        p.classify("s", "u")


# ── ClaudeCodeProvider ──────────────────────────────────────────────────


def _mock_completed(returncode: int, stdout: str = "", stderr: str = ""):
    """Build a fake subprocess.CompletedProcess for patching subprocess.run."""
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


def _claude_envelope(result_text: str, cost: float = 0.0007) -> str:
    """Build the JSON envelope `claude -p --output-format json` returns."""
    return json.dumps(
        {
            "type": "result",
            "result": result_text,
            "total_cost_usd": cost,
            "duration_ms": 1234,
        }
    )


def test_claude_code_check_available_binary_missing():
    with patch("mempalace.llm_client.shutil.which", return_value=None):
        p = ClaudeCodeProvider(model="claude-haiku-4-5")
        ok, msg = p.check_available()
    assert not ok
    assert "`claude` CLI not found" in msg


def test_claude_code_check_available_not_authenticated():
    with patch("mempalace.llm_client.shutil.which", return_value="/usr/local/bin/claude"):
        with patch(
            "mempalace.llm_client.subprocess.run",
            return_value=_mock_completed(1, stdout="", stderr="not logged in"),
        ):
            p = ClaudeCodeProvider(model="claude-haiku-4-5")
            ok, msg = p.check_available()
    assert not ok
    assert "Run `claude auth login`" in msg


def test_claude_code_check_available_ready():
    with patch("mempalace.llm_client.shutil.which", return_value="/usr/local/bin/claude"):
        with patch(
            "mempalace.llm_client.subprocess.run",
            return_value=_mock_completed(0, stdout="logged in as user@example.com"),
        ):
            p = ClaudeCodeProvider(model="claude-haiku-4-5")
            ok, msg = p.check_available()
    assert ok
    assert msg == "ok"


def test_claude_code_classify_command_line():
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _mock_completed(0, stdout=_claude_envelope('{"ok": true}'))

    with patch("mempalace.llm_client.subprocess.run", side_effect=fake_run):
        p = ClaudeCodeProvider(model="claude-haiku-4-5", timeout=99)
        p.classify("system text", "user text", json_mode=True)

    assert captured["cmd"][0] == "claude"
    assert "-p" in captured["cmd"]
    # `--bare` is intentionally NOT passed: it would force ANTHROPIC_API_KEY
    # auth and disable OAuth / keychain, defeating the subscription path.
    assert "--bare" not in captured["cmd"]
    assert "--no-session-persistence" in captured["cmd"]
    assert "--output-format" in captured["cmd"]
    assert "json" in captured["cmd"]
    assert "--model" in captured["cmd"]
    assert "claude-haiku-4-5" in captured["cmd"]
    # System prompt is augmented with a JSON-only instruction in json_mode
    sys_idx = captured["cmd"].index("--system-prompt")
    assert captured["cmd"][sys_idx + 1].startswith("system text")
    assert "JSON only" in captured["cmd"][sys_idx + 1]
    # User content goes via stdin
    assert captured["kwargs"]["input"] == "user text"
    assert captured["kwargs"]["timeout"] == 99
    # cwd must be a temp dir so claude does not pick up a project-level CLAUDE.md
    assert captured["kwargs"]["cwd"] == tempfile.gettempdir()


def test_claude_code_classify_json_mode_off_keeps_system_clean():
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _mock_completed(0, stdout=_claude_envelope("plain text reply"))

    with patch("mempalace.llm_client.subprocess.run", side_effect=fake_run):
        p = ClaudeCodeProvider(model="claude-haiku-4-5")
        resp = p.classify("system text", "user", json_mode=False)

    sys_idx = captured["cmd"].index("--system-prompt")
    assert captured["cmd"][sys_idx + 1] == "system text"
    assert resp.text == "plain text reply"


def test_claude_code_classify_parses_envelope():
    with patch(
        "mempalace.llm_client.subprocess.run",
        return_value=_mock_completed(0, stdout=_claude_envelope("classified")),
    ):
        p = ClaudeCodeProvider(model="claude-haiku-4-5")
        resp = p.classify("s", "u")

    assert resp.text == "classified"
    assert resp.provider == "claude-code"
    assert resp.model == "claude-haiku-4-5"
    assert resp.raw["total_cost_usd"] == pytest.approx(0.0007)


def test_claude_code_classify_timeout_raises_llm_error():
    with patch(
        "mempalace.llm_client.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=1),
    ):
        p = ClaudeCodeProvider(model="claude-haiku-4-5", timeout=1)
        with pytest.raises(LLMError, match="timed out after 1s"):
            p.classify("s", "u")


def test_claude_code_classify_spawn_failure_raises_llm_error():
    with patch(
        "mempalace.llm_client.subprocess.run",
        side_effect=FileNotFoundError("no such file: claude"),
    ):
        p = ClaudeCodeProvider(model="claude-haiku-4-5")
        with pytest.raises(LLMError, match="failed to spawn"):
            p.classify("s", "u")


def test_claude_code_classify_nonzero_raises_llm_error():
    with patch(
        "mempalace.llm_client.subprocess.run",
        return_value=_mock_completed(1, stdout="", stderr="boom: bad model"),
    ):
        p = ClaudeCodeProvider(model="claude-haiku-4-5")
        with pytest.raises(LLMError, match=r"`claude -p` exited 1: boom"):
            p.classify("s", "u")


def test_claude_code_classify_malformed_json_raises_llm_error():
    with patch(
        "mempalace.llm_client.subprocess.run",
        return_value=_mock_completed(0, stdout="not valid json"),
    ):
        p = ClaudeCodeProvider(model="claude-haiku-4-5")
        with pytest.raises(LLMError, match="non-JSON envelope"):
            p.classify("s", "u")


def test_claude_code_classify_empty_result_raises_llm_error():
    with patch(
        "mempalace.llm_client.subprocess.run",
        return_value=_mock_completed(0, stdout=_claude_envelope("")),
    ):
        p = ClaudeCodeProvider(model="claude-haiku-4-5")
        with pytest.raises(LLMError, match="empty result"):
            p.classify("s", "u")


@pytest.mark.skipif(
    os.environ.get("MEMPAL_TEST_CLAUDE_CLI") != "1",
    reason="set MEMPAL_TEST_CLAUDE_CLI=1 to run live `claude -p` integration test",
)
def test_claude_code_real_invocation():
    """End-to-end probe: spawns the real `claude` binary if available + authenticated."""
    p = ClaudeCodeProvider(model="claude-haiku-4-5")
    ok, msg = p.check_available()
    if not ok:
        pytest.skip(f"`claude` not ready: {msg}")
    resp = p.classify(
        "Reply with a single short JSON object.",
        'Reply with {"hello": "world"} and nothing else.',
        json_mode=True,
    )
    assert resp.provider == "claude-code"
    assert resp.text  # non-empty
    assert resp.raw.get("type") == "result"
