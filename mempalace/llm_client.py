"""
llm_client.py — Minimal provider abstraction for LLM-assisted entity refinement.

Four providers cover the useful space:

- ``ollama`` (default): local models via http://localhost:11434. Works fully
  offline. Honors MemPalace's "zero-API required" principle.
- ``openai-compat``: any OpenAI-compatible ``/v1/chat/completions`` endpoint.
  Covers OpenRouter, LM Studio, llama.cpp server, vLLM, Groq, Fireworks,
  Together, and most self-hosted setups.
- ``anthropic``: the official Messages API. Opt-in for users who want Haiku
  quality without setting up a local model.
- ``claude-code``: the local ``claude`` CLI binary. Routes through the user's
  existing Claude Pro/Max subscription via ``claude auth login`` -- no API
  key required. Subprocess-based, zero new pip deps. Subject to Anthropic
  policy on subscription use from third-party tools.

All providers expose the same ``classify(system, user, json_mode)`` method and
the same ``check_available()`` probe. No external SDK dependencies -- stdlib
``urllib`` plus ``subprocess`` (for ``claude-code``) only.

JSON mode matters here: we always ask for structured output. Providers
differ on how to request it (Ollama: ``format: json``; OpenAI-compat:
``response_format``; Anthropic: prompt-level instruction; claude-code:
prompt-level instruction) and this module normalizes that away from the
caller.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMError(RuntimeError):
    """Raised for any provider failure — transport, parse, auth, missing model."""


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    raw: dict


# ==================== BASE ====================


class LLMProvider:
    name: str = "base"

    def __init__(
        self,
        model: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def classify(self, system: str, user: str, json_mode: bool = True) -> LLMResponse:
        raise NotImplementedError

    def check_available(self) -> tuple[bool, str]:
        """Return ``(ok, message)``. Fast probe that the provider is reachable."""
        raise NotImplementedError


def _http_post_json(url: str, body: dict, headers: dict, timeout: int) -> dict:
    """POST JSON and return the parsed response. Raises LLMError on any failure."""
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise LLMError(f"HTTP {e.code} from {url}: {detail or e.reason}") from e
    except (URLError, OSError) as e:
        raise LLMError(f"Cannot reach {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise LLMError(f"Malformed response from {url}: {e}") from e


# ==================== OLLAMA ====================


class OllamaProvider(LLMProvider):
    name = "ollama"
    DEFAULT_ENDPOINT = "http://localhost:11434"

    def __init__(
        self,
        model: str,
        endpoint: Optional[str] = None,
        timeout: int = 180,
        **_: object,
    ):
        super().__init__(
            model=model,
            endpoint=endpoint or self.DEFAULT_ENDPOINT,
            timeout=timeout,
        )

    def check_available(self) -> tuple[bool, str]:
        try:
            with urlopen(f"{self.endpoint}/api/tags", timeout=5) as resp:
                data = json.loads(resp.read())
        except (URLError, HTTPError, OSError, json.JSONDecodeError) as e:
            return False, f"Cannot reach Ollama at {self.endpoint}: {e}"
        names = {m.get("name", "") for m in data.get("models", []) or []}
        # Ollama tags may or may not include ':latest' — accept either form
        wanted = {self.model, f"{self.model}:latest"}
        if not names & wanted:
            return (
                False,
                f"Model '{self.model}' not loaded in Ollama. Run: ollama pull {self.model}",
            )
        return True, "ok"

    def classify(self, system: str, user: str, json_mode: bool = True) -> LLMResponse:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if json_mode:
            body["format"] = "json"
        data = _http_post_json(f"{self.endpoint}/api/chat", body, headers={}, timeout=self.timeout)
        text = (data.get("message") or {}).get("content", "")
        if not text:
            raise LLMError(f"Empty response from Ollama (model={self.model})")
        return LLMResponse(text=text, model=self.model, provider=self.name, raw=data)


# ==================== OPENAI-COMPAT ====================


class OpenAICompatProvider(LLMProvider):
    """Any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Supply ``--llm-endpoint http://host:port`` (with or without ``/v1``).
    API key via ``--llm-api-key`` or the ``OPENAI_API_KEY`` env var.
    """

    name = "openai-compat"

    def __init__(
        self,
        model: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        **_: object,
    ):
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        super().__init__(model=model, endpoint=endpoint, api_key=resolved_key, timeout=timeout)

    def _resolve_url(self) -> str:
        if not self.endpoint:
            raise LLMError("openai-compat provider requires --llm-endpoint")
        url = self.endpoint.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        return f"{url}/chat/completions"

    def check_available(self) -> tuple[bool, str]:
        if not self.endpoint:
            return False, "no --llm-endpoint configured"
        base = self.endpoint.rstrip("/")
        base = base.removesuffix("/chat/completions").removesuffix("/v1")
        try:
            req = Request(f"{base}/v1/models")
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            with urlopen(req, timeout=5):
                pass
        except (URLError, HTTPError, OSError) as e:
            return False, f"Cannot reach {self.endpoint}: {e}"
        return True, "ok"

    def classify(self, system: str, user: str, json_mode: bool = True) -> LLMResponse:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = _http_post_json(self._resolve_url(), body, headers=headers, timeout=self.timeout)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response shape: {e}") from e
        if not text:
            raise LLMError(f"Empty response from {self.name} (model={self.model})")
        return LLMResponse(text=text, model=self.model, provider=self.name, raw=data)


# ==================== ANTHROPIC ====================


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    DEFAULT_ENDPOINT = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 120,
        **_: object,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        super().__init__(
            model=model,
            endpoint=endpoint or self.DEFAULT_ENDPOINT,
            api_key=key,
            timeout=timeout,
        )

    def check_available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "ANTHROPIC_API_KEY not set (use --llm-api-key or env)"
        # Don't probe — a live request would cost money. First real call will
        # surface auth errors if the key is invalid.
        return True, "ok"

    def classify(self, system: str, user: str, json_mode: bool = True) -> LLMResponse:
        if not self.api_key:
            raise LLMError("Anthropic provider requires ANTHROPIC_API_KEY env or --llm-api-key")
        sys_prompt = system
        if json_mode:
            sys_prompt += "\n\nRespond with valid JSON only, no prose."
        body = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": 0.1,
            "system": sys_prompt,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "X-API-Key": self.api_key,
            "anthropic-version": self.API_VERSION,
        }
        data = _http_post_json(
            f"{self.endpoint}/v1/messages", body, headers=headers, timeout=self.timeout
        )
        try:
            text = "".join(
                b.get("text", "") for b in data.get("content", []) or [] if b.get("type") == "text"
            )
        except (AttributeError, TypeError) as e:
            raise LLMError(f"Unexpected response shape: {e}") from e
        if not text:
            raise LLMError(f"Empty response from Anthropic (model={self.model})")
        return LLMResponse(text=text, model=self.model, provider=self.name, raw=data)


# ==================== CLAUDE CODE (CLI subprocess) ====================


class ClaudeCodeProvider(LLMProvider):
    """Routes through the local ``claude`` CLI binary using subscription auth.

    Auth happens once via ``claude auth login`` (stored in the user's keychain
    by Claude Code itself); we shell out to ``claude -p`` for each call. No
    API key, no new pip dependencies -- the CLI itself is the bundled
    transport.

    Going direct via ``subprocess`` rather than the ``claude-agent-sdk``
    Python wrapper is deliberate: the SDK is async-only, requires
    Python >= 3.10 (we still support 3.9), and itself spawns the same binary.
    Skipping the wrapper avoids a dependency and an asyncio bridge.

    Subscription use from third-party harnesses is governed by Anthropic's
    policy, which has changed in 2026. The ``claude -p`` CLI invocation
    pattern is currently sanctioned for first-party tools but may be
    restricted later; ``check_available()`` will surface auth errors at that
    point so callers can fall back.
    """

    name = "claude-code"
    DEFAULT_MODEL = "claude-haiku-4-5"

    def __init__(
        self,
        model: str,
        timeout: int = 120,
        **_: object,  # endpoint/api_key ignored -- auth comes from `claude auth login`
    ):
        super().__init__(model=model, timeout=timeout)

    def check_available(self) -> tuple[bool, str]:
        binary = shutil.which("claude")
        if not binary:
            return (
                False,
                "`claude` CLI not found in PATH. "
                "Install Claude Code: https://claude.com/product/claude-code",
            )
        try:
            r = subprocess.run(
                ["claude", "auth", "status", "--text"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            return False, f"`claude auth status` failed: {e}"
        if r.returncode != 0:
            return (
                False,
                "Not authenticated. Run `claude auth login` to use your Claude subscription.",
            )
        return True, "ok"

    def classify(self, system: str, user: str, json_mode: bool = True) -> LLMResponse:
        sys_prompt = system
        if json_mode:
            sys_prompt += "\n\nRespond with valid JSON only, no prose."
        # `--bare` would skip hooks, plugins, CLAUDE.md auto-discovery, but it
        # also forces claude to use ANTHROPIC_API_KEY only and ignore OAuth /
        # keychain. That defeats this provider's whole point (subscription
        # auth), so we omit it. To keep the surrounding context minimal we
        # invoke from a temp cwd so claude does not pick up a project-level
        # CLAUDE.md it does not need.
        cmd = [
            "claude",
            "-p",
            "--no-session-persistence",  # don't pollute Claude Code session history
            "--output-format",
            "json",
            "--system-prompt",
            sys_prompt,
            "--model",
            self.model,
        ]
        try:
            r = subprocess.run(
                cmd,
                input=user,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired as e:
            raise LLMError(f"`claude -p` timed out after {self.timeout}s") from e
        except OSError as e:
            raise LLMError(f"`claude -p` failed to spawn: {e}") from e
        if r.returncode != 0:
            stderr = (r.stderr or "").strip()[:500]
            raise LLMError(f"`claude -p` exited {r.returncode}: {stderr or 'no stderr'}")
        try:
            envelope = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"`claude -p` returned non-JSON envelope: {e}") from e
        # `--output-format json` returns:
        # {"type":"result","result":"<text>","total_cost_usd":...,...}
        text = envelope.get("result", "")
        if not text:
            raise LLMError(f"`claude -p` returned empty result: {envelope}")
        return LLMResponse(text=text, model=self.model, provider=self.name, raw=envelope)


# ==================== FACTORY ====================


PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "openai-compat": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
    "claude-code": ClaudeCodeProvider,
}


def get_provider(
    name: str,
    model: str,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> LLMProvider:
    """Build a provider by name. Raises LLMError on unknown provider."""
    cls = PROVIDERS.get(name)
    if cls is None:
        raise LLMError(f"Unknown provider '{name}'. Choices: {sorted(PROVIDERS.keys())}")
    return cls(model=model, endpoint=endpoint, api_key=api_key, timeout=timeout)
