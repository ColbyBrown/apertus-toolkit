"""
Ollama-compatible proxy that fronts PublicAI's Apertus-70B endpoint with a
local LM Studio Apertus-8B as fallback when the remote call fails.

Why this exists: PublicAI's API is *almost* OpenAI-compatible but doesn't play
nicely with tools like GitHub Copilot, the VSCode Continue extension, or the
Hermes Agent --- those tools refuse to connect. A raw httpx call to PublicAI
works most of the time but the endpoint is unreliable enough that a local
fallback is worth having.

This proxy:
  - Speaks the Ollama HTTP API so any Ollama client can use it.
  - Advertises a single model: `apertus-hybrid`.
  - Routes every request to PublicAI (Apertus-70B-Instruct) first.
  - On connection/HTTP/timeout error (and only if no bytes have been streamed
    to the client yet), retries against a local LM Studio server hosting
    Apertus-8B-Instruct.

Configuration via env (.env loaded from script directory):
    PUBLICAI_API_KEY     required
    PUBLICAI_BASE        default https://api.publicai.co/v1
    PUBLICAI_MODEL       default swiss-ai/apertus-70b-instruct
    LMSTUDIO_BASE        default http://localhost:1234/v1
    LMSTUDIO_MODEL       default apertus-8b-instruct
    PORT                 default 11436

Run:
    python apertus_proxy_fallback.py
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    override=True,
)

PUBLICAI_BASE = os.environ.get("PUBLICAI_BASE", "https://api.publicai.co/v1")
PUBLICAI_MODEL = os.environ.get("PUBLICAI_MODEL", "swiss-ai/apertus-70b-instruct")
LMSTUDIO_BASE = os.environ.get("LMSTUDIO_BASE", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "apertus-8b-instruct")
OLLAMA_MODEL_NAME = "apertus-hybrid"

# Errors that should trigger fallback. Anything else (auth failures, malformed
# requests, etc.) propagates -- a bad request to the local model would fail too.
FALLBACK_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.HTTPStatusError,
)


def _get_api_key() -> str:
    key = os.environ.get("PUBLICAI_API_KEY", "")
    if not key:
        raise HTTPException(status_code=500, detail="PUBLICAI_API_KEY env var not set")
    return key


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Ollama-options -> OpenAI-options translation
# ---------------------------------------------------------------------------

_OLLAMA_TO_OPENAI = {
    "temperature": "temperature",
    "top_p": "top_p",
    "num_predict": "max_tokens",
    "stop": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
}


def _ollama_options_to_openai(options: dict | None) -> dict:
    if not options:
        return {}
    return {
        _OLLAMA_TO_OPENAI[k]: v
        for k, v in options.items()
        if k in _OLLAMA_TO_OPENAI
    }


# ---------------------------------------------------------------------------
# Apertus -> Hermes format translator
# ---------------------------------------------------------------------------
#
# Apertus models emit:
#   <|inner_prefix|>...<|inner_suffix|>                            reasoning
#   <|tools_prefix|>[{"toolname": <args_json>}, ...]<|tools_suffix|>  tool calls
#
# Hermes/Ollama-friendly equivalent (what most clients expect):
#   <think>...</think>
#   <tool_call>{"name": "...", "arguments": {...}}</tool_call>     (one per call)
#
# Some upstreams (notably PublicAI's OpenAI-compat layer) may surface tool
# calls via OpenAI's structured `tool_calls` field instead of inline tokens;
# we render those into <tool_call> XML too. Same for `reasoning_content`,
# which some providers expose as a separate field rather than inline.

TRANSLATE_FORMAT = os.environ.get("TRANSLATE_FORMAT", "1") not in ("0", "false", "False", "")

_INNER_PREFIX = "<|inner_prefix|>"
_INNER_SUFFIX = "<|inner_suffix|>"
_TOOLS_PREFIX = "<|tools_prefix|>"
_TOOLS_SUFFIX = "<|tools_suffix|>"

_TOKEN_TRANSLATIONS = {
    _INNER_PREFIX: "<think>",
    _INNER_SUFFIX: "</think>",
}

_DROP_TOKENS = {
    "<|assistant_start|>", "<|assistant_end|>",
    "<|user_start|>", "<|user_end|>",
    "<|system_start|>", "<|system_end|>",
    "<|developer_start|>", "<|developer_end|>",
}

_MAX_TOKEN_LEN = max(
    [len(k) for k in _TOKEN_TRANSLATIONS]
    + [len(_TOOLS_PREFIX), len(_TOOLS_SUFFIX)]
    + [len(t) for t in _DROP_TOKENS]
)


def _apertus_tool_array_to_hermes(arr_text: str) -> str:
    """
    Apertus tool array: [{"name1": <args>}, {"name2": <args>}]
    The key in each single-key object IS the function name; the value is the
    arguments (often a pre-serialized JSON string).
    """
    try:
        arr = json.loads(arr_text)
        blocks: list[str] = []
        for call in arr:
            if not isinstance(call, dict) or len(call) != 1:
                continue
            ((name, args),) = call.items()
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass  # leave raw if it isn't valid JSON
            block = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
            blocks.append(f"<tool_call>\n{block}\n</tool_call>")
        if blocks:
            return "\n".join(blocks)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fallback: don't drop data, just wrap raw text in one tool_call
    return f"<tool_call>\n{arr_text.strip()}\n</tool_call>"


def _render_openai_tool_calls(tool_calls: list[dict]) -> str:
    """Render OpenAI-style tool_calls array as Hermes <tool_call> blocks."""
    blocks: list[str] = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        block = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        blocks.append(f"<tool_call>\n{block}\n</tool_call>")
    return "\n".join(blocks)


class StreamingTranslator:
    """
    Streaming state machine that converts Apertus special tokens to Hermes XML.
    Buffers up to ~_MAX_TOKEN_LEN chars to detect token boundaries; flushes
    plain text immediately. Inside a <|tools_prefix|>...<|tools_suffix|> span,
    accumulates the full JSON array before emitting <tool_call> blocks.
    """

    def __init__(self):
        self._buf = ""
        self._in_tools = False
        self._tools_acc = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if self._in_tools:
            return self._feed_in_tools(chunk)
        self._buf += chunk
        return self._drain_normal()

    def _drain_normal(self) -> str:
        out: list[str] = []
        while self._buf:
            tok_start = self._buf.find("<|")
            if tok_start == -1:
                # No potential token; flush everything except a trailing "<"
                # (which might start the next chunk's "<|...|>").
                if self._buf.endswith("<"):
                    out.append(self._buf[:-1])
                    self._buf = "<"
                else:
                    out.append(self._buf)
                    self._buf = ""
                break
            if tok_start > 0:
                out.append(self._buf[:tok_start])
                self._buf = self._buf[tok_start:]
            tok_end = self._buf.find("|>")
            if tok_end == -1:
                # Incomplete token; wait for more (with a sanity cap).
                if len(self._buf) > _MAX_TOKEN_LEN * 4:
                    out.append(self._buf[0])
                    self._buf = self._buf[1:]
                    continue
                break
            token = self._buf[: tok_end + 2]
            self._buf = self._buf[tok_end + 2 :]
            if token in _TOKEN_TRANSLATIONS:
                out.append(_TOKEN_TRANSLATIONS[token])
            elif token == _TOOLS_PREFIX:
                self._in_tools = True
                out.append(self._feed_in_tools(""))  # in case suffix already buffered
                break
            elif token in _DROP_TOKENS:
                pass  # silently drop role envelopes
            else:
                # Unknown special token: keep verbatim so nothing is lost.
                out.append(token)
        return "".join(out)

    def _feed_in_tools(self, chunk: str) -> str:
        self._tools_acc += chunk
        # Drain any pending normal-mode buffer into the tools accumulator
        # (we may have entered tools mode while bytes remained in _buf).
        if self._buf:
            self._tools_acc += self._buf
            self._buf = ""
        suffix_idx = self._tools_acc.find(_TOOLS_SUFFIX)
        if suffix_idx == -1:
            return ""
        array_text = self._tools_acc[:suffix_idx].strip()
        remainder = self._tools_acc[suffix_idx + len(_TOOLS_SUFFIX) :]
        self._tools_acc = ""
        self._in_tools = False
        rendered = _apertus_tool_array_to_hermes(array_text)
        # Push any trailing content back through normal mode.
        self._buf = remainder
        return rendered + self._drain_normal()

    def flush(self) -> str:
        """Emit any trailing buffered content (called at end of stream)."""
        out = ""
        if self._in_tools:
            # Stream ended inside a tools block — emit raw to avoid silent loss.
            out += "<tool_call>\n" + self._tools_acc.strip() + "\n</tool_call>"
            self._tools_acc = ""
            self._in_tools = False
        out += self._buf
        self._buf = ""
        return out


def translate_text(text: str) -> str:
    """One-shot translator for non-streaming content."""
    if not TRANSLATE_FORMAT or not text:
        return text or ""
    t = StreamingTranslator()
    return t.feed(text) + t.flush()


def render_message_to_hermes(msg: dict) -> str:
    """
    Combine an OpenAI-style assistant message into a single Hermes-XML string:
      - reasoning_content (if present) -> <think>...</think>
      - content (after inline-token translation)
      - tool_calls (if present)        -> <tool_call>...</tool_call> blocks
    """
    parts: list[str] = []
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if reasoning:
        parts.append(f"<think>{reasoning}</think>" if TRANSLATE_FORMAT else reasoning)
    content = msg.get("content") or ""
    parts.append(translate_text(content))
    tool_calls = msg.get("tool_calls") or []
    if tool_calls and TRANSLATE_FORMAT:
        rendered = _render_openai_tool_calls(tool_calls)
        if rendered:
            if parts and parts[-1] and not parts[-1].endswith("\n"):
                parts.append("\n")
            parts.append(rendered)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Hermes -> Apertus inbound translator (for assistant turns in chat history)
# ---------------------------------------------------------------------------
#
# When clients submit a multi-turn conversation, prior assistant turns may
# contain Hermes-style markup that Apertus models weren't trained to interpret
# as such -- if we just forward them as raw strings, the model sees the XML as
# literal text content rather than reasoning/tool-call signals.
#
# Translation:
#   <think>X</think>        -> <|inner_prefix|>X<|inner_suffix|>   (stays in content)
#   <tool_call>{...}</tool_call>  -> extracted into message.tool_calls (OpenAI
#                                    structured form). Apertus' chat template
#                                    renders that into <|tools_prefix|>[...]<|tools_suffix|>
#                                    AND sets the waiting_for_tool_outputs flag,
#                                    which keeps the assistant span open for the
#                                    next role:tool message. Embedding the calls
#                                    in content would skip that flag and break
#                                    multi-turn tool flows.
#
# Only assistant-role messages are translated; user/system/tool roles pass through.

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def _hermes_assistant_to_apertus(msg: dict) -> dict:
    """
    Rewrite a Hermes-flavoured assistant message into the shape Apertus
    expects. Returns the original dict unchanged if no Hermes tags are found
    (or if translation is disabled).
    """
    if not TRANSLATE_FORMAT:
        return msg
    content = msg.get("content")
    if not isinstance(content, str) or ("<think>" not in content and "<tool_call>" not in content):
        return msg

    # <think> -> <|inner_prefix|>...<|inner_suffix|>
    new_content = _THINK_RE.sub(r"<|inner_prefix|>\1<|inner_suffix|>", content)

    # <tool_call> -> structured tool_calls; remove from content.
    extracted: list[dict] = []

    def _take(match):
        raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return match.group(0)  # leave malformed blocks in place
        if not isinstance(obj, dict) or "name" not in obj:
            return match.group(0)
        args = obj.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        extracted.append({
            "id": f"call_{len(extracted)}",
            "type": "function",
            "function": {"name": obj["name"], "arguments": args},
        })
        return ""

    new_content = _TOOL_CALL_RE.sub(_take, new_content).strip()

    if new_content == content and not extracted:
        return msg

    out = dict(msg)
    out["content"] = new_content
    if extracted:
        out["tool_calls"] = list(msg.get("tool_calls") or []) + extracted
    return out


def _hermes_tool_to_apertus(msg: dict) -> list[dict]:
    """
    Unwrap Hermes <tool_response>...</tool_response> wrappers in tool-role
    messages. The inner JSON has {tool_call_id, name, content}; lift those
    into OpenAI's standard tool-message shape (bare content + tool_call_id
    + optional name). Apertus' chat template renders bare content for tool
    messages, so the wrapper would otherwise be passed through as literal text.

    A single Hermes tool turn may contain multiple <tool_response> blocks
    (one per parallel call result); each becomes its own OpenAI tool message.
    Returns [msg] unchanged if no <tool_response> blocks are found.
    """
    if not TRANSLATE_FORMAT:
        return [msg]
    content = msg.get("content")
    if not isinstance(content, str) or "<tool_response>" not in content:
        return [msg]

    unwrapped: list[dict] = []
    for match in _TOOL_RESPONSE_RE.finditer(content):
        raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        new_msg: dict = {"role": "tool"}
        if "tool_call_id" in obj:
            new_msg["tool_call_id"] = obj["tool_call_id"]
        elif msg.get("tool_call_id"):
            new_msg["tool_call_id"] = msg["tool_call_id"]
        inner = obj.get("content", "")
        new_msg["content"] = inner if isinstance(inner, str) else json.dumps(inner, ensure_ascii=False)
        if "name" in obj:
            new_msg["name"] = obj["name"]
        unwrapped.append(new_msg)

    return unwrapped if unwrapped else [msg]


def translate_messages_inbound(messages: list[dict]) -> list[dict]:
    """
    Apply Hermes->Apertus translation:
      - assistant: <think> -> inner_prefix/suffix; <tool_call> -> structured tool_calls
      - tool:      unwrap <tool_response>; may split one input into multiple outputs
      - user/system/developer: pass through unchanged
    """
    if not TRANSLATE_FORMAT:
        return messages
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            out.append(_hermes_assistant_to_apertus(m))
        elif role == "tool":
            out.extend(_hermes_tool_to_apertus(m))
        else:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Upstream callers (both OpenAI-compatible)
# ---------------------------------------------------------------------------

def _payload(model: str, messages: list[dict], stream: bool, extra: dict) -> dict:
    return {"model": model, "messages": messages, "stream": stream, **extra}


async def _publicai_nonstream(api_key: str, messages: list[dict], extra: dict) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{PUBLICAI_BASE}/chat/completions",
            headers=headers,
            json=_payload(PUBLICAI_MODEL, messages, False, extra),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]


async def _lmstudio_nonstream(messages: list[dict], extra: dict) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{LMSTUDIO_BASE}/chat/completions",
            json=_payload(LMSTUDIO_MODEL, messages, False, extra),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]


async def complete_nonstream(api_key: str, messages: list[dict], extra: dict) -> tuple[str, str]:
    """Return (hermes_text, backend_used)."""
    try:
        msg = await _publicai_nonstream(api_key, messages, extra)
        return render_message_to_hermes(msg), "publicai"
    except FALLBACK_ERRORS as e:
        print(f"[proxy] PublicAI non-stream failed: {e!r} -- falling back to LM Studio", file=sys.stderr)
        msg = await _lmstudio_nonstream(messages, extra)
        return render_message_to_hermes(msg), "lmstudio"


async def _iter_openai_stream(url: str, headers: dict, payload: dict) -> AsyncIterator[str]:
    """
    Yield Hermes-translated content deltas from an OpenAI-compatible SSE stream.
    At end of stream, flush the translator and emit any accumulated
    reasoning_content (as <think>...</think>) and tool_calls (as <tool_call>...
    </tool_call> blocks) that arrived in structured form rather than inline.
    """
    translator = StreamingTranslator() if TRANSLATE_FORMAT else None
    reasoning = ""
    tool_calls_acc: dict[int, dict] = {}  # index -> {"name", "arguments"}

    timeout = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta") or {}
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

                rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if rc:
                    reasoning += rc

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls_acc.setdefault(idx, {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]

                content = delta.get("content") or ""
                if content:
                    out = translator.feed(content) if translator else content
                    if out:
                        yield out

            # End-of-stream housekeeping
            if translator:
                tail = translator.flush()
                if tail:
                    yield tail
            if reasoning and TRANSLATE_FORMAT:
                yield f"<think>{reasoning}</think>"
            if tool_calls_acc and TRANSLATE_FORMAT:
                for idx in sorted(tool_calls_acc):
                    slot = tool_calls_acc[idx]
                    args = slot["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    block = json.dumps(
                        {"name": slot["name"], "arguments": args}, ensure_ascii=False
                    )
                    yield f"<tool_call>\n{block}\n</tool_call>"


async def stream_with_fallback(api_key: str, messages: list[dict], extra: dict) -> AsyncIterator[str]:
    """
    Try PublicAI first; fall back to LM Studio only if PublicAI fails BEFORE
    yielding any deltas. Mid-stream failures end the stream as-is (we cannot
    restart without confusing the client).
    """
    sent_any = False
    try:
        async for delta in _iter_openai_stream(
            f"{PUBLICAI_BASE}/chat/completions",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            _payload(PUBLICAI_MODEL, messages, True, extra),
        ):
            sent_any = True
            yield delta
        if sent_any:
            return
        # Connected but never produced content -> treat as transient failure.
        raise httpx.ReadError("no deltas from PublicAI stream")
    except FALLBACK_ERRORS as e:
        if sent_any:
            print(f"[proxy] PublicAI stream errored mid-stream: {e!r}", file=sys.stderr)
            return
        print(f"[proxy] PublicAI stream failed before any deltas: {e!r} -- falling back to LM Studio", file=sys.stderr)

    async for delta in _iter_openai_stream(
        f"{LMSTUDIO_BASE}/chat/completions",
        {"Content-Type": "application/json"},
        _payload(LMSTUDIO_MODEL, messages, True, extra),
    ):
        yield delta


# ---------------------------------------------------------------------------
# FastAPI app (Ollama-compatible surface)
# ---------------------------------------------------------------------------

app = FastAPI(title="Apertus Fallback Proxy")


@app.get("/")
async def root():
    return "Ollama is running"


@app.get("/api/version")
async def version():
    return {"version": "0.1.0"}


@app.get("/api/tags")
async def list_tags():
    return {
        "models": [
            {
                "name": OLLAMA_MODEL_NAME,
                "model": OLLAMA_MODEL_NAME,
                "modified_at": _now_iso(),
                "size": 8_000_000_000,
                "digest": "sha256:apertushybrid000000000000000000000000000000000000000000000000000",
                "details": {
                    "parent_model": PUBLICAI_MODEL,
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "70B",
                    "quantization_level": "FP16",
                },
            }
        ]
    }


@app.post("/api/show")
async def show_model(request: Request):
    return JSONResponse({
        "modelfile": f"FROM {PUBLICAI_MODEL}",
        "parameters": "",
        "template": "",
        "details": {
            "parent_model": PUBLICAI_MODEL,
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "70B",
            "quantization_level": "FP16",
        },
        "model_info": {
            "general.architecture": "llama",
            "general.parameter_count": 70_000_000_000,
        },
    })


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    api_key = _get_api_key()
    messages: list[dict] = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages required")
    messages = translate_messages_inbound(messages)
    do_stream: bool = body.get("stream", True)
    extra = _ollama_options_to_openai(body.get("options"))

    if do_stream:
        async def _ndjson():
            async for delta in stream_with_fallback(api_key, messages, extra):
                yield json.dumps({
                    "model": OLLAMA_MODEL_NAME,
                    "created_at": _now_iso(),
                    "message": {"role": "assistant", "content": delta},
                    "done": False,
                }) + "\n"
            yield json.dumps({
                "model": OLLAMA_MODEL_NAME,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            }) + "\n"

        return StreamingResponse(_ndjson(), media_type="application/x-ndjson")

    text, _ = await complete_nonstream(api_key, messages, extra)
    return JSONResponse({
        "model": OLLAMA_MODEL_NAME,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": text},
        "done": True,
        "done_reason": "stop",
    })


@app.post("/api/generate")
async def generate(request: Request):
    """Legacy single-turn endpoint (Ollama clients use this for title gen, etc.)."""
    body = await request.json()
    api_key = _get_api_key()
    prompt: str = body.get("prompt", "")
    system: str = body.get("system", "")
    do_stream: bool = body.get("stream", True)
    extra = _ollama_options_to_openai(body.get("options"))

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if do_stream:
        async def _ndjson():
            async for delta in stream_with_fallback(api_key, messages, extra):
                yield json.dumps({
                    "model": OLLAMA_MODEL_NAME,
                    "created_at": _now_iso(),
                    "response": delta,
                    "done": False,
                }) + "\n"
            yield json.dumps({
                "model": OLLAMA_MODEL_NAME,
                "created_at": _now_iso(),
                "response": "",
                "done": True,
            }) + "\n"

        return StreamingResponse(_ndjson(), media_type="application/x-ndjson")

    text, _ = await complete_nonstream(api_key, messages, extra)
    return JSONResponse({
        "model": OLLAMA_MODEL_NAME,
        "created_at": _now_iso(),
        "response": text,
        "done": True,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 11436))
    print(f"Apertus fallback proxy listening on port {port}", file=sys.stderr)
    print(f"  primary:  {PUBLICAI_BASE}  ({PUBLICAI_MODEL})", file=sys.stderr)
    print(f"  fallback: {LMSTUDIO_BASE}  ({LMSTUDIO_MODEL})", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=port)
