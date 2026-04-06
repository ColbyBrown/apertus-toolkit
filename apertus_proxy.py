"""
Ollama-compatible proxy server for the Apertus 70B model via PublicAI.

Implements a multi-step agentic pipeline:
  1. Reasoning self-dialogue: the model reasons about how to respond
  2. Decision step: determines whether to call a tool or respond directly
  3. Tool execution (web_search or wikipedia_search) if needed
  4. Final response generation with full context

Usage:
    PUBLICAI_API_KEY=your_key python apertus_proxy.py

Listens on port 11435. Compatible with any Ollama client.
"""

import json
import os
import sys
import asyncio
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv
import uvicorn

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _ensure_protocol(url: str) -> str:
    """Ensure URL has http:// or https:// protocol prefix."""
    if not url or not isinstance(url, str):
        return "http://localhost:11434"
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url

# ---------------------------------------------------------------------------
# Tool implementations (inlined from minimal_mcp)
# ---------------------------------------------------------------------------

def _tool_web_search(query: str, max_results: int = 10) -> str:
    from ddgs import DDGS
    results = DDGS().text(query, max_results=max_results)
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n")
    return "\n".join(lines)


def _tool_wikipedia_search(topic: str, language: str = "en") -> str:
    import wikipediaapi
    wiki = wikipediaapi.Wikipedia(user_agent="apertus-proxy/1.0", language=language)
    page = wiki.page(topic)
    if not page.exists():
        return f"No Wikipedia page found for '{topic}'."
    return f"Title: {page.title}\nURL: {page.fullurl}\n\nSummary:\n{page.summary}"


def _tool_python_repl(code: str, timeout: int = 10) -> str:
    """Execute Python code in a subprocess and return stdout + stderr."""
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        if not parts:
            parts.append("(no output)")
        output = "\n".join(parts)
        # Truncate to avoid flooding the context
        if len(output) > 3000:
            output = output[:3000] + "\n…(truncated)"
        return output
    except subprocess.TimeoutExpired:
        return f"[error] Execution timed out after {timeout}s."
    except Exception as e:
        return f"[error] {e}"


_TA_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TA_prompt_v1.txt")
_TA_PROMPT_CACHE: str | None = None

def _load_ta_prompt() -> str:
    global _TA_PROMPT_CACHE
    if _TA_PROMPT_CACHE is None:
        with open(_TA_PROMPT_PATH, "r") as f:
            _TA_PROMPT_CACHE = f.read()
    return _TA_PROMPT_CACHE


def _tool_starcoder(
    query: str = "",
    mode: str = "chat",
    code_prefix: str = "",
    code_suffix: str = "",
    max_tokens: int = 512,
) -> str:
    """
    Call the local starcoder:latest model via Ollama (port 11434).

    mode="chat"     — Technical Assistant style: uses TA_prompt_v1.txt as a few-shot
                      prefix, then appends 'Human: {query}\\n\\nAssistant:' and completes.
    mode="complete" — Fill-in-the-Middle (FIM): uses code_prefix and code_suffix with
                      StarCoder's <fim_prefix>/<fim_suffix>/<fim_middle> tokens, or
                      plain prefix completion if no suffix is given.
    """
    import httpx

    ollama_base = _ensure_protocol(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

    if mode == "chat":
        ta_prompt = _load_ta_prompt()
        # The TA prompt already ends with '-----\n\n'; append the new exchange.
        prompt = ta_prompt.rstrip() + f"\n\nHuman: {query}\n\nAssistant:"
    else:
        # FIM / completion mode
        if code_suffix:
            prompt = f"<fim_prefix>{code_prefix}<fim_suffix>{code_suffix}<fim_middle>"
        else:
            prompt = code_prefix

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{ollama_base}/api/generate",
                json={
                    "model": "starcoder:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.2,
                        "stop": ["\nHuman:", "\n-----"],
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
    except Exception as e:
        return f"[error calling starcoder] {e}"


# ---------------------------------------------------------------------------
# URL fetch helpers (used after web_search)
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    _SKIP_TAGS = {"script", "style", "nav", "head", "noscript", "footer", "iframe"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.chunks.append(text)


def _extract_text(html: str) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return " ".join(extractor.chunks)


async def _fetch_url_text(url: str, max_chars: int = 8000) -> str:
    try:
        # Ensure URL has a protocol
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        headers = {"User-Agent": "ApertusProxy/1.0 (research bot)"}
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            text = _extract_text(resp.text) if "html" in content_type else resp.text
            return text[:max_chars]
    except Exception as e:
        return f"[fetch error: {e}]"


async def _decide_fetch_urls(api_key: str, search_results: str, query: str) -> list[str]:
    system = (
        "You are a research assistant. Given web search results and a user query, "
        "decide which URLs (if any) are worth fetching in full for a more detailed answer. "
        "Output ONLY a JSON array of up to 2 URL strings (e.g. [\"https://...\", \"https://...\"]). "
        "If the snippets are sufficient, output an empty array: []. "
        "No explanation, no markdown fences — only the JSON array."
    )
    prompt = f"Query: {query}\n\nSearch results:\n{search_results}"
    raw = await _complete(
        api_key,
        system,
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=128,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    try:
        urls = json.loads(raw)
        if isinstance(urls, list):
            return [u for u in urls if isinstance(u, str)]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


TOOL_REGISTRY = {
    "web_search": _tool_web_search,
    "wikipedia_search": _tool_wikipedia_search,
    "python_repl": _tool_python_repl,
    "starcoder": _tool_starcoder,
}

TOOL_DESCRIPTIONS = {
    "web_search": {
        "description": (
            "Search the web for current events, recent news, or any topic not likely "
            "in training data. Returns titles, URLs, and snippets."
        ),
        "parameters": {
            "query": "string — the search query",
            "max_results": "integer (optional, default 10) — number of results to return",
        },
    },
    "wikipedia_search": {
        "description": (
            "Look up a well-known topic, concept, person, or place on Wikipedia. "
            "Best for factual background information. Returns a summary and URL."
        ),
        "parameters": {
            "topic": "string — the Wikipedia article title to look up",
            "language": "string (optional, default 'en') — language code",
        },
    },
    "python_repl": {
        "description": (
            "Execute Python code and return the output. Use for calculations, "
            "data transformations, testing snippets, or anything that benefits "
            "from running real code rather than reasoning about it."
        ),
        "parameters": {
            "code": "string — valid Python code to execute",
            "timeout": "integer (optional, default 10) — max seconds to allow",
        },
    },
    "starcoder": {
        "description": (
            "Delegate a coding task to the local StarCoder model. "
            "Use mode='chat' for Q&A, debugging, explanation, or code generation from a description. "
            "Use mode='complete' for fill-in-the-middle or prefix completion of existing code."
        ),
        "parameters": {
            "query": "string — the coding question or instruction (for mode='chat')",
            "mode": "'chat' (default) or 'complete'",
            "code_prefix": "string — code before the insertion point (for mode='complete')",
            "code_suffix": "string (optional) — code after the insertion point for FIM completion",
            "max_tokens": "integer (optional, default 512) — maximum tokens to generate",
        },
    },
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PUBLICAI_BASE = "https://api.publicai.co/v1"
UPSTREAM_MODEL = "swiss-ai/apertus-70b-instruct"
OLLAMA_MODEL_NAME = "apertus:70b"

CHARTER = """\
You are Apertus, an open, capable, and honest AI assistant developed as part of the \
Swiss AI initiative. Your core commitments:

- Truthfulness: be accurate; acknowledge uncertainty rather than confabulating
- Helpfulness: focus on what the user actually needs
- Clarity: prefer concise, well-structured responses
- Tool use: use the right tool for the job —
    web_search for recent/time-sensitive information,
    wikipedia_search for encyclopedic background,
    python_repl for calculations or verifying code snippets,
    starcoder for coding tasks (generation, debugging, explanation, completion)
- Autonomy: respect the user's goals and do not over-explain or moralise
"""


def _charter() -> str:
    now = datetime.now(timezone.utc).strftime("%A, %d %B %Y %H:%M UTC")
    return f"Current date and time: {now}\n\n" + CHARTER


# ---------------------------------------------------------------------------
# PublicAI client
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.environ.get("PUBLICAI_API_KEY", "")
    if not key:
        raise HTTPException(status_code=500, detail="PUBLICAI_API_KEY env var not set")
    return key


async def _complete(
    api_key: str,
    system: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Non-streaming call to PublicAI; returns assistant text."""
    full_messages = [{"role": "system", "content": system}] + messages
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "ApertusProxy/1.0",
        "Content-Type": "application/json",
    }
    payload = {
        "model": UPSTREAM_MODEL,
        "messages": full_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{PUBLICAI_BASE}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def _stream_complete(
    api_key: str,
    system: str,
    messages: list[dict],
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> AsyncIterator[str]:
    """Streaming call; yields text deltas."""
    full_messages = [{"role": "system", "content": system}] + messages
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "ApertusProxy/1.0",
        "Content-Type": "application/json",
    }
    payload = {
        "model": UPSTREAM_MODEL,
        "messages": full_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", f"{PUBLICAI_BASE}/chat/completions", headers=headers, json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

# ---------------------------------------------------------------------------
# Agentic pipeline
# ---------------------------------------------------------------------------

def _format_conversation(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m["role"].upper()
        content = m.get("content") or ""
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


async def run_pipeline(
    api_key: str,
    messages: list[dict],
    stream: bool = False,
) -> tuple[str | None, AsyncIterator[str] | None]:
    """
    Run the full reasoning → decision → tool → respond pipeline.

    Returns (full_text, None) if stream=False,
            (None, async_iterator) if stream=True.
    """
    # Filter out any existing system messages from the user conversation;
    # we'll inject our own.
    user_messages = [m for m in messages if m.get("role") != "system"]
    last_user_msg = next(
        (m["content"] for m in reversed(user_messages) if m["role"] == "user"), ""
    )
    conv_str = _format_conversation(user_messages)
    tools_str = json.dumps(TOOL_DESCRIPTIONS, indent=2)

    # ------------------------------------------------------------------
    # Step 1: Reasoning self-dialogue
    # ------------------------------------------------------------------
    reasoning_system = f"""{_charter()}
## Your tools
{tools_str}

## Task
You are engaging in an internal reasoning dialogue BEFORE replying to the user.
Play both sides of a short conversation (2-4 exchanges) between an inner USER \
(who asks clarifying questions about the task) and an inner ASSISTANT (who reasons \
through the best approach, including whether any tool should be called and why).

Label each turn clearly:
  INNER USER: ...
  INNER ASSISTANT: ...

The real conversation so far:
{conv_str}
"""
    reasoning = await _complete(
        api_key,
        reasoning_system,
        [{"role": "user", "content": f"Think through how to best respond to: {last_user_msg}"}],
        temperature=0.7,
        max_tokens=1024,
    )
    print("\n" + "="*60, file=sys.stderr)
    print("[pipeline] REASONING:", file=sys.stderr)
    print(reasoning, file=sys.stderr)
    print("="*60, file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 2: Tool call decision
    # ------------------------------------------------------------------
    decision_system = f"""{_charter()}
## Your tools
{tools_str}

## Your reasoning so far
{reasoning}

## Task
Decide: should a tool be called, or should you respond directly?

- If a tool should be called, output ONLY valid JSON with exactly these keys:
  {{"tool": "<tool_name>", "args": {{<key>: <value>, ...}}}}
- If no tool is needed, output ONLY the single word: DIRECT

No explanation. No markdown fences. Output only the JSON object or the word DIRECT.
"""
    decision_raw = await _complete(
        api_key,
        decision_system,
        [{"role": "user", "content": "Tool call or direct response?"}],
        temperature=0.1,
        max_tokens=256,
    )
    decision = decision_raw.strip()
    print(f"\n[pipeline] TOOL DECISION: {decision}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 3: Execute tool (if decided)
    # ------------------------------------------------------------------
    tool_block = ""
    executed_tool = ""
    tool_result = ""
    # Strip markdown code fences if the model wrapped the JSON
    if decision.startswith("```"):
        decision = "\n".join(
            line for line in decision.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    if decision.upper() != "DIRECT" and decision.startswith("{"):
        try:
            call = json.loads(decision)
            tool_name = call.get("tool", "")
            tool_args = call.get("args", {})
            if tool_name in TOOL_REGISTRY:
                print(f"[pipeline] Calling tool: {tool_name}({tool_args})", file=sys.stderr)
                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: TOOL_REGISTRY[tool_name](**tool_args)
                )
                executed_tool = tool_name
                tool_result = result
                tool_block = (
                    f"\n\n## Tool results: {tool_name}({tool_args})\n"
                    f"```\n{result}\n```"
                )
            else:
                print(f"[pipeline] Unknown tool requested: {tool_name}", file=sys.stderr)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[pipeline] Decision parse error: {e} — treating as DIRECT", file=sys.stderr)

    # ── Step 3b: fetch URLs if web_search was used ───────────────────────
    if executed_tool == "web_search" and tool_result:
        fetch_urls = await _decide_fetch_urls(api_key, tool_result, last_user_msg)
        for url in fetch_urls:
            page_text = await _fetch_url_text(url)
            if not page_text.startswith("[fetch error"):
                summary = await _complete(
                    api_key,
                    f"Summarise the following web page, focusing on what is relevant "
                    f"to the query: {last_user_msg!r}\nBe concise (200–300 words).",
                    [{"role": "user", "content": page_text}],
                    temperature=0.3,
                    max_tokens=512,
                )
                tool_block += f"\n\n## Page summary: {url}\n{summary}"

    # ------------------------------------------------------------------
    # Step 4: Final response
    # ------------------------------------------------------------------
    final_system = f"""{_charter()}
## Your reasoning
{reasoning}
{tool_block}

Use the above reasoning (and tool results, if any) to compose your reply. \
Do not repeat your reasoning verbatim. Just give a clear, helpful response.

## Web search guidance
If web_search results are present above, apply these rules:
- Check whether snippets or fetched page summaries mention a publication or event date.
  Compare it against today's date shown at the top of this prompt.
- For time-sensitive queries (recent events, current status, latest news): if you cannot
  confirm the results are current, say so explicitly — e.g. "As of [date in results], …"
  or "I found results from [date] but cannot confirm this is still current."
- Never state time-sensitive information as confirmed current fact if the evidence is
  undated or older than a few weeks relative to today.
- If the search returned no relevant or recent results, say so rather than speculating.
"""
    if stream:
        return None, _stream_complete(api_key, final_system, user_messages)
    else:
        text = await _complete(api_key, final_system, user_messages, temperature=0.8, max_tokens=4096)
        return text, None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Apertus 70B Ollama Proxy")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@app.get("/api/tags")
async def list_tags():
    """Ollama GET /api/tags — returns available models."""
    return {
        "models": [
            {
                "name": OLLAMA_MODEL_NAME,
                "model": OLLAMA_MODEL_NAME,
                "modified_at": _now_iso(),
                "size": 70_000_000_000,
                "digest": "sha256:apertus70b000000000000000000000000000000000000000000000000000000",
                "details": {
                    "parent_model": UPSTREAM_MODEL,
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "70B",
                    "quantization_level": "Q4_K_M",
                },
            }
        ]
    }


@app.post("/api/chat")
async def chat(request: Request):
    """Ollama POST /api/chat"""
    body = await request.json()
    api_key = _get_api_key()
    messages: list[dict] = body.get("messages", [])
    do_stream: bool = body.get("stream", True)

    if not messages:
        raise HTTPException(status_code=400, detail="messages required")

    created_at = _now_iso()

    if do_stream:
        full_text, stream_iter = await run_pipeline(api_key, messages, stream=True)

        async def _ollama_stream():
            async for delta in stream_iter:
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

        return StreamingResponse(_ollama_stream(), media_type="application/x-ndjson")
    else:
        full_text, _ = await run_pipeline(api_key, messages, stream=False)
        return JSONResponse({
            "model": OLLAMA_MODEL_NAME,
            "created_at": created_at,
            "message": {"role": "assistant", "content": full_text},
            "done": True,
            "done_reason": "stop",
        })


@app.post("/api/generate")
async def generate(request: Request):
    """Ollama POST /api/generate (legacy single-turn format)"""
    body = await request.json()
    api_key = _get_api_key()
    prompt: str = body.get("prompt", "")
    system: str = body.get("system", "")
    do_stream: bool = body.get("stream", True)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    created_at = _now_iso()

    if do_stream:
        _, stream_iter = await run_pipeline(api_key, messages, stream=True)

        async def _ollama_gen_stream():
            async for delta in stream_iter:
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

        return StreamingResponse(_ollama_gen_stream(), media_type="application/x-ndjson")
    else:
        full_text, _ = await run_pipeline(api_key, messages, stream=False)
        return JSONResponse({
            "model": OLLAMA_MODEL_NAME,
            "created_at": created_at,
            "response": full_text,
            "done": True,
        })


@app.get("/api/version")
async def version():
    return {"version": "0.1.0"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 11435))
    print(f"Starting Apertus 70B proxy on port {port}", file=sys.stderr)
    print(f"Upstream model: {UPSTREAM_MODEL}", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=port)
