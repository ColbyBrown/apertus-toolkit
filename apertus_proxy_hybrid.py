"""
Hybrid Ollama-compatible proxy for Apertus.

Reasoning + tool use → local apertus-tulu-xlam:8b via Ollama  [fine-tuned for this task]
Final response draft → PublicAI (swiss-ai/apertus-70b-instruct) [high-quality generation]

Pipeline:
  1. Reasoning self-dialogue   → local 8B
  2. Tool-call decision        → local 8B
  3. Tool execution            → Python (no LLM)
     3a. URL-fetch decision    → local 8B
     3b. Page summarisation    → local 8B
  4. Final response            → PublicAI 70B (streaming or non-streaming)

Usage:
    PUBLICAI_API_KEY=your_key python apertus_proxy_local.py

Listens on port 11436. Compatible with any Ollama client.
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
# Tool implementations
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
    ollama_base = _ensure_protocol(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

    if mode == "chat":
        ta_prompt = _load_ta_prompt()
        prompt = ta_prompt.rstrip() + f"\n\nHuman: {query}\n\nAssistant:"
    else:
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
# URL fetch helpers
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


TOOL_REGISTRY = {
    "web_search": _tool_web_search,
    "wikipedia_search": _tool_wikipedia_search,
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
UPSTREAM_MODEL = "swiss-ai/apertus-70b-instruct"   # used for final response 
LOCAL_MODEL = "qwen3:8b"                           # used for reasoning
OLLAMA_MODEL_NAME = "apertus-hybrid"             # name advertised to Ollama clients

CHARTER = """\
You are Apertus, an open, capable, and honest AI assistant developed as part of the \
Swiss AI initiative. Your core commitments:

- Truthfulness: be accurate; acknowledge uncertainty rather than confabulating
- Helpfulness: focus on what the user actually needs
- Format: plain prose only — no markdown headers, bullet lists, or bold/italic text \
unless the user explicitly asks for formatted output or a list. Write in flowing sentences.
- Length: answer in as few words as the question requires. Do not pad responses with \
context the user did not ask for. If a one-sentence answer suffices, use one sentence.
- Tool use: use the right tool for the job —
    web_search for recent/time-sensitive information,
    wikipedia_search for encyclopedic background,
    starcoder for coding tasks (generation, debugging, explanation, completion)
- Autonomy: respect the user's goals and do not over-explain or moralise

Example of incorrect format: "Here are the key points:\\n- Point one\\n- Point two"
Example of correct format: "Point one. Point two."
"""


def _charter() -> str:
    now = datetime.now(timezone.utc).strftime("%A, %d %B %Y %H:%M UTC")
    return f"Current date and time: {now}\n\n" + CHARTER


# ---------------------------------------------------------------------------
# Backend: PublicAI
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.environ.get("PUBLICAI_API_KEY", "")
    if not key:
        raise HTTPException(status_code=500, detail="PUBLICAI_API_KEY env var not set")
    return key


def _truncate_on_repetition(text: str, window: int = 5) -> str:
    """
    Detect paragraph-level repetition and truncate at the first repeated block.
    Splits on double-newlines; if any paragraph seen before reappears, stop there.
    Falls back to sentence-level detection using a sliding window.
    """
    # Paragraph-level check
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    seen: set[str] = set()
    for i, para in enumerate(paragraphs):
        if para in seen:
            truncated = "\n\n".join(paragraphs[:i])
            print(f"[pipeline] Repetition detected at paragraph {i} — truncating", file=sys.stderr)
            return truncated
        seen.add(para)

    # Sentence-level sliding-window check
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > window * 2:
        for i in range(window, len(sentences)):
            recent = tuple(sentences[i - window:i])
            for j in range(i - window):
                if tuple(sentences[j:j + window]) == recent:
                    truncated = " ".join(sentences[:i - window + 1])
                    print(f"[pipeline] Sentence-level repetition at index {i} — truncating", file=sys.stderr)
                    return truncated
    return text


async def _complete_remote(
    api_key: str,
    system: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Non-streaming call to PublicAI (used for the final response step)."""
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
        "frequency_penalty": 0.4,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{PUBLICAI_BASE}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return _truncate_on_repetition(data["choices"][0]["message"]["content"])


async def _stream_complete_remote(
    api_key: str,
    system: str,
    messages: list[dict],
    temperature: float = 0.4,
    max_tokens: int = 4096,
) -> AsyncIterator[str]:
    """Streaming call to PublicAI (used for the final response step); yields text deltas."""
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
        "frequency_penalty": 0.4,
        "stream": True,
    }
    accumulated = ""
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
                    if not delta:
                        continue
                    accumulated += delta
                    # Check for paragraph-level repetition every ~200 chars
                    if len(accumulated) % 200 < len(delta):
                        checked = _truncate_on_repetition(accumulated)
                        if checked != accumulated:
                            # Yield only the newly truncated portion then stop
                            already_yielded = len(accumulated) - len(delta)
                            tail = checked[already_yielded:]
                            if tail:
                                yield tail
                            return
                    yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


# ---------------------------------------------------------------------------
# Backend: local Ollama (all pre-response steps)
# ---------------------------------------------------------------------------

def _ollama_base() -> str:
    return _ensure_protocol(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))


def _strip_think(text: str) -> str:
    """Print <think>...</think> blocks to stderr, then remove them from the returned text."""
    import re
    blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    for block in blocks:
        content = block.strip()
        if content:
            print(f"[think]\n{content}\n[/think]", file=sys.stderr)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def _complete_local(
    system: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Non-streaming call to the local model."""
    full_messages = [{"role": "system", "content": system}] + messages
    payload = {
        "model": LOCAL_MODEL,
        "messages": full_messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{_ollama_base()}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return _strip_think(data["message"]["content"])


async def _stream_complete_local(
    system: str,
    messages: list[dict],
    temperature: float = 0.8,
    max_tokens: int = 4096,
) -> AsyncIterator[str]:
    """Streaming call to the local model; yields text deltas."""
    full_messages = [{"role": "system", "content": system}] + messages
    payload = {
        "model": LOCAL_MODEL,
        "messages": full_messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream("POST", f"{_ollama_base()}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break
                except (json.JSONDecodeError, KeyError):
                    continue


# ---------------------------------------------------------------------------
# URL-fetch helpers (use local model for decisions + summarisation)
# ---------------------------------------------------------------------------

async def _decide_fetch_urls(search_results: str, query: str) -> list[str]:
    system = (
        "You are a research assistant. Given web search results and a user query, "
        "decide which URLs (if any) are worth fetching in full for a more detailed answer. "
        "Output ONLY a JSON array of up to 2 URL strings (e.g. [\"https://...\", \"https://...\"]). "
        "If the snippets are sufficient, output an empty array: []. "
        "No explanation, no markdown fences — only the JSON array."
    )
    prompt = f"Query: {query}\n\nSearch results:\n{search_results}"
    raw = await _complete_local(
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
    user_messages = [m for m in messages if m.get("role") != "system"]
    last_user_msg = next(
        (m["content"] for m in reversed(user_messages) if m["role"] == "user"), ""
    )
    tools_str = json.dumps(TOOL_DESCRIPTIONS, indent=2)

    # ------------------------------------------------------------------
    # Step 1: Tool call decision  →  local 8B (native reasoning)
    # ------------------------------------------------------------------
    decision_system = f"""{_charter()}
## Your tools
{tools_str}

## Task
You are a routing step. Given the user's latest message, decide whether a tool must be \
called or whether the question can be answered directly from knowledge.

Output rules (follow exactly):
- If a tool is needed: output ONLY valid JSON → {{"tool": "<tool_name>", "args": {{...}}}}
- If no tool is needed: output ONLY the single word → DIRECT

Examples:
  User: What is 2 + 2?
  Output: DIRECT

  User: Who won the 2024 US election?
  Output: {{"tool": "web_search", "args": {{"query": "2024 US election winner"}}}}

  User: Explain recursion.
  Output: DIRECT

No explanation. No prose. No markdown fences. Output only the JSON object or the word DIRECT.

The user's latest message is:
{last_user_msg}"""
    decision_raw = await _complete_local(
        decision_system,
        [{"role": "user", "content": "Tool call or direct response?"}],
        temperature=0.1,
        max_tokens=1024,
    )
    decision = decision_raw.strip()

    # Strip markdown fences
    if decision.startswith("```"):
        decision = "\n".join(
            line for line in decision.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    # Normalise: if first word is "DIRECT" treat the whole response as DIRECT
    first_word = decision.split()[0].upper().rstrip(".,;:") if decision else ""
    if first_word == "DIRECT":
        decision = "DIRECT"

    # If the model buried JSON inside prose, extract the first {...} block
    if decision != "DIRECT" and "{" in decision:
        start = decision.index("{")
        # Find the matching closing brace
        depth, end = 0, -1
        for i, ch in enumerate(decision[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            decision = decision[start:end]

    print(f"\n[pipeline] TOOL DECISION: {decision}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 3: Execute tool (if decided)
    # ------------------------------------------------------------------
    tool_block = ""
    executed_tool = ""
    tool_result = ""
    fetched_urls: list[str] = []

    if decision.upper() != "DIRECT" and decision.startswith("{"):
        try:
            call = json.loads(decision.replace("\n", " ").replace("\r", " "))
            tool_name = call.get("tool", "")
            tool_args = call.get("args", {})
            if tool_name in TOOL_REGISTRY:
                print(f"[pipeline] Calling tool: {tool_name}({tool_args})", file=sys.stderr)
                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: TOOL_REGISTRY[tool_name](**tool_args)
                )
                executed_tool = tool_name
                tool_result = result
                print(f"[pipeline] Tool result:\n{result}", file=sys.stderr)
                tool_block = (
                    f"\n\n## Tool results: {tool_name}({tool_args})\n"
                    f"```\n{result}\n```"
                )
            else:
                print(f"[pipeline] Unknown tool requested: {tool_name}", file=sys.stderr)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[pipeline] Decision parse error: {e} — treating as DIRECT", file=sys.stderr)

    # ── Step 3b: fetch URLs if web_search was used  →  local 8B ─────────
    if executed_tool == "web_search" and tool_result:
        fetch_urls = await _decide_fetch_urls(tool_result, last_user_msg)
        for url in fetch_urls:
            page_text = await _fetch_url_text(url)
            if not page_text.startswith("[fetch error"):
                summary = await _complete_local(
                    f"Summarise the following web page, focusing on what is relevant "
                    f"to the query: {last_user_msg!r}\nBe concise (200–300 words).",
                    [{"role": "user", "content": page_text}],
                    temperature=0.3,
                    max_tokens=512,
                )
                tool_block += f"\n\n## Page summary: {url}\n{summary}"
                fetched_urls.append(url)

    # ------------------------------------------------------------------
    # Step 3: Final response  →  PublicAI 70B
    # ------------------------------------------------------------------
    tool_section = f"\n\n## Tool results\n{tool_block}" if tool_block else ""
    final_system = f"""{_charter()}{tool_section}

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
    source_lines: list[str] = []
    if executed_tool == "wikipedia_search" and tool_result:
        for line in tool_result.splitlines():
            if line.startswith("URL:"):
                wiki_url = line[len("URL:"):].strip()
                source_lines.append(f"- {wiki_url} (Wikipedia)")
                break
        if not source_lines:
            source_lines.append("- *Wikipedia article (URL unavailable)*")
    source_lines.extend(f"- {u}" for u in fetched_urls)

    if source_lines:
        sources_footer = "\n\n---\n**Sources:**\n" + "\n".join(source_lines)
    elif executed_tool:
        sources_footer = "\n\n---\n*Web search used; no individual pages were fetched.*"
    else:
        sources_footer = ""

    if stream:
        base_stream = _stream_complete_remote(api_key, final_system, user_messages)

        async def _stream_with_sources():
            async for delta in base_stream:
                yield delta
            if sources_footer:
                yield sources_footer

        return None, _stream_with_sources()
    else:
        text = await _complete_remote(api_key, final_system, user_messages, temperature=0.4, max_tokens=4096)

        # Format gate: ask local 8B if the response used markdown formatting.
        # If yes, send one corrective retry to the 70B.
        # Completeness gate: check if the response fully answers the question.
        # If not, fetch more information and retry.
        gate_system = (
            "You are a quality checker. Given a user question and an assistant response, "
            "decide whether the response fully and accurately answers the question.\n"
            "If it does, output exactly: {\"complete\": true}\n"
            "If it does not — e.g. the response is vague, says it lacks information, "
            "or could clearly be improved by a quick lookup — output JSON naming the tool to call:\n"
            "  {\"complete\": false, \"tool\": \"web_search\", \"args\": {\"query\": \"...\"}}\n"
            "  or\n"
            "  {\"complete\": false, \"tool\": \"wikipedia_search\", \"args\": {\"topic\": \"...\"}}\n"
            "No explanation. No markdown fences. Output only valid JSON."
        )
        gate_prompt = f"User question: {last_user_msg}\n\nAssistant response:\n{text}"
        gate_raw = await _complete_local(
            gate_system,
            [{"role": "user", "content": gate_prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        gate_raw = gate_raw.strip()
        if gate_raw.startswith("```"):
            gate_raw = "\n".join(
                line for line in gate_raw.splitlines()
                if not line.strip().startswith("```")
            ).strip()
        try:
            gate = json.loads(gate_raw.replace("\n", " ").replace("\r", " "))
        except (json.JSONDecodeError, ValueError):
            gate = {"complete": True}

        if not gate.get("complete", True):
            tool_name = gate.get("tool", "")
            tool_args = gate.get("args", {})
            if tool_name in TOOL_REGISTRY:
                print(f"[pipeline] Completeness gate — fetching more info via {tool_name}({tool_args})", file=sys.stderr)
                extra_result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: TOOL_REGISTRY[tool_name](**tool_args)
                )
                extra_block = f"\n\n## Additional information: {tool_name}({tool_args})\n```\n{extra_result}\n```"
                text = await _complete_remote(api_key, final_system + extra_block, user_messages, temperature=0.4, max_tokens=4096)

        return text + sources_footer, None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Apertus Local Hybrid Proxy")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@app.get("/")
async def root():
    """Ollama GET / — health check."""
    return "Ollama is running"


@app.get("/api/tags")
async def list_tags():
    """Ollama GET /api/tags — returns available models."""
    return {
        "models": [
            {
                "name": OLLAMA_MODEL_NAME,
                "model": OLLAMA_MODEL_NAME,
                "modified_at": _now_iso(),
                "size": 8_000_000_000,
                "digest": "sha256:apertustuluxlam8b00000000000000000000000000000000000000000000000",
                "details": {
                    "parent_model": LOCAL_MODEL,
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "8B",
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
        _, stream_iter = await run_pipeline(api_key, messages, stream=True)

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


@app.post("/api/show")
async def show_model(request: Request):
    """Ollama POST /api/show — returns model details."""
    return JSONResponse({
        "modelfile": f"FROM {LOCAL_MODEL}",
        "parameters": "",
        "template": "",
        "details": {
            "parent_model": LOCAL_MODEL,
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "8B",
            "quantization_level": "Q4_K_M",
        },
        "model_info": {
            "general.architecture": "llama",
            "general.parameter_count": 8_000_000_000,
        },
    })


@app.get("/api/version")
async def version():
    return {"version": "0.1.0"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 11436))
    print(f"Starting Apertus local hybrid proxy on port {port}", file=sys.stderr)
    print(f"Reasoning + tools : {LOCAL_MODEL} (Ollama local)", file=sys.stderr)
    print(f"Final response    : {UPSTREAM_MODEL} (PublicAI)", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=port)
