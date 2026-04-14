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
from transformers import pipeline as _hf_pipeline
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
        with open(_TA_PROMPT_PATH, "r", encoding="utf-8") as f:
            _TA_PROMPT_CACHE = f.read()
    return _TA_PROMPT_CACHE


_CG_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "completeness_prompt_v1.txt")
_CG_PROMPT_CACHE: str | None = None

def _load_cg_prompt() -> str:
    global _CG_PROMPT_CACHE
    if _CG_PROMPT_CACHE is None:
        with open(_CG_PROMPT_PATH, "r", encoding="utf-8") as f:
            _CG_PROMPT_CACHE = f.read()
    return _CG_PROMPT_CACHE


# ---------------------------------------------------------------------------
# MNLI zero-shot classifier — tool routing
# ---------------------------------------------------------------------------
print("[init] Loading MNLI classifier...", file=sys.stderr)
_nli_pipe = _hf_pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=-1,  # CPU
)
print("[init] MNLI classifier ready.", file=sys.stderr)

# (label, tool_name, threshold) — higher threshold = more conservative
_ROUTING_LABELS: list[tuple[str, str, float]] = [
    ("current events, live data, recent news, or time-sensitive information", "web_search",       0.5),
    ("encyclopedic or historical facts, biography, or well-known concepts",   "wikipedia_search",  0.5),
    ("programming, coding, software development, or debugging",               "starcoder",         0.5),
    ("complex reasoning, philosophical or ethical analysis, or strategic tradeoffs", "think",      0.5),
]


def _mnli_decision(last_user_msg: str) -> str:
    """
    Zero-shot NLI routing. Returns a JSON tool-call string or "DIRECT".
    Synchronous — safe to call from run_in_executor.
    """
    if not last_user_msg.strip():
        return "DIRECT"
    _MAX_MSG = 200
    msg = last_user_msg if len(last_user_msg) <= _MAX_MSG else last_user_msg[:_MAX_MSG] + "..."
    candidate_labels = [label for label, _, _ in _ROUTING_LABELS]
    result = _nli_pipe(msg, candidate_labels=candidate_labels, multi_label=False)

    top_label = result["labels"][0]
    top_score = result["scores"][0]

    for label, tool_name, threshold in _ROUTING_LABELS:
        if label == top_label and top_score >= threshold:
            if tool_name == "web_search":
                args: dict = {"query": msg}
            elif tool_name == "wikipedia_search":
                args = {"topic": msg}
            elif tool_name == "starcoder":
                args = {"query": msg, "mode": "chat"}
            else:  # think
                args = {"query": msg}
            print(f"[pipeline] MNLI → {tool_name} (score={top_score:.2f})", file=sys.stderr)
            return json.dumps({"tool": tool_name, "args": args})

    print(f"[pipeline] MNLI → DIRECT (top={top_label!r}, score={top_score:.2f})", file=sys.stderr)
    return "DIRECT"


def _nli_pick_urls(query: str, search_results: str, max_urls: int = 2, threshold: float = 0.50) -> list[str]:
    """
    Score each search result's (title + snippet) against the query using NLI entailment.
    Returns up to max_urls URLs whose snippets score above the relevance threshold,
    sorted by score descending. Synchronous — safe to call from run_in_executor.
    """
    candidates: list[tuple[str, str]] = []  # (url, scoring_text)
    current: dict = {}
    for line in search_results.splitlines():
        if line.startswith("Title:"):
            current["title"] = line[6:].strip()
        elif line.startswith("URL:"):
            current["url"] = line[4:].strip()
        elif line.startswith("Snippet:"):
            current["snippet"] = line[8:].strip()
        elif not line.strip() and "url" in current and "snippet" in current:
            text = f"{current.get('title', '')}. {current['snippet']}"[:400]
            candidates.append((current["url"], text))
            current = {}
    if "url" in current and "snippet" in current:  # flush last block
        text = f"{current.get('title', '')}. {current['snippet']}"[:400]
        candidates.append((current["url"], text))

    if not candidates:
        return []

    scored: list[tuple[float, str]] = []
    for url, text in candidates:
        try:
            result = _nli_pipe(
                text,
                candidate_labels=[query],
                hypothesis_template="This page contains information relevant to: {}",
            )
            score = result["scores"][0]
            print(f"[pick_urls] {score:.2f} {url}", file=sys.stderr)
            if score >= threshold:
                scored.append((score, url))
        except Exception as e:
            print(f"[pick_urls] NLI error for {url}: {e}", file=sys.stderr)

    scored.sort(reverse=True)
    return [url for _, url in scored[:max_urls]]


def _starcoder_check_complete(question: str, response: str) -> dict:
    """
    Use StarCoder + completeness_prompt_v1.txt to check whether the 70B response
    fully answered the question. Returns a dict with key 'complete' (bool) and
    optionally 'tool' and 'args' for a follow-up lookup.
    """
    ollama_base = _ensure_protocol(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    cg_prompt = _load_cg_prompt()
    _MAX_Q, _MAX_R = 200, 500
    rq = question if len(question) <= _MAX_Q else question[:_MAX_Q] + "..."
    rr = response if len(response) <= _MAX_R else response[:_MAX_R] + "..."
    call_line = f"check_complete({json.dumps(rq)}, {json.dumps(rr)})\n# Returns:"
    prompt = cg_prompt.rstrip() + "\n\n" + call_line
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{ollama_base}/api/generate",
                json={
                    "model": "starcoder:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 128,
                        "temperature": 0.0,
                        "stop": ["\ncheck_complete(", "\n\n"],
                    },
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip().replace("\n", " ").replace("\r", " ")
            # Extract first {...} block — StarCoder may append trailing text
            if "{" in raw:
                start = raw.index("{")
                depth, end = 0, -1
                for i, ch in enumerate(raw[start:], start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end != -1:
                    raw = raw[start:end]
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
    except Exception as e:
        print(f"[pipeline] StarCoder check_complete error: {e} — assuming complete", file=sys.stderr)
    return {"complete": True}


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


THINK_TURNS = 2  # number of 8B turns; 70B turns = THINK_TURNS - 1


def _sync_call_70b(system: str, messages: list[dict], max_tokens: int = 512) -> str:
    """Synchronous 70B call via PublicAI. For use inside thread-pool-executed tools."""
    api_key = os.environ.get("PUBLICAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("PUBLICAI_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "ApertusProxy/1.0",
        "Content-Type": "application/json",
    }
    payload = {
        "model": UPSTREAM_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{PUBLICAI_BASE}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def _sync_call_8b(system: str, messages: list[dict], max_tokens: int = 256) -> str:
    """Synchronous local 8B call via Ollama. For use inside thread-pool-executed tools."""
    ollama_base = _ensure_protocol(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    payload = {
        "model": LOCAL_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
        "options": {"temperature": 0.8, "top_p": 0.9, "num_predict": max_tokens},
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{ollama_base}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


def _tool_think(query: str, conversation: str = "", base_system: str = "") -> dict:
    """
    Collaborative reasoning via structured 8B ↔ 70B dialogue.

    1. Summarize the user-assistant conversation (70B)
    2. Run THINK_TURNS exchanges between 8B and 70B
    3. Summarize the discussion transcript (70B)
    4. Return the summary as context for the 8B's final response

    Synchronous — runs in TOOL_REGISTRY's executor pattern.
    """
    try:
        # ── Step 1: Summarize the dialogue ────────────────────────────────
        if conversation:
            dialogue_summary = _sync_call_70b(
                "Summarize the following conversation in 1-3 sentences, focusing on "
                "what the user is trying to understand or decide.",
                [{"role": "user", "content": conversation}],
                max_tokens=128,
            )
        else:
            dialogue_summary = query
        print(f"[think] Dialogue summary: {dialogue_summary}", file=sys.stderr)

        # ── Step 1b: Pre-fetch grounding context for the 8B only ─────────
        grounding_block = ""
        g_tool = ""
        g_result = ""
        try:
            g_decision = _mnli_decision(dialogue_summary)
            if g_decision != "DIRECT" and g_decision.startswith("{"):
                g_call = json.loads(g_decision.replace("\n", " ").replace("\r", " "))
                _gt = g_call.get("tool", "")
                g_args = g_call.get("args", {})
                if _gt in ("web_search", "wikipedia_search"):
                    g_tool = _gt
                    g_result = TOOL_REGISTRY[g_tool](**g_args)
                    grounding_block = (
                        f"\n\nBackground information (use this to ground your reasoning):\n"
                        f"```\n{g_result}\n```"
                    )
                    print(f"[think] Pre-fetched {g_tool} for 8B grounding", file=sys.stderr)
        except Exception as e:
            print(f"[think] Grounding fetch error: {e}", file=sys.stderr)

        # ── Step 2: Discussion loop ───────────────────────────────────────
        _persona = base_system if base_system else _charter()
        system_8b = (
            f"{_persona}\n\n"
            f"You are now in an internal reasoning dialogue with a larger version of yourself "
            f"(Apertus 70B) before composing your final response to the user. "
            f"Respond in exactly 2 sentences. Stay strictly within your actual capabilities — "
            f"do not suggest or offer actions you cannot perform.\n"
            f"Discussion topic: {dialogue_summary}"
            f"{grounding_block}"
        )
        system_70b = (
            f"You are engaged in a collaborative reasoning discussion. "
            f"Respond to the other participant's points in exactly 2 sentences.\n"
            f"Topic: {dialogue_summary}"
        )

        history_8b: list[dict] = [{"role": "user", "content": "Begin the discussion."}]
        history_70b: list[dict] = []
        turns: list[tuple[str, str]] = []  # (speaker, text)

        for i in range(THINK_TURNS):
            # 8B turn
            text_8b = _sync_call_8b(system_8b, history_8b, max_tokens=80)
            print(f"[think] 8B: {text_8b}", file=sys.stderr)
            turns.append(("8B", text_8b))
            history_8b.append({"role": "assistant", "content": text_8b})

            if i < THINK_TURNS - 1:
                # 70B turn (not after the last 8B turn)
                history_70b.append({"role": "user", "content": text_8b})
                text_70b = _sync_call_70b(system_70b, history_70b, max_tokens=120)
                print(f"[think] 70B: {text_70b}", file=sys.stderr)
                turns.append(("70B", text_70b))
                history_70b.append({"role": "assistant", "content": text_70b})
                history_8b.append({"role": "user", "content": text_70b})

        # ── Step 3: Summarize the discussion ─────────────────────────────
        transcript = "\n".join(f"{speaker}: {text}" for speaker, text in turns)
        summary = _sync_call_70b(
            "Summarize the key insights and conclusions from this discussion in 2-4 sentences.",
            [{"role": "user", "content": transcript}],
            max_tokens=256,
        )
        print(f"[think] Summary: {summary}", file=sys.stderr)
        return {"summary": summary, "grounding_tool": g_tool, "grounding_result": g_result}

    except Exception as e:
        print(f"[think] Error: {e}", file=sys.stderr)
        return {"summary": f"[think error] {e}", "grounding_tool": "", "grounding_result": ""}


TOOL_REGISTRY = {
    "web_search": _tool_web_search,
    "wikipedia_search": _tool_wikipedia_search,
    "starcoder": _tool_starcoder,
    "think": _tool_think,
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
    "think": {
        "description": (
            "Invoke an extended reasoning pass on a complex, open-ended, or multi-step question. "
            "Use for philosophical, ethical, strategic, or analytical questions where deliberation "
            "clearly adds value. Do not use for factual lookups, coding tasks, or simple questions."
        ),
        "parameters": {
            "query": "string — the question or topic to reason about",
        },
    },
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PUBLICAI_BASE = "https://api.publicai.co/v1"
UPSTREAM_MODEL = "swiss-ai/apertus-70b-instruct"   # used by think tool + URL summarisation
LOCAL_MODEL = "MichelRosselli/apertus"          # local model for final responses
OLLAMA_MODEL_NAME = "apertus-hybrid"               # name advertised to Ollama clients

PERSONA = """\
You are Apertus, an open, capable, and honest AI assistant developed as part of the \
Swiss AI initiative. Your core commitments:

- Truthfulness: be accurate; acknowledge uncertainty rather than confabulating
- Helpfulness: focus on what the user actually needs\
"""

FORMAT_RULES = """\
- Brevity (highest priority): be as short as possible. One sentence is better than two. \
Never restate the question. Never explain what you are about to do — just do it. \
Never add caveats, disclaimers, or context the user did not ask for.
- Format: plain prose only. No markdown headers, bullet lists, or bold/italic text \
unless the user explicitly requests them. No preamble ("Great question!", "Sure!", \
"Of course!"). No closing summary or sign-off.

Bad: "That's a great question! The boiling point of water is 100°C at standard atmospheric pressure. I hope that helps!"
Good: "100°C at standard pressure."

Bad: "Here are the key points: first, X. Second, Y. In summary, X and Y matter."
Good: "X. Y."\
"""

CHARTER = PERSONA + "\n\n" + FORMAT_RULES


def _charter() -> str:
    now = datetime.now(timezone.utc).strftime("%A, %d %B %Y %H:%M UTC")
    return f"Current date and time: {now}\n\n{CHARTER}"


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
    temperature: float = 0.8,
    top_p: float = 0.9,
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
            "top_p": top_p,
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
    top_p: float = 0.9,
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
            "top_p": top_p,
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
    return await asyncio.get_running_loop().run_in_executor(
        None, lambda: _nli_pick_urls(query, search_results)
    )


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
    think_enabled: bool = True,
) -> tuple[str | None, AsyncIterator | None, str]:
    """
    Run the full reasoning → decision → tool → respond pipeline.

    Returns (full_text, None) if stream=False,
            (None, async_iterator) if stream=True.
    """
    client_system = next(
        (m["content"] for m in messages if m.get("role") == "system"), ""
    )
    user_messages = [m for m in messages if m.get("role") != "system"]
    last_user_msg = next(
        (m["content"] for m in reversed(user_messages) if m["role"] == "user"), ""
    )
    # ------------------------------------------------------------------
    # Step 1: Tool call decision  →  StarCoder (in-context learning)
    # ------------------------------------------------------------------
    decision_raw = await asyncio.get_running_loop().run_in_executor(
        None, lambda: _mnli_decision(last_user_msg)
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
    # Step 3: Parse pending tool (execution deferred to step 4)
    # ------------------------------------------------------------------
    pending_tool_name = ""
    pending_tool_args: dict = {}

    if decision.upper() != "DIRECT" and decision.startswith("{"):
        try:
            call = json.loads(decision.replace("\n", " ").replace("\r", " "))
            _ptn = call.get("tool", "")
            if _ptn in TOOL_REGISTRY:
                pending_tool_name = _ptn
                pending_tool_args = call.get("args", {})
                if pending_tool_name == "think" and not think_enabled:
                    print("[pipeline] think disabled by client — treating as DIRECT", file=sys.stderr)
                    pending_tool_name = ""
                    pending_tool_args = {}
            else:
                print(f"[pipeline] Unknown tool requested: {_ptn}", file=sys.stderr)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[pipeline] Decision parse error: {e} — treating as DIRECT", file=sys.stderr)

    # ------------------------------------------------------------------
    # Helper: execute the pending tool and build context strings
    # ------------------------------------------------------------------
    async def _execute_tool_and_build_context():
        """Run the pending tool (if any) and return (executed_tool, tool_result,
        tool_block, fetched_urls, sources_footer, final_system)."""
        executed_tool = ""
        tool_result = ""
        tool_block = ""
        fetched_urls: list[str] = []
        grounding_tool = ""
        grounding_result = ""

        if pending_tool_name:
            t_args = {**pending_tool_args}
            if pending_tool_name == "think":
                t_args["conversation"] = _format_conversation(user_messages)
                t_args["base_system"] = client_system if client_system else _charter()
            print(f"[pipeline] Calling tool: {pending_tool_name}({t_args})", file=sys.stderr)
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: TOOL_REGISTRY[pending_tool_name](**t_args)
            )
            executed_tool = pending_tool_name

            # think returns a dict; extract summary and grounding metadata
            if pending_tool_name == "think" and isinstance(result, dict):
                grounding_tool = result.get("grounding_tool", "")
                grounding_result = result.get("grounding_result", "")
                tool_result = result.get("summary", "")
            else:
                grounding_tool = ""
                grounding_result = ""
                tool_result = result

            print(f"[pipeline] Tool result:\n{tool_result}", file=sys.stderr)
            tool_block = (
                f"\n\n## Tool results: {pending_tool_name}({t_args})\n"
                f"```\n{tool_result}\n```"
            )

        # URL fetch for web_search
        if executed_tool == "web_search" and tool_result:
            fetch_urls = await _decide_fetch_urls(tool_result, last_user_msg)
            for url in fetch_urls:
                try:
                    page_text = await _fetch_url_text(url)
                    if not page_text.startswith("[fetch error"):
                        summary = await _complete_remote(
                            api_key,
                            f"Summarise the following web page, focusing on what is relevant "
                            f"to the query: {last_user_msg!r}\nBe concise (200–300 words).",
                            [{"role": "user", "content": page_text}],
                            temperature=0.3,
                            max_tokens=512,
                        )
                        tool_block += f"\n\n## Page summary: {url}\n{summary}"
                        fetched_urls.append(url)
                except Exception as e:
                    print(f"[pipeline] Page summarisation error for {url}: {e}", file=sys.stderr)

        # Build final_system
        tool_section = tool_block if tool_block else ""
        base_system = client_system if client_system else _charter()
        think_instruction = (
            "\n\nDo not reference the reasoning process, the discussion, or any internal"
            " analysis in your reply. Answer naturally and directly as if the context"
            " above is simply what you know."
            if executed_tool == "think" else ""
        )
        final_system = f"""{base_system}{tool_section}{think_instruction}

## Web search guidance
If web_search results are present above, apply these rules:
- When your response draws on a fetched page summary (sections labelled "Page summary: <url>"),
  include that URL inline in your reply so the user can follow up. For example: "According to
  [example.com/article](https://example.com/article), …"
- Check whether snippets or fetched page summaries mention a publication or event date.
  Compare it against today's date shown at the top of this prompt.
- For time-sensitive queries (recent events, current status, latest news): if you cannot
  confirm the results are current, say so explicitly — e.g. "As of [date in results], …"
  or "I found results from [date] but cannot confirm this is still current."
- Never state time-sensitive information as confirmed current fact if the evidence is
  undated or older than a few weeks relative to today.
- If the search returned no relevant or recent results, say so rather than speculating.
"""

        # Build sources footer — only list pages actually fetched, not all search results
        source_lines: list[str] = []

        if executed_tool == "web_search":
            # Only show URLs that were fetched in full; snippet-only search gets a note
            source_lines = [f"- {u}" for u in fetched_urls]
        elif executed_tool == "wikipedia_search" and tool_result:
            for line in tool_result.splitlines():
                if line.startswith("URL:"):
                    source_lines.append(f"- {line[len('URL:'):].strip()} (Wikipedia)")
                    break
            if not source_lines:
                source_lines.append("- *Wikipedia article (URL unavailable)*")

        # Sources from think's grounding fetch
        if grounding_tool == "wikipedia_search" and grounding_result:
            for line in grounding_result.splitlines():
                if line.startswith("URL:"):
                    source_lines.append(f"- {line[len('URL:'):].strip()} (Wikipedia)")
                    break
            if not any("Wikipedia" in s for s in source_lines):
                source_lines.append("- *Wikipedia article (URL unavailable)*")
        # grounding web_search: no individual pages fetched, note it if nothing else listed
        grounding_search_used = grounding_tool == "web_search" and grounding_result

        if source_lines:
            sources_footer = "\n\n---\n**Sources:**\n" + "\n".join(source_lines)
        elif executed_tool == "web_search" or grounding_search_used:
            sources_footer = "\n\n---\n*Web search used; no individual pages were fetched.*"
        else:
            sources_footer = ""

        think_summary = tool_result if executed_tool == "think" else ""
        return executed_tool, tool_result, tool_block, fetched_urls, sources_footer, final_system, think_summary

    # ------------------------------------------------------------------
    # Step 4: Streaming path — status message first, tool runs inside generator
    # ------------------------------------------------------------------
    if stream:
        async def _stream_with_sources():
            # Yield status message BEFORE the (potentially slow) tool runs.
            # Use thinking field (not content) so clients don't store it in history.
            if pending_tool_name == "think":
                yield {"thinking": "Entering deep reasoning mode..."}
            elif pending_tool_name == "starcoder":
                yield {"thinking": "Entering code assistant mode..."}

            # Now execute tool and build context
            _, _, _, _, sources_footer, final_system, think_summary = (
                await _execute_tool_and_build_context()
            )

            # Emit thinking content before response content
            if think_summary:
                yield {"thinking": think_summary}

            async for delta in _stream_complete_local(final_system, user_messages):
                yield delta
            if sources_footer:
                yield {"thinking": sources_footer}

        return None, _stream_with_sources(), ""

    # ------------------------------------------------------------------
    # Step 4: Non-streaming path
    # ------------------------------------------------------------------
    executed_tool, tool_result, tool_block, _, sources_footer, final_system, think_summary = (
        await _execute_tool_and_build_context()
    )

    text = await _complete_local(final_system, user_messages, temperature=0.7, max_tokens=4096)

    # Completeness gate: use StarCoder to check if the response fully answers
    # the question. If not, fire a follow-up tool call and retry with local Apertus.
    gate = await asyncio.get_running_loop().run_in_executor(
        None, lambda: _starcoder_check_complete(last_user_msg, text)
    )

    if not gate.get("complete", True):
        gate_tool_name = gate.get("tool", "")
        gate_tool_args = gate.get("args", {})
        if gate_tool_name in TOOL_REGISTRY:
            print(f"[pipeline] Completeness gate — fetching more info via {gate_tool_name}({gate_tool_args})", file=sys.stderr)
            extra_result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: TOOL_REGISTRY[gate_tool_name](**gate_tool_args)
            )
            extra_block = f"\n\n## Additional information: {gate_tool_name}({gate_tool_args})\n```\n{extra_result}\n```"
            text = await _complete_local(final_system + extra_block, user_messages, temperature=0.7, max_tokens=4096)

    thinking_out = "\n\n".join(filter(None, [think_summary, sources_footer]))
    return text, None, thinking_out


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
    think_flag: bool = body.get("think", True)

    if not messages:
        raise HTTPException(status_code=400, detail="messages required")

    created_at = _now_iso()

    if do_stream:
        _, stream_iter, _ = await run_pipeline(api_key, messages, stream=True, think_enabled=think_flag)

        async def _ollama_stream():
            async for item in stream_iter:
                if isinstance(item, dict) and "thinking" in item:
                    yield json.dumps({
                        "model": OLLAMA_MODEL_NAME,
                        "created_at": _now_iso(),
                        "message": {"role": "assistant", "content": "", "thinking": item["thinking"]},
                        "done": False,
                    }) + "\n"
                else:
                    yield json.dumps({
                        "model": OLLAMA_MODEL_NAME,
                        "created_at": _now_iso(),
                        "message": {"role": "assistant", "content": item},
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
        full_text, _, think_summary = await run_pipeline(api_key, messages, stream=False, think_enabled=think_flag)
        msg: dict = {"role": "assistant", "content": full_text}
        if think_summary:
            msg["thinking"] = think_summary
        return JSONResponse({
            "model": OLLAMA_MODEL_NAME,
            "created_at": created_at,
            "message": msg,
            "done": True,
            "done_reason": "stop",
        })


@app.post("/api/generate")
async def generate(request: Request):
    """Ollama POST /api/generate (legacy single-turn format).

    Used by Open WebUI for housekeeping tasks (title generation, topic summaries, etc.).
    Routed directly to the 70B — no tool decision or reasoning pipeline.
    """
    body = await request.json()
    api_key = _get_api_key()
    prompt: str = body.get("prompt", "")
    system: str = body.get("system", "")
    do_stream: bool = body.get("stream", True)

    messages = [{"role": "user", "content": prompt}]
    system_prompt = system or _charter()

    created_at = _now_iso()

    if do_stream:
        stream_iter = _stream_complete_remote(api_key, system_prompt, messages)

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
        full_text = await _complete_remote(api_key, system_prompt, messages)
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
    uvicorn.run(app, host="0.0.0.0", port=port)
