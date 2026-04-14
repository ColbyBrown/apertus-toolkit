"""
Gradio chatbot UI for the Apertus hybrid proxy.

Connects to apertus_proxy_hybrid.py via HTTP (Ollama-compatible /api/chat endpoint),
parses the NDJSON stream, and renders thinking content as collapsible blocks using
Gradio's reasoning_tags feature.

Usage:
    python apertus_chat_hybrid.py
    # or with a custom proxy URL:
    APERTUS_PROXY_URL=http://localhost:11435 python apertus_chat_hybrid.py
"""

import asyncio
import json
import os

import gradio as gr
import httpx

HYBRID_PROXY_URL = os.environ.get("APERTUS_PROXY_URL", "http://localhost:11436")


# ---------------------------------------------------------------------------
# Pipeline — streams from hybrid proxy, yields (display_content, clean_response)
# ---------------------------------------------------------------------------

async def _run_pipeline(messages: list[dict], think: bool = True):
    """
    Async generator yielding (display_content, clean_response) tuples.

    display_content  — full string with <reasoning>…</reasoning> tags for Gradio
    clean_response   — None on intermediate yields; plain response text on final yield
    """
    payload = {
        "model": "apertus-hybrid",
        "messages": messages,
        "stream": True,
        "think": think,
    }
    accumulated_thinking = ""
    response_text = ""

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", f"{HYBRID_PROXY_URL}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg = chunk.get("message", {})
                    thinking_delta = msg.get("thinking", "")
                    content_delta = msg.get("content", "")

                    if thinking_delta:
                        accumulated_thinking += thinking_delta
                    if content_delta:
                        response_text += content_delta

                    display = ""
                    if accumulated_thinking:
                        display += f"<reasoning>\n{accumulated_thinking}\n</reasoning>\n\n"
                    display += response_text or "⏳ *Generating…*"

                    if chunk.get("done"):
                        yield display, response_text
                    else:
                        yield display, None

    except httpx.ConnectError:
        msg = (
            f"⚠️ Could not connect to the hybrid proxy at `{HYBRID_PROXY_URL}`. "
            "Make sure it is running (`uvicorn apertus_proxy_hybrid:app --port 11435`)."
        )
        yield msg, ""
    except httpx.HTTPStatusError as e:
        yield f"⚠️ Proxy returned HTTP {e.response.status_code}: {e.response.text[:200]}", ""
    except Exception as e:
        yield f"⚠️ Unexpected error: {e}", ""


# ---------------------------------------------------------------------------
# Gradio event handler
# ---------------------------------------------------------------------------

async def respond(
    message: str,
    chatbot_display: list,
    clean_history: list,
    think_toggle: bool,
):
    """Async generator: streams pipeline updates to chatbot + hidden clean history."""
    if not message.strip():
        yield chatbot_display, clean_history, ""
        return

    user_msg = {"role": "user", "content": message}
    new_display = chatbot_display + [user_msg]
    new_clean = clean_history + [user_msg]

    # Show placeholder immediately
    yield (
        new_display + [{"role": "assistant", "content": "⏳ *Connecting to proxy…*"}],
        clean_history,
        "",
    )

    last_display = "⚠️ *No output received.*"
    final_response = ""

    async for display_content, clean_response in _run_pipeline(new_clean, think=think_toggle):
        last_display = display_content
        if clean_response is not None:
            final_response = clean_response
        yield (
            new_display + [{"role": "assistant", "content": display_content}],
            clean_history,  # hold — don't update clean history mid-stream
            "",
        )

    # Commit only the plain response text to clean history (no tags, no thinking)
    final_msg = {"role": "assistant", "content": final_response}
    yield (
        new_display + [{"role": "assistant", "content": last_display}],
        new_clean + [final_msg],
        "",
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
#chatcol { min-width: 0; }
.chatbot-wrap .message-wrap { max-width: 100% !important; }
footer { display: none !important; }
"""


def build_ui():
    with gr.Blocks(title="Apertus Hybrid Chat", css=CSS, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
        gr.Markdown(
            "# Apertus Hybrid\n"
            "Reasoning and tool activity shown inline — click **▶ Reasoning** to expand."
        )

        with gr.Row():
            with gr.Column(scale=3, elem_id="chatcol"):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=580,
                    reasoning_tags=[("<reasoning>", "</reasoning>")],
                    render_markdown=True,
                    show_label=False,
                    placeholder=(
                        "<div style='text-align:center; padding: 2rem; opacity:.5'>"
                        "Ask anything — reasoning and tool use will appear inline.</div>"
                    ),
                )

                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Type a message…",
                        show_label=False,
                        scale=5,
                        submit_btn=True,
                        stop_btn=True,
                        autofocus=True,
                    )

                with gr.Row():
                    clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### Settings")

                think_toggle = gr.Checkbox(
                    label="Enable deep reasoning (think)",
                    value=True,
                    info="Runs the internal think-tool pipeline before responding.",
                )

                gr.Markdown(
                    f"**Proxy URL:** `{HYBRID_PROXY_URL}`\n\n"
                    "Override with the `APERTUS_PROXY_URL` environment variable."
                )

                gr.Markdown("---")
                gr.Markdown(
                    "### What the ▶ Reasoning block contains\n\n"
                    "- **Status** — which tool was called\n"
                    "- **Reasoning dialogue** — the internal 8B ↔ 70B think exchange\n"
                    "- **Sources** — pages fetched and summarised\n\n"
                    "The response below the block is what gets stored in chat history "
                    "and sent to the model on future turns."
                )

        # Hidden state: clean conversation history (no tags, no thinking)
        clean_history = gr.State([])

        # Wiring
        submit_event = msg_box.submit(
            fn=respond,
            inputs=[msg_box, chatbot, clean_history, think_toggle],
            outputs=[chatbot, clean_history, msg_box],
        )

        clear_btn.click(
            fn=lambda: ([], [], ""),
            outputs=[chatbot, clean_history, msg_box],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", 7861)),
        share=False,
        show_error=True,
    )
