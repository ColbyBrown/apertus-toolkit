"""
Gradio chatbot UI for the Apertus 70B proxy.

Runs the full reasoning → decision → tool → response pipeline and displays
each step inline in the chat as collapsible sections.

Usage:
    PUBLICAI_API_KEY=your_key python apertus_chat.py
"""

import asyncio
import json
import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apertus_proxy import (
    CHARTER,
    TOOL_DESCRIPTIONS,
    TOOL_REGISTRY,
    _complete,
    _format_conversation,
    _stream_complete,
)

# ---------------------------------------------------------------------------
# Pipeline (verbose, yields intermediate states)
# ---------------------------------------------------------------------------

async def _run_verbose_pipeline(api_key: str, user_messages: list):
    """
    Async generator that yields (display_content, clean_response) tuples.

    display_content  — full assistant content string with <reasoning>/<tool> tags
    clean_response   — only the final reply text (set on last yield)
    """
    last_user_msg = next(
        (m["content"] for m in reversed(user_messages) if m["role"] == "user"), ""
    )
    conv_str = _format_conversation(user_messages)
    tools_str = json.dumps(TOOL_DESCRIPTIONS, indent=2)

    # ── Step 1: Reasoning ────────────────────────────────────────────────
    yield "<reasoning>\n⏳ *Thinking…*", None

    reasoning_system = f"""{CHARTER}
## Your tools
{tools_str}

## Task
You are engaging in an internal reasoning dialogue BEFORE replying to the user.
Play both sides of a short conversation (2–4 exchanges) between an inner USER \
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

    display = f"<reasoning>\n{reasoning}\n</reasoning>\n\n⏳ *Deciding next step…*"
    yield display, None

    # ── Step 2: Decision ─────────────────────────────────────────────────
    decision_system = f"""{CHARTER}
## Your tools
{tools_str}

## Your reasoning so far
{reasoning}

## Task
Decide: should a tool be called, or should you respond directly?

- If a tool should be called, output ONLY valid JSON:
  {{"tool": "<tool_name>", "args": {{<key>: <value>, ...}}}}
- If no tool is needed, output ONLY the single word: DIRECT

No explanation. No markdown fences. Output only the JSON or the word DIRECT.
"""
    decision_raw = await _complete(
        api_key,
        decision_system,
        [{"role": "user", "content": "Tool call or direct response?"}],
        temperature=0.1,
        max_tokens=256,
    )
    decision = decision_raw.strip()
    if decision.startswith("```"):
        decision = "\n".join(
            line for line in decision.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    # ── Step 3: Tool execution ───────────────────────────────────────────
    tool_block_for_system = ""
    display_prefix = f"<reasoning>\n{reasoning}\n</reasoning>\n"

    if decision.upper() != "DIRECT" and decision.startswith("{"):
        try:
            call = json.loads(decision)
            tool_name = call.get("tool", "")
            tool_args = call.get("args", {})

            if tool_name in TOOL_REGISTRY:
                running_display = (
                    display_prefix
                    + f"<tool>\n**🔧 {tool_name}**\n`{json.dumps(tool_args)}`\n\n"
                    + "⏳ *Running tool…*\n</tool>\n\n⏳ *Generating response…*"
                )
                yield running_display, None

                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: TOOL_REGISTRY[tool_name](**tool_args)
                )
                tool_block_for_system = (
                    f"\n\n## Tool results: {tool_name}({json.dumps(tool_args)})\n"
                    f"```\n{result}\n```"
                )
                # Truncate for display only
                result_preview = result if len(result) <= 2000 else result[:2000] + "\n…(truncated)"
                display_prefix = (
                    display_prefix
                    + f"<tool>\n**🔧 {tool_name}**\n`{json.dumps(tool_args)}`\n\n"
                    + f"```\n{result_preview}\n```\n</tool>\n"
                )
            else:
                display_prefix += f"<tool>\n⚠️ Unknown tool requested: `{tool_name}`\n</tool>\n"

        except (json.JSONDecodeError, TypeError) as e:
            display_prefix += f"<tool>\n⚠️ Could not parse tool call: {e}\n</tool>\n"
    else:
        # DIRECT — note the decision quietly in the reasoning block
        pass

    yield display_prefix + "\n⏳ *Generating response…*", None

    # ── Step 4: Stream final response ────────────────────────────────────
    final_system = f"""{CHARTER}
## Your reasoning
{reasoning}
{tool_block_for_system}

Use the above reasoning (and tool results, if any) to compose your reply. \
Do not repeat your reasoning verbatim. Just give a clear, helpful response.
"""
    response_text = ""
    async for delta in _stream_complete(
        api_key, final_system, user_messages, temperature=0.8, max_tokens=4096
    ):
        response_text += delta
        yield display_prefix + response_text, None

    # Final yield — signal completion with clean response text
    yield display_prefix + response_text, response_text


# ---------------------------------------------------------------------------
# Gradio event handler
# ---------------------------------------------------------------------------

async def respond(message: str, chatbot_display: list, clean_history: list, api_key_input: str):
    """Async generator: streams pipeline updates to chatbot + hidden state."""
    api_key = api_key_input.strip() or os.environ.get("PUBLICAI_API_KEY", "")
    if not api_key:
        error_msg = (
            "⚠️ No API key found. Enter your PublicAI key in the settings panel "
            "or set the `PUBLICAI_API_KEY` environment variable."
        )
        new_display = chatbot_display + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg},
        ]
        yield new_display, clean_history, ""
        return

    # Append user turn to both display and clean history
    user_msg = {"role": "user", "content": message}
    new_display = chatbot_display + [user_msg]
    new_clean = clean_history + [user_msg]

    # Placeholder while pipeline starts
    in_progress = new_display + [{"role": "assistant", "content": "⏳ *Starting pipeline…*"}]
    yield in_progress, clean_history, ""  # don't update clean_history yet

    final_clean_response = ""
    last_display_content = "⚠️ *Pipeline produced no output.*"
    async for display_content, clean_response in _run_verbose_pipeline(api_key, new_clean):
        last_display_content = display_content
        current_display = new_display + [{"role": "assistant", "content": display_content}]
        if clean_response is not None:
            final_clean_response = clean_response
        yield current_display, clean_history, ""  # still holding clean_history

    # Finalise: commit the completed assistant turn to clean history
    final_clean_msg = {"role": "assistant", "content": final_clean_response}
    final_display = new_display + [{"role": "assistant", "content": last_display_content}]
    yield final_display, new_clean + [final_clean_msg], ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
#chatcol { min-width: 0; }
.chatbot-wrap .message-wrap { max-width: 100% !important; }
footer { display: none !important; }
"""

def build_ui():
    with gr.Blocks(title="Apertus 70B Chat") as demo:
        gr.Markdown(
            "# Apertus 70B\n"
            "Powered by [PublicAI](https://platform.publicai.co) · "
            "Reasoning and tool use shown inline — click the arrows to expand."
        )

        with gr.Row():
            with gr.Column(scale=3, elem_id="chatcol"):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=560,
                    reasoning_tags=[
                        ("<reasoning>", "</reasoning>"),
                        ("<tool>", "</tool>"),
                    ],
                    avatar_images=(None, "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg"),
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

            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Settings")
                api_key_box = gr.Textbox(
                    label="PublicAI API key",
                    placeholder="Leave blank to use env var",
                    type="password",
                    value=os.environ.get("PUBLICAI_API_KEY", ""),
                )
                gr.Markdown(
                    "Get a key at [platform.publicai.co](https://platform.publicai.co/settings/api-keys)",
                    elem_classes=["small-text"],
                )
                gr.Markdown("---")
                gr.Markdown(
                    "### How it works\n"
                    "Each message triggers a 4-step pipeline:\n\n"
                    "1. **Reasoning** — internal self-dialogue\n"
                    "2. **Decision** — tool or direct reply?\n"
                    "3. **Tool call** — web search or Wikipedia\n"
                    "4. **Response** — streamed final answer\n\n"
                    "Click **▶ Reasoning** / **▶ Tool** in a message to expand."
                )

        # ── Hidden state ────────────────────────────────────────────────
        clean_history = gr.State([])

        # ── Wiring ──────────────────────────────────────────────────────
        submit_event = msg_box.submit(
            fn=respond,
            inputs=[msg_box, chatbot, clean_history, api_key_box],
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
        server_port=int(os.environ.get("GRADIO_PORT", 7860)),
        share=False,
        show_error=True,
        css=CSS,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
