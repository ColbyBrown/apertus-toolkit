#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.36.0",
#     "accelerate>=0.24.0",
#     "bitsandbytes>=0.41.0",
#     "datasets>=2.0.0",
#     "trackio",
# ]
# ///

"""
Fine-tune swiss-ai/Apertus-8B-2509 on marcodsn/SOC-2508 (Synthetic Online Conversations).

Preserves the full multi-participant chat structure: each conversation is formatted
as ChatML with custom roles (persona usernames) rather than collapsing to user/assistant.
Loss is computed on ALL tokens so the model learns every participant's voice.
"""

import trackio
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "swiss-ai/Apertus-8B-2509"
DATASET_ID = "marcodsn/SOC-2508"
OUTPUT_REPO = "Colby/apertus-8b-soc"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Apertus-8B tokenizer has no chat_template set; define ChatML explicitly.
# Custom roles (persona usernames) are preserved as-is in the im_start tag.
tokenizer.chat_template = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
)

print("Loading dataset...")
dataset = load_dataset(DATASET_ID, split="train")
print(f"Loaded {len(dataset)} conversations")


def format_conversation(example):
    """
    Convert a SOC conversation to a ChatML text string for training.

    Structure:
      - system turn: full persona bios, relationship, and situation context
      - one turn per chat_parts entry, role = sender's username, content = all messages joined

    Using apply_chat_template + dataset_text_field trains on all tokens (all participants),
    which is correct for multi-participant chat — there is no single "assistant" role.
    """
    exp = example["experience"]
    p1, p2 = exp["persona1"], exp["persona2"]
    id_to_username = {
        p1["id"]: p1["username"],
        p2["id"]: p2["username"],
    }

    system_content = (
        f"Participants:\n"
        f"- {p1['name']} (@{p1['username']}, age {p1['age']}): {p1['background']} "
        f"Chatting style: {p1['chatting_style']}\n"
        f"- {p2['name']} (@{p2['username']}, age {p2['age']}): {p2['background']} "
        f"Chatting style: {p2['chatting_style']}\n"
        f"Relationship: {exp['relationship']}\n"
        f"Situation: {exp['situation']}"
    )

    messages = [{"role": "system", "content": system_content}]
    for turn in example["chat_parts"]:
        username = id_to_username.get(turn["sender"], turn["sender"])
        content = "\n".join(turn["messages"])
        messages.append({"role": username, "content": content})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


print("Formatting conversations to ChatML...")
dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"  Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

config = SFTConfig(
    # Hub push — ephemeral environment, must push or results are lost
    output_dir="apertus-8b-soc",
    push_to_hub=True,
    hub_model_id=OUTPUT_REPO,
    hub_strategy="every_save",

    # Train on ALL tokens (all participant voices, not just "assistant")
    dataset_text_field="text",
    max_length=2048,

    # Hyperparameters
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,   # eval disables grad checkpointing; keep small to avoid OOM
    gradient_accumulation_steps=8,  # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    gradient_checkpointing=True,

    # Checkpointing
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,

    # Monitoring
    report_to="trackio",
    project="apertus-soc-finetune",
    run_name="apertus-8b-soc-v1",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

print("Loading model with 4-bit quantization (QLoRA)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=config,
)

print("Starting training...")
trainer.train()

print("Pushing to Hub...")
trainer.push_to_hub()
trackio.finish()

print(f"Done! Model at: https://huggingface.co/{OUTPUT_REPO}")
print(f"Metrics at: https://huggingface.co/spaces/Colby/trackio")
