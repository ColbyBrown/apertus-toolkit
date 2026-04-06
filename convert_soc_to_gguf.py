#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.36.0",
#     "peft>=0.7.0",
#     "torch>=2.0.0",
#     "accelerate>=0.24.0",
#     "huggingface_hub>=0.20.0",
#     "sentencepiece>=0.1.99",
#     "protobuf>=3.20.0",
#     "numpy",
#     "gguf",
# ]
# ///

"""
Convert Colby/apertus-8b-soc (LoRA adapter) to GGUF for Ollama.

Merges the adapter into swiss-ai/Apertus-8B-2509 base, then produces
Q4_K_M, Q5_K_M, and Q8_0 quantizations via llama.cpp.
Output repo: Colby/apertus-8b-soc-gguf
"""

import os
import sys
import subprocess
import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_MODEL = "Colby/apertus-8b-soc"
BASE_MODEL = "swiss-ai/Apertus-8B-2509"
OUTPUT_REPO = "Colby/apertus-8b-soc-gguf"
MODEL_NAME = "apertus-8b-soc"
MERGED_DIR = "/tmp/merged_model"
GGUF_DIR = "/tmp/gguf_output"
LLAMA_CPP_DIR = "/tmp/llama.cpp"


def run(cmd, desc):
    print(f"  {desc}...")
    result = subprocess.run(cmd, capture_output=True)  # raw bytes — avoid UTF-8 decode errors from binary progress output
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[:600]
        print(f"  FAILED: {stderr}")
        sys.exit(1)
    if result.stdout:
        stdout = result.stdout.decode("utf-8", errors="replace")[:200]
        print(f"  {stdout}")
    return True


# --- Step 0: Install build tools (MUST happen before cloning llama.cpp) ---
print("Step 0: Installing build tools...")
subprocess.run(["apt-get", "update", "-qq"], check=True, capture_output=True)
subprocess.run(
    ["apt-get", "install", "-y", "-qq", "build-essential", "cmake"],
    check=True, capture_output=True
)
print("  Build tools ready.")

# --- Step 1: Load base model and merge LoRA adapter ---
print("\nStep 1: Loading base model and merging LoRA adapter...")
print(f"  Base:    {BASE_MODEL}")
print(f"  Adapter: {ADAPTER_MODEL}")

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print("  Base model loaded.")

model = PeftModel.from_pretrained(base, ADAPTER_MODEL)
print("  Adapter loaded.")

merged = model.merge_and_unload()
print("  Merged.")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL, trust_remote_code=True)

# --- Step 2: Save merged model to disk ---
print(f"\nStep 2: Saving merged model to {MERGED_DIR}...")
os.makedirs(MERGED_DIR, exist_ok=True)
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print("  Saved.")

# Free GPU memory before llama.cpp conversion
del merged, model, base
torch.cuda.empty_cache()

# --- Step 3: Clone llama.cpp ---
print("\nStep 3: Cloning llama.cpp...")
run(
    ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_DIR],
    "Cloning"
)
run(
    ["pip", "install", "-q", "-r", f"{LLAMA_CPP_DIR}/requirements.txt"],
    "Installing llama.cpp Python requirements"
)

# --- Step 4: Convert to FP16 GGUF ---
print("\nStep 4: Converting to FP16 GGUF...")
os.makedirs(GGUF_DIR, exist_ok=True)
fp16_gguf = f"{GGUF_DIR}/{MODEL_NAME}-f16.gguf"

run(
    [sys.executable, f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py",
     MERGED_DIR, "--outfile", fp16_gguf, "--outtype", "f16"],
    "Converting"
)
print(f"  FP16 GGUF: {os.path.getsize(fp16_gguf) / 1024**3:.1f} GB")

# --- Step 5: Build llama-quantize with CMake ---
print("\nStep 5: Building llama-quantize...")
build_dir = f"{LLAMA_CPP_DIR}/build"
os.makedirs(build_dir, exist_ok=True)

run(
    ["cmake", "-B", build_dir, "-S", LLAMA_CPP_DIR, "-DGGML_CUDA=OFF"],
    "CMake configure"
)
run(
    ["cmake", "--build", build_dir, "--target", "llama-quantize", "-j", "4"],
    "CMake build"
)
quantize_bin = f"{build_dir}/bin/llama-quantize"
print(f"  Binary: {quantize_bin}")

# --- Step 6: Quantize ---
print("\nStep 6: Quantizing...")
quant_formats = [
    ("Q4_K_M", "4-bit medium (recommended for Ollama)"),
    ("Q5_K_M", "5-bit medium"),
    ("Q8_0",   "8-bit"),
]

quantized = []
for qtype, desc in quant_formats:
    qfile = f"{GGUF_DIR}/{MODEL_NAME}-{qtype.lower()}.gguf"
    run([quantize_bin, fp16_gguf, qfile, qtype], f"{qtype} ({desc})")
    size_mb = os.path.getsize(qfile) / 1024**2
    print(f"  {qtype}: {size_mb:.0f} MB")
    quantized.append((qfile, qtype))

# --- Step 7: Upload to Hub ---
print(f"\nStep 7: Uploading to {OUTPUT_REPO}...")
api = HfApi()
api.create_repo(repo_id=OUTPUT_REPO, repo_type="model", exist_ok=True)

for path, qtype in [(fp16_gguf, "F16")] + quantized:
    fname = os.path.basename(path)
    print(f"  Uploading {fname}...")
    api.upload_file(path_or_fileobj=path, path_in_repo=fname, repo_id=OUTPUT_REPO)
    print(f"  Done: {fname}")

readme = f"""---
base_model: {BASE_MODEL}
tags:
- gguf
- apertus
- multi-turn-chat
- sft
---

# {MODEL_NAME}-gguf

GGUF conversion of [{ADAPTER_MODEL}](https://huggingface.co/{ADAPTER_MODEL}),
a LoRA fine-tune of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})
on [marcodsn/SOC-2508](https://huggingface.co/datasets/marcodsn/SOC-2508)
(Synthetic Online Conversations).

## Quantizations

| File | Format | Size |
|------|--------|------|
| {MODEL_NAME}-f16.gguf | FP16 | ~16 GB |
| {MODEL_NAME}-q8_0.gguf | Q8_0 | ~8 GB |
| {MODEL_NAME}-q5_k_m.gguf | Q5_K_M | ~5 GB |
| {MODEL_NAME}-q4_k_m.gguf | Q4_K_M | ~4 GB |

## Ollama usage

```bash
hf download {OUTPUT_REPO} {MODEL_NAME}-q4_k_m.gguf
ollama create apertus-soc:8b -f Modelfile   # FROM ./{MODEL_NAME}-q4_k_m.gguf
ollama run apertus-soc:8b
```
"""
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=OUTPUT_REPO,
)

print(f"\nDone! GGUF repo: https://huggingface.co/{OUTPUT_REPO}")
