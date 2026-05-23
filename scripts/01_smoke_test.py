"""
Phase 2: Gemma 4 E2B smoke test.
Loads the model, runs text-only and vision+text generation,
then prints the module tree for export planning.

Usage:
    scripts/run.sh scripts/01_smoke_test.py
"""

import _paths  # noqa: F401 — must come before transformers/torch imports

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
from io import BytesIO

MODEL_ID = "google/gemma-4-e2b-it"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading processor and model (first run downloads ~10GB)...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.float16
).to(device).eval()

# --- Test 1: text only ---
print("\n=== Test 1: text generation ===")
messages = [{"role": "user", "content": [{"type": "text", "text": "The capital of France is"}]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# --- Test 2: vision + text ---
print("\n=== Test 2: image + text ===")
img = Image.open(BytesIO(requests.get(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
).content))
messages = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": "Describe this image in one sentence."},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# --- Module tree (top 3 levels) for Phase 3 planning ---
print("\n=== Module tree (for export wrapper design) ===")
for name, _ in model.named_modules():
    if name and name.count(".") <= 2:
        print(name)
