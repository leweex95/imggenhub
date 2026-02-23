"""Rebuild kaggle-flux-schnell-bf16.ipynb with correct structure."""
import json
from pathlib import Path

cell_setup = """\
import gc
import os
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
"""

cell_imports = """\
from diffusers import FluxPipeline
from datetime import datetime
import torch
import re
"""

cell_hf_token = """\
import json
import subprocess

# -------- LOAD HF TOKEN FROM KAGGLE DATASET ---------
HF_TOKEN_PATH = "/kaggle/input/imggenhub-hf-token/hf_token.json"
HF_TOKEN = None

if os.path.exists(HF_TOKEN_PATH):
    with open(HF_TOKEN_PATH, "r") as f:
        HF_TOKEN = json.load(f)["HF_TOKEN"]
    print("HF_TOKEN loaded from mounted dataset")
else:
    print("Dataset not mounted, downloading via kaggle API...")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["kaggle", "datasets", "download", "leventecsibi/imggenhub-hf-token",
                 "-p", tmpdir, "--unzip"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                token_file = os.path.join(tmpdir, "hf_token.json")
                with open(token_file, "r") as f:
                    HF_TOKEN = json.load(f)["HF_TOKEN"]
                print("HF_TOKEN downloaded and loaded successfully")
            else:
                print(f"Failed to download dataset: {result.stderr}")
    except Exception as e:
        print(f"Error downloading token: {e}")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF_TOKEN set in environment")
else:
    print("CRITICAL ERROR: Could not load HF_TOKEN")
    raise RuntimeError("HF_TOKEN is required but could not be loaded")
"""

# Parameters cell: ONLY simple assignments - no imports, no functions
# edit_notebook_params uses AST line-based replacement - safe with simple assignments
cell_params = """\
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
PROMPTS = ["a beautiful sunset", "a cute dog", "a fast car"]
OUTPUT_DIR = "."
IMG_SIZE = (1024, 1024)
GUIDANCE = 1.5
STEPS = 4
SEED = 42
PRECISION = "bf16"
INDEX_OFFSET = 0
KERNEL_ID = "unknown"
LORA_REPO_ID = None
LORA_FILENAME = None
LORA_SCALE = 0.8
"""

cell_generate = """\
import gc


def slugify(text):
    text = text.lower()
    text = re.sub(r'[^\\w\\s-]', '', text)
    text = re.sub(r'[\\s_-]+', '_', text)
    text = re.sub(r'^-+|-+$', '', text)
    return text[:30]


os.makedirs(OUTPUT_DIR, exist_ok=True)

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
torch_dtype = dtype_map.get(PRECISION, torch.bfloat16)
print(f"Device: cuda" if torch.cuda.is_available() else "Device: cpu")


def get_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 3
    return 0.0


print("Loading model...")
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch_dtype, token=HF_TOKEN)

# Load LoRA BEFORE sequential CPU offload (fusing requires weights on model)
if LORA_REPO_ID:
    print(f"Loading LoRA weights from {LORA_REPO_ID} / {LORA_FILENAME} (scale={LORA_SCALE})...")
    load_kwargs = {"adapter_name": "photorealism"}
    if LORA_FILENAME:
        load_kwargs["weight_name"] = LORA_FILENAME
    pipe.load_lora_weights(LORA_REPO_ID, token=HF_TOKEN, **load_kwargs)
    pipe.set_adapters(["photorealism"], adapter_weights=[LORA_SCALE])
    pipe.fuse_lora(lora_scale=LORA_SCALE)
    print("LoRA weights fused into model.")
else:
    print("No LoRA specified. Generating without LoRA.")

pipe.enable_vae_tiling()
pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=False)
pipe.enable_sequential_cpu_offload()

print(f"Model loaded (VRAM: {get_vram_gb():.2f} GB)")

for i, prompt in enumerate(PROMPTS):
    print(f"[{i + 1}/{len(PROMPTS)}] {prompt}")
    generator = torch.Generator(device="cpu").manual_seed(SEED + i)
    image = pipe(
        prompt,
        height=IMG_SIZE[0],
        width=IMG_SIZE[1],
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        generator=generator,
        max_sequence_length=256,
    ).images[0]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_kernel_name = str(KERNEL_ID).split('/')[-1]
    safe_prompt = slugify(prompt)
    filename = f"gen_{safe_kernel_name}_p{i + 1 + INDEX_OFFSET}_{safe_prompt}_{timestamp}.png"

    image.save(os.path.join(OUTPUT_DIR, filename))
    print(f"Saved: {filename}")
    torch.cuda.empty_cache()
    gc.collect()

print(f"Complete! {len(PROMPTS)} images in {OUTPUT_DIR}")
"""

cell_display = """\
from IPython.display import display, Markdown
from PIL import Image
import glob
import natsort

# Collect all generated PNGs (gen_ naming convention)
image_paths = natsort.natsorted(glob.glob(os.path.join(OUTPUT_DIR, "gen_*.png")))

print(f"Displaying {len(image_paths)} generated images with prompts:")

for i, path in enumerate(image_paths):
    prompt = PROMPTS[i] if i < len(PROMPTS) else "Unknown prompt"
    display(Markdown(f"**Prompt {i + 1}:** {prompt}"))
    img = Image.open(path)
    display(img)
    print("-" * 50)
"""


def make_cell(source_str, cell_id):
    # Store source as list of lines (standard Jupyter format)
    # edit_notebook_params uses node.lineno for line-based replacement,
    # which requires source_lines to be indexed by line number.
    lines = source_str.split('\n')
    source_list = [line + '\n' for line in lines[:-1]]
    if lines[-1]:  # last line without trailing newline
        source_list.append(lines[-1])
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_list,
    }


notebook = {
    "cells": [
        make_cell(cell_setup, "VSC-75781ea9"),
        make_cell(cell_imports, "VSC-242b980c"),
        make_cell(cell_hf_token, "VSC-b23645d4"),
        make_cell(cell_params, "VSC-11182635"),
        make_cell(cell_generate, "VSC-36368182"),
        make_cell(cell_display, "VSC-7db62c5a"),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path("src/imggenhub/kaggle/notebooks/kaggle-flux-schnell-bf16.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Written {out_path} ({out_path.stat().st_size} bytes)")

# Verify
nb_check = json.load(open(out_path))
print(f"Cells: {len(nb_check['cells'])}")
for i, c in enumerate(nb_check['cells']):
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    print(f"  cell {i} ({c['id']}): {src[:60]!r}...")
