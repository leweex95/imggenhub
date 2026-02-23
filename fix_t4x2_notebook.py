"""Fix T4x2 notebook: pip install in Cell 0, try/except in Cell 4, CPU generator/VAE in Cell 5."""
import json
from pathlib import Path

nb_path = Path('src/imggenhub/kaggle/notebooks/kaggle-flux-dual-t4.ipynb')

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c.get('cell_type') == 'code']
print(f"Found {len(code_cells)} code cells")

# --- Cell 0: add pip install accelerate/diffusers ---
cell0_source = [
    "import gc\n",
    "import os\n",
    "import subprocess\n",
    "import sys\n",
    "os.environ[\"PYDEVD_DISABLE_FILE_VALIDATION\"] = \"1\"\n",
    "\n",
    "# Ensure accelerate is up-to-date (required for device_map='balanced')\n",
    "print(\"Installing/upgrading accelerate and diffusers...\", flush=True)\n",
    "result = subprocess.run(\n",
    "    [sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"--upgrade\", \"accelerate\", \"diffusers\"],\n",
    "    capture_output=True, text=True\n",
    ")\n",
    "if result.returncode != 0:\n",
    "    print(f\"pip install failed: {result.stderr[:200]}\", flush=True)\n",
    "else:\n",
    "    print(\"accelerate and diffusers ready\", flush=True)\n",
]
code_cells[0]['source'] = cell0_source
print("Updated Cell 0 (pip install)")

# --- Cell 4: model loading with try/except fallback ---
cell4_source = [
    "import gc\n",
    "\n",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
    "\n",
    "dtype_map = {\"bf16\": torch.bfloat16, \"fp16\": torch.float16, \"fp32\": torch.float32}\n",
    "torch_dtype = dtype_map.get(PRECISION, torch.bfloat16)\n",
    "\n",
    "gpu_count = torch.cuda.device_count()\n",
    "print(f\"Available GPUs: {gpu_count}\", flush=True)\n",
    "for i in range(gpu_count):\n",
    "    props = torch.cuda.get_device_properties(i)\n",
    "    print(f\"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM\", flush=True)\n",
    "\n",
    "print(f\"Loading model with device_map='balanced' across {gpu_count} GPU(s)...\", flush=True)\n",
    "try:\n",
    "    pipe = FluxPipeline.from_pretrained(\n",
    "        MODEL_ID,\n",
    "        torch_dtype=torch_dtype,\n",
    "        device_map=\"balanced\",\n",
    "        token=HF_TOKEN\n",
    "    )\n",
    "    pipe.set_progress_bar_config(disable=False)\n",
    "    print(\"Model loaded with device_map='balanced'.\", flush=True)\n",
    "except Exception as e:\n",
    "    print(f\"device_map='balanced' failed: {e}\", flush=True)\n",
    "    print(\"Falling back to sequential CPU offload...\", flush=True)\n",
    "    pipe = FluxPipeline.from_pretrained(\n",
    "        MODEL_ID,\n",
    "        torch_dtype=torch_dtype,\n",
    "        token=HF_TOKEN\n",
    "    )\n",
    "    pipe.enable_sequential_cpu_offload()\n",
    "    pipe.set_progress_bar_config(disable=False)\n",
    "    print(\"Fallback model loaded with CPU offload.\", flush=True)\n",
    "\n",
    "if LORA_REPO_ID:\n",
    "    print(f\"Loading LoRA weights from {LORA_REPO_ID} / {LORA_FILENAME} (scale={LORA_SCALE})...\", flush=True)\n",
    "    load_kwargs = {\"adapter_name\": \"photorealism\"}\n",
    "    if LORA_FILENAME:\n",
    "        load_kwargs[\"weight_name\"] = LORA_FILENAME\n",
    "    pipe.load_lora_weights(LORA_REPO_ID, token=HF_TOKEN, **load_kwargs)\n",
    "    pipe.set_adapters([\"photorealism\"], adapter_weights=[LORA_SCALE])\n",
    "    print(\"LoRA weights loaded and activated.\", flush=True)\n",
    "else:\n",
    "    print(\"No LoRA specified. Generating without LoRA.\", flush=True)\n",
    "\n",
    "print(f\"Model ready. dtype={torch_dtype}, GPUs={gpu_count}\", flush=True)\n",
]
if len(code_cells) >= 5:
    code_cells[4]['source'] = cell4_source
    print("Updated Cell 4 (model loading with try/except)")
else:
    print(f"WARNING: expected >=5 code cells, found {len(code_cells)}")

# --- Cell 5: generation with CPU generator and VAE optimizations ---
cell5_source = [
    'print(f"Generating {len(PROMPTS)} image(s) at {IMG_SIZE[1]}x{IMG_SIZE[0]}", flush=True)\n',
    'print(f"guidance={GUIDANCE}, steps={STEPS}, precision={PRECISION}", flush=True)\n',
    'if LORA_REPO_ID:\n',
    '    print(f"LoRA: {LORA_REPO_ID} (scale={LORA_SCALE})", flush=True)\n',
    '\n',
    '# VAE memory optimizations (helps prevent OOM during decode on T4x2)\n',
    'try:\n',
    '    pipe.vae.enable_slicing()\n',
    '    pipe.vae.enable_tiling()\n',
    '    print("VAE slicing/tiling enabled.", flush=True)\n',
    'except Exception:\n',
    '    pass\n',
    '\n',
    'for i, prompt in enumerate(PROMPTS):\n',
    '    print(f"[{i+1}/{len(PROMPTS)}] {prompt}", flush=True)\n',
    '    # Use CPU generator to avoid device conflicts with device_map="balanced"\n',
    '    generator = torch.Generator(device="cpu").manual_seed(SEED + i)\n',
    '    gc.collect()\n',
    '    torch.cuda.empty_cache()\n',
    '    image = pipe(\n',
    '        prompt,\n',
    '        height=IMG_SIZE[0],\n',
    '        width=IMG_SIZE[1],\n',
    '        guidance_scale=GUIDANCE,\n',
    '        num_inference_steps=STEPS,\n',
    '        generator=generator,\n',
    '        max_sequence_length=256,\n',
    '    ).images[0]\n',
    '\n',
    '    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")\n',
    '    safe_kernel_name = str(KERNEL_ID).split("/")[-1]\n',
    '    safe_prompt = slugify(prompt)\n',
    '    filename = f"gen_{safe_kernel_name}_p{i+1+INDEX_OFFSET}_{safe_prompt}_{timestamp}.png"\n',
    '\n',
    '    image.save(os.path.join(OUTPUT_DIR, filename))\n',
    '    print(f"Saved: {filename}", flush=True)\n',
    '    gc.collect()\n',
    '\n',
    'print(f"Complete! {len(PROMPTS)} images in {OUTPUT_DIR}", flush=True)\n',
]
if len(code_cells) >= 6:
    code_cells[5]['source'] = cell5_source
    print("Updated Cell 5 (generation with CPU generator + VAE)")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written {nb_path} ({nb_path.stat().st_size} bytes)")

# Verify
with open(nb_path) as f:
    nb2 = json.load(f)
code_cells2 = [c for c in nb2['cells'] if c.get('cell_type') == 'code']
for i, cell in enumerate(code_cells2):
    src = ''.join(cell.get('source', []))
    flags = []
    if 'pip install' in src:
        flags.append('pip-install')
    if 'device_map' in src:
        flags.append('device_map')
    if 'try:' in src and 'except' in src:
        flags.append('try/except')
    if 'enable_slicing' in src:
        flags.append('vae-slicing')
    if 'cpu' in src and 'Generator' in src:
        flags.append('cpu-generator')
    if flags:
        print(f"  Cell {i}: {', '.join(flags)}")
print("Done.")
