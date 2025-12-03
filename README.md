![Python Version](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/github/license/leweex95/imggenhub) [![Kaggle pipeline regression test](https://github.com/leweex95/imggenhub/actions/workflows/kaggle-regression-test.yml/badge.svg)](https://github.com/leweex95/imggenhub/actions/workflows/kaggle-regression-test.yml) [![Image generation](https://github.com/leweex95/imggenhub/actions/workflows/image-generation.yml/badge.svg)](https://github.com/leweex95/imggenhub/actions/workflows/image-generation.yml) [![Unit Tests](https://github.com/leweex95/imggenhub/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/leweex95/imggenhub/actions/workflows/unit-tests.yml) [![codecov](https://codecov.io/gh/leweex95/imggenhub/branch/master/graph/badge.svg)](https://codecov.io/gh/leweex95/imggenhub)

# ImgGenHub

**A personal, cost-efficient AI image generation platform.**

ImgGenHub is a personal image generation hub that connects to web-based image generation services, currently featuring a fully automated Kaggle-based pipeline with plans for multi-platform support.

---

## Features

### **Current functionality**

#### **Kaggle-powered image generation**
- **Automated pipeline**: Deploy → Monitor → Download workflow
- **Automated secret management**: Local `.env` secret detected and auto-uploaded to Kaggle dataset
- **Multiple models**: Supports Stable Diffusion variants, Flux.1-schnell GGUF quantized (Q4) version and Flux.1-schnell bf16 version.

#### **Vast.ai support** (todo)

_Planned in Dec 2025 / Jan 2026._

---

## Requirements

- **3.11+ Python** and **Poetry**  
- **Kaggle Account**: with API credentials
- **HuggingFace token**: for accessing gated models (e.g., FLUX.1-schnell)

---

## Usage

### Using FLUX.1-schnell

[FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) is commercially usable. The full fp32 model doesn't fit into our available VRAM on Kaggle kernels but its lossless bf16 compression does. 

### FLUX.1-schnell with bf16

```bash
poetry run imggenhub \
  --prompt "A photorealistic restaurant scene: a single customer ordering food, a waiter presenting the menu, natural lighting, sharp focus, realistic textures" \
  --model_name "black-forest-labs/FLUX.1-schnell" \
  --steps 4 \
  --guidance 0.75 \
  --img_width 1024 \
  --img_height 1024 \
  --precision bf16 \
  --gpu
```

### FLUX.1-schnell quantized version (Q4_0 GGUF)

```bash
poetry run imggenhub \
  --prompt "A photorealistic restaurant scene: a single customer ordering food, a waiter presenting the menu, natural lighting, sharp focus, realistic textures" \
  --diffusion_filename "flux1-schnell-Q4_0.gguf" \
  --diffusion_repo_id "city96/FLUX.1-schnell-gguf" \
  --steps 4 \
  --guidance 0.8 \
  --img_width 512 \
  --img_height 512 \
  --precision q4 \
  --gpu
```

### Stable diffusion XL with refiner 

```bash
poetry run imggenhub \
  --prompt "photorealistic parliament building along river in summer" \
  --model_name stabilityai/stable-diffusion-xl-base-1.0 \
  --refiner_model_name stabilityai/stable-diffusion-xl-refiner-1.0 \
  --guidance 8.0 \
  --steps 30 \
  --refiner_guidance 7.0 \
  --refiner_steps 15 \
  --precision fp16 \
  --refiner_precision fp16 \
  --two_stage_refiner \
  --gpu
```

### **Supported flags**
- `--dest DEST`: Custom name prefix for the output folder (default: timestamp only)
- Outputs are always saved under `output/` with automatic timestamping
- All logs from the generation process are saved in the same folder as the images
- `--prompt`: Single prompt string
- `--prompts_file`: JSON file with multiple prompts  
- `--model_name`: Hugging Face model ID
- `--refiner_model_name`: SDXL refiner model for enhanced photorealism
- `--gpu`: Enable GPU acceleration (required for FLUX.1 models)
- `--precision`: Model precision (fp32/fp16/int8/int4)
- `--guidance`: Prompt adherence strength (7-12 recommended for photorealism)
- `--steps`: Inference steps (50-100 for stable diffusion models, ~4 for FLUX)
- `--negative_prompt`: Quality control prompts
- `--two_stage_refiner`: Use VRAM-optimized two-stage approach (base → unload → refiner)
- `--refiner_guidance`: Guidance scale for refiner (defaults to same as --guidance)
- `--refiner_steps`: Inference steps for refiner (defaults to 20)
- `--refiner_precision`: Precision for refiner (defaults to same as --precision)
- `--refiner_negative_prompt`: Negative prompt for refiner (defaults to same as --negative_prompt)
- `--notebook`: Custom notebook path
- `--kernel_path`: Kaggle kernel configuration directory

## Custom Kaggle datasets

To speed up inference, we upload the largest model files as a custom Kaggle dataset. This massively speeds up model inference as there is no need to fetch tens of GBs before every inference run.

Note: temporarily switched off as we found that for FLUX.1-schnell models (and hence, likely for any large model) it actually takes more time to read these models from Kaggle datasets inside of a Kaggle notebook than downloading directly from Hugging Face.
