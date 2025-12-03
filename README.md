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
  --prompts "photorealistic indoor restaurant scene" \
  --model_id "black-forest-labs/FLUX.1-schnell" \
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
  --prompts "photorealistic indoor restaurant scene" \
  --model_id "city96/FLUX.1-schnell-gguf" \
  --model_filename "flux1-schnell-Q4_0.gguf" \
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
  --prompts "photorealistic indoor restaurant scene" \
  --model_id stabilityai/stable-diffusion-xl-base-1.0 \
  --refiner_model_id stabilityai/stable-diffusion-xl-refiner-1.0 \
  --steps 30 \
  --guidance 8.0 \
  --precision fp16 \
  --refiner_steps 15 \
  --refiner_guidance 7.0 \
  --refiner_precision fp16 \
  --gpu
```

### **Supported flags**

#### **General flags** (all models)
- `--prompts`: Single prompt string or list of strings
- `--prompts_file`: JSON file with multiple prompts  
- `--gpu`: Enable GPU acceleration (required for FLUX.1 models)
- `--steps`: Inference steps (50-100 for Stable Diffusion models, ~4 for FLUX)
- `--guidance`: Prompt adherence strength (7-12 recommended for photorealism, 0.75-1.0 for FLUX)
- `--precision`: Model precision (fp32/fp16/int8/int4 for base models; q4/q5/q6 for GGUF quantized models)
- `--img_width` / `--img_height`: Image dimensions (must be multiples of 64)
- `--negative_prompt`: Quality control prompts
- `--dest DEST`: Custom name prefix for the output folder (default: timestamp only)
- `--notebook`: Custom notebook path
- `--kernel_path`: Kaggle kernel configuration directory
- Outputs are always saved under `output/` with automatic timestamping
- All logs from the generation process are saved in the same folder as the images

#### **Stable Diffusion XL flags** (model_id: "stabilityai/*")
- `--model_id`: Hugging Face model ID (e.g., `stabilityai/stable-diffusion-xl-base-1.0`)
- `--refiner_model_id`: SDXL refiner model for enhanced photorealism (e.g., `stabilityai/stable-diffusion-xl-refiner-1.0`)
- `--refiner_steps`: Inference steps for refiner (defaults to 20)
- `--refiner_guidance`: Guidance scale for refiner (defaults to same as --guidance)
- `--refiner_precision`: Precision for refiner (defaults to same as --precision)
- `--refiner_negative_prompt`: Negative prompt for refiner (defaults to same as --negative_prompt)

#### **FLUX.1-schnell bf16 flags** (model_id: "black-forest-labs/FLUX.1-schnell")
- `--model_id`: Must be `"black-forest-labs/FLUX.1-schnell"`

Note: Refiner-related flags are ignored for FLUX models

#### **FLUX.1-schnell GGUF (quantized) flags** (model_id: "city96/FLUX.1-schnell-gguf")
- `--model_id`: Hugging Face repository containing quantized model (e.g., `city96/FLUX.1-schnell-gguf`)
- `--model_filename`: GGUF model filename (e.g., `flux1-schnell-Q4_0.gguf`)
- `--vae_repo_id`: VAE model repository (auto-resolved if not specified)
- `--vae_filename`: VAE model filename (auto-resolved if not specified)
- `--clip_l_repo_id`: CLIP-L component repository (auto-resolved if not specified)
- `--clip_l_filename`: CLIP-L component filename (auto-resolved if not specified)
- `--t5xxl_repo_id`: T5-XXL component repository (auto-resolved if not specified)
- `--t5xxl_filename`: T5-XXL component filename (auto-resolved if not specified)

Note: Refiner-related flags are ignored for FLUX models

## Custom Kaggle datasets

To speed up inference, we upload the largest model files as a custom Kaggle dataset. This massively speeds up model inference as there is no need to fetch tens of GBs before every inference run.

Note: temporarily switched off as we found that for FLUX.1-schnell models (and hence, likely for any large model) it actually takes more time to read these models from Kaggle datasets inside of a Kaggle notebook than downloading directly from Hugging Face.
