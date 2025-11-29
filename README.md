![Python Version](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/github/license/leweex95/imggenhub) [![Kaggle pipeline regression test](https://github.com/leweex95/imggenhub/actions/workflows/kaggle-regression-test.yml/badge.svg)](https://github.com/leweex95/imggenhub/actions/workflows/kaggle-regression-test.yml) [![Image generation](https://github.com/leweex95/imggenhub/actions/workflows/image-generation.yml/badge.svg)](https://github.com/leweex95/imggenhub/actions/workflows/image-generation.yml) [![Unit Tests](https://github.com/leweex95/imggenhub/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/leweex95/imggenhub/actions/workflows/unit-tests.yml) [![codecov](https://codecov.io/gh/leweex95/imggenhub/branch/master/graph/badge.svg)](https://codecov.io/gh/leweex95/imggenhub)

# ImgGenHub

**A personal, cost-efficient AI image generation platform.**

ImgGenHub is a personal image generation hub that connects to web-based image generation services, currently featuring a fully automated Kaggle-based pipeline with plans for multi-platform support.

---

## Features

### **Current functionality**

#### **Kaggle-powered image generation**
- **Automated pipeline**: Deploy → Monitor → Download workflow
- **GPU/CPU support**: Configurable hardware acceleration via Kaggle's free GPUs (30 hours/week)
- **Multiple models**: Support for popular Stable Diffusion variants and Flux.1-schnell quantized (Q4) version
- **Flexible prompting**: Command-line prompts or JSON file batch processing

---

## Requirements

- **Python**: 3.11 or higher
- **Poetry**: For dependency management  
- **Kaggle Account**: With API credentials configured
- **Git**: For repository management

---

## Usage

### **Local usage**

#### **Simple single prompt**
```bash
poetry run imggenhub \
  --model_name stabilityai/stable-diffusion-xl-base-1.0 \
  --refiner_model_name stabilityai/stable-diffusion-xl-refiner-1.0 \
  --gpu \
  --prompt "photorealistic parliament building along river in summer" \
  --guidance 8.0 --steps 30 \
  --refiner_guidance 7.0 --refiner_steps 15 \
  --precision fp16 --refiner_precision fp16 --two_stage_refiner
```

#### **Advanced photorealistic generation**
```bash
poetry run imggenhub \
  --gpu \
  --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
  --prompt "A highly detailed photorealistic portrait of a young woman with long flowing hair, professional studio lighting, sharp focus on eyes, cinematic composition, 8K resolution, masterpiece quality" \
  --guidance 10.0 \
  --steps 75 \
  --precision fp16 \
  --negative_prompt "blurry, low quality, distorted, watermark, duplicate, multiple identical people, clones, repetition, cartoon, anime, painting, drawing, sketch, low resolution, pixelated, noisy, grainy, artifacts, overexposed, underexposed, bad anatomy, deformed, ugly, disfigured, poorly lit, bad composition" \
  --dest "photorealistic_portrait"
```
*Output will be saved to: `output/photorealistic_portrait_YYYYMMDD_HHMMSS/`*

#### **FLUX.1-schnell Q4 GGUF quantized generation**
FLUX GGUF uses quantized models for faster generation with lower memory requirements:

```bash
poetry run imggenhub \
  --model_name "flux-gguf-q4" \
  --prompt "A beautiful landscape with mountains and a lake, photorealistic, high detail, cinematic lighting" \
  --guidance 3.5 \
  --steps 4 \
  --precision q4 \
  --img_width 1024 \
  --img_height 1024
```

**FLUX GGUF Features:**
- **Automatic GPU enforcement**: CPU mode not supported (too slow)
- **Quantized Q4 model**: Reduced memory footprint
- **Fast generation**: 4 steps typical
- **Default image size**: 1024x1024 (customizable via `--img_width` and `--img_height`)
- **Model sources**: Supports both Kaggle datasets (default) and direct HuggingFace download
  - Set `MODEL_SOURCE=huggingface` and `HF_TOKEN=your_token` in `.env` for HF downloads

*Output will be saved to: `output/output_images_YYYYMMDD_HHMMSS/` or custom destination with `--dest`*

#### **All supported flags**
- `--dest DEST`: Custom name prefix for the output folder (default: timestamp only)
- Outputs are always saved under `output/` with automatic timestamping
- All logs from the generation process are saved in the same folder as the images
- `--prompt`: Single prompt string
- `--prompts_file`: JSON file with multiple prompts  
- `--model_name`: Hugging Face model ID
- `--refiner_model_name`: SDXL refiner model for enhanced photorealism
- `--gpu`: Enable GPU acceleration
- `--precision`: Model precision (fp32/fp16/int8/int4)
- `--guidance`: Prompt adherence strength (7-12 recommended for photorealism)
- `--steps`: Inference steps (50-100 for quality)
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
