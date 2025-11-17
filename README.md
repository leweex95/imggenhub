![Python Version](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/github/license/leweex95/imggenhub) [![Kaggle pipeline regression test](https://github.com/leweex95/imggenhub/actions/workflows/kaggle-regression-test.yml/badge.svg)](https://github.com/leweex95/imggenhub/actions/workflows/kaggle-regression-test.yml) [![Image generation](https://github.com/leweex95/imggenhub/actions/workflows/image-generation.yml/badge.svg)](https://github.com/leweex95/imggenhub/actions/workflows/image-generation.yml) [![Unit Tests](https://github.com/leweex95/imggenhub/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/leweex95/imggenhub/actions/workflows/unit-tests.yml) [![codecov](https://codecov.io/gh/leweex95/imggenhub/branch/master/graph/badge.svg)](https://codecov.io/gh/leweex95/imggenhub)

# ImgGenHub

**A personal, cost-efficient AI image generation platform.**

ImgGenHub is a personal image generation hub that connects to web-based image generation services, currently featuring a fully automated Kaggle-based pipeline with plans for multi-platform support.

---

## Features

### **Current functionality**

#### **Kaggle-powered image generation**
- **Automated pipeline**: Deploy → Monitor → Download workflow
- **GPU/CPU support**: Configurable hardware acceleration via Kaggle's free T4×2 GPUs (30 hours/week)
- **Multiple models**: Support for popular Stable Diffusion variants
- **Flexible prompting**: Command-line prompts or JSON file batch processing

#### **GitHub Actions automation**
- **On-Demand generation**: Manual workflow triggers with custom parameters
- **Automated testing**: Daily regression tests to ensure pipeline stability
- **Email notifications**: Success/failure alerts with generation details

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
poetry run python -m imggenhub.kaggle.main \
  --model_name stabilityai/stable-diffusion-xl-base-1.0 \
  --refiner_model_name stabilityai/stable-diffusion-xl-refiner-1.0 \
  --gpu \
  --prompt "photorealistic parliament building along river in summer" \
  --guidance 8.0 --steps 30 \
  --refiner_guidance 7.0 --refiner_steps 15 \
  --precision fp16 --refiner_precision fp16 --two_stage_refiner
```

#### **Advanced photorealistic generation (tested & working)**
```bash
poetry run python -m imggenhub.kaggle.main \
  --gpu \
  --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
  --prompt "A highly detailed photorealistic portrait of a young woman with long flowing hair, professional studio lighting, sharp focus on eyes, cinematic composition, 8K resolution, masterpiece quality" \
  --guidance 10.0 \
  --steps 75 \
  --precision int8 \
  --negative_prompt "blurry, low quality, distorted, watermark, duplicate, multiple identical people, clones, repetition, cartoon, anime, painting, drawing, sketch, low resolution, pixelated, noisy, grainy, artifacts, overexposed, underexposed, bad anatomy, deformed, ugly, disfigured, poorly lit, bad composition" \
  --dest "photorealistic_portrait"
```
*Output will be saved to: `output/photorealistic_portrait_YYYYMMDD_HHMMSS/`*

#### **Available CLI options**
```bash
poetry run python -m imggenhub.kaggle.main --help
```

#### **Output directory structure**
All generated images and logs are automatically organized in timestamped subfolders under the `output/` directory:

```
output/
├── final_working_test_20251114_121500/  # --dest "final_working_test" + timestamp
│   ├── generated_images/                 # Images from the notebook
│   ├── kaggle_cli_stdout.log            # Kaggle CLI output logs
│   ├── kaggle_cli_stderr.log            # Kaggle CLI error logs
│   └── stable-diffusion-batch-generator.log  # Kernel execution logs (if any)
└── 20251114_121600/                     # Default timestamp-only naming
    ├── generated_images/
    └── ...
```

**Key parameters:**
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
