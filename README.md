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

## Installation

### **1. Clone the repository**
```bash
git clone https://github.com/leweex95/imggenhub.git
cd imggenhub
```

### **2. Install dependencies**
```bash
pip install poetry

# Install project dependencies
poetry install
```

### **3. Configure Kaggle credentials**
```bash
# Create Kaggle credentials directory
mkdir -p ~/.kaggle

# Add your credentials to ~/.kaggle/kaggle.json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json
```

> **Tip**: Get your Kaggle API credentials from [kaggle.com/settings](https://www.kaggle.com/settings) → Create New API Token

---

## Usage

### **Local usage**

#### **Simple single prompt**
```bash
poetry run python -m imggenhub.kaggle.main \
  --prompt "A photorealistic cat in space, 4K quality" \
  --gpu \
  --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
  --dest ./my_images
```

#### **Batch processing with JSON**
```bash
# Create prompts.json
cat > prompts.json << EOF
[
  "A futuristic city skyline at sunset, cyberpunk style",
  "A serene mountain landscape with aurora borealis",
  "A vintage cafe scene with warm lighting"
]
EOF

# Run batch generation
poetry run python -m imggenhub.kaggle.main \
  --prompts_file prompts.json \
  --gpu \
  --dest ./batch_images
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

#### **⚠️ SDXL Base + Refiner pipeline (VRAM intensive)**
> **Warning**: This configuration requires ~30GB VRAM and will **fail on Kaggle** (max 15GB). Use local GPU with sufficient memory.

```bash
poetry run python -m imggenhub.kaggle.main \
  --gpu \
  --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
  --refiner_model_name "stabilityai/stable-diffusion-xl-refiner-1.0" \
  --prompt "A highly detailed photorealistic portrait of a young woman with long flowing hair, professional studio lighting, sharp focus on eyes, cinematic composition, 8K resolution, masterpiece quality" \
  --guidance 10.0 \
  --steps 75 \
  --precision fp16 \
  --negative_prompt "blurry, low quality, distorted, watermark, duplicate, multiple identical people, clones, repetition, cartoon, anime, painting, drawing, sketch, low resolution, pixelated, noisy, grainy, artifacts, overexposed, underexposed, bad anatomy, deformed, ugly, disfigured, poorly lit, bad composition" \
  --dest "photorealistic_portrait_refined"
```
*Output will be saved to: `output/photorealistic_portrait_refined_YYYYMMDD_HHMMSS/`*

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

### **☁️ GitHub Actions usage**

#### **🎨 On-demand image generation**
1. Go to **Actions** → **Image generation**
2. Click **Run workflow**  
3. Configure parameters:
   - **Prompt**: Your image description
   - **Platform**: Currently "kaggle" (more coming soon!)
   - **Model**: Choose from supported Stable Diffusion models
   - **GPU**: Enable for faster generation
   - **Destination**: Output folder name

#### **🔍 Automated testing**
- **Daily regression tests** run automatically at midnight UTC
- **Manual testing** available via **Actions** → **Kaggle pipeline regression test**
- Tests validate deployment, monitoring, and cleanup without consuming resources

---

## Architecture

### **Project structure**
```
imggenhub/
├── src/imggenhub/kaggle/           # Kaggle platform integration
│   ├── main.py                     # CLI entry point and pipeline orchestration
│   ├── core/                       # Core functionality
│   │   ├── deploy.py              # Kernel deployment logic
│   │   └── download.py             # Output retrieval
│   ├── utils/                      # Utilities
│   │   ├── poll_status.py         # Status monitoring
│   │   └── prompts.py             # Prompt processing
│   ├── config/                     # Configuration files
│   │   ├── kernel-metadata.json   # Kaggle kernel settings
│   │   ├── prompts.json           # Sample prompts
│   │   └── *.ipynb                # Jupyter notebook template
│   └── data/                       # Data files
├── .github/workflows/              # CI/CD automation
│   ├── image_generation.yml       # On-demand generation workflow
│   ├── kaggle_regression_test.yml  # Daily testing workflow
│   └── dist_cleanup.yml           # Maintenance workflow (cleanup)
├── output_images/                  # Generated images (auto-created)
├── tests/                          # Test suite
└── pyproject.toml                  # Project configuration
```

### **Pipeline flow**
```mermaid
graph LR
    A[Input Prompt] --> B[Deploy to Kaggle]
    B --> C[Monitor Status]
    C --> D{Complete?}
    D -->|No| C
    D -->|Yes| E[Download Images]
    E --> F[Auto-commit to Repo]
    F --> G[Send Notification]
```

### **Supported models**
- `stabilityai/stable-diffusion-xl-base-1.0` (default)
- `runwayml/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-2-1`  
- `CompVis/stable-diffusion-v1-4`

> **⚠️ Note**: Only public Hugging Face models are supported. Gated models require authentication that's currently incompatible with Kaggle's containerized environment.

---

## Configuration

### **Required secrets (for GitHub Actions)**
```bash
# Kaggle API credentials
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# Email notifications  
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Repository access
PAT_TOKEN=your_github_personal_access_token
```
