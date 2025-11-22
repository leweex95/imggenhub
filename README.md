![Python Version](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/github/license/leweex95/imggenhub) [![Kaggle pipeline regression test](https://github.com/leweex95/imggenhub/actions/workflows/kaggle-regression-test.yml/badge.svg)](https://github.com/leweex95/imggenhub/actions/workflows/kaggle-regression-test.yml) [![Image generation](https://github.com/leweex95/imggenhub/actions/workflows/image-generation.yml/badge.svg)](https://github.com/leweex95/imggenhub/actions/workflows/image-generation.yml) [![Unit Tests](https://github.com/leweex95/imggenhub/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/leweex95/imggenhub/actions/workflows/unit-tests.yml) [![codecov](https://codecov.io/gh/leweex95/imggenhub/branch/master/graph/badge.svg)](https://codecov.io/gh/leweex95/imggenhub)

# ImgGenHub

**A personal, cost-efficient AI image generation platform.**

ImgGenHub is a personal image generation hub that connects to web-based image generation services, currently featuring a fully automated Kaggle-based pipeline with paid Vast.ai GPU support for high-speed generation.

---

## Features

### **Current functionality**

#### **Kaggle-powered image generation**
- **Automated pipeline**: Deploy → Monitor → Download workflow
- **GPU/CPU support**: Configurable hardware acceleration via Kaggle's free T4×2 GPUs (30 hours/week)
- **Multiple models**: Support for popular Stable Diffusion variants including SDXL with refiner
- **Batch processing**: Multiple prompts in a single execution
- **Flexible prompting**: Command-line prompts or JSON file batch processing

#### **Vast.ai GPU deployment**
- **On-demand GPU rental**: Rent Tesla P40, RTX 3090, RTX 4090, and other GPUs by the hour
- **Real-time streaming**: Console output and progress logs stream live to your local terminal
- **Multiple models**: Stable Diffusion, Flux, and Forge UI support
- **Cost-effective**: $0.10-0.50/hr depending on GPU selection

#### **GitHub Actions automation**
- **On-Demand generation**: Manual workflow triggers with custom parameters
- **Automated testing**: Daily regression tests to ensure pipeline stability
- **Email notifications**: Success/failure alerts with generation details

---

## Usage

### **Local usage**

#### **Simple single prompt**
```bash
python src/imggenhub/kaggle/main.py \
  --model_name stabilityai/stable-diffusion-xl-base-1.0 \
  --gpu \
  --prompt "photorealistic parliament building along river in summer" \
  --guidance 10.0 --steps 30
```

#### **Multiple prompts with refiner**
```bash
python src/imggenhub/kaggle/main.py \
  --gpu \
  --prompts "bombed out high rise soviet apartment in kharkiv 2022 ukraine war" "haunted house frightening horror" "peaceful forest autumn sunlight" \
  --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
  --refiner_model_name "stabilityai/stable-diffusion-xl-refiner-1.0" \
  --guidance 10.0 --steps 30 \
  --refiner_guidance 7.0 --refiner_steps 15
```

#### **Advanced photorealistic generation (tested & working)**
```bash
python src/imggenhub/kaggle/main.py \
  --gpu \
  --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
  --prompt "A highly detailed photorealistic portrait of a young woman with long flowing hair, professional studio lighting, sharp focus on eyes, cinematic composition, 8K resolution, masterpiece quality" \
  --guidance 10.0 \
  --steps 75 \
  --precision fp16 \
  --negative_prompt "blurry, low quality, distorted, watermark, duplicate, multiple identical people, clones, repetition, cartoon, anime, painting, drawing, sketch, low resolution, pixelated, noisy, grainy, artifacts, overexposed, underexposed, bad anatomy, deformed, ugly, disfigured, poorly lit, bad composition" \
  --dest "photorealistic_portrait"
```
*Output will be saved to: `output/photorealistic_portrait_YYYYMMDD_HHMMSS/`*. If `--dest` is not provided, the subfolder will only be the datetime one.

#### **Available CLI options**
```bash
python src/imggenhub/kaggle/main.py --help
```

**Key parameters:**
- `--dest DEST`: Custom name prefix for the output folder (default: timestamp only)
- Outputs are always saved under `output/` with automatic timestamping
- All logs from the generation process are saved in the same folder as the images
- `--prompt`: Single prompt string
- `--prompts`: Multiple prompts as space-separated arguments
- `--prompts_file`: JSON file with multiple prompts
- `--model_name`: Hugging Face model ID (required)
- `--refiner_model_name`: SDXL refiner model for enhanced photorealism
- `--gpu`: Enable GPU acceleration on Kaggle
- `--precision`: Model precision (fp32/fp16/int8/int4, required)
- `--guidance`: Prompt adherence strength (7-12 recommended for photorealism, required)
- `--steps`: Inference steps (50-100 for quality, required)
- `--negative_prompt`: Quality control prompts
- `--refiner_guidance`: Guidance scale for refiner (defaults to same as --guidance)
- `--refiner_steps`: Inference steps for refiner (defaults to 20)
- `--refiner_precision`: Precision for refiner (REQUIRED when using --refiner_model_name)
- `--refiner_negative_prompt`: Negative prompt for refiner (defaults to same as --negative_prompt)
- `--hf_token`: HuggingFace API token for accessing gated models
- `--notebook`: Custom notebook path (default: ./config/kaggle-notebook-image-generation.ipynb)
- `--kernel_path`: Kaggle kernel configuration directory (default: ./config)

---

## Setup

### **Vast.ai Account (for GPU deployment)**

1. **Create account**: Sign up at [vast.ai](https://vast.ai/)
2. **Add API key**: Get your API key from Account → API Keys  
3. **Add credit**: Fund your account with at least $5-10 (required for GPU rental)
   - Credit card or crypto accepted
   - Costs start at ~$0.10/hour for basic GPUs
4. **Configure environment**: Set `VAST_API_KEY` in your `.env` file

### **Local Installation**

```bash
# Clone repository
git clone https://github.com/leweex95/imggenhub.git
cd imggenhub

# Install dependencies
poetry install

# Configure SSH key (for Vast.ai instance access)
ssh-keygen -t ed25519 -C "your-email@example.com"
```

---

## Vast.ai GPU Deployment

### **Automatic GPU deployment (recommended)**

Let ImgGenHub find the cheapest qualifying GPU, rent it, run your prompt, and shut it down when finished:

```bash
# Auto-search for cheapest P40+ GPU under $0.50/hr and generate
poetry run imggenhub-vast auto \
  --prompt "a photorealistic landscape" \
  --max-hourly-price 0.50 \
  --steps 30 \
  --guidance 7.5

# Focus on RTX 4090 offers with 40GB+ VRAM
poetry run imggenhub-vast auto \
  --gpu-name "RTX 4090" \
  --min-vram 40 \
  --max-hourly-price 0.80 \
  --prompt "a detailed portrait" \
  --steps 45

# Flux model (just change --model-name)
poetry run imggenhub-vast auto \
  --model-name black-forest-labs/FLUX.1-schnell \
  --prompt "abstract artwork" \
  --steps 25 \
  --guidance 3.5
```

Use `--keep-instance` to leave the machine running for additional jobs. Without it, ImgGenHub automatically destroys the instance after the run finishes (or immediately on failure unless `--preserve-on-failure` is set).

#### **Auto-deployment search criteria**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--min-vram` | 24 GB | Minimum VRAM required |
| `--max-hourly-price` | $1.00/hr | Maximum hourly cost |
| `--gpu-name` | Any | Filter by GPU type (e.g., RTX 4090) |
| `--min-reliability` | 95% | Minimum uptime reliability |
| `--no-spot` | Disabled | Only on-demand (no spot instances) |

#### **Automatic vs Manual comparison**

| Task | Manual | Auto |
|------|--------|------|
| Search GPUs | ✋ Dashboard filters | 🤖 `imggenhub-vast auto` search |
| Rent GPU | ✋ Click "Rent" | 🤖 API rental |
| Wait for SSH | ✋ Manual polling | 🤖 Auto-polling |
| Deploy | ✋ SSH + scripts | 🤖 Automatic upload & run |
| Cleanup | ✋ Destroy instance | 🤖 Automatic unless `--keep-instance` |
| **Typical cost** | ~$0.113 (P40 for 1 hr) | ~$0.03 (P40 for 15 min) |

---

### **Manual GPU workflow (advanced)**

Sometimes you want full control: inspect available offers, pick one, rent it, then decide which model to run. The unified `imggenhub-vast` CLI exposes each step.

#### 1. List available offers
```bash
poetry run imggenhub-vast list \
  --max-hourly-price 0.20 \
  --min-vram 24 \
  --limit 15
```

**Example output:**
```
Offer ID      GPU                  VRAM  $/hr    Reliab%  Location
   22589934  Tesla P40            24GB  0.1100    99.88  EU-DE
   22589945  RTX 2080 Ti          11GB  0.0850    98.50  US-CA
   22589956  V100                 32GB  0.2500    99.90  US-NY
   22589967  RTX 4090             24GB  0.1950    99.75  US-CA
```

**Note:** The "Offer ID" is a marketplace identifier (different from the instance ID you'll get after reserving).

#### 2. Reserve a specific offer
```bash
poetry run imggenhub-vast reserve \
  --offer-id 22589934 \
  --disk-size 40 \
  --image pytorch/pytorch:latest
```
The command prints the instance ID, SSH endpoint, and hourly cost. **Note:** The offer ID (from step 1) is different from the instance ID (created in this step).

#### 3. Run any model by supplying its `--model-name`

Stable Diffusion example:
```bash
poetry run imggenhub-vast run \
  --instance-id 28116358 \
  --ssh-key ~/.ssh/id_ed25519 \
  --model-name stabilityai/stable-diffusion-3.5-large \
  --prompt "a photorealistic landscape with mountains and lake at sunset" \
  --steps 30 --guidance 7.5
```

Flux example (same command, different model/guidance):
```bash
poetry run imggenhub-vast run \
  --instance-id 28116358 \
  --ssh-key ~/.ssh/id_ed25519 \
  --model-name black-forest-labs/FLUX.1-schnell \
  --prompt "a portrait with professional studio lighting, sharp focus" \
  --steps 25 --guidance 3.5
```

Use the same `run` command for any Hugging Face model—only the `--model-name`, `--steps`, and `--guidance` need to change.

#### 4. Destroy instances to stop billing

View all active instances:
```bash
poetry run imggenhub-vast instances
```

Destroy one specific instance:
```bash
poetry run imggenhub-vast destroy --instance-id 28116358
```

Destroy all instances (emergency cleanup):
```bash
poetry run imggenhub-vast destroy-all
```

**Important:** Instances continue billing as long as they're running, even if idle. Always destroy instances when done.

#### **Deploy Forge UI (web interface)**
```bash
poetry run imggenhub-vast-forge \
  --instance-id 28116358 \
  --ssh-key ~/.ssh/id_ed25519
```
Then access the web interface at the displayed URL (typically `http://host:7860`).

#### **Test SSH connectivity (dummy run)**
```bash
poetry run imggenhub-vast-dummy \
  --instance-id 28116358 \
  --ssh-key ~/.ssh/id_ed25519
```

### **Real-time Output Streaming**

All Vast.ai commands stream console output live as the remote GPU processes:
- ? Progress updates and logs appear instantly
- ? Errors are caught and displayed immediately
- ? Output is saved locally for review

### **Supported GPU Models**

Tested configurations:

| GPU | VRAM | Cost/hr | Best For |
|-----|------|---------|----------|
| **Tesla P40** | 24GB GDDR5 | $0.11-0.15 | Stable Diffusion, basic Flux |
| **RTX 3090** | 24GB GDDR6X | $0.20-0.30 | Fast SD generation, Flux |
| **RTX 4090** | 24GB GDDR6X | $0.40-0.50 | High-speed SD/Flux, batches |
| **A100** | 40GB HBM2 | $0.50-1.00 | Multi-batch, large models |

### **Cost-Effective Tips**

- ?? Use **spot instances** for 30-70% savings
- ?? **Filter by location** to reduce latency and cost
- ?? **Batch multiple generations** in one session
- ? Use **`--steps 20-30`** for cost-effective quality
- ?? Download generated images immediately to free GPU memory

### **SSH Connection Details**

For direct SSH access:
```bash
# Direct connection
ssh -p 56836 root@183.89.209.74

# Via proxy (if direct fails)
ssh -p 36359 root@ssh4.vast.ai
```

### **Troubleshooting**

**Insufficient credit error?**
- ImgGenHub automatically destroys all instances to prevent further charges
- Add credit to your Vast.ai account
- Retry the operation

**Connection timeout?**
- Verify SSH key permissions: `chmod 600 ~/.ssh/id_ed25519`
- Check firewall: Vast.ai instances require outbound SSH access
- Verify instance is still running on Vast.ai dashboard

**GPU out of memory?**
- Reduce `--steps` parameter
- Use smaller models (SD 1.5 vs SDXL)

**Slow generation?**
- Check GPU utilization: `nvidia-smi` on the instance
- Consider using RTX 4090 for faster speeds

**Unexpected charges/instances running?**
- Use `poetry run imggenhub-vast instances` to see all active instances
- Use `poetry run imggenhub-vast destroy-all` for emergency cleanup

````
