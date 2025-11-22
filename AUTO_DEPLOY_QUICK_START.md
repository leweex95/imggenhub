# Vast.ai GPU Deployment: Complete Workflow

## Setup (one-time)

1. Create a [Vast.ai account](https://vast.ai)
2. Get your API key from [console.vast.ai/account](https://console.vast.ai/account)
3. Create `.env` in the repository root:
   ```bash
   VAST_AI_API_KEY=your-api-key-here
   SSH_PRIVATE_KEY_PATH=~/.ssh/id_ed25519
   ```
4. Generate or use an existing SSH key at `~/.ssh/id_ed25519`

---

## 🚀 Quick Start: Automatic (Recommended)

**One command: search → rent → deploy → cleanup**

```bash
# Generate with default Stable Diffusion 3.5
poetry run imggenhub-vast auto \
  --prompt "A photorealistic landscape with mountains and lake" \
  --steps 30 \
  --guidance 7.5
```

**What happens automatically:**
1. Searches for GPUs matching your criteria (default: P40+, max $1.00/hr)
2. Finds the cheapest available instance
3. Rents it (~30 sec)
4. Waits for SSH (~30-60 sec)
5. Uploads code and generates image (~10-30 sec depending on GPU)
6. **Automatically destroys the instance** when done (saves money!)
7. Downloads results to `output/vast_ai_results/`

**Total time:** ~2-3 minutes  
**Cost:** ~$0.03 for P40 GPU

---

## 🎨 Choosing Models

Use any Hugging Face model by changing `--model-name`:

**Stable Diffusion (default, fastest):**
```bash
poetry run imggenhub-vast auto \
  --model-name stabilityai/stable-diffusion-3.5-large \
  --prompt "your prompt" \
  --steps 30 \
  --guidance 7.5
```

**Flux (highest quality, slower):**
```bash
poetry run imggenhub-vast auto \
  --model-name black-forest-labs/FLUX.1-schnell \
  --prompt "your prompt" \
  --steps 25 \
  --guidance 3.5
```

---

## 💰 Cost Examples

### Example 1: Budget-conscious (P40, ~2 min)
```bash
poetry run imggenhub-vast auto \
  --prompt "a landscape" \
  --max-hourly-price 0.15
```
- GPU: Tesla P40 (~$0.11/hr)
- Time: ~2 minutes
- **Cost: ~$0.004**

### Example 2: Quality-focused (RTX 4090, ~1 min)
```bash
poetry run imggenhub-vast auto \
  --gpu-name "RTX 4090" \
  --min-vram 24 \
  --max-hourly-price 1.00 \
  --prompt "a detailed portrait"
```
- GPU: RTX 4090 (3-5x faster)
- Time: ~1 minute
- **Cost: ~$0.015** (still cheaper than P40 for 1 hour!)

### Example 3: Keep instance running for multiple jobs
```bash
# Rent once, use multiple times
poetry run imggenhub-vast auto \
  --prompt "image 1" \
  --keep-instance

# Use same instance
poetry run imggenhub-vast run \
  --instance-id <from-output> \
  --prompt "image 2"

# Done, destroy it
poetry run imggenhub-vast reserve --offer-id <id>  # Later: destroy via API
```

---

## 🔍 Manual Workflow: Fine-grained Control

If you want to manually select a GPU instead of auto-searching:

### Step 1: List available GPUs

Filter by price and VRAM:
```bash
poetry run imggenhub-vast list \
  --min-vram 24 \
  --max-hourly-price 0.20 \
  --limit 15
```

Output:
```
Offer ID      GPU                  VRAM  $/hr    Reliab%  Location
   22589934  Tesla P40           24GB  0.1100    99.88  EU-DE
   22589945  RTX 2080 Ti         11GB  0.0850    98.50  US-CA
   22589956  V100                 32GB  0.2500    99.90  US-NY
```

### Step 2: Reserve a specific GPU

Pick one from the list (using the Offer ID):
```bash
poetry run imggenhub-vast reserve \
  --offer-id 22589934 \
  --disk-size 40 \
  --image pytorch/pytorch:latest
```

Output:
```
Instance created:
  Instance ID : 28116358
  GPU         : Tesla P40
  Hourly cost : $0.1100
  SSH command : ssh -p 56836 root@183.89.209.74
```

### Step 3: Deploy and generate

Use the instance ID from step 2:
```bash
poetry run imggenhub-vast run \
  --instance-id 28116358 \
  --model-name stabilityai/stable-diffusion-3.5-large \
  --prompt "a photorealistic landscape" \
  --steps 30 \
  --guidance 7.5
```

### Step 4: Cleanup

Manually destroy when done (via Vast.ai dashboard or API)

---

## ⚙️ Advanced Options

### All search filters
```bash
poetry run imggenhub-vast auto \
  --gpu-name "RTX 4090"          # Exact GPU match
  --min-vram 32                  # Minimum VRAM in GB
  --max-hourly-price 0.50               # Maximum $/hour
  --min-reliability 95            # Uptime percentage
  --no-spot                      # Only on-demand (not spot)
```

### Preserve instance on failure
```bash
poetry run imggenhub-vast auto \
  --prompt "your prompt" \
  --preserve-on-failure          # Don't destroy if generation fails
  --keep-instance                # Keep running after success
```

### Use custom Hugging Face token (for gated models)
```bash
poetry run imggenhub-vast auto \
  --model-name meta-llama/Llama-2-7b \
  --hf-token your-hf-token \
  --prompt "your prompt"
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No offers matched" | Increase `--max-hourly-price` or decrease `--min-vram` |
| SSH timeout | GPU still booting; wait a minute and try `run` again |
| "Permission denied" | Verify SSH key at `SSH_PRIVATE_KEY_PATH` in `.env` |
| Instance left running | Use `--keep-instance` sparingly; manually destroy in dashboard |

---

## 🎯 Search criteria (all optional)

```bash
# Find Tesla P40 under $0.15/hr
--gpu-name "P40" --max-hourly-price 0.15

# Find 40GB+ VRAM GPU under $1/hr
--min-vram 40 --max-hourly-price 1.0

# Only high-reliability (99%+) instances
--min-reliability 99.0

# Avoid spot instances (more stable, slightly pricier)
--no-spot

# Custom inference settings
--steps 50 --guidance 8.5
```

---

## 🔄 Keep instance for multiple generations

```bash
# First generation
poetry run imggenhub-auto-sd sd \
  --prompt "First image" \
  --keep-instance

# Output will show instance ID, e.g., 28116358

# Use same instance for more images
poetry run imggenhub-vast-sd \
  --instance-id 28116358 \
  --ssh-key ~/.ssh/id_ed25519 \
  --prompt "Second image"

# When done, clean up
poetry run imggenhub-vast-destroy 28116358
```

---

## 📈 Real-time monitoring

Auto-deployment streams output in real-time:

```
[INFO] Searching Vast.ai marketplace...
[INFO] Found 12 matching offers
[INFO]   1. RTX 4090 (24GB) @ $0.45/hr (reliability: 99.8%)
[INFO]   2. RTX 3090 (24GB) @ $0.28/hr (reliability: 99.2%)
[INFO]   3. Tesla P40 (24GB) @ $0.11/hr (reliability: 99.88%)
[INFO] Renting Tesla P40 (24GB) ($0.11/hr)...
[INFO] ✓ Instance rented: 28116358
[INFO] Waiting for SSH to be ready...
[INFO] ✓ SSH ready after 3 attempts
[INFO] Deploying Stable Diffusion...
[INFO] [PROGRESS] Loading model...
[INFO] [PROGRESS] Step 1/30...
[INFO] ✓ Generation completed in 45.2s
[INFO] Cost: $0.0014
[INFO] Destroying instance 28116358...
```

---

## 💡 Tips for best results

1. **Off-peak times** → More GPUs available, better prices
2. **Batch generations** → Keep instance alive for multiple images
3. **Lower steps** → Use `--steps 20-25` instead of 50+ for drafts
4. **Check specs** → `--gpu-name "RTX 4090"` forces high-end only
5. **Test first** → Try with `--max-hourly-price 0.20` before exploring expensive GPUs

---

## ❌ Troubleshooting

**"No suitable GPU offers found"**
- Relax criteria: increase `--max-hourly-price`, decrease `--min-vram`
- Try: `--max-hourly-price 1.5 --min-vram 20`

**SSH connection timeout**
- Increase timeout: `--ssh-timeout 600` (10 minutes)
- Check API key: verify VASTAI_API_KEY in .env

**Generation failed but instance still running**
- Use: `poetry run imggenhub-vast-destroy <instance-id>`
- Check: `ssh -p PORT root@IP` to verify manually

---

## 📚 Full CLI options

```bash
poetry run imggenhub-auto-sd --help
```

Shows all available parameters for fine-tuning searches and deployments.
