# Using ImgGenHub CLI: Step-by-Step Walkthrough

This guide shows exactly what you'll see when using the `imggenhub-vast` CLI for GPU management on Vast.ai.

## Prerequisites

### Vast.ai Account Setup
1. **Create account**: Sign up at [vast.ai](https://vast.ai/)
2. **Add API key**: Get your API key from Account → API Keys
3. **Add credit**: Fund your account with at least $5-10 (required for GPU rental)
   - Credit card or crypto accepted
   - Costs start at ~$0.10/hour for basic GPUs
4. **Configure environment**: Set `VAST_API_KEY` in your `.env` file

### Local Setup
```bash
# Install dependencies
poetry install

# Configure SSH key (for instance access)
ssh-keygen -t ed25519 -C "your-email@example.com"
```

## Command Overview

```
poetry run imggenhub-vast {list|instances|reserve|destroy|destroy-all|run|auto} [options]
```

### **imggenhub-vast list** — Find available GPUs

```bash
poetry run imggenhub-vast list \
  --min-vram 24 \
  --max-hourly-price 0.20 \
  --limit 15
```

**What you'll see:**
```
Offer ID      GPU                  VRAM  $/hr    Reliab%  Location
   22589934  Tesla P40            24GB  0.1100    99.88  EU-DE
   22589945  RTX 2080 Ti          11GB  0.0850    98.50  US-CA
   22589956  V100                 32GB  0.2500    99.90  US-NY
   22589967  RTX 4090             24GB  0.1950    99.75  US-CA
   22589978  A100                 40GB  0.3200    99.95  US-VA
```

**Key columns:**
- **Offer ID**: Unique marketplace ID for this rental offer (use this in `reserve`)
- **GPU**: GPU model name
- **VRAM**: Total memory in GB
- **$/hr**: Hourly rental cost
- **Reliab%**: Uptime reliability score (higher = more stable)
- **Location**: Data center region

### **imggenhub-vast reserve** — Pick and rent a GPU

Once you've found a GPU from `list`, reserve it using the Offer ID:

```bash
poetry run imggenhub-vast reserve \
  --offer-id 22589934 \
  --disk-size 40 \
  --image pytorch/pytorch:latest
```

**What you'll see:**
```
Instance created:
  Instance ID : 28116358
  GPU         : Tesla P40
  Hourly cost : $0.1100
  SSH command : ssh -p 56836 root@183.89.209.74
```

**Save the Instance ID** (`28116358`) — you'll use it in the next step.

### **imggenhub-vast instances** — View all active instances

List all currently running instances (useful for checking costs):

```bash
poetry run imggenhub-vast instances
```

**What you'll see:**
```
Instance ID     GPU                  Status          Cost/hr SSH
  28116358     Tesla P40             running          $0.1100 root@183.89.209.74:56836
  28116359     RTX 3090              running          $0.2500 root@192.168.1.50:42891
```

### **imggenhub-vast destroy** — Stop and destroy a specific instance

Manually destroy one instance to stop billing:

```bash
poetry run imggenhub-vast destroy --instance-id 28116358
```

**Output:**
```
Successfully destroyed instance 28116358
```

### **imggenhub-vast destroy-all** — Emergency cleanup (destroy all instances)

Destroy ALL active instances at once. **Use this when you see unexpected charges:**

```bash
poetry run imggenhub-vast destroy-all
```

**What happens:**
- Lists all active instances
- Destroys each one
- Stops all billing immediately

This is automatically triggered if you encounter an `insufficient_credit` error during `reserve` or `auto` commands.

### **imggenhub-vast run** — Deploy and generate on existing instance

Use the instance ID from `reserve`:

```bash
poetry run imggenhub-vast run \
  --instance-id 28116358 \
  --model-name stabilityai/stable-diffusion-3.5-large \
  --prompt "a photorealistic landscape with mountains" \
  --steps 30 \
  --guidance 7.5
```

**What happens:**
1. Connects via SSH to the instance
2. Uploads code and config
3. Installs dependencies
4. Runs the model
5. Downloads results to `output/vast_ai_results/`

**Output:**
```
Run finished with exit code 0
Log file      : vast_ai_results/generation_log.txt
Output folder : vast_ai_results/
```

### **imggenhub-vast auto** — All-in-one (recommended for beginners)

```bash
poetry run imggenhub-vast auto \
  --prompt "a detailed portrait with professional lighting" \
  --steps 40 \
  --guidance 8.0 \
  --max-price 0.15
```

**What happens automatically:**
1. Searches marketplace for matching GPUs
2. Picks the cheapest option
3. Rents it
4. Waits for SSH (~30-60 seconds)
5. Deploys and generates
6. **Automatically destroys the instance** (saves money!)
7. Downloads results

**Output:**
```
2025-11-22 18:35:12 [INFO] Searching Vast.ai marketplace...
2025-11-22 18:35:14 [INFO] Found 127 offers matching criteria
2025-11-22 18:35:16 [INFO] Selected: Tesla P40 (Offer 22589934, $0.11/hr)
2025-11-22 18:35:46 [INFO] Rented instance 28116358
2025-11-22 18:36:08 [INFO] SSH ready, deploying...
2025-11-22 18:36:45 [INFO] Generating image...
2025-11-22 18:37:15 [INFO] Generation completed in 30.2s (~$0.001)
2025-11-22 18:37:25 [INFO] Destroying instance...
Generation completed in 30.2s (~$0.001)
```

---

## Common Workflows

### Workflow 1: Budget-conscious (fully automatic)
```bash
poetry run imggenhub-vast auto \
  --prompt "your prompt" \
  --max-hourly-price 0.10
```
- Cost: ~$0.003 for P40
- No manual steps needed

### Workflow 2: Quality-focused (automatic with GPU filter)
```bash
poetry run imggenhub-vast auto \
  --gpu-name "RTX 4090" \
  --min-vram 24 \
  --max-hourly-price 0.50 \
  --prompt "your prompt"
```
- Searches specifically for RTX 4090
- 3-5x faster generation
- Still cheaper than manual on-demand

### Workflow 3: Manual selection (control exactly which GPU)
```bash
# Step 1: See available options
poetry run imggenhub-vast list --max-hourly-price 0.20 --min-vram 24 --limit 15

# Step 2: Pick one from the list (e.g., Offer 22589934)
poetry run imggenhub-vast reserve --offer-id 22589934

# Step 3: Use the instance ID you got from reserve (e.g., 28116358)
poetry run imggenhub-vast run --instance-id 28116358 --prompt "your prompt"
```
- Maximum control
- See all options before committing
- Useful for exploring different GPUs

### Workflow 4: Reuse instance for multiple jobs (save setup time)
```bash
# Rent once
poetry run imggenhub-vast auto \
  --prompt "image 1" \
  --keep-instance

# Use same instance (note: no need to reserve again)
poetry run imggenhub-vast run \
  --instance-id 28116358 \
  --prompt "image 2"

# Do it again
poetry run imggenhub-vast run \
  --instance-id 28116358 \
  --prompt "image 3"

# When done, delete from Vast.ai console (costs still accumulate!)
```
- Saves 1-2 minutes per job (no new rental, no setup)
- Remember: **instance still costs money even if idle**
- Always destroy manually when done

---

## Search Filters Reference

All `list` and `auto` commands support these filters:

```bash
poetry run imggenhub-vast {list|auto} \
  --gpu-name "RTX 4090"          # Exact GPU model filter
  --min-vram 32                  # Minimum VRAM in GB
  --max-hourly-price 0.50               # Maximum hourly cost in USD
  --min-reliability 95            # Minimum uptime percentage
  --no-spot                      # Only on-demand (exclude spot instances)
```

---

## Model Selection

Any Hugging Face model can be used. Pass it via `--model-name`:

**Text-to-image models:**
- `stabilityai/stable-diffusion-3.5-large` (default, balanced)
- `black-forest-labs/FLUX.1-schnell` (high quality, slower)
- `stabilityai/stable-diffusion-2-1` (older, faster)

**Use with auto or run:**
```bash
poetry run imggenhub-vast auto \
  --model-name black-forest-labs/FLUX.1-schnell \
  --prompt "your prompt" \
  --steps 25 \
  --guidance 3.5
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No offers matched` | Increase `--max-hourly-price` or decrease `--min-vram` |
| `SSH timeout` | Instance might still be booting; wait 60s and retry |
| `Permission denied (publickey)` | Verify SSH key path in `.env` matches actual key location |
| `CUDA out of memory` | Use fewer `--steps` or try a different GPU with more VRAM |
| Instance still running after auto | Use `--preserve-on-failure` for debugging; normally auto cleans up |

---
