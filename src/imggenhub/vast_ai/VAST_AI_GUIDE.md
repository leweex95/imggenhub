# Vast.ai Integration Guide

This guide explains how to use the Vast.ai GPU cloud integration to run image generation pipelines on remote GPUs without owning local hardware.

## Prerequisites

- Vast.ai account with API key (get one at https://vast.ai/)
- Local repository with `imggenhub` project
- SSH access capability (most systems have this built-in)
- Valid HuggingFace token (optional, for gated models like FLUX.1)

## Quick Start

### 1. Get Your Vast.ai API Key

1. Log in to https://cloud.vast.ai/
2. Go to Settings → API
3. Copy your API key

### 2. Search for Available GPUs

Before renting, check available GPUs and pricing:

```bash
poetry run python -c "
from imggenhub.vast_ai.client import VastAiClient

api_key = 'YOUR_API_KEY'
client = VastAiClient(api_key)

# Search for GPUs under $1/hr
offers = client.search_offers(max_price=1.0, limit=10)
for offer in offers:
    print(f\"GPU: {offer['gpu_name']}, Price: \${offer['price_per_hour']}/hr\")
"
```

### 3. Rent a GPU Instance

Note the `id` (ask_id) from the search results, then create an instance:

```bash
poetry run python -c "
from imggenhub.vast_ai.client import VastAiClient

api_key = 'YOUR_API_KEY'
offer_id = 12345  # From search results
image = 'pytorch/pytorch:latest'

client = VastAiClient(api_key)
instance = client.create_instance(
    offer_id=offer_id,
    image=image,
    disk_size=20,
    label='imggenhub-run'
)
print(f'Instance created: {instance.id}')
print(f'SSH connection: {instance.ssh_user}@{instance.ssh_host}:{instance.ssh_port}')
"
```

### 4. Run Image Generation Pipeline

Execute the pipeline on your rented GPU:

```bash
poetry run imggenhub-vast \
  --api_key YOUR_API_KEY \
  --instance_id INSTANCE_ID \
  --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
  --guidance 7.5 \
  --steps 50 \
  --precision fp16 \
  --prompt "A beautiful landscape" \
  --config_path ./src/imggenhub/kaggle/config \
  --repo_path .
```

### 5. Monitor and Download Results

Results are automatically downloaded to `output/vast_ai_results/`.

## Full CLI Reference

### Remote Pipeline Execution

```bash
poetry run imggenhub-vast [OPTIONS]
```

**Required Arguments:**
- `--api_key`: Vast.ai API key
- `--instance_id`: ID of the rented instance
- `--model_name`: Base model (e.g., "stabilityai/stable-diffusion-xl-base-1.0")
- `--guidance`: Guidance scale (float, 7-12 recommended)
- `--steps`: Number of inference steps (int, 50-100 for quality)
- `--precision`: Precision level (fp32|fp16|int8|int4)
- `--config_path`: Path to config directory with prompts.json

**Optional Arguments:**
- `--prompt`: Single prompt string
- `--prompts_file`: Path to prompts JSON file
- `--negative_prompt`: Negative prompt for better control
- `--refiner_model_name`: Optional refiner model
- `--refiner_guidance`: Guidance for refiner (required if using refiner)
- `--refiner_steps`: Steps for refiner (required if using refiner)
- `--refiner_precision`: Precision for refiner (required if using refiner)
- `--hf_token`: HuggingFace API token (for gated models)
- `--ssh_key`: Path to SSH private key (default: uses system SSH agent)
- `--repo_path`: Path to repository root (default: current directory)

## Example Workflows

### Single Image Generation

```bash
poetry run imggenhub-vast \
  --api_key $VAST_API_KEY \
  --instance_id 5678 \
  --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
  --guidance 7.5 \
  --steps 50 \
  --precision fp16 \
  --prompt "A serene mountain landscape at sunset" \
  --config_path ./src/imggenhub/kaggle/config
```

### Batch Generation with Prompts File

Create `prompts.json`:

```json
[
  "A futuristic city at night",
  "A peaceful forest with waterfalls",
  "An abstract colorful pattern"
]
```

Then run:

```bash
poetry run imggenhub-vast \
  --api_key $VAST_API_KEY \
  --instance_id 5678 \
  --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
  --guidance 7.5 \
  --steps 50 \
  --precision fp16 \
  --prompts_file ./prompts.json \
  --config_path ./src/imggenhub/kaggle/config
```

### With Refiner for Enhanced Quality

```bash
poetry run imggenhub-vast \
  --api_key $VAST_API_KEY \
  --instance_id 5678 \
  --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
  --guidance 7.5 \
  --steps 50 \
  --precision fp16 \
  --prompt "High-quality professional photo" \
  --refiner_model_name "stabilityai/stable-diffusion-xl-refiner-1.0" \
  --refiner_guidance 7.5 \
  --refiner_steps 25 \
  --refiner_precision fp16 \
  --config_path ./src/imggenhub/kaggle/config
```

### Using FLUX (Requires HuggingFace Token)

```bash
poetry run imggenhub-vast \
  --api_key $VAST_API_KEY \
  --instance_id 5678 \
  --model_name "black-forest-labs/FLUX.1-schnell" \
  --guidance 3.5 \
  --steps 4 \
  --precision fp16 \
  --prompt "A masterpiece photograph" \
  --hf_token $HF_TOKEN \
  --config_path ./src/imggenhub/kaggle/config
```

## Cost Management

### Estimate Costs

Each GPU is priced per hour. Example:
- RTX 4090: ~$0.40/hour
- H100: ~$1.50/hour
- A100: ~$0.80/hour

**Calculation:**
```
Cost = GPU hourly rate × (setup time + inference time)
```

Average inference times:
- Stable Diffusion: 2-5 minutes
- FLUX.1-schnell: 1-2 minutes
- With refiner: +50% time

### Stop Instance When Done

Always terminate instances when finished to avoid charges:

```bash
poetry run python -c "
from imggenhub.vast_ai.client import VastAiClient

api_key = 'YOUR_API_KEY'
instance_id = 5678

client = VastAiClient(api_key)
client.destroy_instance(instance_id)
print('Instance terminated')
"
```

## Advanced Usage

### Using Python API Directly

```python
from imggenhub.vast_ai.client import VastAiClient
from imggenhub.vast_ai.executor import RemoteExecutor

# Initialize client
api_key = 'YOUR_API_KEY'
client = VastAiClient(api_key)

# Get instance details
instance = client.get_instance(5678)

# Create executor
executor = RemoteExecutor(api_key, instance)

# Setup and run pipeline
executor.setup_environment()
executor.upload_codebase('/path/to/repo')
executor.upload_config('/path/to/config')
executor.install_dependencies()

exit_code, stdout, stderr = executor.run_pipeline(
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    guidance=7.5,
    steps=50,
    precision="fp16",
    prompt="Your prompt here"
)

executor.download_results('/workspace/output', './output')
```

## Troubleshooting

### Connection Timeout

- Verify instance is running: `client.get_instance(instance_id)`
- Check firewall/NAT on your network
- Try with explicit SSH key: `--ssh_key /path/to/key`

### Out of Memory

- Reduce `--steps` (e.g., 30 instead of 50)
- Use lower precision: `--precision int8` or `--precision int4`
- Use smaller model: Stable Diffusion instead of XL
- Reduce batch size in config

### Slow Performance

- Check GPU utilization on instance details
- Switch to faster GPU (e.g., RTX 4090 over A100 for SDXL)
- Use `--precision int8` for faster inference
- Try FLUX.1-schnell (fastest model)

### SSH Connection Issues

Generate SSH key if needed:

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/vast_ai
poetry run imggenhub-vast ... --ssh_key ~/.ssh/vast_ai
```

## Best Practices

1. **Start small**: Test with 1-2 prompts before batch processing
2. **Monitor costs**: Set alerts on Vast.ai dashboard
3. **Keep instances minimal**: Don't rent multiple instances if one suffices
4. **Use spot pricing**: Cheaper but can be interrupted (no SLA)
5. **Clean up**: Always destroy instances when done
6. **Backup results**: Download and store important outputs locally
7. **Test locally first**: Validate your prompts and settings locally if possible

## Integration with Kaggle

The Vast.ai integration complements Kaggle integration:
- Use Kaggle when you have small workloads and want free GPU
- Use Vast.ai when you need specific GPUs, better performance, or have large batches

Switch between them by using different CLI commands:
- Kaggle: `poetry run imggenhub ...`
- Vast.ai: `poetry run imggenhub-vast ...`

## Support and Resources

- Vast.ai Docs: https://docs.vast.ai/
- API Reference: https://docs.vast.ai/api
- Discord Community: https://discord.gg/vast
