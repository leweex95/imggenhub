# Running Your Custom Pipeline on Tesla P40 (Vast.ai)

## Quick Start

### 1. Get Your API Key
1. Go to https://www.vast.ai/account/settings/
2. Copy your API key
3. Save it securely

### 2. Prepare Your Configuration

Create a file `tesla_p40_config.json`:
```json
{
  "api_key": "your-vast-ai-api-key",
  "gpu_type": "nvidia-tesla-p40",
  "max_price": 0.15,
  "target_specs": {
    "vram": 24,
    "location": "TH1",
    "reliability": 0.99
  }
}
```

### 3. Run on Your Target GPU

```bash
# Using the specific Tesla P40 in Thailand
poetry run python -m imggenhub.vast_ai.test_remote_pipeline `
  --api-key "your-api-key" `
  --gpu-type "nvidia-tesla-p40" `
  --max-price 0.15 `
  --model-name "stabilityai/stable-diffusion-3.5-large" `
  --guidance 7.5 `
  --steps 30 `
  --precision fp16 `
  --prompt "your custom prompt here"
```

### 4. For FLUX AI (Future)

Once your basic pipeline is working, upgrade to FLUX.1:
```bash
poetry run python -m imggenhub.vast_ai.test_remote_pipeline `
  --api-key "your-api-key" `
  --gpu-type "nvidia-tesla-p40" `
  --max-price 0.15 `
  --model-name "black-forest-labs/FLUX.1-schnell" `
  --guidance 3.5 `
  --steps 10 `
  --precision fp16 `
  --prompt "ultra-detailed futuristic landscape" `
  --hf-token "your-hf-token"
```

## Why This GPU Works

Your target Tesla P40 specs:
- **24 GB VRAM**: Handles Stable Diffusion 3.5, FLUX.1 schnell, and many refiners
- **fp16 precision**: Cuts VRAM usage in half while maintaining quality
- **$0.113/hr**: Cost-effective for iterative development
- **12-core CPU**: Fast data loading and preprocessing
- **4TB SSD**: Room for multiple models

## Optimization Tips

1. **Start with fp16**: Always try fp16 first, fall back to int8 if needed
2. **Adjust steps**: Lower guidance (3-5) and fewer steps (10-20) for faster iteration
3. **Batch prompts**: Generate multiple images in one rental session
4. **Model caching**: The setup script caches models to `/mnt/models` for faster reuse

## Cost Analysis

Estimated costs for common workflows:
- Single Stable Diffusion image (50 steps): $0.03
- Single FLUX.1 schnell image (20 steps): $0.05
- Batch of 20 images: $0.40-$0.80
- Full training/optimization session (4 hours): $0.45

## Monitoring

Check instance status at https://www.vast.ai/dashboard:
- Instance should boot within 1-2 minutes
- SSH access typically available within 3-5 minutes
- Setup takes 3-5 minutes depending on model size

## Troubleshooting

### Instance Won't Boot
- Try increasing max_price slightly (GPUs fill up quickly)
- Check Vast.ai dashboard for system issues
- Verify your account has available credits

### SSH Connection Refused
- Wait additional 2-3 minutes for instance to fully initialize
- Verify your SSH key is in Vast.ai settings
- Check firewall isn't blocking port access

### Out of Memory During Generation
- Reduce steps or batch size
- Switch to int8 or int4 precision
- Use a larger GPU model (but will cost more)

### Model Download Too Slow
- Setup script downloads models once and caches them
- Large models (10GB+) may take 5-10 minutes first run
- Subsequent runs reuse cached models instantly

## Advanced Usage

### Custom GPU Filtering
```bash
# Find the absolute cheapest P40
poetry run python -m imggenhub.vast_ai.cli search `
  --api-key "your-api-key" `
  --gpu-type "nvidia-tesla-p40" `
  --max-price 0.20 `
  --sort-by price
```

### Batch Processing
```bash
# Process multiple prompts in one rental session
poetry run python -m imggenhub.vast_ai.test_remote_pipeline `
  --api-key "your-api-key" `
  --prompts-file "batch_prompts.json" `
  --gpu-type "nvidia-tesla-p40" `
  --max-price 0.15
```

### Performance Monitoring
The output directory will contain:
- `generation_log.json`: Timing and resource usage
- `system_stats.txt`: GPU/CPU/Memory stats
- `images/`: Generated images
- `models_info.json`: Model versions and precisions used

## Next: Forge UI Integration

After successful testing with your custom pipeline, the next phase is integrating Forge UI for:
- Web-based UI for easier iteration
- Advanced features (ControlNet, upscaling, etc.)
- Faster batch processing
- Real-time generation preview

See `docs/FORGE_UI_INTEGRATION.md` for details.
