# Vast.ai Integration

This project now fully supports GPU-accelerated image generation on Vast.ai rental GPUs, enabling cost-effective access to powerful GPUs without owning hardware.

## Features

✅ **Custom Pipeline on Remote GPU**: Run your existing image generation pipeline on rented Vast.ai GPUs  
✅ **Forge UI Integration**: Deploy and use Forge UI for optimized, faster inference  
✅ **Cost Tracking & Budgeting**: Estimate costs and track spending across runs  
✅ **Automatic Instance Management**: Provision and cleanup instances automatically  
✅ **Multi-Model Support**: Stable Diffusion 3.5, FLUX.1-schnell, FLUX.1-pro  
✅ **Batch Processing**: Generate multiple images in a single rental session  

## Quick Start

### 1. Set Up Your API Key
```bash
$env:VAST_API_KEY = "your-vast-ai-api-key"
```

### 2. Test Your Custom Pipeline
```bash
poetry run python -m imggenhub.vast_ai.test_remote_pipeline `
  --api-key $env:VAST_API_KEY `
  --gpu-type "nvidia-tesla-p40" `
  --model-name "stabilityai/stable-diffusion-3.5-large" `
  --prompt "a beautiful landscape"
```

### 3. Run Forge UI for Faster Generation
```bash
poetry run imggenhub-forge `
  --api-key $env:VAST_API_KEY `
  --model-name "black-forest-labs/FLUX.1-schnell" `
  --prompt "ultra-detailed futuristic city"
```

## Cost Estimation

```bash
# How many images can I generate with $5?
poetry run imggenhub-costs estimate budget --budget 5.0 --gpu-dph 0.113

# Cost for 50 images with FLUX.1-schnell?
poetry run imggenhub-costs estimate generation \
  --num-images 50 \
  --avg-time-sec 30 \
  --gpu-dph 0.113
```

## Documentation

- 📖 **[Full Integration Guide](docs/VAST_AI_INTEGRATION.md)** - Complete setup and usage
- ⚡ **[Forge UI Guide](docs/FORGE_UI_INTEGRATION.md)** - Optimize for speed and quality
- 💰 **[Cost Estimation](docs/TESLA_P40_QUICK_START.md)** - Budget planning and cost tracking
- 🚀 **[Quick Reference](docs/VAST_AI_QUICK_REFERENCE.md)** - Cheat sheet and common commands

## Supported GPU Types

The integration works with all Vast.ai GPU offerings. Recommended for this project:

| GPU | VRAM | Price/hr | Best For |
|-----|------|----------|----------|
| Tesla P40 | 24GB | $0.10-0.15 | **Recommended** for FLUX.1 and SD3.5 |
| RTX 3090 | 24GB | $0.15-0.25 | High performance, larger batches |
| A40 | 48GB | $0.30-0.50 | Large models, maximum quality |

## Supported Models

| Model | Speed | VRAM | Quality | Use Case |
|-------|-------|------|---------|----------|
| FLUX.1-schnell | 🚀 30s | 15GB | 7/10 | Fast iteration, prototyping |
| Stable Diffusion 3.5 | 🟡 60s | 19GB | 8.5/10 | Balanced quality and speed |
| FLUX.1-pro | 🐌 90s | 22GB | 9.5/10 | Final renders, max quality |

## CLI Commands

### Image Generation
```bash
# Custom pipeline on remote GPU
poetry run python -m imggenhub.vast_ai.test_remote_pipeline ...

# Forge UI (faster inference)
poetry run imggenhub-forge ...

# Search available GPUs
poetry run imggenhub-vast search ...
```

### Cost Management
```bash
# Estimate generation costs
poetry run imggenhub-costs estimate generation ...

# Plan budget for models
poetry run imggenhub-costs estimate budget ...

# Track run history
poetry run imggenhub-costs log view ...
poetry run imggenhub-costs log add ...
```

## Architecture

```
imggenhub/vast_ai/
├── client.py              # Vast.ai API integration
├── ssh.py                 # SSH remote execution
├── executor.py            # Pipeline execution orchestration
├── forge.py               # Forge UI management
├── cost_tracking.py       # Cost estimation and logging
├── test_remote_pipeline.py # E2E testing
├── forge_cli.py           # Forge UI CLI
├── cost_cli.py            # Cost estimation CLI
├── cli.py                 # Main Vast.ai CLI
├── setup.sh               # Remote environment setup
└── forge_deploy.sh        # Forge UI deployment script
```

## Project Structure

```
imggenhub/
├── kaggle/                # Original Kaggle pipeline
├── vast_ai/               # Vast.ai GPU integration
│   ├── core/             # API and core modules
│   └── utils/            # Utilities and helpers
└── ...
```

## Workflow

### Using Custom Pipeline
```
1. User runs: imggenhub.vast_ai.test_remote_pipeline
2. Client searches for GPU on Vast.ai
3. Instance is rented and provisioned
4. Custom pipeline is uploaded and executed
5. Results are downloaded locally
6. Instance is automatically destroyed
```

### Using Forge UI
```
1. User runs: imggenhub-forge
2. Vast.ai instance is rented
3. Forge UI is deployed with all dependencies
4. Model is loaded into Forge UI
5. Images are generated via Forge UI
6. Results are downloaded locally
7. Instance is destroyed (optional: --keep-instance)
```

## Troubleshooting

### Connection Issues
- **SSH timeout**: Wait 3-5 minutes for instance to initialize
- **API key invalid**: Verify at https://www.vast.ai/account/settings/
- **No instances available**: Try higher max-price or different GPU type

### Performance Issues
- **Out of memory**: Reduce `--steps`, reduce `--guidance`, use `--precision int8`
- **Slow generation**: Enable `--precision fp16` (slightly lower quality, 2x faster)
- **Slow model download**: First download caches model; subsequent runs use cache

### Cost Issues
- **Unexpected charges**: Use `--keep-instance false` (default) to auto-cleanup
- **High billing**: Monitor active instances at https://www.vast.ai/dashboard
- **Over budget**: Use `imggenhub-costs estimate budget` to plan ahead

## Cost Reference

### Tesla P40 @ $0.113/hr
- **FLUX.1-schnell**: $0.02 per image (~30 seconds)
- **SD 3.5**: $0.04 per image (~60 seconds)
- **FLUX.1-pro**: $0.07 per image (~90 seconds)

### Example Costs
- 10 images with FLUX.1-schnell: ~$0.20
- 100 images with SD 3.5: ~$4.00
- 24-hour session: ~$2.70

## Security Notes

1. **API Keys**: Store securely, don't commit to repository
2. **SSH Keys**: Vast.ai handles SSH key management
3. **Instance Access**: Only accessible via SSH from your machine
4. **Data**: Upload/download via secure SSH tunnels
5. **Cleanup**: Always verify instances are destroyed after use

## Future Enhancements

- [ ] Web UI for instance management
- [ ] Advanced scheduling for batch jobs
- [ ] Model caching and optimization
- [ ] Multi-GPU instance support
- [ ] Real-time monitoring dashboard

## Support & Resources

- **Vast.ai Docs**: https://vast.ai/docs/
- **Forge UI**: https://github.com/lllyasviel/stable-diffusion-webui-forge
- **HuggingFace Models**: https://huggingface.co/black-forest-labs
- **Project Issues**: [GitHub Issues](https://github.com/leweex95/imggenhub/issues)

---

**Ready to start?** See [VAST_AI_QUICK_REFERENCE.md](docs/VAST_AI_QUICK_REFERENCE.md) for quick commands!
