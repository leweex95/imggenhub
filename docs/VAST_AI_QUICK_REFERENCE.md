# Vast.ai Integration - Quick Reference

## Setup (One-time)

1. Create Vast.ai account: https://www.vast.ai
2. Generate API key: https://www.vast.ai/account/settings/
3. Get your Tesla P40 instance ready: Search for GPU type "nvidia-tesla-p40" with max price $0.15/hr

## CLI Commands

### Test Your Custom Pipeline on Vast.ai
```bash
poetry run python -m imggenhub.vast_ai.test_remote_pipeline \
  --api-key "your-api-key" \
  --gpu-type "nvidia-tesla-p40" \
  --max-price 0.15 \
  --model-name "stabilityai/stable-diffusion-3.5-large" \
  --guidance 7.5 \
  --steps 30 \
  --precision fp16 \
  --prompt "your prompt here"
```

### Run Forge UI for Image Generation
```bash
poetry run imggenhub-forge \
  --api-key "your-api-key" \
  --gpu-type "nvidia-tesla-p40" \
  --max-price 0.15 \
  --model-name "black-forest-labs/FLUX.1-schnell" \
  --guidance 3.5 \
  --steps 20 \
  --precision fp16 \
  --prompt "your prompt"
```

### Estimate Generation Costs
```bash
# Cost for 50 images at 30sec each on P40 ($0.113/hr)
poetry run imggenhub-costs estimate generation \
  --num-images 50 \
  --avg-time-sec 30 \
  --gpu-dph 0.113

# Budget planning for $5
poetry run imggenhub-costs estimate budget \
  --budget 5.0 \
  --gpu-dph 0.113

# Get model info and costs
poetry run imggenhub-costs estimate model \
  --model flux-schnell \
  --gpu-dph 0.113
```

### Log and Track Runs
```bash
# View previous runs
poetry run imggenhub-costs log view --log-file vast_ai_runs.json

# Add a new run
poetry run imggenhub-costs log add \
  --instance-id 12345 \
  --instance-type "nvidia-tesla-p40" \
  --model "black-forest-labs/FLUX.1-schnell" \
  --num-images 10 \
  --duration-sec 300 \
  --gpu-dph 0.113 \
  --notes "First test with P40"
```

## Environment Variables
```bash
# PowerShell
$env:VAST_API_KEY = "your-api-key"

# Or in batch file
set VAST_API_KEY=your-api-key
```

## Model Quick Reference

| Model | Speed | Quality | VRAM | Time/Img | Guidance | Steps | Cost/10 |
|-------|-------|---------|------|----------|----------|-------|---------|
| FLUX.1-schnell | 🚀 Fast | 7/10 | 15GB | 30s | 3.5-5 | 10-20 | $0.02 |
| SD 3.5 Large | 🟡 Med | 8.5/10 | 19GB | 60s | 7.5-9 | 30-40 | $0.04 |
| FLUX.1-pro | 🐌 Slow | 9.5/10 | 22GB | 90s | 7.5-12 | 30-50 | $0.07 |

## File Locations
- Custom pipeline test: `src/imggenhub/vast_ai/test_remote_pipeline.py`
- Forge UI CLI: `src/imggenhub/vast_ai/forge_cli.py`
- Cost estimation: `src/imggenhub/vast_ai/cost_cli.py`
- Vast.ai API client: `src/imggenhub/vast_ai/client.py`
- SSH wrapper: `src/imggenhub/vast_ai/ssh.py`
- Forge UI manager: `src/imggenhub/vast_ai/forge.py`
- Setup script: `src/imggenhub/vast_ai/setup.sh`

## Documentation
- **Full guide**: `docs/VAST_AI_INTEGRATION.md`
- **Test guide**: `docs/VAST_AI_TEST_GUIDE.md`
- **Tesla P40 quick start**: `docs/TESLA_P40_QUICK_START.md`
- **Forge UI guide**: `docs/FORGE_UI_INTEGRATION.md`

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| SSH timeout | Wait 3-5 min, verify API key in settings |
| Out of memory | Reduce steps, reduce guidance, use int8 precision |
| Model download slow | First download is slow (~5-10min), cached after |
| Instance won't boot | Try higher max-price, check Vast.ai status |
| High bill | Use --keep-instance false (auto-destroy), monitor dashboard |

## Your Target GPU
- **Type**: NVIDIA Tesla P40 (24GB VRAM)
- **Location**: Thailand (TH1)
- **Price**: $0.113/hr (DLP-optimized)
- **Reliability**: 99.88%
- **Best for**: FLUX.1, Stable Diffusion 3.5, batch generation

## Next Steps
1. Get your Vast.ai API key
2. Run the test pipeline to verify everything works
3. Try Forge UI for faster iteration
4. Generate batches of images with your preferred model
5. Monitor costs with the cost tracking tool
