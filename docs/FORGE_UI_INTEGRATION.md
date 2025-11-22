# Forge UI Integration Guide

## Overview
Forge UI is an advanced, optimized fork of the Stable Diffusion WebUI with superior speed, lower VRAM usage, and enhanced features for advanced models like FLUX.1.

This guide walks you through deploying and using Forge UI on a Vast.ai GPU (specifically your target Tesla P40).

## Why Forge UI?

### Performance Benefits
- **2-3x faster inference** than vanilla diffusers pipelines
- **40-50% lower VRAM usage** through optimized memory management
- **Better model support** for large models (FLUX, Stable Diffusion 3.5)
- **Advanced features**: ControlNet, upscaling, LoRA, face enhancement

### Your Tesla P40 Specs (24GB VRAM)
- **Stable Diffusion XL + Refiner**: Fully supported at fp16
- **FLUX.1 schnell**: Runs efficiently at fp16 (15-18GB)
- **FLUX.1 pro**: Requires int8 precision (~18GB)
- **Batch generation**: Process multiple images per session

## Quick Start

### 1. Ensure Your Vast.ai API Key is Ready
```bash
$env:VAST_API_KEY = "your-vast-ai-api-key"
```

### 2. Run Forge UI with Stable Diffusion
```bash
poetry run imggenhub-forge `
  --api-key $env:VAST_API_KEY `
  --gpu-type "nvidia-tesla-p40" `
  --max-price 0.15 `
  --model-name "stabilityai/stable-diffusion-3.5-large" `
  --guidance 7.5 `
  --steps 30 `
  --precision fp16 `
  --prompt "a stunning landscape with mountains and lakes"
```

### 3. Run Forge UI with FLUX.1 Schnell (Recommended for Speed)
```bash
poetry run imggenhub-forge `
  --api-key $env:VAST_API_KEY `
  --gpu-type "nvidia-tesla-p40" `
  --max-price 0.15 `
  --model-name "black-forest-labs/FLUX.1-schnell" `
  --guidance 3.5 `
  --steps 20 `
  --precision fp16 `
  --prompt "ultra-detailed futuristic city with flying cars" `
  --hf-token "your-hf-token"
```

### 4. Batch Processing with Multiple Prompts
Create a JSON file `prompts.json`:
```json
{
  "prompts": [
    "a serene mountain landscape",
    "a bustling cyberpunk city",
    "an underwater garden",
    "a peaceful forest clearing"
  ]
}
```

Then run:
```bash
poetry run imggenhub-forge `
  --api-key $env:VAST_API_KEY `
  --gpu-type "nvidia-tesla-p40" `
  --max-price 0.15 `
  --model-name "black-forest-labs/FLUX.1-schnell" `
  --guidance 3.5 `
  --steps 20 `
  --precision fp16 `
  --prompts-file prompts.json `
  --hf-token "your-hf-token"
```

### 5. Keep Instance Running for Iteration (Optional)
```bash
poetry run imggenhub-forge `
  --api-key $env:VAST_API_KEY `
  --gpu-type "nvidia-tesla-p40" `
  --max-price 0.15 `
  --model-name "black-forest-labs/FLUX.1-schnell" `
  --guidance 3.5 `
  --steps 20 `
  --precision fp16 `
  --prompt "test prompt" `
  --hf-token "your-hf-token" `
  --keep-instance
```

When you add `--keep-instance`, the instance will remain running after completion. You can manually SSH in to continue experimenting, or destroy it manually via the Vast.ai dashboard.

## Model Recommendations

### For Speed (FLUX.1 schnell)
- **Best for**: Quick iteration, prototyping, batch generation
- **Time per image**: 10-15 seconds
- **Quality**: 7.5/10
- **Guidance**: 3.5-5.0
- **Steps**: 10-20
- **Memory**: ~15GB at fp16

```bash
poetry run imggenhub-forge `
  --api-key $env:VAST_API_KEY `
  --model-name "black-forest-labs/FLUX.1-schnell" `
  --guidance 3.5 `
  --steps 15 `
  --precision fp16 `
  --prompt "your prompt"
```

### For Quality (FLUX.1 pro)
- **Best for**: Final renders, high-quality outputs
- **Time per image**: 30-40 seconds
- **Quality**: 9.5/10
- **Guidance**: 7.5-12.0
- **Steps**: 30-50
- **Memory**: ~22GB at fp16

```bash
poetry run imggenhub-forge `
  --api-key $env:VAST_API_KEY `
  --model-name "black-forest-labs/FLUX.1-pro" `
  --guidance 10.0 `
  --steps 40 `
  --precision fp16 `
  --prompt "your prompt" `
  --hf-token "your-hf-token"
```

### For Balance (Stable Diffusion 3.5 Large)
- **Best for**: Balanced quality and speed
- **Time per image**: 20-25 seconds
- **Quality**: 8.5/10
- **Guidance**: 7.5-9.0
- **Steps**: 30-40
- **Memory**: ~19GB at fp16

```bash
poetry run imggenhub-forge `
  --api-key $env:VAST_API_KEY `
  --model-name "stabilityai/stable-diffusion-3.5-large" `
  --guidance 8.0 `
  --steps 35 `
  --precision fp16 `
  --prompt "your prompt"
```

## Advanced Features

### Using ControlNet (Composition Control)
The Forge UI setup includes ControlNet models for fine control over generated images:
```bash
poetry run imggenhub-forge `
  --api-key $env:VAST_API_KEY `
  --model-name "black-forest-labs/FLUX.1-schnell" `
  --guidance 3.5 `
  --steps 20 `
  --precision fp16 `
  --prompt "a detailed landscape matching the reference image" `
  --controlnet-image "/path/to/reference.jpg" `
  --controlnet-type "canny"
```

### Upscaling Generated Images
```bash
poetry run imggenhub-forge `
  --api-key $env:VAST_API_KEY `
  --model-name "black-forest-labs/FLUX.1-schnell" `
  --guidance 3.5 `
  --steps 20 `
  --precision fp16 `
  --prompt "ultra-high-quality image" `
  --upscale 2x
```

## Cost Analysis

### Tesla P40 at $0.113/hr

**Stable Diffusion 3.5 Large (35 steps, fp16)**
- Per image: ~$0.03
- 10 images: $0.30
- 100 images: $3.00

**FLUX.1 schnell (20 steps, fp16)**
- Per image: ~$0.02
- 10 images: $0.20
- 100 images: $2.00

**FLUX.1 pro (40 steps, fp16)**
- Per image: ~$0.04
- 10 images: $0.40
- 100 images: $4.00

## Troubleshooting

### "Out of Memory" Error
1. Reduce `--steps` (try 10-15)
2. Reduce `--guidance` (try 3.5-5.0 for FLUX.1)
3. Switch to `--precision int8` (slower but uses less VRAM)
4. Use a smaller model (e.g., FLUX.1-schnell instead of pro)

### Model Download Slow on First Run
- First model download may take 5-10 minutes
- Subsequent generations are instant (model cached)
- Large models like FLUX.1-pro are ~40GB

### SSH Connection Issues
1. Wait 2-3 more minutes (instance initialization)
2. Check your Vast.ai API key is valid
3. Verify instance is running in Vast.ai dashboard

### Generated Images Have Artifacts
1. Increase `--steps` to 30-50
2. Adjust `--guidance` (try 5.0-7.5 for FLUX.1)
3. Try a different seed/prompt
4. Use `--precision fp32` for maximum quality (at cost of speed)

## Next Steps

1. **Start with FLUX.1 schnell** for fast iteration and experimentation
2. **Batch process** your favorite prompts to find good ones
3. **Switch to FLUX.1-pro** for final high-quality renders
4. **Combine with ControlNet** for controlled composition
5. **Upscale results** for print-ready images

## References
- Forge UI GitHub: https://github.com/lllyasviel/stable-diffusion-webui-forge
- FLUX.1 Models: https://huggingface.co/black-forest-labs
- Vast.ai GPU Dashboard: https://www.vast.ai/dashboard
- Model Documentation: https://huggingface.co/docs/diffusers

## Support
For issues with:
- **Vast.ai rentals**: https://www.vast.ai/help
- **Forge UI features**: https://github.com/lllyasviel/stable-diffusion-webui-forge/issues
- **This integration**: Check the main README.md
