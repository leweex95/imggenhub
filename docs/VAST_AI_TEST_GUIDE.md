# Vast.ai Remote Pipeline Test Guide

## Overview
This guide walks you through testing your custom image generation pipeline on a remote Vast.ai GPU (specifically the Tesla P40 you identified in Thailand).

## Prerequisites
1. **Vast.ai Account**: Create one at https://www.vast.ai/
2. **API Key**: Generate from https://www.vast.ai/account/settings/
3. **SSH Key Pair**: Vast.ai requires SSH for remote access

## Target GPU Specifications
Your target instance (#22589933):
- **Type**: NVIDIA Tesla P40 (24GB VRAM)
- **Location**: Thailand (TH1)
- **Specs**: 
  - 24 GB VRAM
  - 277.3 GB/s bandwidth
  - Xeon Gold 6146 CPU (12 cores, 48 threads)
  - 64 GB RAM
  - 4TB SSD
  - $0.113/hr (DLP-optimized pricing)
- **Reliability**: 99.88%
- **Max Duration**: 27 days

## Running the Test

### Step 1: Set Your Vast.ai API Key
```bash
# Store your API key (keep this secure)
$env:VAST_API_KEY = "your-vast-ai-api-key-here"
```

### Step 2: Run the Test
```bash
# Basic test with default settings
poetry run python -m imggenhub.vast_ai.test_remote_pipeline `
  --api-key $env:VAST_API_KEY `
  --gpu-type "nvidia-tesla-p40" `
  --max-price 0.15

# Custom test with your model and prompt
poetry run python -m imggenhub.vast_ai.test_remote_pipeline `
  --api-key $env:VAST_API_KEY `
  --gpu-type "nvidia-tesla-p40" `
  --max-price 0.15 `
  --model-name "stabilityai/stable-diffusion-3.5-large" `
  --guidance 7.5 `
  --steps 30 `
  --precision fp16 `
  --prompt "a stunning Tesla P40 GPU rendering futuristic landscapes" `
  --hf-token "your-hf-token"
```

### Step 3: Monitor Progress
The test will:
1. Search for available Tesla P40 instances under $0.15/hr
2. Rent the best matching instance
3. Wait for SSH access (up to 5 minutes)
4. Set up the remote environment (Python, PyTorch, dependencies)
5. Upload your test prompts
6. Execute the image generation pipeline remotely
7. Download results to `output_from_vast_ai/`
8. Automatically destroy the instance

### Expected Output
```
=== Vast.ai Remote Pipeline Test ===
GPU Type: nvidia-tesla-p40
Max Price: $0.15/hr
Model: stabilityai/stable-diffusion-3.5-large
Prompt: a stunning landscape...

Step 1: Searching for available instances...
Selected instance: 1234567
  GPU: NVIDIA Tesla P40
  VRAM: 24 GB
  Price: $0.113/hr

Step 2: Renting instance...
Instance rented: 1234567

Step 3: Waiting for SSH access...
SSH available: 1.2.3.4:22

Step 4: Setting up remote environment...
Remote environment ready

Step 5: Uploading test prompts...

Step 6: Running image generation pipeline...
Pipeline execution completed successfully

Step 7: Downloading results...
Results downloaded to: output_from_vast_ai

Step 8: Verifying results...
Generated 1 image(s)
  - output_from_vast_ai/image_0.png

=== Test Completed Successfully ===
```

## Troubleshooting

### SSH Connection Timeout
- Vast.ai instances take 1-5 minutes to start and configure
- If SSH fails after 5 minutes, manually check instance status in the Vast.ai dashboard
- Verify your SSH key is properly configured in Vast.ai account settings

### Remote Setup Failures
- Check that dependencies install correctly on the target OS (Ubuntu 20.04 LTS typical)
- Verify PyTorch is compatible with the P40 (CUDA 11.8 or 12.x recommended)
- Check available disk space (minimum 50GB recommended for large models)

### Pipeline Execution Errors
- Ensure HuggingFace token is valid if using gated models
- Check that model is available for the specified precision
- Verify VRAM is sufficient (24GB P40 handles most models, but check model requirements)

### Cost Overruns
- The test automatically destroys the instance after completion
- If the test fails, the instance will also be destroyed (see finally block)
- Monitor https://www.vast.ai/dashboard for unexpected running instances

## Next Steps

After a successful test:
1. **Refine Parameters**: Adjust guidance, steps, precision based on quality/speed tradeoff
2. **Test FLUX AI**: Switch to FLUX.1 models if they fit in VRAM
3. **Deploy Forge UI**: Proceed to Forge UI integration for faster iteration
4. **Automate Workflows**: Schedule recurring image generation jobs on Vast.ai

## Cost Estimation

Your target GPU at $0.113/hr:
- 1 image generation (10 min): ~$0.02
- Batch of 10 images (100 min): ~$0.19
- Daily quota: ~$2.72/day (24 hrs continuous)

## References
- Vast.ai API Docs: https://vast.ai/docs/
- Account Settings: https://www.vast.ai/account/settings/
- Instance Search: https://www.vast.ai/search/offers
- Support: https://vast.ai/help
