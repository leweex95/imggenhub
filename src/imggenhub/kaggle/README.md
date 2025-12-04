# Fully automated AI image generation using Kaggle kernels

_Kaggle offers 30 hours of GPU time (T4×2 GPUs) per week._

## **Full pipeline** 

Only run the following to have the e2e image generation with Kaggle functioning. 

Full pipeline (all steps chained)

Run the main script to perform all steps automatically:

    poetry run python -m imggenhub.kaggle.main  --model_id "stabilityai/stable-diffusion-xl-base-1.0"--gpu true --dest .

**Note**: don't forget to update the `prompts.json` with the appropriate prompts!

For more details, read further.

## Pipeline steps

### Prompts

Pre-requisite: Create a JSON file with prompts (one or more) like:

```json
[
  "a photorealistic painting of a cat in space",
  "futuristic city skyline at sunset"
]
```

Alternatively, you can also pass simpler prompts in command line. The idea is that to allow users to phrase very complex prompts as well - which is more comfortable if prompts are in a separate `prompts.json` file. But especially for PoCs, we might just want to execute one simple prompt directly from the command line. For that, the `--prompts` flag is also available. 

Note: if the `--prompts` flag is set, the content of the `prompts.json` file are ignored. 

### Deploy image generation to the remote kernel 

Push the notebook to the remote Kaggle kernel to start execution:

    poetry run python -m kaggle.cli kernels push -p .

Or use the deploy script with GPU override to combine steps 1 & 2:

    poetry run python -m deploy --prompts_file prompts.json --gpu true --model_id "stabilityai/stable-diffusion-xl-base-1.0"

### Poll for completion

Use the polling script to wait until the kernel finishes:

    poetry run python -m poll_status

### Fetch the produced output

Download generated images from Kaggle:

    poetry run python -m download -p .

where `-p .` refers to the destination path being in the current working directory.

## Note about Hugging Face model usage

Note that in order to use the current Kaggle pipeline, one must opt for a publicly accessible Hugging Face model. 

There are so-called gated (restricted) models that require that users first log in and accept the terms prior to use, and then configure a custom Hugging Face token to gain read permissions for inference with the selected model. I tested this approach and it works perfectly **locally**, i.e., we are able to fetch any gated model as well. But unfortunately integrating these models into the full Kaggle pipeline means deploying it to Kaggle server via the Kaggle CLI and run it in a containerized environment which fails. I encountered the followimg limitations:
- To access the gated model, we need a Hugging Face token (configurable here: https://huggingface.co/settings/tokens). This can't directly be passed via Kaggle CLI as it doesn't support environment variable passing, neither custom arguments in the metadata. 
- As a workaround, I auto-uploaded a secret "Kaggle dataset" under my personal Kaggle kernel, including only one single json file with the Hugging Face token. The benefit of this is that the Kaggle pipeline deployment can freely reach Kaggle datasets and hence, access the Hugging Face token environment variable. I confirmed that this token is passable this way and can be accessed from within the containerized workflow. 
- However despite this, I received a 401 Error. Presumably, Kaggle intentionally disabled algorithmic Hugging Face access from within their containers. To verify this officially, I submitted a ticket inquiring about this. If there is a workaround provided, I will integrate that into this codebase. 

Until then, we must always only use **public** Hugging Face models only.
