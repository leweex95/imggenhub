# Running partly automated AI image generation for free using Kaggle

_Kaggle offers 30 hours of GPU time (T4×2 GPUs) per week._

## **Full pipeline** 

Only run the following to have the e2e inage generation with Kaggle functioning. 

```
python -m main --prompts_file prompts.json --gpu true
```

Note: don't forget to update the `prompts.json` with the appropriate prompts!

For more details, read further.

## 1. Update input arguments automatically

Create a JSON file with prompts (one or more) like:

```json
[
  "a photorealistic painting of a cat in space",
  "futuristic city skyline at sunset"
]
```

Then run the deploy script to inject prompts into the notebook and optionally enable GPU:

```
python deploy.py --prompts_file prompts.json --gpu true
```

2. Run the notebook automatically

Push the notebook to Kaggle to start execution:

```
python -m kaggle.cli kernels push -p .
```

Or use the deploy script with GPU override to combine steps 1 & 2:

```
python deploy --prompts_file prompts.json --gpu true
```

3. Poll for completion

Use the polling script to wait until the kernel finishes:

```
python poll_status
```

4. Fetch the produced output

Download generated images from Kaggle:

```
python download -o /path/to/dest
```

5. Full pipeline (all steps chained)

Run the main script to perform all steps automatically:

```
python -m main --prompts_file prompts.json --gpu true --dest .
```
