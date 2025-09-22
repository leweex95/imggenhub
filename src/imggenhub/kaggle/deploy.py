import json
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--prompts_file", type=str,
                    help="Path to a JSON file containing a list of prompts", default="prompts.json")
parser.add_argument("--notebook", type=str, default="kaggle-notebook-image-generation.ipynb")
parser.add_argument("--kernel_path", type=str, default=".")
parser.add_argument("--gpu", type=str, choices=["true", "false"],
                    help="Override enable_gpu in kernel-metadata.json")
args = parser.parse_args()

# Load notebook
nb_path = Path(args.notebook)
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Load prompts from JSON file
prompts_file = Path(args.prompts_file)
if not prompts_file.exists():
    raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

with open(prompts_file, "r", encoding="utf-8") as f:
    prompts_list = json.load(f)

if not prompts_list or not isinstance(prompts_list, list):
    raise ValueError(f"Prompts JSON must be a non-empty list. Got: {prompts_list}")

# Update PROMPTS in the first code cell (adjust index if needed)
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        for i, line in enumerate(cell["source"]):
            if line.strip().startswith("PROMPTS ="):
                cell["source"][i] = f"PROMPTS = {prompts_list}\n"
                break
        else:
            continue
        break

# Save updated notebook
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

# Load and update kernel metadata
metadata_path = Path(args.kernel_path) / "kernel-metadata.json"
with open(metadata_path, "r", encoding="utf-8") as f:
    kernel_meta = json.load(f)

if args.gpu:
    kernel_meta["enable_gpu"] = args.gpu

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(kernel_meta, f, indent=2)

# Push via Kaggle CLI
subprocess.run(["python", "-m", "kaggle.cli", "kernels", "push", "-p", args.kernel_path])
