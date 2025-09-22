# main.py
import subprocess
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--prompts_file", type=str, default="prompts.json")
parser.add_argument("--notebook", type=str, default="config/kaggle-notebook-image-generation.ipynb")
parser.add_argument("--kernel_path", type=str, default="./config")
parser.add_argument("--gpu", type=str, choices=["true", "false"], default=None)
parser.add_argument("--dest", type=str, default="output_images")
args = parser.parse_args()

# Step 1: Deploy
deploy_cmd = [
    "python", "-m", "deploy",
    "--prompts_file", args.prompts_file,
    "--notebook", args.notebook,
    "--kernel_path", args.kernel_path,
]

deploy_cmd += ["--gpu", args.gpu if args.gpu else "false"]

print("Deploying kernel...")
subprocess.run(deploy_cmd, check=True)

# Step 2: Poll status
poll_cmd = ["python", "-m", "poll_status"]
print("Polling kernel status...")
subprocess.run(poll_cmd, check=True)

# Step 3: Download output
download_cmd = ["python", "-m", "download", "--dest", args.dest]
print("Downloading output artifacts...")
subprocess.run(download_cmd, check=True)

print("Pipeline completed!")
print("You can check your remaining GPU quota here:")
print("https://www.kaggle.com/settings#quotas")
