# main.py
import argparse
import subprocess
import logging
from pathlib import Path
from imggenhub.kaggle.core import deploy, download
from imggenhub.kaggle.utils import poll_status
from imggenhub.kaggle.utils.prompts import resolve_prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_pipeline(prompts_file, notebook, kernel_path, gpu=False, dest="output_images", model_name=None, prompt=None, prompts=None):
    """Run Kaggle image generation pipeline: deploy -> poll -> download"""
    print("Initializing run_pipeline in main.py...")
    cwd = Path(__file__).parent

    # Resolve paths
    if prompts_file:
        prompts_file = Path(prompts_file)
        if not prompts_file.is_absolute():
            prompts_file = cwd / prompts_file

    notebook = Path(notebook)
    if not notebook.is_absolute():
        notebook = cwd / notebook

    kernel_path = Path(kernel_path)
    if not kernel_path.is_absolute():
        kernel_path = cwd / kernel_path

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    prompts_list = resolve_prompts(prompts_file, prompt, prompts)

    logging.debug(f"Resolved paths:\n prompts_file={prompts_file}\n notebook={notebook}\n kernel_path={kernel_path}\n dest={dest}")

    # Step 1: Deploy
    logging.info("Deploying kernel...")
    deploy.run(prompts_list, notebook, kernel_path, gpu, model_id=model_name)
    logging.debug("Deploy step completed")

    # Step 2: Poll status
    logging.info("Polling kernel status...")
    poll_status.run()
    logging.debug("Poll status completed")

    # Step 3: Download output
    logging.info("Downloading output artifacts...")
    download.run(dest)
    logging.debug("Download completed")

    logging.info("Pipeline completed!")
    logging.info("Check remaining GPU quota: https://www.kaggle.com/settings#quotas")


def main():
    parser = argparse.ArgumentParser(description="Kaggle image generation pipeline")
    parser.add_argument("--prompts_file", type=str, default="./config/prompts.json")
    parser.add_argument("--notebook", type=str, default="./config/kaggle-notebook-image-generation.ipynb")
    parser.add_argument("--kernel_path", type=str, default="./config")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for the kernel")
    parser.add_argument("--dest", type=str, default="output_images")
    parser.add_argument("--model_name", type=str, default=None, help="Image generation model to use")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt string")
    parser.add_argument("--prompts", type=str, nargs="+", default=None, help="Multiple prompts")

    args = parser.parse_args()

    run_pipeline(
        prompts_file=args.prompts_file,
        notebook=args.notebook,
        kernel_path=args.kernel_path,
        gpu=args.gpu,
        dest=args.dest,
        model_name=args.model_name,
        prompt=args.prompt,
        prompts=args.prompts
    )


if __name__ == "__main__":
    main()
