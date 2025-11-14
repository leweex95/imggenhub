# main.py
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from imggenhub.kaggle.core import deploy, download
from imggenhub.kaggle.utils import poll_status
from imggenhub.kaggle.utils.prompts import resolve_prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_pipeline(prompts_file, notebook, kernel_path, gpu=False, dest="output_images", model_name=None, refiner_model_name=None, prompt=None, prompts=None, guidance=None, steps=None, precision="fp16", negative_prompt=None):
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
        notebook = cwd / "config" / notebook.name  # Always resolve relative to config directory

    kernel_path = Path(kernel_path)
    if not kernel_path.is_absolute():
        kernel_path = cwd / "config"  # Always use config directory

    # Create timestamp-based output directory under output/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path("output")
    output_base.mkdir(exist_ok=True)
    
    # Use dest as a descriptive name, combine with timestamp
    if dest and dest != "output_images":  # If custom dest provided, use it as prefix
        run_name = f"{dest}_{timestamp}"
    else:
        run_name = timestamp
    
    dest_path = output_base / run_name
    dest_path.mkdir(parents=True, exist_ok=True)

    prompts_list = resolve_prompts(prompts_file, prompt, prompts)

    logging.debug(f"Resolved paths:\n prompts_file={prompts_file}\n notebook={notebook}\n kernel_path={kernel_path}\n dest={dest_path}")

    # Step 1: Deploy
    logging.info("Deploying kernel...")
    deploy.run(prompts_list, notebook, kernel_path, gpu, model_id=model_name, refiner_model_id=refiner_model_name, guidance=guidance, steps=steps, precision=precision, negative_prompt=negative_prompt, output_dir=str(dest_path))
    logging.debug("Deploy step completed")

    # Step 2: Poll status
    logging.info("Polling kernel status...")
    status = poll_status.run()
    logging.debug("Poll status completed")

    if status == "kernelworkerstatus.error":
        log_path = dest_path / "stable-diffusion-batch-generator.log"
        logging.error(f"Kernel failed. See log: {log_path}")
        raise RuntimeError("Kaggle kernel failed during image generation. Aborting pipeline.")

    # Step 3: Download output
    logging.info("Downloading output artifacts...")
    download.run(dest_path)
    logging.debug("Download completed")

    logging.info(f"Pipeline completed! Output saved to: {dest_path}")
    logging.info("Check remaining GPU quota: https://www.kaggle.com/settings#quotas")


def main():
    parser = argparse.ArgumentParser(description="Kaggle image generation pipeline")
    parser.add_argument("--prompts_file", type=str, default="./config/prompts.json")
    parser.add_argument("--notebook", type=str, default="./config/kaggle-notebook-image-generation.ipynb")
    parser.add_argument("--kernel_path", type=str, default="./config")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for the kernel")
    parser.add_argument("--dest", type=str, default="output_images")
    parser.add_argument("--model_name", type=str, default=None, help="Image generation model to use")
    parser.add_argument("--refiner_model_name", type=str, default=None, help="Refiner model to use (defaults to SDXL refiner)")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt string")
    parser.add_argument("--prompts", type=str, nargs="+", default=None, help="Multiple prompts")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance scale (7-12 recommended for photorealism)")
    parser.add_argument("--steps", type=int, default=None, help="Number of inference steps (50-100 for better quality)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8", "int4"], 
                        help="Precision level: fp32 (highest quality), fp16 (balanced), int8 (faster), int4 (fastest, lowest memory)")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Custom negative prompt for better quality control")

    args = parser.parse_args()

    # Automatically enable refiner if refiner_model_name is specified
    use_refiner = (args.refiner_model_name is not None)

    run_pipeline(
        prompts_file=args.prompts_file,
        notebook=args.notebook,
        kernel_path=args.kernel_path,
        gpu=args.gpu,
        dest=args.dest,
        model_name=args.model_name,
        refiner_model_name=args.refiner_model_name,
        prompt=args.prompt,
        prompts=args.prompts,
        guidance=args.guidance,
        steps=args.steps,
        precision=args.precision,
        negative_prompt=args.negative_prompt
    )


if __name__ == "__main__":
    main()
