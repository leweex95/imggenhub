# parallel_deploy.py
"""
Parallel deployment module for distributing prompts across multiple Kaggle kernels.

When prompts list > 4 items, splits into two roughly equal batches and deploys
them to separate Kaggle kernels for parallel execution.
"""

import json
import logging
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kaggle_connector import JobManager, SelectiveDownloader
from imggenhub.kaggle.utils.config_loader import load_kaggle_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Constants for parallel execution
PARALLEL_THRESHOLD = 4


def get_deployment_ids(base_kernel_id: str) -> Tuple[str, str]:
    """Generate deployment IDs based on base kernel ID."""
    return f"{base_kernel_id}-deployment-1", f"{base_kernel_id}-deployment-2"


def split_prompts(prompts: List[str]) -> Tuple[List[str], List[str]]:
    """
    Split prompts into two roughly equal lists.
    First list gets the smaller half if odd count.
    
    Args:
        prompts: List of prompts to split
        
    Returns:
        Tuple of (first_half, second_half)
    """
    mid = len(prompts) // 2
    return prompts[:mid], prompts[mid:]


def should_use_parallel(prompts: List[str]) -> bool:
    """Check if parallel deployment should be used based on prompt count."""
    return len(prompts) > PARALLEL_THRESHOLD



def _deploy_single_kernel(
    prompts_list: List[str],
    notebook: Path,
    kernel_path: Path,
    kernel_id: str,
    deploy_kwargs: Dict[str, Any],
    accelerator: str = None,
    index_offset: int = 0
) -> str:
    """
    Deploy a single kernel with given prompts using Kaggle Connector.
    
    Args:
        prompts_list: Prompts for this kernel
        notebook: Notebook path
        kernel_path: Kernel config directory
        kernel_id: Kernel identifier
        deploy_kwargs: Additional kwargs for JobManager
        accelerator: Kaggle accelerator type
        index_offset: Offset for image numbering
        
    Returns:
        Kernel ID that was deployed
    """
    logging.info(f"Deploying {len(prompts_list)} prompts to kernel: {kernel_id}")
    
    # 1. Create a truly temporary deployment directory for this specific deployment
    # This prevents cross-contamination and side effects on the source notebooks
    with tempfile.TemporaryDirectory(prefix=f"kaggle_deploy_{kernel_id.split('/')[-1]}_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        nb_name = notebook.name
        tmp_nb_path = tmp_dir_path / nb_name
        
        # Copy notebook to tmp dir
        shutil.copy2(notebook, tmp_nb_path)
        
        manager = JobManager(kernel_id)
        
        # Prepare parameters for injection
        params = {
            "PROMPTS": prompts_list,
            "OUTPUT_DIR": ".",
            "KERNEL_ID": kernel_id,
            "INDEX_OFFSET": index_offset
        }
        
        # Only add optional params if they are provided to avoid overwriting notebook defaults with None
        optional_params = [
            ("MODEL_ID", "model_id"),
            ("GUIDANCE", "guidance"),
            ("STEPS", "steps"),
            ("PRECISION", "precision"),
            ("IMG_SIZE", "img_size"),
            ("MODEL_FILENAME", "model_filename"),
            ("VAE_REPO_ID", "vae_repo_id"),
            ("VAE_FILENAME", "vae_filename"),
            ("CLIP_L_REPO_ID", "clip_l_repo_id"),
            ("CLIP_L_FILENAME", "clip_l_filename"),
            ("T5XXL_REPO_ID", "t5xxl_repo_id"),
            ("T5XXL_FILENAME", "t5xxl_filename"),
        ]
        
        for nb_key, kw_key in optional_params:
            val = deploy_kwargs.get(kw_key)
            if val is not None:
                params[nb_key] = val
        
        manager.edit_notebook_params(str(tmp_nb_path), params)
        
        # 2. Metadata configuration
        gpu = deploy_kwargs.get("gpu", True)
        username = deploy_kwargs.get("username", "leventecsibi")
        dataset_sources = [f"{username}/imggenhub-hf-token"]
        if "flux-gguf" in str(notebook).lower():
             dataset_sources.extend([
                f"{username}/flux1-schnell-q4-zip",
                f"{username}/vae-zip",
                f"{username}/clip-l-zip",
                f"{username}/t5xxl-zip",
                f"{username}/sd-build-zip"
             ])
        
        kernel_type = "notebook" if nb_name.endswith(".ipynb") else "script"
        manager.create_metadata(
            str(tmp_dir_path),
            kernel_id=kernel_id,
            code_file=nb_name,
            kernel_type=kernel_type,
            enable_gpu=gpu,
            dataset_sources=dataset_sources,
            accelerator=accelerator
        )
        
        # 3. Deploy
        wait_min = deploy_kwargs.get("wait_timeout", 30)
        manager.deploy(
            str(tmp_dir_path), 
            wait=True, 
            timeout_min=wait_min
        )
        
        logging.info(f"Successfully deployed kernel: {kernel_id}")
        return kernel_id


def _poll_kernel(kernel_id: str, max_wait: int = 1800, poll_interval: int = None) -> str:
    """
    Poll a single kernel using JobManager.
    """
    if poll_interval is None:
        config = load_kaggle_config()
        poll_interval = config.get("polling_interval_seconds", 60)
    
    logging.info(f"Starting poll for kernel: {kernel_id}")
    manager = JobManager(kernel_id)
    status = manager.poll_until_complete(poll_interval=poll_interval)
    
    # Check for library-specific unknown status error
    if "error_unknown_status" in status:
        logging.error(f"Kernel {kernel_id} failed with unknown status (likely 404).")
        raise RuntimeError(f"Kernel {kernel_id} unreachable: {status}")
        
    return status


def _download_kernel_output(kernel_id: str, dest_path: Path, expected_count: int = 0) -> None:
    """
    Download output from a single kernel using SelectiveDownloader.
    """
    logging.info(f"Downloading output from kernel: {kernel_id} to {dest_path}")
    downloader = SelectiveDownloader(kernel_id, dest=str(dest_path))
    success = downloader.download_images(
        expected_image_count=expected_count,
        stable_count_patience=4
    )
    if not success:
        logging.warning(f"Selective download from {kernel_id} reported failure or timed out.")


def run_parallel_pipeline(
    dest_path: Path,
    prompts_list: List[str],
    notebook: Path,
    kernel_path: Path,
    wait_timeout: int = None,
    retry_interval: int = None,
    polling_interval: int = None,
    accelerator: str = None,
    **deploy_kwargs
) -> None:
    """
    Run parallel image generation pipeline with split prompts.
    
    Splits prompts into two batches and deploys them to separate kernels.
    Polls both kernels in parallel, then downloads and merges results.
    
    Args:
        dest_path: Output destination path
        prompts_list: Full list of prompts
        notebook: Notebook to use
        kernel_path: Kernel config directory
        wait_timeout: Maximum wait time in minutes for GPU availability
        retry_interval: Interval in seconds between retries
        polling_interval: Interval in seconds between status polls
        accelerator: Kaggle accelerator type
        **deploy_kwargs: Additional arguments for deploy.run
    """
    # Load config for defaults if not provided
    if wait_timeout is None or retry_interval is None or polling_interval is None:
        config = load_kaggle_config()
        if wait_timeout is None:
            wait_timeout = config.get("deployment_timeout_minutes", 30)
        if retry_interval is None:
            retry_interval = config.get("retry_interval_seconds", 60)
        if polling_interval is None:
            polling_interval = config.get("polling_interval_seconds", 60)

    # Resolve deployment IDs
    base_kernel_id = deploy_kwargs.get("base_kernel_id", "leventecsibi/stable-diffusion-batch-generator")
    deployment1_kernel_id, deployment2_kernel_id = get_deployment_ids(base_kernel_id)

    # Split prompts
    first_batch, second_batch = split_prompts(prompts_list)
    logging.info(f"Splitting {len(prompts_list)} prompts across 2 kernels:")
    logging.info(f"  Kernel 1 ({deployment1_kernel_id}): {len(first_batch)} prompts")
    for idx, p in enumerate(first_batch):
        logging.info(f"    - [{idx+1}] {p}")
    logging.info(f"  Kernel 2 ({deployment2_kernel_id}): {len(second_batch)} prompts")
    for idx, p in enumerate(second_batch):
        logging.info(f"    - [{idx+1}] {p}")
    
    try:
        # Deploy both kernels
        logging.info("="*80)
        logging.info("PARALLEL DEPLOYMENT: Deploying to 2 Kaggle kernels")
        logging.info("="*80)
        
        # Deploy deployment1 kernel first (no offset)
        _deploy_single_kernel(
            prompts_list=first_batch,
            notebook=notebook,
            kernel_path=kernel_path,
            kernel_id=deployment1_kernel_id,
            deploy_kwargs={**deploy_kwargs, "wait_timeout": wait_timeout, "retry_interval": retry_interval},
            accelerator=accelerator,
            index_offset=0
        )
        
        # Wait before deploying deployment2 to avoid API conflicts
        logging.info("Waiting 15 seconds before deploying deployment2 kernel...")
        time.sleep(15)
        
        # Deploy deployment2 kernel (with offset = size of first batch)
        _deploy_single_kernel(
            prompts_list=second_batch,
            notebook=notebook,
            kernel_path=kernel_path,
            kernel_id=deployment2_kernel_id,
            deploy_kwargs={**deploy_kwargs, "wait_timeout": wait_timeout, "retry_interval": retry_interval},
            accelerator=accelerator,
            index_offset=len(first_batch)
        )
        
        logging.info("="*80)
        logging.info("Both kernels deployed! Waiting 30 seconds before polling...")
        logging.info("="*80)
        
        time.sleep(30)
        
        # Poll both kernels in parallel using threads
        statuses = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(_poll_kernel, deployment1_kernel_id, poll_interval=polling_interval): deployment1_kernel_id,
                executor.submit(_poll_kernel, deployment2_kernel_id, poll_interval=polling_interval): deployment2_kernel_id
            }
            
            for future in as_completed(futures):
                kernel_id = futures[future]
                try:
                    status = future.result()
                    statuses[kernel_id] = status
                except Exception as e:
                    logging.error(f"Error polling kernel {kernel_id}: {e}")
                    statuses[kernel_id] = f"error: {e}"
        
        # Log status summary
        logging.info("="*80)
        logging.info("POLLING COMPLETE - Status Summary:")
        for kid, status in statuses.items():
            logging.info(f"  {kid}: {status}")
        logging.info("="*80)
        
        # Check for errors
        errors = [kid for kid, status in statuses.items() if "error" in status.lower()]
        if errors:
            logging.error(f"The following kernels failed: {errors}")
            # Identify first error message
            err_kid = errors[0]
            manager = JobManager(err_kid)
            try:
                logs = manager.get_logs()
                msg = logs.strip() if logs else f"no logs available (status: {statuses[err_kid]})"
            except Exception as e:
                msg = f"could not retrieve logs: {e} (status: {statuses[err_kid]})"
                
            raise RuntimeError(f"Parallel kernel execution failed: {err_kid}\n{msg}")
        
        logging.info("="*80)
        logging.info("Both kernels completed! Downloading outputs...")
        logging.info("="*80)
        
        # Create temporary directories for downloads
        deployment1_download_path = dest_path / "_deployment1_temp"
        deployment2_download_path = dest_path / "_deployment2_temp"
        deployment1_download_path.mkdir(parents=True, exist_ok=True)
        deployment2_download_path.mkdir(parents=True, exist_ok=True)
        
        # Download from both kernels sequentially
        download_errors = []
        
        logging.info(f"Downloading from deployment1 kernel: {deployment1_kernel_id}")
        try:
            _download_kernel_output(deployment1_kernel_id, deployment1_download_path, expected_count=len(first_batch))
        except Exception as e:
            logging.warning(f"Failed to download from deployment1: {e}")
            download_errors.append(("deployment1", str(e)))
        
        logging.info(f"Downloading from deployment2 kernel: {deployment2_kernel_id}")
        try:
            _download_kernel_output(deployment2_kernel_id, deployment2_download_path, expected_count=len(second_batch))
        except Exception as e:
            logging.warning(f"Failed to download from deployment2: {e}")
            download_errors.append(("deployment2", str(e)))
        
        # If both downloads failed, raise error
        if len(download_errors) >= 2:
            logging.error(f"Both kernel downloads failed: {download_errors}")
            raise RuntimeError(f"All kernel downloads failed: {download_errors}")
        
        if len(download_errors) == 1:
            logging.info(f"One kernel download failed, but continuing with available output: {download_errors}")
        
        # Merge results into final images folder
        final_images_path = dest_path / "images"
        final_images_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = (".png", ".jpg", ".jpeg")
        unique_images = {} # Map filename -> Path
        
        for temp_path in [deployment1_download_path, deployment2_download_path]:
            if not temp_path.exists(): continue
            for image_file in temp_path.rglob("*"):
                # ONLY collect files that start with 'gen_' or 'generated_' to avoid pulling stale artifacts
                is_image = image_file.is_file() and image_file.suffix.lower() in image_extensions
                is_valid_prefix = image_file.name.startswith("gen_") or image_file.name.startswith("generated_")
                if is_image and is_valid_prefix:
                    # Deduplicate by filename
                    if image_file.name not in unique_images:
                        unique_images[image_file.name] = image_file
        
        image_count = 0
        for name, source_path in unique_images.items():
            target = final_images_path / name
            # Handle name collisions if they occur across runs, but for this run we deduplicated
            counter = 1
            while target.exists():
                target = final_images_path / f"{source_path.stem}_{counter}{source_path.suffix}"
                counter += 1
            shutil.move(str(source_path), str(target))
            image_count += 1
            logging.info(f"  Collected: {target.name}")
        
        # Check if we got the expected number of images
        expected_images = len(prompts_list)
        if image_count == 0:
            logging.error("No images were downloaded from any kernel")
            raise RuntimeError("Image generation failed: no images downloaded from any kernel")
        elif image_count != expected_images:
            logging.error(f"Incomplete image generation: expected {expected_images} images but got {image_count}")
            raise RuntimeError(
                f"Image generation incomplete: expected {expected_images} images "
                f"but only got {image_count}. Some prompts failed to generate images."
            )
        
        # Clean up temporary directories
        shutil.rmtree(deployment1_download_path, ignore_errors=True)
        shutil.rmtree(deployment2_download_path, ignore_errors=True)
        
        logging.info("="*80)
        logging.info(f"PARALLEL PIPELINE COMPLETE!")
        logging.info(f"Total images: {image_count}")
        logging.info(f"Output location: {final_images_path}")
        logging.info("="*80)
        
    finally:
        # Temp directories are already cleaned up or handled
        pass
