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

from imggenhub.kaggle.core import deploy, download
from imggenhub.kaggle.utils import poll_status
from imggenhub.kaggle.utils.config_loader import load_kaggle_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Kernel IDs for parallel execution
DEPLOYMENT1_KERNEL_ID = "leventecsibi/stable-diffusion-batch-generator-deployment-1"
DEPLOYMENT2_KERNEL_ID = "leventecsibi/stable-diffusion-batch-generator-deployment-2"

# Threshold for parallel deployment
PARALLEL_THRESHOLD = 4


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


def _create_parallel_kernel_dir(kernel_path: Path, notebook: Path, kernel_id: str, title: str) -> Path:
    """
    Create a temporary kernel directory for parallel deployment.
    
    Args:
        kernel_path: Original kernel config directory
        notebook: Notebook path to use
        kernel_id: Target Kaggle kernel ID
        title: Target Kaggle kernel title
        
    Returns:
        Path to temporary kernel directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=f"kaggle_{kernel_id.split('/')[-1]}_"))
    
    # Copy notebook
    notebook_name = notebook.name
    source_notebook = kernel_path / notebook_name
    if not source_notebook.exists():
        # Fallback to absolute path or search in notebooks directory
        if notebook.exists():
            source_notebook = notebook
        else:
            # Check src/imggenhub/kaggle/notebooks/
            pkg_root = Path(__file__).parent.parent
            source_notebook = pkg_root / "notebooks" / notebook_name

    if not source_notebook.exists():
        raise FileNotFoundError(f"Source notebook not found: {notebook_name}")
        
    shutil.copy2(source_notebook, temp_dir / notebook_name)
    
    # Load original metadata
    metadata_path = kernel_path / "kernel-metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        # Default metadata fallback
        metadata = {
            "id": kernel_id,
            "title": title,
            "code_file": notebook_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "model_sources": [],
            "metadata": {
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3",
                    "language": "python"
                }
            }
        }

    # Override for this specific deployment
    metadata["id"] = kernel_id
    metadata["title"] = title
    metadata["code_file"] = notebook_name
    
    with open(temp_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    return temp_dir


def _deploy_single_kernel(
    prompts_list: List[str],
    notebook: Path,
    kernel_path: Path,
    kernel_id: str,
    deploy_kwargs: Dict[str, Any]
) -> str:
    """
    Deploy a single kernel with given prompts.
    
    Args:
        prompts_list: Prompts for this kernel
        notebook: Notebook path
        kernel_path: Kernel config directory
        kernel_id: Kernel identifier
        deploy_kwargs: Additional kwargs for deploy.run
        
    Returns:
        Kernel ID that was deployed
    """
    logging.info(f"Deploying {len(prompts_list)} prompts to kernel: {kernel_id}")
    
    # Deploy using standard deploy module
    deploy.run(
        prompts_list=prompts_list,
        notebook=notebook,
        kernel_path=kernel_path,
        kernel_id=kernel_id,
        **deploy_kwargs
    )
    logging.info(f"Successfully deployed kernel: {kernel_id}")
    return kernel_id


def _poll_kernel(kernel_id: str, max_wait: int = 1800, poll_interval: int = None) -> str:
    """
    Poll a single kernel until completion with timeout.
    
    Args:
        kernel_id: Kernel to poll
        max_wait: Maximum wait time in seconds (default 30 min)
        poll_interval: Interval in seconds between status polls
        
    Returns:
        Final status
    """
    if poll_interval is None:
        config = load_kaggle_config()
        poll_interval = config.get("polling_interval_seconds", 60)
    
    logging.info(f"Starting poll for kernel: {kernel_id}")
    start_time = time.time()
    
    while (time.time() - start_time) < max_wait:
        try:
            status = poll_status.run(kernel_id=kernel_id, poll_interval=poll_interval)
            logging.info(f"Kernel {kernel_id} finished with status: {status}")
            return status
        except Exception as e:
            # If polling fails, wait and retry
            logging.warning(f"Poll error for {kernel_id}: {e}. Retrying...")
            time.sleep(poll_interval)
    
    logging.error(f"Kernel {kernel_id} timed out after {max_wait}s")
    return "timeout"


def _download_kernel_output(kernel_id: str, dest_path: Path) -> None:
    """
    Download output from a single kernel.
    
    Args:
        kernel_id: Kernel to download from
        dest_path: Destination directory
        
    Raises:
        RuntimeError: If download fails
    """
    logging.info(f"Downloading output from kernel: {kernel_id}")
    success = download.run(dest=dest_path, kernel_id=kernel_id)
    if not success:
        raise RuntimeError(f"Failed to download images from kernel {kernel_id}")


def run_parallel_pipeline(
    dest_path: Path,
    prompts_list: List[str],
    notebook: Path,
    kernel_path: Path,
    wait_timeout: int = None,
    retry_interval: int = None,
    polling_interval: int = None,
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

    # Split prompts
    first_batch, second_batch = split_prompts(prompts_list)
    logging.info(f"Splitting {len(prompts_list)} prompts: {len(first_batch)} (deployment1) + {len(second_batch)} (deployment2)")
    
    # Create temporary kernel directories for both deployments
    deployment1_kernel_path = _create_parallel_kernel_dir(
        kernel_path, notebook, DEPLOYMENT1_KERNEL_ID, "Stable Diffusion Batch Generator Deployment 1"
    )
    deployment2_kernel_path = _create_parallel_kernel_dir(
        kernel_path, notebook, DEPLOYMENT2_KERNEL_ID, "Stable Diffusion Batch Generator Deployment 2"
    )
    logging.info(f"Created temp deployment dirs:\n  1: {deployment1_kernel_path}\n  2: {deployment2_kernel_path}")
    
    try:
        # Deploy both kernels
        logging.info("="*80)
        logging.info("PARALLEL DEPLOYMENT: Deploying to 2 Kaggle kernels")
        logging.info("="*80)
        
        # Deploy deployment1 kernel first
        _deploy_single_kernel(
            prompts_list=first_batch,
            notebook=deployment1_kernel_path / notebook.name,
            kernel_path=deployment1_kernel_path,
            kernel_id=DEPLOYMENT1_KERNEL_ID,
            deploy_kwargs={**deploy_kwargs, "wait_timeout": wait_timeout, "retry_interval": retry_interval, "index_offset": 0}
        )
        
        # Wait longer before deploying deployment2 to avoid API conflicts or resource race
        logging.info("Waiting 30 seconds before deploying deployment2 kernel...")
        time.sleep(30)
        
        # Deploy deployment2 kernel
        _deploy_single_kernel(
            prompts_list=second_batch,
            notebook=deployment2_kernel_path / notebook.name,
            kernel_path=deployment2_kernel_path,
            kernel_id=DEPLOYMENT2_KERNEL_ID,
            deploy_kwargs={**deploy_kwargs, "wait_timeout": wait_timeout, "retry_interval": retry_interval, "index_offset": len(first_batch)}
        )
        
        logging.info("="*80)
        logging.info("Both kernels deployed! Polling for completion...")
        logging.info("="*80)
        
        # Poll both kernels in parallel using threads
        statuses = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(_poll_kernel, DEPLOYMENT1_KERNEL_ID, poll_interval=polling_interval): DEPLOYMENT1_KERNEL_ID,
                executor.submit(_poll_kernel, DEPLOYMENT2_KERNEL_ID, poll_interval=polling_interval): DEPLOYMENT2_KERNEL_ID
            }
            
            for future in as_completed(futures):
                kernel_id = futures[future]
                try:
                    status = future.result()
                    statuses[kernel_id] = status
                except Exception as e:
                    logging.error(f"Error polling kernel {kernel_id}: {e}")
                    statuses[kernel_id] = "error"
        
        # Log status summary
        logging.info("="*80)
        logging.info("POLLING COMPLETE - Status Summary:")
        for kid, status in statuses.items():
            logging.info(f"  {kid}: {status}")
        logging.info("="*80)
        
        # Check for errors (only hard errors, not unknown/timeout which might still produce output)
        errors = [kid for kid, status in statuses.items() 
                  if "error" in status.lower()]
        if errors:
            logging.error(f"The following kernels failed: {errors}")
            raise RuntimeError(f"Kernel execution failed for: {errors}")
        
        logging.info("="*80)
        logging.info("Both kernels completed! Downloading outputs...")
        logging.info("="*80)
        
        # Create temporary directories for downloads
        deployment1_download_path = dest_path / "_deployment1_temp"
        deployment2_download_path = dest_path / "_deployment2_temp"
        deployment1_download_path.mkdir(parents=True, exist_ok=True)
        deployment2_download_path.mkdir(parents=True, exist_ok=True)
        
        # Download from both kernels sequentially, handling failures gracefully
        download_errors = []
        
        logging.info(f"Downloading from deployment1 kernel: {DEPLOYMENT1_KERNEL_ID}")
        try:
            _download_kernel_output(DEPLOYMENT1_KERNEL_ID, deployment1_download_path)
        except Exception as e:
            logging.warning(f"Failed to download from deployment1: {e}")
            download_errors.append(("deployment1", str(e)))
        
        logging.info(f"Downloading from deployment2 kernel: {DEPLOYMENT2_KERNEL_ID}")
        try:
            _download_kernel_output(DEPLOYMENT2_KERNEL_ID, deployment2_download_path)
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
        image_count = 0
        
        for temp_path in [deployment1_download_path, deployment2_download_path]:
            for image_file in temp_path.rglob("*"):
                if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                    target = final_images_path / image_file.name
                    # Handle name collisions
                    counter = 1
                    while target.exists():
                        target = final_images_path / f"{image_file.stem}_{counter}{image_file.suffix}"
                        counter += 1
                    shutil.move(str(image_file), str(target))
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
        # Always clean up temporary kernel directories
        for path in [deployment1_kernel_path, deployment2_kernel_path]:
            if path and path.exists():
                shutil.rmtree(path, ignore_errors=True)
                logging.info(f"Cleaned up temp directory: {path}")
